import torch
from typing import Union
import numpy as np
# from nnsight import LanguageModel
from transformer_lens import HookedTransformer
import torch.nn.functional as F
import random
import os
import json
import gc
import pickle
import einops

from wrapper import get_accessor_config, get_model_specs, ModelAccessor

def free_unused_cuda_memory():
    """Free unused cuda memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("not using cuda")
    gc.collect()

def compute_top_k_accuracy(target_token_ranks, k=10) -> float:
    """
    Evaluation to compute topk accuracy.

    Parameters:
    target_token_ranks: the distribution of output token ranks
    k: how many tokens we're looking at (top K)

    Return:
    The accuracy of the token in the top k of tokens
    """

    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks) 

def group_ranked_locations_by_layer(layer_module_ranking:list
) -> dict[int, list[int]]:
    """
    Groups indices together by layer index where the module indices represent a particular module component (attn head, neuron, etc.) in a layer.

    Parameters:
    layer_module_ranking: a list of tuples/lists, where the first two indices of the tuple/list are a layer index (int) and a module index (int) 
        (e.g. [(layer, module_index, score), ...]) 

    Returns:
    grouped_dict: a dict where all module elements of a particular layer are grouped under its layer key. Useful for interventions
    """
    grouped_dict = {}
    for layer_index, module_index in [x[:2] for x in layer_module_ranking]:
        if layer_index not in grouped_dict:
            grouped_dict[layer_index] = [module_index]
        else:
            grouped_dict[layer_index].append(module_index)

    return dict(sorted(grouped_dict.items()))

def compute_token_rank_prob(prob_dist, target_ids:Union[torch.Tensor, int, list[int]]):
    """
    Computes the rank of a token and its probability given a probability distribution

    Args:
    prob_dist (torch.Tensor): logits distribution (batch_size, d_vocab)
    target_ids (torch.Tensor, int, list[int]), a list of integers or a tensor of shape (batch_size,)

    """
    if len(prob_dist.shape) == 1:
        prob_dist = prob_dist.unsqueeze(0)
    if isinstance(target_ids, int):
        target_ids = torch.Tensor([target_ids])
    elif isinstance(target_ids, list):
        target_ids = torch.Tensor(target_ids)
    assert len(target_ids) == len(prob_dist)
    
    values, indices = prob_dist.softmax(-1).topk(k=prob_dist.shape[-1])
    where_result = torch.where(indices == torch.Tensor(target_ids)[:,None].to(indices.device))
    token_ranks = where_result[1]
    token_probs = values[where_result]

    return token_ranks, token_probs

def compute_token_rank_logit(prob_dist, target_ids:Union[torch.Tensor, int, list[int]]):
    """
    Computes the rank of a token and its probability given a probability distribution

    Args:
    prob_dist (torch.Tensor): logits distribution (batch_size, d_vocab)
    target_ids (torch.Tensor, int, list[int]), a list of integers or a tensor of shape (batch_size,)

    """
    if len(prob_dist.shape) == 1:
        prob_dist = prob_dist.unsqueeze(0)
    if isinstance(target_ids, int):
        target_ids = torch.Tensor([target_ids])
    elif isinstance(target_ids, list):
        target_ids = torch.Tensor(target_ids)
    assert len(target_ids) == len(prob_dist)
    
    values, indices = prob_dist.topk(k=prob_dist.shape[-1])
    where_result = torch.where(indices == torch.Tensor(target_ids)[:,None].to(indices.device))
    token_ranks = where_result[1]
    token_logits = values[where_result]

    return token_ranks, token_logits

def decode_vec(vec, model:HookedTransformer, topk=10, apply_ln=True) -> list:
  """
  Takes a vector, unembeds it, and returns the topk tokens
  IMPORTANT: When using this, make sure to set apply_ln properly. This
    applies the final layernorm before unembedding (model.ln_final) if set to true
  Args:
    vec: a tensor (d_model,)
    model: a HookedTransformer model instance. We use the unembedding matrix (W_U) from this
    
  Returns:
    topk_tokens: a list of the topk tokens as strings
  """

  if apply_ln:
    vec = model.ln_final(vec)
  
  logits = model.unembed(vec)
  prob = F.softmax(logits, dim=-1)
  topk_values, topk_indices = torch.topk(prob, topk)
  topk_tokens = [model.tokenizer.decode([i]) for i in topk_indices]
  return topk_tokens

def get_head_activation_ranks_probs_mean(
    model: HookedTransformer,
    prompts: list[str],
    answers: list[str],
    batch_size: int = 8,
    remote: bool = False, 
): 
    """
    Get the mean activation and output (ranks and probs) of the heads for a set of prompts.
    Return:
    head_activation_all_tensor: (n_layers, n_heads, d_head)
    ranks_all: (total_samples, 
    probs_all: (total_samples, 
    """
    
    total_samples = len(prompts)
    # Get tokens
    all_tokens = model.tokenizer(
        prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).input_ids
    token_len = all_tokens.shape[1]

    # Get token IDs for answers
    answer_tokens = [model.tokenizer(ans, add_special_tokens=False)["input_ids"][0] for ans in answers]

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]


    # Initialize lists to store results 
    head_activation_all = []
    ranks_all = []
    probs_all = []
    
    # Process in batches
    #for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        tokens = all_tokens[batch_start:batch_end]
        batch_answer_tokens = answer_tokens[batch_start:batch_end]
        attn_output_layers = []

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(tokens): 
                for layer in range(n_layers):
                    # Get attention output
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_output_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    attn_output_layers.append(attn_output_reshaped.save())
                
                logits = model.output.logits[:,-1].save()

        # Retrieve the actual attention output tensors from proxies 
        attn_output_layers = [a.value for a in attn_output_layers]
        # Stack along layers 
        assert attn_output_layers[0].shape == (current_batch_size, n_heads, d_head)
        assert len(attn_output_layers) == n_layers
        attn_output_layers_tensor = torch.stack(attn_output_layers) # (n_layers, current_batch_size, n_heads, d_head)
        assert attn_output_layers_tensor.shape == (n_layers, current_batch_size, n_heads, d_head)
        head_activation_all.append(attn_output_layers_tensor)

        # Get ranks and probs 
        batch_ranks, batch_probs = compute_token_rank_prob(logits, batch_answer_tokens)
        ranks_all.append(batch_ranks)
        probs_all.append(batch_probs)

    # Activation 
    head_activation_all = torch.cat(head_activation_all, dim=1) # Stack along batches 
    assert head_activation_all.shape == (n_layers, total_samples, n_heads, d_head)
    # Get mean across samples 
    head_activation_all_tensor = torch.mean(head_activation_all, dim=1) # (n_layers, n_heads, d_head)
    assert head_activation_all_tensor.shape == (n_layers, n_heads, d_head)

    # Ranks and probs 
    ranks_all = torch.cat(ranks_all) 
    probs_all = torch.cat(probs_all)

    return head_activation_all_tensor, ranks_all, probs_all

def get_head_activation_mean(
    model: HookedTransformer,
    prompts: list[str],
    batch_size: int = 8,
    remote: bool = False, 
): 
    """
    Get the mean activation of the heads for a set of prompts.
    Return:
    head_activation_all_tensor: (n_layers, n_heads, d_head)
    ranks_all: (total_samples, 
    probs_all: (total_samples, 
    """
    
    total_samples = len(prompts)
    # Get tokens
    all_tokens = model.tokenizer(
        prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).input_ids
    token_len = all_tokens.shape[1]

    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]


    # Initialize lists to store results 
    head_activation_all = []
    
    # Process in batches
    #for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        # Get current batch tokens from pre-tokenized data
        tokens = all_tokens[batch_start:batch_end]
        attn_output_layers = []

        # Get accessor for the model
        accessor_config = get_accessor_config(model)
        accessor = ModelAccessor(model, accessor_config)
        with accessor.trace(remote=remote) as tracer:
            with tracer.invoke(tokens): 
                for layer in range(n_layers):
                    # Get attention output
                    attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                    attn_output_reshaped = attn_output.reshape(current_batch_size, n_heads, d_head)
                    attn_output_layers.append(attn_output_reshaped.save())
                
                #logits = model.output.logits[:,-1].save()

        # Retrieve the actual attention output tensors from proxies 
        attn_output_layers = [a.value for a in attn_output_layers]
        # Stack along layers 
        assert attn_output_layers[0].shape == (current_batch_size, n_heads, d_head)
        assert len(attn_output_layers) == n_layers
        attn_output_layers_tensor = torch.stack(attn_output_layers) # (n_layers, current_batch_size, n_heads, d_head)
        assert attn_output_layers_tensor.shape == (n_layers, current_batch_size, n_heads, d_head)
        head_activation_all.append(attn_output_layers_tensor)

    # Activation 
    head_activation_all = torch.cat(head_activation_all, dim=1) # Stack along batches 
    assert head_activation_all.shape == (n_layers, total_samples, n_heads, d_head)
    # Get mean across samples 
    head_activation_all_tensor = torch.mean(head_activation_all, dim=1) # (n_layers, n_heads, d_head)
    assert head_activation_all_tensor.shape == (n_layers, n_heads, d_head)

    return head_activation_all_tensor


def sample_random_heads(model: HookedTransformer,
    num_samples: int, seed: int = None
) -> list[tuple[int, int, float]]:
    """
    Samples a list of tuples, where each tuple contains (layer, head, score).

    The first integer in each tuple is sampled from the range [0, n_layers-1] (inclusive).
    The second integer in each tuple is sampled from the range [0, n_heads-1] (inclusive).

    Args:
        n_layers: The upper bound (inclusive) for the first integer in the tuple.
        n_heads: The upper bound (inclusive) for the second integer in the tuple.
        num_samples: The number of tuples to sample.
        seed: An optional integer seed for reproducibility. If None, a random seed will be used.

    Returns:
        A list of sampled tuples.
    """
    spec = get_model_specs(model)
    n_layers, n_heads = spec["n_layers"], spec["n_heads"]

    if not all(isinstance(arg, int) and arg >= 0 for arg in [n_layers, n_heads, num_samples]):
        raise ValueError("n_layers, n_heads, and num_samples must be non-negative integers.")
    if num_samples == 0:
        return []

    if seed is not None:
        random.seed(seed)

    sampled_list = []
    for _ in range(num_samples):
        # Randomly select the first item (layer index)
        layer_index = random.randint(0, n_layers - 1)
        # Randomly select the second item (head index)
        head_index = random.randint(0, n_heads - 1)
        score = random.random()
        sampled_list.append((layer_index, head_index, score))

    return sampled_list

def check_intersection(dict1, dict2):
    """
    dict1: a dictionary keyed by layer (interger), value is a list of components 
    dict2: a dictionary keyed by layer (integer), value is a list of components 
    intersection_set: a dic keyed by layer (integer), value is a list of shared components 
    """
    intersection_set = {}
    total=0
    max_layer = max(list([int(x) for x in dict1.keys()]) + list([int(x) for x in dict2.keys()]))
    for k in range(max_layer + 1):
        if k in list(dict1.keys()) and k in list(dict2.keys()):
            intersection_set[k] = list(set(dict1[k]).intersection(dict2[k]))

            total+=len(intersection_set[k])
            
    return intersection_set, total 

def get_ranked_heads_from_MAPS_score(
    target_task:str=None, other_task:str=None,
    prompt_type:str=None, prompt_template_index:int=None,
    correct_incorrect:str="correct", model_name:str=None,
    save_root:str=None, save_path:str=None, 
    component_type:str=None, n_match:int=None,
):
    if save_path is None:
        save_path = os.path.join(save_root, model_name, target_task,
            "Heads", "MAPS", f"{component_type}_across_tasks",)
        
    with open(os.path.join(save_path, 
        f"{target_task}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_incorrect}.pkl"
    ), "rb") as f:
        result = pickle.load(f)
        
    maps_scores_2_analyze = result[n_match][other_task].mean(axis=0)

    ranked_heads = rank_heads(maps_scores_2_analyze, threshold=None)

    return ranked_heads

def get_MAPS_head(
    model_name:str,
    target_task:str,
    other_task:str,
    k_or_n_match:int,
    component_type:str="Relation",
    MAPS_score_threshold:float=0.1,
    save_path:str=None,
    save_root:str=None,
):
    if save_path is None:
        save_path = os.path.join(save_root, model_name, target_task,
                "Heads", "MAPS", f"{component_type}_across_tasks",)
     
    with open(os.path.join(save_path, 
        f"{target_task}_MAPS_{component_type}_heads_across_tasks.pkl"
    ), "rb") as f:
        result = pickle.load(f)
        
    maps_scores_2_analyze = result[k_or_n_match][other_task].mean(axis=0)

    indices_array = np.argwhere(maps_scores_2_analyze >= MAPS_score_threshold)
    list_of_heads = [(int(layer), int(head)) for (layer, head) in indices_array]
        
    return list_of_heads

def get_task_heads_list(
    target_task:str=None, other_task:str=None,
    prompt_type:str=None, prompt_template_index:int=None,
    n_match:int=None, MAPS_score_threshold:float=None, 
    correct_incorrect:str="correct", model_name:str=None,
    save_root:str=None, save_path:str=None, 
    component_type:str="Relation",

):
    if save_path is None:
        save_path = os.path.join(save_root, model_name, target_task,
            "Heads", "MAPS", f"{component_type}_across_tasks",)
       
    with open(os.path.join(save_path, 
        f"{target_task}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_incorrect}.pkl"
    ), "rb") as f:
        result = pickle.load(f)
        
    maps_scores_2_analyze = result[n_match][other_task].mean(axis=0)

    indices_array = np.argwhere(maps_scores_2_analyze >= MAPS_score_threshold)
    list_of_heads = [(int(layer), int(head)) for (layer, head) in indices_array]

    return list_of_heads

def get_task_heads_list_k(
    target_task:str=None, other_task:str=None,
    prompt_type:str=None, prompt_template_index:int=None,
    k:int=None, MAPS_score_threshold:float=None, 
    correct_incorrect:str="correct", model_name:str=None,
    save_root:str=None, save_path:str=None, 
    component_type:str="Relation",

):
    if save_path is None:
        save_path = os.path.join(save_root, model_name, target_task,
            "Heads", "MAPS", f"{component_type}_across_tasks_vary_k",)
       
    with open(os.path.join(save_path, 
        f"{target_task}_MAPS_{component_type}_heads_across_tasks_{prompt_type}_{prompt_template_index}_{correct_incorrect}.pkl"
    ), "rb") as f:
        result = pickle.load(f)
        
    maps_scores_2_analyze = result[k][other_task].mean(axis=0)

    indices_array = np.argwhere(maps_scores_2_analyze >= MAPS_score_threshold)
    list_of_heads = [(int(layer), int(head)) for (layer, head) in indices_array]

    return list_of_heads

def rank_heads(scores_tensor, threshold=None):
    """
    Ranks heads by their score from a 2D tensor of shape [layer, head].

    Args:
        scores_tensor (np.ndarray): A 2D numpy array where the value at
                                    scores_tensor[i, j] is the score for
                                    layer i and head j.

    Returns:
        list: A list of tuples (layer, head, score), sorted in descending
              order of the score.
    """
    # Get the dimensions of the tensor
    num_layers, num_heads = scores_tensor.shape

    # Create an empty list to store the (layer, head, score) tuples
    ranked_head_list = []

    # Iterate through each layer and head to build the list
    for layer in range(num_layers):
        for head in range(num_heads):
            score = scores_tensor[layer, head]
            # If a threshold is provided, only add the head if its score is greater than the threshold
            if threshold is None or score >= threshold:
                ranked_head_list.append((layer, head, score))
            

    # reverse=True ensures the sorting is from high to low.
    ranked_head_list.sort(key=lambda item: item[2], reverse=True)

    return ranked_head_list

def load_multi_MAPS_top_shared_heads(result_root_path:str, model_name:str, 
    head_type:str, dataset_name:str, n_top_heads_ini:int = None,
    n_prompts_EP:int = 100, n_prompts_IP:int = 100, n_shared_heads:list[int] = None,
) -> dict[int, dict[int, list[int]]]:
    """
    Find n shared top heads (in a dict) by scanning through n top heads from EP & IP 
    Args:

        head_type: "Attribute_Extraction_heads" or "Relation_Propagation_heads"
        k: top k accuracy for MAPS score
        n_top_heads_ini: maximum number of top heads from EP & IP to find intersection 
        n_shared_heads: a list of number of shared top heads to find 

    Returns:
        top_shared_heads: a dict of dicts, outer dict keyed by number of shared heads, 
            inner dict keyed by layer (integer), value is a list of shared components
    """
    model_name = model_name.split('/')[-1]
    # Load MAPS ranked top heads json file 
    ranked_components_file_path_EP = os.path.join(result_root_path,
        model_name, dataset_name, "Heads", "MAPS", 
        head_type, "EP",
    )
    ranked_components_file_path_IP = os.path.join(result_root_path,
        model_name, dataset_name, "Heads", "MAPS", 
        head_type, "IP",
    )
    with open(os.path.join(ranked_components_file_path_EP, 
        f"MAPS_ranked_top_heads_{n_prompts_EP}prompts.json"), 'r'
    ) as results_file:        
        components_dict_EP = json.load(results_file)
    with open(os.path.join(ranked_components_file_path_IP, 
        f"MAPS_ranked_top_heads_{n_prompts_IP}prompts.json"), 'r'
    ) as results_file:        
        components_dict_IP = json.load(results_file)
    
    # Iterate over number of top heads from EP & IP to find intersection  
    top_shared_heads = {}
    for n_top_heads in range(1, n_top_heads_ini):
        grouped_heads_IP = group_ranked_locations_by_layer(components_dict_IP["top_components"][:n_top_heads])
        grouped_heads_EP = group_ranked_locations_by_layer(components_dict_EP["top_components"][:n_top_heads])

        intersection_set, intersection_count = check_intersection(grouped_heads_IP, grouped_heads_EP)
        if intersection_count in n_shared_heads:
            #print(f" {n_top_heads} top heads -> {intersection_count} shared heads: {intersection_set} ")
            top_shared_heads[intersection_count] = intersection_set

    #assert len(top_shared_heads) == len(n_shared_heads)
    return top_shared_heads



def load_MAPS_ranked_top_heads(result_root_path:str, model_name:str, 
    head_type:str, k:int, dataset_name:str, n_top_heads:int = None,
) -> dict[int, list[int]]:
    """
    #TODO: consider get heads with score above certain threshold rather than top n heads from IP & EP 
    Args:

        head_type: "Attribute_Extraction_heads" or "Relation_Propagation_heads"
        k: top k accuracy for MAPS score
        n_top_heads: number of top heads from EP & IP to find intersection  
    Returns:
        intersection_set: a dict keyed by layer (integer), value is a list of shared components
    """
    model_name = model_name.split('/')[-1]
    # Load MAPS ranked top heads json file 
    ranked_components_file_path_EP = os.path.join(result_root_path,
        model_name, dataset_name, "Heads", "MAPS", 
        head_type, "EP",
    )
    ranked_components_file_path_IP = os.path.join(result_root_path,
        model_name, dataset_name, "Heads", "MAPS", 
        head_type, "IP",
    )
    with open(os.path.join(ranked_components_file_path_EP, "MAPS_ranked_top_heads.json"), 'r') as results_file:        
        components_dict_EP = json.load(results_file)
    with open(os.path.join(ranked_components_file_path_IP, "MAPS_ranked_top_heads.json"), 'r') as results_file:        
        components_dict_IP = json.load(results_file)

    grouped_heads_IP = group_ranked_locations_by_layer(components_dict_IP["top_components"][:n_top_heads])
    grouped_heads_EP = group_ranked_locations_by_layer(components_dict_EP["top_components"][:n_top_heads])

    intersection_set, intersection_count = check_intersection(grouped_heads_IP, grouped_heads_EP)

    print(f"Num of shared heads among top {n_top_heads} {head_type} in IP & EP: {intersection_count} heads")
    return intersection_set

def load_MAPS_grouped_top_heads(result_root_path:str, model_name:str, 
    head_type:str, n_top_heads=None,
    threshold:float = None, n_max_heads:int = None,
    prompts_type:str = "EP", d_name:str = None,
) -> dict[int, list[int]]:
    """
    Load heads with top mapping scores above a threshold and at most n_max_heads
    or Load top n_top_heads heads
    
    Args:
        head_type: "Attribute_Extraction_heads" or "Relation_Propagation_heads"
        k: top k accuracy for MAPS score
        n_top_heads: number of top heads 
        If using threshold, n_top_heads is ignored
            threshold: float, threshold for MAPS score
            n_max_heads: int, maximum number of heads

    Returns:
        top_heads: a dict keyed by layer (integer), value is a list of component index
    """
    if d_name == "capitalize":
        if prompts_type == "IP":
            n_prompts = 51
        else:
            n_prompts = 100
    elif d_name == "prev_item":
        if prompts_type == "IP":
            n_prompts = 46
        else:
            n_prompts = 100
    elif d_name == "synonym":
        if prompts_type == "IP":
            n_prompts = 58
        else:
            n_prompts = 81
    else: 
        if prompts_type == "IP":
            n_prompts = 100
        else:
            n_prompts = 100
    
    # Load MAPS ranked top heads json file 
    ranked_components_file_path = os.path.join(result_root_path,
        model_name, d_name, "Heads", "MAPS", 
        head_type, prompts_type,
    )
    if head_type == "Relation_Propagation_heads":
        file_name = f"MAPS_ranked_top_heads_{n_prompts}prompts.json"
    else:
        file_name = f"MAPS_ranked_top_heads.json"
    with open(os.path.join(ranked_components_file_path, 
        file_name), 'r'
    ) as results_file:        
        components_dict = json.load(results_file)
    
    if n_top_heads is not None and threshold is None and n_max_heads is None:
        ranked_heads = components_dict["top_components"][:n_top_heads]
        #print("num of top Mapping score heads: ", len(ranked_heads))
    elif threshold is not None and n_max_heads is not None and n_top_heads is None:
        ranked_heads = [item for item in components_dict["top_components"] if item[2] >= threshold]
        ranked_heads = ranked_heads[:n_max_heads]
        #print(f"num of top Mapping score heads above {threshold}: {len(ranked_heads)}")
    else:
        raise ValueError("Either n_top_heads or threshold and n_max_heads must be provided")
    
    grouped_heads = group_ranked_locations_by_layer(ranked_heads)
        
    return grouped_heads

def load_multi_top_other_task_recognition_heads(result_root_path:str=None, 
    prompts_type:str=None, d_name:str=None, n_heads_list:list[int] = None,
    act_patching_n_prompts:int = 40,
    model_name:str = None, n_heads:int = 32,
    n_top_execution_heads:int = 25,
    relation_head_threshold:float = 0.1, 
    n_max_other_task_recognition_heads:int = 25,
    universal_FV_heads:dict[int, list[int]] = None,
    print_process:bool = False,
) -> dict[int, dict[int, list[int]]]:
    """
    Start from top n heads from activation patching 
    Exclude top 25 execution heads from path ptching 
    Exclude relation propagation heads that have a mapping score > 0.1 
    Args:
        prompts_type: "EP" or "IP"
        n_heads_list: list of number of top heads from activation patching 
        n_heads: number of heads in the model 
        act_patching_n_prompts: number of prompts for calculating activation patching scores 
        n_top_execution_heads: number of top execution heads to exclude 
        relation_head_threshold: mapping score threshold for relation head 
        n_max_other_task_recognition_heads: maximum number of other task recognition heads to get 
    Returns:
        top_heads: a dict of dicts, outer dict keyed by number of heads, 
            inner dict keyed by layer (integer), value is a list of haead indices 
    """
    # Load top heads from activation patching 
    file_path = os.path.join(result_root_path, model_name, d_name, "Heads",
        "causal_mediation", "activation_patching", f"act_patching_ZS_{prompts_type}_{act_patching_n_prompts}samples.npy")
    act_score = np.load(file_path)
    ranked_heads = rank_heads(act_score)

    # Load top execution heads from path patching 
    execution_file_path = os.path.join(result_root_path, model_name, d_name, "Heads",
        "causal_mediation", "path_patching", f"component_to_logits_random_query_{prompts_type}.pt")
    execution_score = torch.load(execution_file_path)
    ranked_heads_execution = rank_heads(execution_score[:, :n_heads]) # the last one of the 2nd dim is the whole MLP layer for path patching result
    execution_heads_to_exclude = ranked_heads_execution[:n_top_execution_heads]

    # Load top relation heads from mapping score 
    if d_name == "capitalize":
        if prompts_type == "IP":
            MAPS_n_prompts = 51
        else:
            MAPS_n_prompts = 100
    elif d_name == "prev_item":
        if prompts_type == "IP":
            MAPS_n_prompts = 46
        else:
            MAPS_n_prompts = 100
    elif d_name == "synonym":
        if prompts_type == "IP":
            MAPS_n_prompts = 58
        else:
            MAPS_n_prompts = 81
    else: 
        if prompts_type == "IP":
            MAPS_n_prompts = 100
        else:
            MAPS_n_prompts = 100
    relation_head_file_path = os.path.join(result_root_path, model_name, d_name, "Heads",
        "MAPS", "Relation_Propagation_heads", prompts_type,
        f"MAPS_ranked_top_heads_{MAPS_n_prompts}prompts.json")
    with open(relation_head_file_path, "r") as f:
        relation_head_dict = json.load(f)
    relation_heads_to_exclude = [item for item in relation_head_dict["top_components"] if item[2] >= relation_head_threshold]

    # Iterate over number of top heads 
    top_grouped_heads = {}
    for n_top_heads in n_heads_list:
        if print_process:
            print("\ninitial heads: ", ranked_heads[:n_top_heads])
        indirect_heads = exclude_overlapping_layers_and_heads(
            ranked_heads[:n_top_heads], execution_heads_to_exclude)
        if print_process:
            print("after excluding execution heads: ", indirect_heads)
        other_task_recognition_heads = exclude_overlapping_layers_and_heads(
            indirect_heads, relation_heads_to_exclude)
        if print_process:
            print("after excluding relation heads: ", other_task_recognition_heads)
        if universal_FV_heads is not None and prompts_type == "EP":
            other_task_recognition_heads = exclude_overlapping_layers_and_heads(
                other_task_recognition_heads, universal_FV_heads)
        if print_process:
            print("after excluding FV heads: ", other_task_recognition_heads)
        grouped_other_task_recognition_heads = group_ranked_locations_by_layer(other_task_recognition_heads)
        if len(other_task_recognition_heads) > 0:
            top_grouped_heads[len(other_task_recognition_heads)] = grouped_other_task_recognition_heads
        if len(other_task_recognition_heads) >= n_max_other_task_recognition_heads:
            break

    return top_grouped_heads

def find_relation_abstract_neuron(
    model:HookedTransformer=None,
    top_components=None, n_neurons=None, topk=None,  words=None
):
    """
    Find the neurons whose topk decoded vocab contain the relation words of the task.
    """
    words = [ " "+ x  for x in words]
    relation_components = []
    relation_neuron_count = 0
    for top_n in range(n_neurons):
        target_layer = top_components[top_n][0]
        neuron_idx = top_components[top_n][1]
        effect = top_components[top_n][2]
        vec = model.blocks[target_layer].mlp.W_out[neuron_idx]
        top10vocab = decode_vec(vec, model, topk=topk)
        # check if the top10vocab contains any of a list of words
        if any(word in top10vocab for word in words):
            relation_components.append([target_layer, neuron_idx, effect])
            relation_neuron_count += 1

    return relation_components, relation_neuron_count


def load_relation_neurons(
    prompts_type=None,
    model_name=None, model:HookedTransformer=None,
    d_name=None,
    n_neurons=2000, topk=10,
    task_relation_dict=None,     
): 
    """
    Load saved top neurons, then filter out the ones that contain the relation/task words of the task.
    Return:
        top_relation_neurons: a list of lists, each list contains [layer, neuron, mapping score]
    
    """
    if prompts_type == "EP":
        neuron_type = "ICL_neuron"
    elif prompts_type == "IP":
        neuron_type = "INST_neuron"
    else:
        raise ValueError(f"Invalid prompts type: {prompts_type}")
   
    # load json file 
    file_path = os.path.join("/oscar/data/epavlick/zyang220/results/fv_comm", 
        model_name, "fingerprint_fv_scale", d_name, neuron_type, 
        "48592", f"ranked_component_{n_neurons}.json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    top_components = data["top_components"]

    top_relation_neurons, relation_neuron_count = find_relation_abstract_neuron(
        top_components=top_components, n_neurons=n_neurons, topk=topk, model=model,
        words=task_relation_dict[d_name])
    
    return top_relation_neurons

def get_component_count_in_grouped_component_dict(component_dict):
    """
    Calculates the total number of elements across all lists in a dictionary.

    Args:
        layers_dict (dict): A dictionary where values are lists.

    Returns:
        int: The sum of the lengths of all lists.
    """
    total_length = 0
    for layer_list in component_dict.values():
        total_length += len(layer_list)
    return total_length

def exclude_overlapping_layers_and_heads(A, B):
    """
    Excludes elements from list A that have overlapping (layer, head) with list B.

    Args:
        A: A list of tuples, where each tuple is (layer, head, score).
        B: A list of tuples, where each tuple is (layer, head, score).

    Returns:
        A new list A' containing elements from A that do not have a 
        (layer, head) pair present in B.
    """
    # Create a set of (layer, head) pairs from list B for efficient lookup.
    b_layers_and_heads = {(layer, head) for layer, head, _ in B}

    # Filter list A, keeping only elements whose (layer, head) pair is not in the set.
    A_prime = [item for item in A if (item[0], item[1]) not in b_layers_and_heads]

    return A_prime

def calculate_component_metrics_per_head(
    projection:torch.Tensor, k:int, dst_tokens:torch.Tensor,
    n_match=1, component_type="retrieval_heads",
):
    """
    Calculate the metrics of each prompt for a given head
    Get several metrics from decoded vectors(projection) 

    Args:
        projection: logits of shape (batch, d_vocab)
        k: int
        dst_tokens: (batch) or (n_options)
        n_match: int, number of top decoded words matched dst_tokens should be >= n_match to get a score of 1
      
        
    Returns: a dict of following metrics:
        MAPS_score: MAPS score of shape (batch,)
        dst_tokens_ranks: ranks of shape (batch,)  
        dst_tokens_logits: logits of shape (batch,)
        top_k_tokens: top k tokens of shape (batch, k)
        
    """
    dst_tokens = dst_tokens.to(projection.device) 
    _, top_k_tokens = torch.topk(projection, k) # indices (batch, k)

    # Calculate MAPS score 
    if component_type == "retrieval_heads":
        matches = (top_k_tokens == dst_tokens.unsqueeze(1)) #(batch, k)
        total_matches_per_batch_item = matches.sum(dim=1) #(batch)
        batch_item_has_n_matches = (total_matches_per_batch_item >= n_match)
        MAPS_score = batch_item_has_n_matches.float() # Convert boolean to float (True=1.0, False=0.0)

    elif component_type == "lexical_task_heads":
        # For each data point, we want to check if any of the top k indices
        # matches any of the n_options dst_tokens. If so, that batch item gets a score of 1.
        
        # Expand dst_tokens (n_options,) to (batch, 1, n_options) for broadcasting
        batch_size = top_k_tokens.shape[0]
        dst_tokens_expanded = dst_tokens.unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)

        # Compare each top-k index with each of the dst_tokens for that batch item
        # (batch, k, 1) == (batch, 1, n_options) results in (batch, k, n_options)
        individual_matches = (top_k_tokens.unsqueeze(2) == dst_tokens_expanded)

        # For each (batch, k) element, check if it matched ANY of the n_options dst_tokens
        # This results in (batch, k) where True means one of the dst_tokens was matched
        any_dst_token_matched_per_k_index = individual_matches.any(dim=2)

        # Count the total number of top-k indices that matched ANY dst_token
        # This results in (batch)
        total_matches_per_batch_item = any_dst_token_matched_per_k_index.sum(dim=1)

        # Check if the total number of matches is greater than or equal to n
        # This results in (batch)
        batch_item_has_n_matches = (total_matches_per_batch_item >= n_match)

        # # For each batch item, check if ANY of its top-k indices matched ANY of the dst_tokens
        # # This results in (batch) where True means at least one top-k index matched for that batch item
        # batch_item_has_match = any_dst_token_matched_per_k_index.any(dim=1)

        # Convert boolean to float (True=1.0, False=0.0)
        MAPS_score = batch_item_has_n_matches.float()

    if component_type == "retrieval_heads":
        dst_tokens_ranks, dst_tokens_logits = compute_token_rank_logit(projection, dst_tokens) 

    if component_type == "retrieval_heads":
        results = {
            "MAPS_score": MAPS_score.cpu().numpy(),
            "dst_tokens_ranks": dst_tokens_ranks.cpu().numpy(),
            "dst_tokens_logits": dst_tokens_logits.cpu().numpy(),
            "top_k_tokens": top_k_tokens.cpu().numpy(),
        }
    elif component_type == "lexical_task_heads":
        results = {
            "MAPS_score": MAPS_score.cpu().numpy(),
            "top_k_tokens": top_k_tokens.cpu().numpy(),
        }

    return results
    
def calculate_component_metrics_from_heads_output(
    model: HookedTransformer, 
    heads_output: torch.Tensor, 
    dst_tokens=None,
    apply_ln: bool = True,
    k=10, n_match=1, 
    component_type="retrieval_heads",
):
    """
    Calculate the metrics of each prompt across all heads
    Args:
        model: HookedTransformer
        heads_output: torch.Tensor, shape (n_examples, n_layers, n_heads, d_model)
        
        apply_ln: bool, whether to apply layer norm to the heads output
        k: int, number of top tokens to consider
        n_match: int, number of top tokens that should match the destination tokens to get a score of 1
        component_type: str, "retrieval_heads" or "lexical_task_heads"
        dst_tokens: torch.Tensor, shape (n_examples,) of "retrieval_heads" 
            or (n_options) of "lexical_task_heads"
       
    """
    n_examples = heads_output.shape[0]
    n_layers = heads_output.shape[1]
    n_heads = heads_output.shape[2]

    MAPS_scores = np.empty((n_examples, n_layers, n_heads))
    dst_tokens_ranks = np.empty((n_examples, n_layers, n_heads))
    dst_tokens_logits = np.empty((n_examples, n_layers, n_heads))
    top_k_tokens = np.empty((n_examples, n_layers, n_heads, k), dtype=int)

    for layer in range(n_layers):
        for head in range(n_heads):
            head_output = heads_output[:, layer, head] # (batch, d_model)
            # apply linear norm 
            if apply_ln:
                head_output = model.model.norm(head_output.to(model.device))
            # project to vocab to get the logits 
            projection = model.lm_head(head_output) # (batch, d_vocab)
         
            #print("projection.shape", projection.shape)
            metrics_per_head = calculate_component_metrics_per_head(
                projection, k, dst_tokens, n_match=n_match,
                component_type=component_type, 
            )
            MAPS_scores[:, layer, head] = metrics_per_head["MAPS_score"]
            top_k_tokens[:, layer, head] = metrics_per_head["top_k_tokens"]

            if component_type == "retrieval_heads":
                dst_tokens_ranks[:, layer, head] = metrics_per_head["dst_tokens_ranks"]
                dst_tokens_logits[:, layer, head] = metrics_per_head["dst_tokens_logits"]
                
    
    if component_type == "retrieval_heads":
        results = {
            "MAPS_scores": MAPS_scores,
            "dst_tokens_ranks": dst_tokens_ranks,
            "dst_tokens_logits": dst_tokens_logits,
            "top_k_tokens": top_k_tokens,
        }
    elif component_type == "lexical_task_heads":
        results = {
            "MAPS_scores": MAPS_scores,
            "top_k_tokens": top_k_tokens,
        }
            
    return results

def cache_act_nnsight(
    model: HookedTransformer, 
    cache_component_type:str="heads", 
    all_tokens: torch.Tensor = None,
    all_answer_tokens: torch.Tensor = None,
    remote: bool = False, batch_size: int = 8,
) -> torch.Tensor:
    """ 
    Cache the specified components' activations. 
    Args:
        model: HookedTransformer
        all_tokens: Pre-tokenized tokens, shape (sample_size, seq)
        all_answer_tokens: Pre-tokenized answer tokens, shape (sample_size,)
        cache_component_type: "heads" or "neurons"
        batch_size: Size of each processing batch
        remote: Whether to run model remotely

    Returns:
        run_probs (torch.Tensor): probability of the correct answer of the runs
        cached_act (torch.Tensor): cached activations of shape (sample_size, n_layers, n_heads, d_model) 
            or (sample_size, n_layers, d_mlp)
    """
    # Get model specs
    spec = get_model_specs(model)
    n_layers, n_heads, d_model, d_head = spec["n_layers"], spec["n_heads"], spec["d_model"], spec["d_head"]

    sample_size = all_tokens.shape[0]
    # Initialize results storage ON CPU to save GPU memory
    run_ranks = []
    run_probs = []
    if cache_component_type == "heads":
        cached_act = torch.empty(sample_size, n_layers, n_heads, d_model)  # Keep on CPU
    elif cache_component_type == "neurons":
        raise NotImplementedError("neurons not implemented")
    
    # Move accessor OUTSIDE the batch loop - create once, reuse
    accessor_config = get_accessor_config(model)
    accessor = ModelAccessor(model, accessor_config)
    
    # Pre-load and cache projection matrices to avoid repeated state_dict access
    W_o_projs = []
    if cache_component_type == "heads":
        for layer in range(n_layers):
            W_o = model.state_dict()[f"model.layers.{layer}.self_attn.o_proj.weight"]
            W_o_reshaped = W_o.reshape(d_model, n_heads, d_head)
            W_o_projs.append(W_o_reshaped)
            del W_o  # Clean up immediately
    
    # Corrupt runs, patch the clean activations of grouped heads
    for batch_start in range(0, sample_size, batch_size):
        batch_end = min(batch_start + batch_size, sample_size)
        current_batch_size = batch_end - batch_start 
        batch_tokens = all_tokens[batch_start:batch_end]
        if all_answer_tokens is not None:
            batch_answer_tokens = all_answer_tokens[batch_start:batch_end]

        with model.trace(remote=remote) as tracer:
            with tracer.invoke(batch_tokens) as invoker:
                for layer in range(n_layers):
                    if cache_component_type == "heads":
                        downstream_attn_output = accessor.layers[layer].attention.output.unwrap().input[:, -1]
                        downstream_reshaped = downstream_attn_output.reshape(current_batch_size, n_heads, d_head)
                        
                        # Use pre-loaded projection matrix
                        downstream_output = einops.einsum(
                            downstream_reshaped, W_o_projs[layer], 
                            "batch n_heads d_head, d_model n_heads d_head -> batch n_heads d_model")
                        
                        # Save the output - nnsight will return it as a tensor
                        cached_act[batch_start:batch_end, layer] = downstream_output.save()
                        
                        # Delete intermediate tensors to free GPU memory
                        del downstream_attn_output, downstream_reshaped, downstream_output
                        
                    elif cache_component_type == "neurons":
                        raise NotImplementedError("neurons not implemented")
                    else:
                        raise ValueError(f"cache_component_type {cache_component_type} not supported")
                
                # Get logits INSIDE the invoke context, AFTER the layer loop
                logits = model.lm_head.output[:, -1].save()
        
        # Move cached activations to CPU immediately after the trace
        cached_act[batch_start:batch_end] = cached_act[batch_start:batch_end].to('cpu')
        
        if all_answer_tokens is not None:
            # Compute ranks and probs
            batch_ranks, batch_probs = compute_token_rank_prob(logits, batch_answer_tokens) 
            # Clean up
            del logits
            torch.cuda.empty_cache()  # More aggressive than your free_unused_cuda_memory()
            run_ranks.append(batch_ranks.to('cpu'))
            run_probs.append(batch_probs.to('cpu'))

    # Clean up projection matrices
    del W_o_projs
    torch.cuda.empty_cache()
    
    if all_answer_tokens is not None:
        run_probs = torch.cat(run_probs)
        run_ranks = torch.cat(run_ranks)
        return run_probs, cached_act
    else: 
        return None, cached_act