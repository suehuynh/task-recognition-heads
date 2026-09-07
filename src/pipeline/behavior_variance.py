import os
import json
import argparse
from transformer_lens import HookedTransformer
import torch
torch.set_grad_enabled(False)

from utils.build_prompts import create_few_shot_prompts, create_instruction_prompts, check_correctness

if __name__ == "__main__":
    """
    Perform inference runs and save the behavior variance 
    of a dataset for a given model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, 
        help="model name e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--d_name", type=str, required=True,)
    parser.add_argument("--save_root", type=str, 
        default="output",)
    parser.add_argument("--project_root", type=str, 
        default="",
        help="directory of the codebase ")
    parser.add_argument("--prompt_type", type=str, required=True, help="prompt type: EP or IP")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size")
    parser.add_argument("--dataset_folder", type=str, default="datasets/abstractive", help="folder of the dataset")
    parser.add_argument("--dtype", type=str, default="float32",
        choices=["float32", "float16", "bfloat16"], help="model dtype")

    args = parser.parse_args()
    model_name = args.model_name
    save_root = args.save_root
    project_root = args.project_root
    d_name = args.d_name
    prompt_type = args.prompt_type
    batch_size = args.batch_size
    dataset_folder = args.dataset_folder
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("model_name", model_name)
    print("dtype", args.dtype)
    print("device", device)
    print("prompt_type", prompt_type)
    print("d_name", d_name)
    print("project_root", project_root)

    # Load model 
    print("To load model")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    print("Model loaded")
    model_name = model_name.split("/")[-1]

    # Load file if exist, otherwise create a new dictionary
    if prompt_type == "EP":
        file_name = "EP_vary_n_shot_behavior.json"
        prompt_temp_idx_list = [1,2,3,4,5,10,20,30]
        # prompt_temp_idx_list = [5,10,20,30]
    elif prompt_type == "IP":
        file_name = "IP_vary_n_inst_behavior.json"
        prompt_temp_idx_list = [0,1,2,3,4]
    else:
        raise ValueError(f"prompt_type {prompt_type} not supported")
    save_path = os.path.join(save_root, model_name, "across_tasks", "Behavior")
    if os.path.exists(os.path.join(save_path, file_name)):
        with open(os.path.join(save_path, file_name), "r"
        ) as f:
            result_dict = json.load(f)
            print(f"Behavior file {file_name} loaded")
    else:
        print("Behavior file does not exist, creating a new dictionary")
        result_dict = {}
    
    # Load instruction_dict
    with open(os.path.join(project_root, "datasets", "dataset_info", 
        f"instruction_dict.json"), "r"
    ) as f:
        instruction_dict = json.load(f)

    # Run inference and check correctness
    result_dict[d_name] = {}
    for prompt_temp_index_idx in prompt_temp_idx_list:
        result_dict[d_name][prompt_temp_index_idx] = {}
        if prompt_type == "EP":
            prompts, answers, _ = create_few_shot_prompts(
                d_name, n_shot=prompt_temp_index_idx, dataset_folder=dataset_folder,
                delimiter=";", q_bos=" ", a_bos=" ", qa_delimiter=":"
            )
        elif prompt_type == "IP":
            with open(os.path.join(dataset_folder, f"{d_name}.json"), encoding="utf-8") as f:
                dataset = json.load(f)
            prompts, answers = create_instruction_prompts(dataset,
                instruction_dict[d_name][str(prompt_temp_index_idx)])

        prompt_dict = check_correctness(model=model, prompts=prompts, answers=answers,
            batch_size=batch_size, return_pred_tokens=False, return_answer_tokens=False,
        )
        correct_index = prompt_dict['correct_index']
        acc = len(correct_index) / len(prompts)
        result_dict[d_name][prompt_temp_index_idx]['accuracy'] = acc
        result_dict[d_name][prompt_temp_index_idx]['correct_index'] = correct_index
        result_dict[d_name][prompt_temp_index_idx]['n_correct_index'] = len(correct_index)
        result_dict[d_name][prompt_temp_index_idx]['n_dataset'] = len(prompts)
    
    # Save after each dataset is done (atomic write)
    os.makedirs(save_path, exist_ok=True)
    final_path = os.path.join(save_path, file_name)
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    os.replace(tmp_path, final_path)
    print(f"Behavior file {final_path} saved")