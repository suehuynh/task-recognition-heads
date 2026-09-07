import re
from typing import Any, Generic, TypeVar, Union
from transformer_lens import HookedTransformer

# Type variable for generic type hints
T = TypeVar("T")

# Configuration mapping for model architecture paths
LLAMA_CONFIG = {
    "base": "model",
    "layers": "{base}.layers",
    "lm_head": "lm_head",
    "tokenizer": "tokenizer",
    "attention": "self_attn",
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "output": "o_proj",
    "mlp": "mlp",
    "down_proj": "down_proj",
}
# model.model.layers[i].mlp.down_proj
# model.model.layers[i].self_attn.o_proj
#  accessor.layers[layer].attention.output.unwrap().input[:, -1] accessor.layers[layer].attention.output.unwrap().input[:, -1]

GPTJ_CONFIG = {
    "base": "transformer",
    "layers": "{base}.h",
    "lm_head": "lm_head",
    "tokenizer": "tokenizer",
    "attention": "attn",
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "output": "out_proj",
}

BLOOM_CONFIG = {
    "base": "transformer",
    "layers": "{base}.h",
    "lm_head": "lm_head",
    "tokenizer": "tokenizer",
    "attention": "self_attention",
    "output": "dense",
}

QWEN_CONFIG = {
    "base": "model",
    "layers": "{base}.layers",
    "lm_head": "lm_head",
    "tokenizer": "tokenizer",
    "attention": "self_attn",
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "output": "o_proj",
    "mlp": "mlp",
    "down_proj": "down_proj",
}

GEMMA_CONFIG = {
    "base": "model",
    "layers": "{base}.layers",
    "lm_head": "lm_head",
    "tokenizer": "tokenizer",
    "attention": "self_attn",
    "query": "q_proj",
    "key": "k_proj",
    "value": "v_proj",
    "output": "o_proj",
    "mlp": "mlp",
    "down_proj": "down_proj",
}


class AttributeProxy(Generic[T]):
    """
    A proxy class that provides flexible attribute access to model components
    while maintaining configuration-based mapping.
    """

    def __init__(
        self,
        obj: T,  # The object being proxied
        config: dict[str, Union[str, list[str]]],  # Configuration mapping
        accessor: "ModelAccessor",  # Reference to parent accessor
    ):
        self._obj = obj
        self._config = config
        self._accessor = accessor

    def __getattr__(self, name):
        """
        Handles attribute access, resolving through config if name is mapped.
        Falls back to direct attribute access if not in config.
        """
        if name in self._config:
            paths = self._config[name] if isinstance(self._config[name], list) else [self._config[name]]
            # Try each possible path for the attribute
            for path in paths:
                try:
                    attr = self._resolve_path(self._obj, path)
                    return AttributeProxy(attr, self._config, self._accessor)
                except AttributeError:
                    continue
            raise AttributeError(f"None of the aliases for '{name}' were found in the object")
        # Direct attribute access for non-configured attributes
        attr = getattr(self._obj, name)
        return (
            AttributeProxy(attr, self._config, self._accessor)
            if isinstance(attr, object) and not isinstance(attr, (str, int, float, bool))
            else attr
        )

    def __getitem__(self, key):
        """
        Enables dictionary-style access to the proxied object.
        """
        item = self._obj[key]
        return (
            AttributeProxy(item, self._config, self._accessor)
            if isinstance(item, object) and not isinstance(item, (str, int, float, bool))
            else item
        )

    def _resolve_path(self, obj: Any, path: str) -> Any:
        """
        Resolves a dot-notation path to an attribute.
        """
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def __call__(self, *args, **kwargs):
        """
        Enables calling the proxied object if it's callable.
        """
        if callable(self._obj):
            return self._obj(*args, **kwargs)
        raise TypeError(f"'{type(self._obj).__name__}' object is not callable")

    def __repr__(self):
        return f"AttributeProxy({self._obj})"

    def unwrap(self) -> T:
        """Returns the original proxied object."""
        return self._obj


class ModelAccessor(Generic[T]):
    """
    Main interface for accessing model components through a unified configuration system.
    Handles path resolution and provides a consistent interface across model architectures.
    """

    def __init__(self, model: T, config: dict[str, Union[str, list[str]]]):
        self._model = model
        self._config = config

    def __getattr__(self, name):
        """
        Resolves attributes through config mapping or falls back to direct model access.
        """
        if name in self._config:
            paths = self._config[name] if isinstance(self._config[name], list) else [self._config[name]]
            for path in paths:
                try:
                    attr = self._resolve_path(self._model, path)
                    return AttributeProxy(attr, self._config, self)
                except AttributeError:
                    continue
            raise AttributeError(f"None of the aliases for '{name}' were found in the model")
        if hasattr(self._model, name):
            attr = getattr(self._model, name)
            if callable(attr):
                return lambda *args, **kwargs: attr(*args, **kwargs)
            return attr
        raise AttributeError(f"'{name}' not found in config or model")

    def _resolve_path(self, obj: Any, path: str) -> Any:
        """
        Resolves a path string to an attribute, handling variable substitution
        and indexed access.
        """

        def replace_key(match):
            """Handles substitution of config variables in paths."""
            key = match.group(1)
            if key not in self._config:
                raise KeyError(f"'{key}' not found in config")
            return self._config[key]

        # Handle variable substitution in paths
        while "{" in path:
            path = re.sub(r"\{(\w+)\}", replace_key, path)

        # Handle attribute access and indexing
        for part in path.split("."):
            if "[" in part:
                attr, idx = part.split("[")
                obj = getattr(obj, attr)
                idx = idx.strip("[]")
                if idx == "{}":
                    return lambda i: AttributeProxy(obj[i], self._config, self)
                obj = obj[int(idx) if idx.isdigit() else idx]
            else:
                obj = getattr(obj, part)
        return obj

    def unwrap(self) -> T:
        """Returns the original model object."""
        return self._model

    def __repr__(self):
        return f"ModelAccessor({self._model})"


def get_accessor_config(model: HookedTransformer) -> dict:
    """
    Returns the appropriate configuration dictionary for the given model type.
    """
    name = model.config._name_or_path
    if name in [
        "meta-llama/Meta-Llama-3.1-8B", 
        "meta-llama/Llama-3.1-8B", 
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.1-405B-Instruct", 
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",

    ]:
        return LLAMA_CONFIG
    elif name in ["EleutherAI/gpt-j-6b"]:
        return GPTJ_CONFIG
    elif name in ["bigscience/bloom-560m"]:
        return BLOOM_CONFIG
    elif name in [
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
    ]:
        return QWEN_CONFIG
    elif name in [
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-9b",
    ]:
        return GEMMA_CONFIG
    else:
        raise NotImplementedError(f"No config found for {name}")


def get_model_specs(model: HookedTransformer) -> dict:
    """
    Extracts key architectural specifications from the model.

    Args:
        model: The language model instance

    Returns:
        Dictionary containing model specifications:
        - n_layers: Number of transformer layers
        - n_heads: Number of attention heads
        - d_model: Hidden dimension size

    Raises:
        NotImplementedError: If model type is not supported
    """
    name = model.config._name_or_path
    if "Llama" in name:
        return dict(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            d_model=model.config.hidden_size,
            d_head=model.config.head_dim,
            d_mlp=model.config.intermediate_size,
        )
    elif "Qwen2.5" in name:
        return dict(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            d_model=model.config.hidden_size,
            #d_head=getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
            d_head=model.config.hidden_size // model.config.num_attention_heads,
            d_mlp=model.config.intermediate_size,
        )
    elif "Qwen3" in name:
        return dict(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            d_model=model.config.hidden_size,
            d_head=model.config.head_dim,
            d_mlp=model.config.intermediate_size,
        )
    elif "gemma" in name.lower():
        return dict(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            d_model=model.config.hidden_size,
            d_head=model.config.head_dim,
            d_mlp=model.config.intermediate_size,
        )
    else:
        raise NotImplementedError(f"No spec found for {name}")