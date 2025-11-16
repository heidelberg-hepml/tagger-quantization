from collections.abc import Mapping

import torch
from lloca.backbone.attention_backends.xformers_attention import (
    BlockDiagonalMask,
    get_xformers_attention_mask,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@torch.compile
def get_flex_attention_mask(batch: torch.Tensor, device: torch.device) -> BlockMask:
    """Returns a mask for the attention mechanism.
    Args:
        batch: Batch vector, maps each token to its sequence in the batch.
        device: Device to create the mask on.
    Returns:
        Block-diagonal BlockMask for flex attention, with one block per sequence.
    """

    N = batch.size(0)

    def jagged_masking(b, h, q_idx, kv_idx):
        return batch[q_idx] == batch[kv_idx]

    mask = create_block_mask(jagged_masking, None, None, N, N, device=device, _compile=True)
    return mask


def get_attention_mask(
    batch: torch.Tensor,
    attention_backend: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | BlockMask | BlockDiagonalMask]:
    """Returns the attention mask according to the backend.
    Args:
        batch: Batch vector, maps each token to its sequence in the batch.
        attention_backend: Attention backend to use ("xformers" or "flex_attention").
        device: Device to create the mask on.
        dtype: Data type of the attention mask (for xformers backend).
    Returns:
        Attention mask for the specified backend.
    """
    if attention_backend == "xformers":
        materialize = device == torch.device("cpu")
        mask = get_xformers_attention_mask(batch=batch, dtype=dtype, materialize=materialize)
        return {"attn_mask" if materialize else "attn_bias": mask}
    elif attention_backend == "flex_attention":
        mask = get_flex_attention_mask(batch=batch, device=device)
        return {"block_mask": mask}
    else:
        raise ValueError(
            f"Unsupported attention backend: {attention_backend}. "
            'Supported backends are "xformers" and "flex_attention".'
        )
