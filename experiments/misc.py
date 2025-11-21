from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from xformers.ops.fmha.attn_bias import BlockDiagonalMask


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


def get_xformers_attention_mask(batch, materialize=False, dtype=torch.float32):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch.tensor
        batch object in the torch_geometric.data naming convention
        contains batch index for each event in a sparse tensor
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts, device=batch.device)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch), len(batch))).to(batch.device, dtype=dtype)
    return mask


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


def assert_finite(fn: Callable) -> Callable:
    """Decorator to assert that all tensors in the inputs and output are finite."""

    def _check(x: Any, label: str) -> None:
        if torch.is_tensor(x):
            assert not torch.isinf(x).any(), f"{label} contains inf."
            assert not torch.isnan(x).any(), f"{label} contains nan."
        elif isinstance(x, Mapping):
            for k, v in x.items():
                _check(v, f"{label}[{k}]")
        elif isinstance(x, (list, tuple)):
            for i, item in enumerate(x):
                _check(item, f"{label}[{i}]")

    def wrapper(*args, **kwargs):
        for i, arg in enumerate(args):
            _check(arg, f"Input argument {i}")

        for k, v in kwargs.items():
            _check(v, f"Input argument {k}")

        output = fn(*args, **kwargs)
        _check(output, "Output")
        return output

    return wrapper
