from collections.abc import Mapping
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask


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

    mask = create_block_mask(
        jagged_masking, None, None, N, N, device=device, _compile=True
    )
    return mask
