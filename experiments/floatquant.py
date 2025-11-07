import torch
from torch import Tensor
from parq.quant import Quantizer


def get_beta(q: Tensor, b: int, e: int, m: int, dim: int | None = None) -> Tensor:
    """Get beta for float quantization.

    Args:
        p: tensor to be quantized
        b: number of bits for quantization
        e: number of exponent bits
        m: number of mantissa bits
        dim: dimension along which to compute q_max

    Returns:
        beta tensor
    """

    assert (
        e + m == b - 1
    ), "Exponent and mantissa bits must sum to total bits - sign bit"

    q_abs = q.abs()
    if dim is not None:
        q_max = torch.max(q_abs, dim=dim, keepdim=True).values
    else:
        q_max = torch.max(q_abs)

    beta = torch.zeros_like(q_max, device=q.device, dtype=q.dtype)
    beta.add_(2 - 2 ** (-m)).div_(q_max).log2_().add_(2**e - 1)

    return beta


def get_gamma(
    q: Tensor, b: int, beta: Tensor, dim: int | None = None, scale_method: str = "max"
) -> Tensor:
    """Get gamma for float quantization.

    Args:
        p: tensor to be quantized
        b: number of bits for quantization
        dim: dimension along which to compute q_max

    Returns:
        gamma tensor
    """

    exp = q.abs().log2_().floor_().add_(beta).floor_()
    if dim is not None:
        x = torch.max(exp, dim=dim, keepdim=True).values
    else:
        x = torch.max(exp)

    gamma = torch.full_like(x, 2, device=q.device, dtype=q.dtype)
    gamma.pow_(torch.max(x, torch.ones_like(x)))

    return gamma


class FloatQuantizer(Quantizer):
    """Float quantizer"""

    def __init__(
        self,
        exponent_bits: int,
        mantissa_bits: int,
        center: bool = False,
    ):
        """Set quantization function parameters.

        Args:
            exponent_bits: number of exponent bits
            mantissa_bits: number of mantissa bits
            center: whether to subtract p.mean() prior to quantization
        """
        self.e = exponent_bits
        self.m = mantissa_bits
        super().__init__(center=center)

    def get_quant_size(self, b: int) -> int:
        return 2 ** (b - 1) + 1

    def quantize(
        self, p: Tensor, b: int, dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize() method"""
        assert b != 0, "Not implemented"
        assert (
            self.e + self.m == b - 1
        ), "Exponent and mantissa bits must sum to total bits - sign bit"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        beta = get_beta(q, b, self.e, self.m, dim=dim)
        gamma = get_gamma(q, b, beta, dim=dim)

        range_absmax = (2 - 2 ** (-self.m)) * 2 ** (2**self.e - 1)  # max float value
        s = gamma / range_absmax

        # scale by 1/s -> round -> scale by s
        q.div_(s).round_().mul_(s)

        # set of all target quantization values
        #### TODO: implement set of quantization values
        Q = torch.zeros_like(q, device=q.device, dtype=q.dtype)

        if dim is not None:
            Q = Q.unsqueeze(0).mul(s)  # broadcasted multiply requires copy
        else:
            Q.mul_(s)

        # return quantized tensor and set of possible quantization values
        if self.center:
            q += mean
            Q += mean
        return q, Q
