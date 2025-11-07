import torch
from torch import Tensor
from parq.quant import Quantizer


def assert_tensor(tensor):
    if torch.isnan(tensor).any():
        raise ValueError("Input tensor contains NaN values.")
    if torch.isinf(tensor).any():
        raise ValueError("Input tensor contains Inf values.")
    return True


def assert_finite(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, Tensor):
            try:
                assert_tensor(result)
            except ValueError as e:
                raise ValueError(
                    f"Output of {func.__name__} is not finite.\n"
                    f"The input arguments were: args={args}, kwargs={kwargs}"
                ) from e
        elif isinstance(result, tuple) or isinstance(result, list):
            for idx, item in enumerate(result):
                if isinstance(item, Tensor):
                    try:
                        assert_tensor(item)
                    except ValueError as e:
                        raise ValueError(
                            f"Output item {idx} of {func.__name__} is not finite.\n"
                            f"The input arguments were: args={args}, kwargs={kwargs}"
                        ) from e
        return result

    return wrapper


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
        assert self.e > 0, "Exponent bits must be positive"
        assert self.m >= 0, "Mantissa bits must be non-negative"
        self.bias = (1 << (self.e - 1)) - 1
        self.emin = 1 - self.bias
        self.mantissa_scale = 1 << self.m
        self.use_finite_top = (self.e, self.m) == (4, 3)
        top_code = (1 << self.e) - 1
        if self.use_finite_top:
            self.max_exponent_code = top_code
            self.max_mantissa_for_max_exp = max(self.mantissa_scale - 2, 0)
        else:
            self.max_exponent_code = top_code - 1
            self.max_mantissa_for_max_exp = self.mantissa_scale - 1
        self.emax = self.max_exponent_code - self.bias
        self._codebook_cache: dict[tuple[torch.dtype, torch.device], Tensor] = {}
        super().__init__(center=center)

    def get_quant_size(self, b: int) -> int:
        assert (
            self.e + self.m == b - 1
        ), "Exponent and mantissa bits must sum to total bits - sign bit"
        regular_exponents = self.max_exponent_code
        top_bonus = 0
        if self.use_finite_top:
            regular_exponents -= 1
            top_bonus = self.max_mantissa_for_max_exp + 1
        regular_normals = regular_exponents * self.mantissa_scale
        subnorm = self.mantissa_scale - 1
        positive_non_zero = regular_normals + subnorm + top_bonus
        return 2 * positive_non_zero + 1

    def get_max_value(self) -> float:
        """Get the maximum representable value in this float format."""
        max_mant = float(self.max_mantissa_for_max_exp)
        max_value = (1 + max_mant / self.mantissa_scale) * (2**self.emax)
        return max_value

    @assert_finite
    def quantize(
        self, p: Tensor, b: int, dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Instantiation of Quantizer.quantize() method"""
        assert b >= 4, "Bit width must be at least 4 for float quantization"
        assert (
            self.e + self.m == b - 1
        ), "Exponent and mantissa bits must sum to total bits - sign bit"

        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        absmax = q.abs().max()
        format_max = self.get_max_value()
        scale = absmax / format_max
        q.div_(scale).clamp_(min=-format_max, max=format_max)

        q_quant = self._quantize_tensor(q)
        q_quant.mul_(scale)

        Q = self._codebook(q_quant.dtype, q_quant.device)
        if dim is not None:
            Q = Q.unsqueeze(0).mul_(scale)
        else:
            Q.mul_(scale)

        if self.center:
            q_quant += mean
            Q += mean

        Q = self._codebook(q_quant.dtype, q_quant.device)
        return q_quant, Q

    def _quantize_tensor(self, q: Tensor) -> Tensor:
        dtype = q.dtype
        device = q.device
        mant_scale_tensor = torch.tensor(
            float(self.mantissa_scale), dtype=dtype, device=device
        )
        min_normal = torch.ldexp(
            torch.ones(1, dtype=dtype, device=device),
            torch.tensor(self.emin, dtype=torch.int32, device=device),
        )
        max_scale = torch.ldexp(
            torch.ones(1, dtype=dtype, device=device),
            torch.tensor(self.emax, dtype=torch.int32, device=device),
        )
        max_mant = torch.tensor(
            float(self.max_mantissa_for_max_exp), dtype=dtype, device=device
        )
        max_value = (1 + max_mant / mant_scale_tensor) * max_scale
        subnorm_step = torch.ldexp(
            torch.ones(1, dtype=dtype, device=device),
            torch.tensor(self.emin - self.m, dtype=torch.int32, device=device),
        )

        q_abs = q.abs()
        quantized_abs = torch.zeros_like(q_abs)

        large_mask = q_abs > max_value
        quantized_abs[large_mask] = max_value

        normal_mask = (~large_mask) & (q_abs >= min_normal)
        if normal_mask.any():
            mant, exp = torch.frexp(q_abs[normal_mask])
            exp = exp - 1
            frac = mant.mul_(2).sub_(1)
            mant_q = torch.round(frac * mant_scale_tensor).to(torch.int32)
            overflow = mant_q == self.mantissa_scale
            if overflow.any():
                mant_q = mant_q.masked_fill(overflow, 0)
                exp = exp + overflow.to(exp.dtype)

            exp = torch.minimum(exp, torch.full_like(exp, self.emax, dtype=exp.dtype))
            if self.use_finite_top:
                max_mant = torch.full_like(mant_q, self.mantissa_scale - 1)
                max_mant = torch.where(
                    exp == self.emax,
                    torch.full_like(mant_q, self.max_mantissa_for_max_exp),
                    max_mant,
                )
                mant_q = torch.minimum(mant_q, max_mant)

            mant_q_float = mant_q.to(dtype) / mant_scale_tensor
            scale = torch.ldexp(torch.ones_like(mant), exp)
            normal_vals = (1 + mant_q_float) * scale
            quantized_abs[normal_mask] = normal_vals

        subnorm_mask = (~large_mask) & (q_abs < min_normal) & (q_abs > 0)
        if subnorm_mask.any():
            mant_q = torch.round(q_abs[subnorm_mask] / subnorm_step)
            mant_q.clamp_(0, self.mantissa_scale - 1)
            subnorm_vals = mant_q * subnorm_step
            quantized_abs[subnorm_mask] = subnorm_vals

        sign = torch.sign(q)
        quantized = quantized_abs * sign
        return quantized

    def _codebook(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        key = (dtype, device)
        if key in self._codebook_cache:
            return self._codebook_cache[key]

        mant_scale = torch.tensor(
            float(self.mantissa_scale), dtype=dtype, device=device
        )
        step = torch.ldexp(
            torch.ones(1, dtype=dtype, device=device),
            torch.tensor(self.emin - self.m, dtype=torch.int32, device=device),
        )
        positives = []
        if self.mantissa_scale > 1:
            positives.append(
                torch.arange(1, self.mantissa_scale, dtype=dtype, device=device) * step
            )

        for exp_code in range(1, self.max_exponent_code + 1):
            max_mant = self.mantissa_scale - 1
            if self.use_finite_top and exp_code == (1 << self.e) - 1:
                max_mant = self.max_mantissa_for_max_exp
            mant = torch.arange(max_mant + 1, dtype=dtype, device=device)
            if mant.numel() == 0:
                continue
            mant = mant / mant_scale
            exp_val = exp_code - self.bias
            scale = torch.ldexp(
                torch.ones(1, dtype=dtype, device=device),
                torch.tensor(exp_val, dtype=torch.int32, device=device),
            )
            positives.append((1 + mant) * scale)

        if positives:
            positives = torch.cat(positives)
        else:
            positives = torch.zeros(0, dtype=dtype, device=device)

        negatives = -positives.flip(0)
        zero = torch.zeros(1, dtype=dtype, device=device)
        codebook = torch.cat([negatives, zero, positives])
        self._codebook_cache[key] = codebook
        return codebook


class TorchFloatQuantizer(Quantizer):
    def __init__(self, e: int, m: int, center: bool = False):
        super().__init__(center=center)
        if e == 4 and m == 3:
            self.dtype = torch.float8_e4m3fn
        elif e == 5 and m == 2:
            self.dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported float8 format")
        self.bits = e + m + 1
        self._codebook_cache = {}

    def get_quant_size(self, b: int) -> int:
        assert b == 8, "Only 8-bit float quantization is supported"
        if self.dtype == torch.float8_e4m3fn:
            return 253
        elif self.dtype == torch.float8_e5m2:
            return 247
        else:
            raise ValueError("Unsupported float8 format")

    def quantize(
        self, p: Tensor, b: int, dim: int | None = None
    ) -> tuple[Tensor, Tensor]:
        assert b == self.bits, "Bit size does not match exponent and mantissa bits"
        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()

        q_quant = q.to(self.dtype).to(torch.float32)

        Q = self._codebook(b, q_quant.dtype, q_quant.device)

        if self.center:
            q_quant += mean

        return q_quant, Q

    def _codebook(self, b: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        key = (b, dtype, device)
        if key in self._codebook_cache:
            return self._codebook_cache[key]
        values = (
            torch.arange(2**b, dtype=torch.uint8, device=device)
            .view(self.dtype)
            .to(dtype)
        )
        finite = values  # [torch.isfinite(values)]
        finite = torch.unique(finite)
        codebook = torch.sort(finite).values
        self._codebook_cache[key] = codebook
        return codebook
