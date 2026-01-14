import torch
from parq.quant import Quantizer
from torch import Tensor


class FloatQuantizer(Quantizer):
    """Float quantizer for FP16, E4M3FN (8-bit), E3M2FN (6-bit) and E2M1FN (4-bit)."""

    def __init__(self, bits: int = 8, center: bool = False):
        """
        Args:
            bits: 4, 6, 8 or 16 bits
            center: subtract mean before quantization
        """
        super().__init__(center=center)

        if bits == 16:
            self.e, self.m = 5, 10
            self.max_val = torch.finfo(torch.float16).max
            self.use_native = True
        elif bits == 8:
            self.e, self.m = 4, 3
            self.max_val = torch.finfo(torch.float8_e4m3fn).max
            self.use_native = True
        elif bits == 6:
            self.e, self.m = 3, 2
            self.max_val = 28.0  # E3M2FN max (no inf, no NaN)
            self.use_native = False
        elif bits == 4:
            self.e, self.m = 2, 1
            self.max_val = 6.0  # E2M1FN max (no inf, no NaN)
            self.use_native = False
        else:
            raise ValueError(f"Unsupported bits={bits}. Use 4, 6, 8 or 16.")

        self.bits = bits
        self.codebook = self._build_codebook()

    def get_quant_size(self, b: int) -> int:
        """Return number of quantization values (excluding NaN)."""
        assert b == self.bits
        return len(self.codebook)

    def quantize(self, p: Tensor, b: int, dim: int | None = None) -> tuple[Tensor, Tensor]:
        """Quantize tensor to float format."""
        assert b == self.bits

        # Mean centering
        if self.center:
            q, mean = super().remove_mean(p.detach(), dim=dim)
        else:
            q = p.detach().clone()
            mean = torch.zeros(1, dtype=p.dtype, device=p.device)

        # Compute scale
        if dim is None:
            absmax = q.abs().max()
        else:
            absmax = q.abs().amax(dim=dim, keepdim=True)
        scale = torch.where(absmax > 0, absmax / self.max_val, torch.ones_like(absmax))

        # Scale codebook
        Q = self.codebook.to(dtype=p.dtype, device=p.device) * scale

        # Quantize
        if self.use_native:
            # Float8/Float16: use native dtype
            target_dtype = torch.float16 if self.bits == 16 else torch.float8_e4m3fn
            q = (q / scale).to(target_dtype).to(p.dtype) * scale
        else:
            # Float4/Float6: snap to codebook
            bin_edges = (Q[:-1] + Q[1:]) / 2.0
            indices = torch.bucketize(q.flatten(), bin_edges, right=True)
            q = Q[indices].view_as(p)

        # Add mean back
        if self.center:
            q = q + mean
            Q = Q + mean

        return q, Q

    def _build_codebook(self) -> Tensor:
        """Build codebook of representable float values."""
        bias = (1 << (self.e - 1)) - 1
        mant_scale = 1 << self.m

        vals = []

        # Iterate through all exponent codes
        max_exp_code = 1 << self.e

        for exp_code in range(max_exp_code):
            if exp_code == 0:
                # Subnormals: exp_code=0
                for m in range(mant_scale):
                    val = m * (2.0 ** (1 - bias - self.m))
                    vals.append(val)
            else:
                # Normals: exp_code >= 1
                if self.bits == 16 and exp_code == max_exp_code - 1:
                    # Skip inf/NaN exponent for FP16
                    continue
                max_mant = mant_scale
                # For E4M3FN: exp=15 only has mantissa 0-6 (mantissa=7 is NaN)
                if self.bits == 8 and exp_code == max_exp_code - 1:
                    max_mant = mant_scale - 1  # Exclude last mantissa (NaN)

                for m in range(max_mant):
                    val = (1.0 + m / mant_scale) * (2.0 ** (exp_code - bias))
                    vals.append(val)

        # Full range: negative + positive (signed zeros)
        full = [-v for v in reversed(vals)] + vals

        return torch.tensor(full, dtype=torch.float32)


class IntQuantizer(Quantizer):
    def __init__(self, bits: int, center: bool = False, signed: bool = True):
        """
        Args:
            bits: 8 bits
            center: subtract mean before quantization
        """
        super().__init__(center=center)

        # Kept for compatibility with parq interface
        assert center is False, "Mean centering not supported for IntQuantizer."

        self.bits = bits
        if signed:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bits - 1

    def get_quant_size(self, b: int) -> int:
        """Return number of quantization values."""
        assert b == self.bits
        return self.qmax - self.qmin + 1

    def quantize(
        self,
        p: Tensor,
        b: int,
        dim: int | None = None,
        min_val: Tensor | None = None,
        max_val: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        assert b == self.bits

        q = p.detach().clone()

        # Compute scale and zero point
        if min_val is None or max_val is None:
            if dim is None:
                min_val, max_val = q.min(), q.max()
            else:
                min_val, max_val = q.amin(dim=dim, keepdim=True), q.amax(dim=dim, keepdim=True)

        scale = (max_val - min_val) / (self.qmax - self.qmin)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))

        zeropoint = torch.round(self.qmin - min_val / scale)
        zeropoint = torch.clamp(zeropoint, self.qmin, self.qmax)

        # Quantize
        q_int = torch.clamp(torch.round(q / scale) + zeropoint, self.qmin, self.qmax)
        q_dequant = (q_int - zeropoint) * scale

        Q = torch.arange(self.qmin, self.qmax + 1, device=p.device, dtype=p.dtype)
        Q = (Q - zeropoint) * scale

        return q_dequant, Q
