import torch

from experiments.floatquant import FloatQuantizer


def _finite_values(dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(256, dtype=torch.uint8).view(dtype).to(torch.float32)
    finite = values[torch.isfinite(values)]
    finite = torch.unique(finite)
    return torch.sort(finite).values


def test_float8_e4m3_matches_torch():
    fq = FloatQuantizer(4, 3)
    bits = 8
    samples = torch.linspace(-400.0, 400.0, steps=2049, dtype=torch.float32)
    quantized, _ = fq.quantize(samples, bits)
    reference = samples.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.allclose(quantized, reference)


def test_float8_e5m2_matches_torch():
    fq = FloatQuantizer(5, 2)
    bits = 8
    samples = torch.linspace(-50000.0, 50000.0, steps=2049, dtype=torch.float32)
    quantized, _ = fq.quantize(samples, bits)
    reference = samples.to(torch.float8_e5m2).to(torch.float32)
    assert torch.allclose(quantized, reference)


def test_codebook_matches_finite_values():
    fq = FloatQuantizer(4, 3)
    _, codebook = fq.quantize(torch.zeros(1, dtype=torch.float32), 8)
    reference = _finite_values(torch.float8_e4m3fn)
    assert codebook.numel() == fq.get_quant_size(8)
    assert torch.allclose(codebook, reference)
