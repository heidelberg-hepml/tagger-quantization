import torch

from experiments.floatquant import FloatQuantizer, TorchFloatQuantizer


def _assert_quantizers_match(e: int, m: int, samples: torch.Tensor) -> None:
    bits = e + m + 1
    fq = FloatQuantizer(e, m)
    tq = TorchFloatQuantizer(e, m)
    fq_q, fq_Q = fq.quantize(samples, bits)
    tq_q, tq_Q = tq.quantize(samples, bits)
    assert torch.allclose(fq_q, tq_q)
    assert torch.allclose(fq_Q, tq_Q, equal_nan=True)


def test_float_quantizers_match_e4m3():
    samples = torch.linspace(-400.0, 400.0, steps=2049, dtype=torch.float32)
    _assert_quantizers_match(4, 3, samples)


def test_float_quantizers_match_e5m2():
    samples = torch.linspace(-50000.0, 50000.0, steps=2049, dtype=torch.float32)
    _assert_quantizers_match(5, 2, samples)
