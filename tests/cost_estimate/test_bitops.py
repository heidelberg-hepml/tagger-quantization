import pytest

from .estimate import estimate_bitops


@pytest.mark.parametrize(
    "bits_a,bits_w", [(32, 32), (16, 16), (8, 8), (32, 2), (16, 2), (8, 2), (8, 1)]
)
@pytest.mark.parametrize(
    "arch,arch_kwargs",
    [
        ("transformer", {"blocks": 1, "channels": 16, "mlp_ratio": 1, "attn_ratio": 1}),
        (
            "particletransformer",
            {
                "blocks": 2,
                "channels": 8,
                "channels_pair": 4,
                "layers_pair": 3,
                "mlp_ratio": 4,
                "attn_ratio": 1,
            },
        ),
        (
            "llocatransformer",
            {
                "blocks": 1,
                "channels": 16,
                "mlp_ratio": 1,
                "attn_ratio": 1,
                "channels_framesnet": 4,
                "layers_framesnet": 2,
            },
        ),
        (
            "lorentztransformer",
            {"blocks": 1, "channels_v": 1, "channels_s": 16, "mlp_ratio": 1, "attn_ratio": 1},
        ),
    ],
)
def test_bitops_1k(arch, arch_kwargs, bits_a, bits_w, seqlen=8, bits_fp=32):
    arch_kwargs["seqlen"] = seqlen
    bitops = estimate_bitops(
        arch,
        arch_kwargs,
        bits_a=bits_a,
        bits_w=bits_w,
        bits_fp=bits_fp,
    )
    print(f"{arch:<20} bits_a={bits_a:>2} bits_w={bits_w:>2}: {bitops:.1e}")
