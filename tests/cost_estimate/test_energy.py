import pytest

from .estimate import estimate_energy


@pytest.mark.parametrize(
    "dtype_a,dtype_w",
    [
        ("float32", "float32"),
        ("float16", "float16"),
        ("float8", "float8"),
        ("int8", "int8"),
        ("float32", "ternary"),
        ("float16", "ternary"),
        ("float8", "ternary"),
        ("int8", "ternary"),
    ],
)
@pytest.mark.parametrize(
    "arch,arch_kwargs",
    [
        ("transformer", {"blocks": 12, "channels": 128, "mlp_ratio": 4, "attn_ratio": 1}),
        (
            "particletransformer",
            {
                "blocks": 12,
                "channels": 128,
                "channels_pair": 64,
                "layers_pair": 3,
                "mlp_ratio": 4,
                "attn_ratio": 1,
            },
        ),
        (
            "llocatransformer",
            {
                "blocks": 12,
                "channels": 128,
                "mlp_ratio": 4,
                "attn_ratio": 1,
                "channels_framesnet": 128,
                "layers_framesnet": 2,
            },
        ),
        (
            "lorentztransformer",
            {"blocks": 12, "channels_v": 32, "channels_s": 64, "mlp_ratio": 4, "attn_ratio": 1},
        ),
    ],
)
def test_energy_1k(arch, arch_kwargs, dtype_a, dtype_w, seqlen=50, dtype_fp="float32"):
    arch_kwargs["seqlen"] = seqlen
    modes = ["literature", "A100-estimate", "H100-estimate"]
    energies = []
    for mode in modes:
        energy = estimate_energy(
            arch,
            arch_kwargs,
            dtype_a=dtype_a,
            dtype_w=dtype_w,
            dtype_fp=dtype_fp,
            mode=mode,
        )
        energies.append(energy)
    print(
        f"{arch:<20} dtype_a={dtype_a:<10} dtype_w={dtype_w:<10}: {energies[0]:.1e} (lit) {energies[1]:.1e} (A100 est) {energies[2]:.1e} (H100 est)"
    )
