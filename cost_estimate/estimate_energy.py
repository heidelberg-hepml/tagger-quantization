import json

from cost_estimate.estimate import estimate_energy

SEQLEN = 50
ARCHNAMES = [
    "transformer",
    "particletransformer",
    "llocatransformer",
    "lgatr-slim",
    "lgatr",
]
DTYPES = [
    ("float32", "float32"),
    ("float16", "float16"),
    ("float8", "float8"),
    ("int8", "int8"),
    ("float32", "ternary"),
    ("float16", "ternary"),
    ("float8", "ternary"),
    ("int8", "ternary"),
]


def get_arch_kwargs(arch):
    if arch == "transformer":
        return "transformer", dict(blocks=12, channels=128, mlp_ratio=4, attn_ratio=1)
    elif arch == "particletransformer":
        return "particletransformer", dict(
            blocks=12, channels=128, channels_pair=64, layers_pair=3, mlp_ratio=4, attn_ratio=1
        )
    elif arch == "llocatransformer":
        return "llocatransformer", dict(
            blocks=12,
            channels=128,
            mlp_ratio=4,
            attn_ratio=1,
            channels_framesnet=128,
            layers_framesnet=2,
        )
    elif arch == "lgatr-slim":
        return "lgatr-slim", dict(
            blocks=12, channels_v=32, channels_s=96, mlp_ratio=4, attn_ratio=1
        )
    elif arch == "lgatr":
        return "lgatr", dict(blocks=12, channels_mv=16, channels_s=32, mlp_ratio=2, attn_ratio=2)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def main(save=True):
    results = dict()
    for archname in ARCHNAMES:
        arch, arch_kwargs = get_arch_kwargs(archname)
        arch_kwargs["seqlen"] = SEQLEN

        results_sub = dict()
        for dtype_a, dtype_w in DTYPES:
            dtype_default = dtype_a if dtype_a == "float32" else "float16"
            results_subsub = []
            for mode in ["Horowitz", "A100-estimate", "H100-estimate"]:
                energy = estimate_energy(
                    arch,
                    arch_kwargs,
                    dtype_default=dtype_default,
                    dtype_a=dtype_a,
                    dtype_w=dtype_w,
                    dtype_fp="float32",
                    mode=mode,
                )
                results_subsub.append(energy)
            print(
                f"{archname:<20} dtype_a={dtype_a:<10} dtype_w={dtype_w:<10}: {results_subsub[0]:.1e} (lit) {results_subsub[1]:.1e} (A100 est) {results_subsub[2]:.1e} (H100 est)"
            )

            results_sub[f"{dtype_a},{dtype_w}"] = results_subsub
        results[archname] = results_sub

    if save:
        with open("cost_estimate/energy.json", "w") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
