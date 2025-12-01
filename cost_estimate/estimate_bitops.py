import json

from cost_estimate.estimate import estimate_bitops, estimate_flops

SEQLEN = 50
ARCHS = [
    "transformer",
    "particletransformer",
    "llocatransformer",
    "llocatransformer-global",
    "lorentztransformer",
    "lgatr",
]
BITS = [
    (32, 32),
    (16, 16),
    (8, 8),
    (32, 2),
    (16, 2),
    (8, 2),
]

MAP = {
    32: "float32",
    16: "float16",
    8: "float8",
    2: "ternary",
}


def get_arch_kwargs(arch):
    if arch == "transformer":
        return "transformer", dict(blocks=1, channels=16, mlp_ratio=1, attn_ratio=1)
    elif arch == "particletransformer":
        return "particletransformer", dict(
            blocks=2, channels=8, channels_pair=4, layers_pair=3, mlp_ratio=4, attn_ratio=1
        )
    elif arch == "llocatransformer":
        return "llocatransformer", dict(
            blocks=1,
            channels=16,
            mlp_ratio=1,
            attn_ratio=1,
            channels_framesnet=4,
            layers_framesnet=2,
            is_global=False,
        )
    elif arch == "llocatransformer-global":
        return "llocatransformer", dict(
            blocks=1,
            channels=16,
            mlp_ratio=1,
            attn_ratio=1,
            channels_framesnet=4,
            layers_framesnet=2,
            is_global=True,
        )
    elif arch == "lorentztransformer":
        return "lorentztransformer", dict(
            blocks=1, channels_v=1, channels_s=16, mlp_ratio=1, attn_ratio=1
        )
    elif arch == "lgatr":
        return "lgatr", dict(blocks=1, channels_mv=3, channels_s=8, mlp_ratio=1, attn_ratio=1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def main(save=True):
    results = dict()
    for archname in ARCHS:
        arch, arch_kwargs = get_arch_kwargs(archname)
        arch_kwargs["seqlen"] = SEQLEN

        results_sub = dict()
        for bits_a, bits_w in BITS:
            bits_default = bits_a if bits_a > 16 else 16
            bitops = estimate_bitops(
                arch,
                arch_kwargs,
                bits_a=bits_a,
                bits_w=bits_w,
                bits_default=bits_default,
                bits_fp=32,
            )
            flops = estimate_flops(arch, arch_kwargs)
            print(
                f"{archname:<20} bits_a={bits_a:>2} bits_w={bits_w:>2}: bitops={bitops:.1e}, flops={flops:.1e}"
            )
            results_sub[f"{MAP[bits_a]},{MAP[bits_w]}"] = bitops
        results[archname] = results_sub

    if save:
        with open("cost_estimate/bitops.json", "w") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
