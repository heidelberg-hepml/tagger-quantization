import json

from cost_estimate.estimate import estimate_bitops, estimate_flops

ARCHS = [
    "transformer",
    "llocatransformer",
    "llocatransformer-global",
]
BITS = [
    (32, 32),
    (16, 16),
    (8, 8),
    (8, 1.58),
]

MAP = {
    32: "float32",
    16: "float16",
    8: "float8",
    1.58: "ternary",
}


def get_arch_kwargs(arch):
    if arch == "transformer":
        return "transformer", dict(blocks=1, channels=16, mlp_ratio=1)
    elif arch == "llocatransformer":
        return "llocatransformer", dict(
            blocks=1,
            channels=16,
            mlp_ratio=1,
            channels_framesnet=4,
            layers_framesnet=2,
            is_global=False,
        )
    elif arch == "llocatransformer-global":
        return "llocatransformer", dict(
            blocks=1,
            channels=16,
            mlp_ratio=1,
            channels_framesnet=4,
            layers_framesnet=2,
            is_global=True,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def main(save=True):
    results = dict()
    for seqlen in [30, 50]:
        for archname in ARCHS:
            arch, arch_kwargs = get_arch_kwargs(archname)
            arch_kwargs["seqlen"] = seqlen

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
                    f"{seqlen}: {archname:<20} bits_a={bits_a:>2} bits_w={bits_w:>2}: bitops={bitops:.1e}, flops={flops:.1e}"
                )
                results_sub[f"{MAP[bits_a]},{MAP[bits_w]}"] = bitops
            results[archname] = results_sub

        if save:
            with open(f"cost_estimate/bitops_{seqlen}.json", "w") as file:
                json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
