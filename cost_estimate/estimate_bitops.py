import json

from cost_estimate.estimate import estimate_bitops, estimate_flops

SEQLEN = 50
ARCHS = ["transformer", "particletransformer", "llocatransformer", "lorentztransformer", "lgatr"]
BITS = [
    (32, 32),
    (16, 16),
    (8, 8),
    (32, 2),
    (16, 2),
    (8, 2),
]


def get_arch_kwargs(arch):
    if arch == "transformer":
        return dict(blocks=1, channels=16, mlp_ratio=1, attn_ratio=1)
    elif arch == "particletransformer":
        return dict(blocks=2, channels=8, channels_pair=4, layers_pair=3, mlp_ratio=4, attn_ratio=1)
    elif arch == "llocatransformer":
        return dict(
            blocks=1,
            channels=16,
            mlp_ratio=1,
            attn_ratio=1,
            channels_framesnet=4,
            layers_framesnet=2,
        )
    elif arch == "lorentztransformer":
        return dict(blocks=1, channels_v=1, channels_s=16, mlp_ratio=1, attn_ratio=1)
    elif arch == "lorentztransformer":
        return dict(blocks=1, channels_v=1, channels_s=16, mlp_ratio=1, attn_ratio=1)
    elif arch == "lgatr":
        return dict(blocks=1, channels_mv=3, channels_s=8, mlp_ratio=1, attn_ratio=1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def main(save=True):
    results = dict()
    for arch in ARCHS:
        arch_kwargs = get_arch_kwargs(arch)
        arch_kwargs["seqlen"] = SEQLEN

        results_sub = dict()
        for bits_a, bits_w in BITS:
            bitops = estimate_bitops(
                arch,
                arch_kwargs,
                bits_a=bits_a,
                bits_w=bits_w,
                bits_fp=32,
            )
            flops = estimate_flops(arch, arch_kwargs)
            print(
                f"{arch:<20} bits_a={bits_a:>2} bits_w={bits_w:>2}: bitops={bitops:.1e}, flops={flops:.1e}"
            )
            results_sub[f"{bits_a},{bits_w}"] = bitops
        results[arch] = results_sub

    if save:
        with open("cost_estimate/bitops.json", "w") as file:
            json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
