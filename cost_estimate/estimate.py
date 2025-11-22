"""
Global comments
- Neglect contributions from linear_in and linear_out
- Neglect terms that are O(channels*seqlen), except if they use bits_fp
- Bitops estimates the number of multiplications; we roughly say that additions = multiplications
"""


def linear_cost(dim_1, dim_2, factor, factor_bias=0):
    cost_mul = dim_1 * dim_2 * factor
    cost_bias = dim_2 * factor_bias  # multiplicative factor
    cost = cost_mul + cost_bias
    return cost


def transformer_cost(
    blocks,
    seqlen,
    channels,
    mlp_ratio=4,
    attn_ratio=1,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # attention projections
    cost_attnproj = linear_cost(
        dim_1=channels, dim_2=channels * attn_ratio, factor=factor_aw, factor_bias=factor_aa
    )
    # - factor 4 for Q, K, V, output
    cost_attnproj *= 4 * seqlen

    # attention
    cost_attn = linear_cost(
        dim_1=seqlen,
        dim_2=seqlen,
        factor=factor_aa,
    )
    # - factor 2 for A=Q*K and O=A*V
    cost_attn *= 2 * channels * attn_ratio

    # MLP projections
    cost_mlp = linear_cost(
        dim_1=channels, dim_2=channels * mlp_ratio, factor=factor_aw, factor_bias=factor_aa
    )
    # - factor 2 for proj_in, proj_out
    cost_mlp *= 2 * seqlen

    # layer normalization
    # - factor 2 for pre-attn and pre-mlp
    # - factor 3 for square, mean, normalization
    cost_ln = 2 * 3 * factor_fpfp * seqlen * channels

    cost = cost_attnproj + cost_attn + cost_mlp + cost_ln
    cost *= blocks
    return cost


def llocatransformer_cost(
    blocks,
    seqlen,
    channels,
    mlp_ratio=4,
    attn_ratio=1,
    channels_framesnet=128,
    layers_framesnet=2,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # llocatransformer uses transformer backbone
    cost_transformer = transformer_cost(
        blocks=blocks,
        seqlen=seqlen,
        channels=channels,
        mlp_ratio=mlp_ratio,
        attn_ratio=attn_ratio,
        factor_aw=factor_aw,
        factor_aa=factor_aa,
        factor_fpfp=factor_fpfp,
    )

    # frame-to-frame transformations
    # - factor 4 for f2f_QKV (3) and f2f_output (1)
    cost_frame2frame = 4 * blocks * seqlen * channels * factor_fpfp

    # estimate this based on FLOPs (uses bits_fp)
    num_edges = (seqlen + 3) * (seqlen + 2)  # because of spurions
    cost_framesnet_in = linear_cost(
        dim_1=15,
        dim_2=channels_framesnet,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    cost_framesnet_out = linear_cost(
        dim_1=channels_framesnet,
        dim_2=3,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    cost_framesnet_middle = linear_cost(
        dim_1=channels_framesnet,
        dim_2=channels_framesnet,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    cost_framesnet_middle *= layers_framesnet - 2
    cost_framesnet = cost_framesnet_in + cost_framesnet_out + cost_framesnet_middle
    cost_framesnet *= num_edges

    cost = cost_transformer + cost_frame2frame + cost_framesnet
    return cost


def lgatr_linear_cost(ch1_mv, ch2_mv, ch1_s, ch2_s, factor, factor_bias=0):
    cost_s2s = ch1_s * ch2_s * factor
    cost_2s2_bias = ch2_s * factor_bias
    # - factor 2 for possibility to go either to scalar or pseudoscalar
    cost_mv2s_s2mv = 2 * (ch1_s * ch2_mv + ch1_mv * ch2_s) * factor
    cost_mv2s_s2mv_bias = 2 * (ch2_mv + ch2_s) * factor_bias
    # - factor 10 for 10 linear maps on multivectors
    cost_mv2mv = 10 * ch1_mv * ch2_mv * factor
    cost_mv2mv_bias = 10 * ch2_mv * factor_bias
    cost = (
        cost_s2s
        + cost_2s2_bias
        + cost_mv2s_s2mv
        + cost_mv2s_s2mv_bias
        + cost_mv2mv
        + cost_mv2mv_bias
    )
    return cost


def lgatr_cost(
    blocks,
    seqlen,
    channels_mv,
    channels_s,
    mlp_ratio=4,
    attn_ratio=1,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # 3 spurions and 1 global token
    seqlen += 4

    # attention projections
    cost_attnproj = lgatr_linear_cost(
        ch1_mv=channels_mv,
        ch2_mv=channels_mv * attn_ratio,
        ch1_s=channels_s,
        ch2_s=channels_s * attn_ratio,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    # - factor 4 for Q, K, V, output
    cost_attnproj *= 4 * seqlen

    # attention
    # - factor 16 from multivector inner product in attention matrix
    cost_attn_QK = factor_aa * seqlen**2 * (channels_s + 16 * channels_mv)
    # - factor 16 from A * mv with scalar A and 16-component mv
    cost_attn_AV = factor_aa * seqlen**2 * (channels_s + 16 * channels_mv)
    cost_attn = cost_attn_QK + cost_attn_AV

    # MLP projections
    # - factor 16**2 from 16x16->16 outer product
    cost_tensorproduct = factor_aa * channels_mv * 16**2
    cost_leftright = lgatr_linear_cost(
        ch1_mv=channels_mv,
        ch2_mv=channels_mv * mlp_ratio,
        ch1_s=channels_s,
        ch2_s=0,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    # - factor 2 for proj_in_left, proj_in_right
    cost_leftright *= 2
    cost_hidden = lgatr_linear_cost(
        ch1_mv=channels_mv * mlp_ratio,
        ch2_mv=channels_mv * mlp_ratio,
        ch1_s=channels_s,
        ch2_s=channels_s * mlp_ratio,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    cost_out = lgatr_linear_cost(
        ch1_mv=channels_mv * mlp_ratio,
        ch2_mv=channels_mv,
        ch1_s=channels_s * mlp_ratio,
        ch2_s=channels_s,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    cost_mlp = seqlen * (cost_tensorproduct + cost_leftright + cost_hidden + cost_out)

    # layer normalization
    # - factor 2 for pre-attn and pre-mlp
    # - factor 3 for square, mean, normalization
    cost_ln = 2 * 3 * factor_fpfp * seqlen * (channels_s + 16 * channels_mv)

    cost = cost_attnproj + cost_attn + cost_mlp + cost_ln
    cost *= blocks
    return cost


def particletransformer_cost(
    blocks,
    seqlen,
    channels,
    channels_pair,
    layers_pair=3,
    mlp_ratio=4,
    attn_ratio=1,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # - neglect difference between self-attention and class-attention blocks
    # - neglect cost of adding edge features to attention scores
    cost_transformer = transformer_cost(
        blocks=blocks,
        seqlen=seqlen,
        channels=channels,
        mlp_ratio=mlp_ratio,
        attn_ratio=attn_ratio,
        factor_aw=factor_aw,
        factor_aa=factor_aa,
        factor_fpfp=factor_fpfp,
    )

    # learnable attention bias
    # - factor 4 for 4 edge features mij, dR2, kT, z
    cost_pairembed = 4 * seqlen**2 * channels_pair**2 * factor_aw
    cost_pairembed *= layers_pair

    cost = cost_transformer + cost_pairembed
    return cost


def lotr_linear_cost(ch1_v, ch2_v, ch1_s, ch2_s, factor, factor_bias=0):
    cost_s2s = ch1_s * ch2_s * factor
    cost_s2s_bias = ch2_s * factor_bias
    # - factor 4 for 4 components of vector
    cost_v2v = 4 * ch1_v * ch2_v * factor
    cost_v2v_bias = 4 * ch2_v * factor_bias
    cost = cost_s2s + cost_s2s_bias + cost_v2v + cost_v2v_bias
    return cost


def lorentztransformer_cost(
    blocks,
    seqlen,
    channels_v,
    channels_s,
    mlp_ratio=4,
    attn_ratio=1,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # 3 spurions and 1 global token
    seqlen += 4

    # attention projections
    cost_attnproj = lotr_linear_cost(
        ch1_v=channels_v,
        ch2_v=channels_v * attn_ratio,
        ch1_s=channels_s,
        ch2_s=channels_s * attn_ratio,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    # - factor 4 for Q, K, V, output
    cost_attnproj *= 4 * seqlen

    # attention
    # - factor 4 from multivector inner product in attention matrix
    cost_attn_QK = factor_aa * seqlen**2 * (channels_s + 4 * channels_v) * attn_ratio
    # - factor 4 from A * mv with scalar A and 4-component mv
    cost_attn_AV = factor_aa * seqlen**2 * (channels_s + 4 * channels_v) * attn_ratio
    # - factor 3 for square, mean, normalization
    # - factor 3 for normalizing Q, K, V
    cost_attn_norm = 3 * 3 * factor_fpfp * seqlen * (channels_s + 4 * channels_v) * attn_ratio
    cost_attn = cost_attn_QK + cost_attn_AV + cost_attn_norm

    # MLP projections
    cost_in = lotr_linear_cost(
        ch1_v=channels_v,
        ch2_v=3 * channels_v * mlp_ratio,
        ch1_s=channels_s,
        ch2_s=2 * channels_s * mlp_ratio,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    # neglect inner products (attention will dominate)
    cost_out = lotr_linear_cost(
        ch1_v=channels_v * mlp_ratio,
        ch2_v=channels_v,
        ch1_s=channels_s * mlp_ratio,
        ch2_s=channels_s,
        factor=factor_aw,
        factor_bias=factor_aa,
    )
    cost_mlp = seqlen * (cost_in + cost_out)

    # layer normalization
    # - factor 2 for pre-attn and pre-mlp
    # - factor 3 for square, mean, normalization
    cost_ln = 2 * 3 * factor_fpfp * seqlen * (channels_s + 4 * channels_v)

    cost = cost_attnproj + cost_attn + cost_mlp + cost_ln
    cost *= blocks
    return cost


def get_cost_func(architecture):
    if architecture == "transformer":
        return transformer_cost
    elif architecture == "llocatransformer":
        return llocatransformer_cost
    elif architecture == "lgatr":
        return lgatr_cost
    elif architecture == "particletransformer":
        return particletransformer_cost
    elif architecture == "lorentztransformer":
        return lorentztransformer_cost
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def estimate_flops(
    architecture: str,
    arch_kwargs,
):
    func = get_cost_func(architecture)
    factors = dict(factor_aw=1, factor_aa=1, factor_fpfp=1)
    mul = func(
        **arch_kwargs,
        **factors,
    )
    # - factor 2 for additions + multiplications
    flops = 2 * mul
    return flops


def estimate_bitops(
    architecture: str,
    arch_kwargs,
    bits_a,
    bits_w,
    bits_fp,
):
    func = get_cost_func(architecture)
    factors = dict(factor_aw=bits_a * bits_w, factor_aa=bits_a**2, factor_fpfp=bits_fp**2)
    bitops = func(
        **arch_kwargs,
        **factors,
    )
    return bitops


def estimate_energy(
    architecture: str,
    arch_kwargs,
    dtype_a="float32",
    dtype_w="float32",
    dtype_fp="float32",
    mode="literature",
):
    if mode == "literature":

        def get_energy(mul_op, dtype):
            return get_energy_cost_7nm_literature(mul_op, dtype=dtype)
    elif mode == "H100-estimate":

        def get_energy(mul_op, dtype):
            return 0.5 * get_energy_cost_estimate(machine="H100-PCle", dtype=dtype)
    elif mode == "A100-estimate":

        def get_energy(mul_op, dtype):
            return 0.5 * get_energy_cost_estimate(machine="A100-PCle", dtype=dtype)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    def get_energy_MAC(dtype):
        return get_energy(mul_op=True, dtype=dtype) + get_energy(mul_op=False, dtype=dtype)

    func = get_cost_func(architecture)
    factor_aa = get_energy_MAC(dtype=dtype_a)
    factor_fpfp = get_energy_MAC(dtype=dtype_fp)
    if dtype_w == "ternary":
        factor_aw = get_energy(mul_op=False, dtype=dtype_a)
    else:
        assert dtype_w == dtype_a
        factor_aw = get_energy_MAC(dtype=dtype_a)
    factors = dict(factor_aw=factor_aw, factor_aa=factor_aa, factor_fpfp=factor_fpfp)
    energy = func(
        **arch_kwargs,
        **factors,
    )
    return energy


def get_energy_cost_7nm_literature(mul_op, dtype):
    # https://arxiv.org/pdf/2112.00133 table 1
    if dtype == "float32":
        return 1.310 if mul_op else 0.380
    elif dtype == "float16":
        return 0.340 if mul_op else 0.160
    elif dtype == "bfloat16" or dtype == "float8":
        return 0.210 if mul_op else 0.110
    elif dtype == "int32":
        return 1.480 if mul_op else 0.030
    elif dtype == "int8":
        return 0.070 if mul_op else 0.007
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_energy_cost_estimate(machine, dtype):
    if machine == "H100-PCle":
        # https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet
        power = 350
        if dtype == "float64":
            tflops = 51
        elif dtype == "float32":
            tflops = 756
        elif dtype in ["float16", "bfloat16"]:
            tflops = 1513
        elif dtype in ["float8", "int8"]:
            tflops = 3026
    elif machine == "A100-PCle":
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
        power = 300
        if dtype == "float64":
            tflops = 19.5
        elif dtype == "float32":
            tflops = 156
        elif dtype in ["float16", "bfloat16", "float8"]:
            tflops = 312
        elif dtype == "int8":
            tflops = 624

    energy_pJ = power / tflops
    return energy_pJ
