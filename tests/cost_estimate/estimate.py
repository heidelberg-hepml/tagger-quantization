"""
Global comments
- Neglect contributions from linear_in and linear_out
- Neglect terms that are O(channels*seqlen), except if they use bits_fp
- Bitops estimates the number of multiplications; we roughly say that additions = multiplications
"""


def linear_cost(dim_1, dim_2, factor):
    mul = dim_1 * dim_2 * factor
    return mul


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
    mul_attnproj = linear_cost(dim_1=channels, dim_2=channels * attn_ratio, factor=factor_aw)
    # - factor 4 for Q, K, V, output
    mul_attnproj *= 4 * seqlen

    # attention
    mul_attn = linear_cost(
        dim_1=seqlen,
        dim_2=seqlen,
        factor=factor_aa,
    )
    # - factor 2 for A=Q*K and O=A*V
    mul_attn *= 2 * channels * attn_ratio

    # MLP projections
    mul_mlp = linear_cost(dim_1=channels, dim_2=channels * mlp_ratio, factor=factor_aw)
    # - factor 2 for proj_in, proj_out
    mul_mlp *= 2 * seqlen

    # layer normalization
    # - factor 2 for pre-attn and pre-mlp
    # - factor 3 for square, mean, normalization
    mul_ln = 2 * 3 * factor_fpfp * seqlen * channels

    mul = mul_attnproj + mul_attn + mul_mlp + mul_ln
    mul *= blocks
    return mul


def llocatransformer_cost(
    blocks,
    seqlen,
    channels,
    mlp_ratio=4,
    attn_ratio=1,
    channels_framesnet=128,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # llocatransformer uses transformer backbone
    mul_transformer = transformer_cost(
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
    mul_frame2frame = 4 * blocks * seqlen * channels * factor_fpfp

    # estimate this based on FLOPs (uses bits_fp)
    num_edges = (seqlen + 3) * (seqlen + 2)  # because of spurions
    mul_framesnet = (
        num_edges
        * (channels_framesnet**2 + 15 * channels_framesnet + 3 * channels_framesnet)
        * factor_fpfp
    )

    mul = mul_transformer + mul_frame2frame + mul_framesnet
    return mul


def lgatr_linear_cost(ch1_mv, ch2_mv, ch1_s, ch2_s, factor):
    s2s = ch1_s * ch2_s
    # - factor 2 for possibility to go either to scalar or pseudoscalar
    mv2s_s2mv = 2 * (ch1_s * ch2_mv + ch1_mv * ch2_s)
    # - factor 10 for 10 linear maps on multivectors
    mv2mv = 10 * ch1_mv * ch2_mv
    mul = factor * (s2s + mv2s_s2mv + mv2mv)
    return mul


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
    # attention projections
    mul_attnproj = lgatr_linear_cost(
        ch1_mv=channels_mv,
        ch2_mv=channels_mv * attn_ratio,
        ch1_s=channels_s,
        ch2_s=channels_s * attn_ratio,
        factor=factor_aw,
    )
    # - factor 4 for Q, K, V, output
    mul_attnproj *= 4 * seqlen

    # attention
    # - factor 16 from multivector inner product in attention matrix
    mul_attn_QK = factor_aa**2 * seqlen**2 * (channels_s + 16 * channels_mv)
    # - factor 16 from A * mv with scalar A and 16-component mv
    mul_attn_AV = factor_aa**2 * seqlen**2 * (channels_s + 16 * channels_mv)
    mul_attn = mul_attn_QK + mul_attn_AV

    # MLP projections
    # - factor 16**2 from 16x16->16 outer product
    mul_tensorproduct = factor_aa * channels_mv * 16**2
    mul_leftright = lgatr_linear_cost(
        ch1_mv=channels_mv,
        ch2_mv=channels_mv * mlp_ratio,
        ch1_s=channels_s,
        ch2_s=0,
        factor=factor_aw,
    )
    # - factor 2 for proj_in_left, proj_in_right
    mul_leftright *= 2
    mul_hidden = lgatr_linear_cost(
        ch1_mv=channels_mv * mlp_ratio,
        ch2_mv=channels_mv * mlp_ratio,
        ch1_s=channels_s,
        ch2_s=channels_s * mlp_ratio,
        factor=factor_aw,
    )
    mul_out = lgatr_linear_cost(
        ch1_mv=channels_mv * mlp_ratio,
        ch2_mv=channels_mv,
        ch1_s=channels_s * mlp_ratio,
        ch2_s=channels_s,
        factor=factor_aw,
    )
    mul_mlp = seqlen * (mul_tensorproduct + mul_leftright + mul_hidden + mul_out)

    # layer normalization
    # - factor 2 for pre-attn and pre-mlp
    # - factor 3 for square, mean, normalization
    mul_ln = 2 * 3 * factor_fpfp * seqlen * (channels_s + 16 * channels_mv)

    mul = mul_attnproj + mul_attn + mul_mlp + mul_ln
    mul *= blocks
    return mul


def particletransformer_cost(
    blocks,
    seqlen,
    channels,
    channels_pair,
    mlp_ratio=4,
    attn_ratio=1,
    factor_aw=1,
    factor_aa=1,
    factor_fpfp=1,
):
    # - neglect difference between self-attention and class-attention blocks
    # - neglect cost of adding edge features to attention scores
    mul_transformer = transformer_cost(
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
    mul_pairembed = 4 * seqlen**2 * channels_pair**2 * factor_aw

    mul = mul_transformer + mul_pairembed
    return mul


def lorentztransformer_cost(
    blocks,
    seqlen,
    channels_v,
    channels_s,
    mlp_ratio=4,
    attn_ratio=1,
    bits_a=1,
    bits_w=1,
    bits_fp=1,
):
    pass


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
    factors = {"factor_aw": 1, "factor_aa": 1, "factor_fpfp": 1}
    mul = func(
        **arch_kwargs,
        **factors,
    )
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
    factors = {"factor_aw": bits_a * bits_w, "factor_aa": bits_a**2, "factor_fpfp": bits_fp**2}
    bitops = func(
        **arch_kwargs,
        **factors,
    )
    return bitops


def estimate_energy(
    architecture: str,
    arch_kwargs,
    bits_a,
    bits_w,
    bits_fp,
):
    func = get_cost_func(architecture)
    factors = {"factor_aw": 1, "factor_aa": 1, "factor_fpfp": 1}
    energy = func(
        **arch_kwargs,
        **factors,
    )
    return energy
