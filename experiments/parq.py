from experiments.logger import LOGGER
from parq.optim import (
    ProxBinaryRelax,
    ProxHardQuant,
    ProxPARQ,
    ProxSoftQuant,
    build_quant_optimizer,
)
from parq.quant import LSBQuantizer, UnifQuantizer


def init_parq_optimizer(base_optimizer, cfg):
    quantizer = UnifQuantizer() if cfg.parq.uniform else LSBQuantizer()

    start_step = cfg.parq.start_step
    end_step = cfg.training.iterations - cfg.parq.final_hard_steps

    if cfg.parq.prox_map == "parq":
        prox_map = ProxPARQ(start_step, end_step, steepness=cfg.parq.steepness)
    elif cfg.parq.prox_map == "soft":
        prox_map = ProxSoftQuant(start_step, end_step)
    elif cfg.parq.prox_map == "hard":
        prox_map = ProxHardQuant()
    elif cfg.parq.prox_map == "binaryrelax":
        prox_map = ProxBinaryRelax(start_step, end_step)
    else:
        raise ValueError(f"Prox map {cfg.parq.prox_map} not implemented")

    optimizer = build_quant_optimizer(
        base_optimizer=base_optimizer,
        quantizer=quantizer,
        prox_map=prox_map,
        warmup_steps=cfg.parq.warmup_steps,
        quant_period=cfg.parq.quant_period,
        quant_per_channel=cfg.parq.quant_per_channel,
        quant_shrink=cfg.parq.quant_shrink,
        anneal_wd_frac=cfg.parq.anneal_wd_frac,
        nm_gamma=cfg.parq.nm_gamma,
    )
    return optimizer


def init_parq_param_groups(model, cfg, modelname, param_groups=None):
    if modelname in ["Transformer", "LGATr"]:
        # Transformer and LGATr use the same high-level module syntax
        assert param_groups is None  # not manually specified for these models
        param_groups = init_param_groups_transformer(model, cfg)
    elif modelname == "ParticleTransformer":
        # params_groups is given, but we have to do it again including the quantization split
        param_groups = init_param_groups_ParticleTransformer(model, cfg)
    else:
        raise NotImplementedError(f"PARQ not implemented for model {modelname}")

    num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_quantized = sum(
        p.numel()
        for group in param_groups
        for p in group["params"]
        if p.requires_grad and "quant_bits" in group
    )
    LOGGER.info(
        f"Fraction of quantized parameters: {num_params_quantized}/{num_params_total} ({num_params_quantized/num_params_total*100:.2f}%)"
    )

    return param_groups


def init_param_groups_transformer(model, cfg):
    # collect parameters in groups
    params_inout = list(model.net.linear_in.parameters()) + list(
        model.net.linear_out.parameters()
    )
    params_attn = []
    params_mlp = []
    for block in model.net.blocks:
        params_attn += list(block.attention.parameters())
        params_mlp += list(block.mlp.parameters())

    def include_params(in_list, out_quant, out_unquant):
        if cfg.parq.bias:
            out_quant += in_list
        else:
            is_bias = lambda param: param.ndim == 1
            out_quant += [p for p in in_list if not is_bias(p)]
            out_unquant += [p for p in in_list if is_bias(p)]

    # divide backbone parameters
    params_q = []
    params_noq = []
    if cfg.parq.inout:
        include_params(params_inout, params_q, params_noq)
    else:
        params_noq += params_inout
    if cfg.parq.attn:
        include_params(params_attn, params_q, params_noq)
    else:
        params_noq += params_attn
    if cfg.parq.mlp:
        include_params(params_mlp, params_q, params_noq)
    else:
        params_noq += params_mlp

    param_groups = [
        {
            # never quantize framesnet
            "params": list(model.framesnet.parameters()),
            "lr": cfg.training.lr_factor_framesnet * cfg.training.lr,
            "weight_decay": cfg.training.weight_decay_framesnet,
        },
        {
            "params": params_q,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "quant_bits": cfg.parq.bits,
        },
        {
            "params": params_noq,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
        },
    ]
    return param_groups


def init_param_groups_ParticleTransformer(model, cfg):
    # Note: be extra careful with distributing weight decay across parameter groups
    # See experiments/tagging/experiment.py in the _init_optimizer() function
    # Mindset: no weight decay for class token, layer normalization, c_attn, w_resid and biases
    # or in other words, weight_decay only for weight matrices in linear layers

    # collect parameters in groups
    params_inout = (
        list(model.net.embed.parameters())
        + list(model.net.fc.parameters())
        + list(model.net.pair_embed.parameters())
    )
    params_attn = []
    params_mlp = []
    params_noq = [model.net.cls_token] + list(model.net.norm.parameters())
    for block in model.net.blocks + model.net.cls_blocks:
        params_attn += list(block.attn.parameters())
        params_mlp += list(block.fc1.parameters()) + list(block.fc2.parameters())
        params_noq += (
            [block.c_attn]
            + [block.w_resid]
            + list(block.pre_attn_norm.parameters())
            + list(block.post_attn_norm.parameters())
            + list(block.pre_fc_norm.parameters())
            + list(block.post_fc_norm.parameters())
        )

    is_bias = lambda param: param.ndim == 1

    def include_params(in_list, out_quant_wd, out_quant_nowd, out_unquant):
        out_quant_wd += [p for p in in_list if not is_bias(p)]
        if cfg.parq.bias:
            out_quant_nowd += [p for p in in_list if is_bias(p)]
        else:
            out_unquant += [p for p in in_list if is_bias(p)]

    # divide backbone parameters
    params_q_nowd = []
    params_q_wd = []
    params_noq_wd = []
    if cfg.parq.inout:
        include_params(params_inout, params_q_wd, params_q_nowd, params_noq)
    else:
        params_noq_wd += [p for p in params_inout if not is_bias(p)]
        params_noq += [p for p in params_inout if is_bias(p)]
    if cfg.parq.attn:
        include_params(params_attn, params_q_wd, params_q_nowd, params_noq)
    else:
        params_noq_wd += [p for p in params_attn if not is_bias(p)]
        params_noq += [p for p in params_attn if is_bias(p)]
    if cfg.parq.mlp:
        include_params(params_mlp, params_q_wd, params_q_nowd, params_noq)
    else:
        params_noq_wd += [p for p in params_mlp if not is_bias(p)]
        params_noq += [p for p in params_mlp if is_bias(p)]

    param_groups = [
        {
            # never quantize framesnet
            "params": list(model.framesnet.parameters()),
            "lr": cfg.training.lr_factor_framesnet * cfg.training.lr,
            "weight_decay": cfg.training.weight_decay_framesnet,
        },
        {
            "params": params_q_wd,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "quant_bits": cfg.parq.bits,
        },
        {
            "params": params_q_nowd,
            "lr": cfg.training.lr,
            "quant_bits": cfg.parq.bits,
        },
        {
            "params": params_noq,
            "lr": cfg.training.lr,
        },
        {
            "params": params_noq_wd,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
        },
    ]
    return param_groups
