from lloca.equivectors import LGATrVectors, MLPVectors, PELICANVectors
from parq.optim import (
    ProxBinaryRelax,
    ProxHardQuant,
    ProxPARQ,
    ProxSoftQuant,
    build_quant_optimizer,
)
from parq.quant import (
    LSBQuantizer,
    MaxUnifQuantizer,
    TernaryUnifQuantizer,
    UnifQuantizer,
)
from parq.quant.uniform import AsymUnifQuantizer

from experiments.floatquant import FloatQuantizer, TorchFloatQuantizer
from experiments.logger import LOGGER


def get_quantizer(name, bits):
    name = name.lower()
    if name == "lsbq":
        return LSBQuantizer()
    elif bits == 0:
        return TernaryUnifQuantizer()
    elif name == "uniform":
        return UnifQuantizer()
    elif name == "asymuniform":
        return AsymUnifQuantizer()
    elif name == "maxuniform":
        return MaxUnifQuantizer()
    elif "float" in name:  # e.g. floate5m2 or Float-E4M3
        em = name.split("e")[-1]
        e, m = em.split("m")
        assert int(e) + int(m) + 1 == bits, "Bits do not match exponent and mantissa"
        if "torch" in name:
            return TorchFloatQuantizer(int(e), int(m))
        else:
            return FloatQuantizer(int(e), int(m))
    else:
        raise ValueError(f"Unknown quantizer {name}")


def init_parq_optimizer(base_optimizer, cfg):
    quantizer = get_quantizer(cfg.weightquant.quantizer, cfg.weightquant.bits)

    start_step = cfg.weightquant.start_step
    end_step = cfg.training.iterations - cfg.weightquant.final_hard_steps

    if cfg.weightquant.prox_map == "parq":
        prox_map = ProxPARQ(start_step, end_step, steepness=cfg.weightquant.steepness)
    elif cfg.weightquant.prox_map == "soft":
        prox_map = ProxSoftQuant(start_step, end_step)
    elif cfg.weightquant.prox_map == "hard":
        prox_map = ProxHardQuant()
    elif cfg.weightquant.prox_map == "binaryrelax":
        prox_map = ProxBinaryRelax(start_step, end_step)
    else:
        raise ValueError(f"Prox map {cfg.weightquant.prox_map} not implemented")

    optimizer = build_quant_optimizer(
        base_optimizer=base_optimizer,
        quantizer=quantizer,
        prox_map=prox_map,
        warmup_steps=cfg.weightquant.warmup_steps,
        quant_period=cfg.weightquant.quant_period,
        quant_per_channel=cfg.weightquant.quant_per_channel,
        quant_shrink=cfg.weightquant.quant_shrink,
        anneal_wd_frac=cfg.weightquant.anneal_wd_frac,
        nm_gamma=cfg.weightquant.nm_gamma,
    )
    return optimizer


def init_parq_param_groups(model, cfg, modelname, param_groups=None):
    if modelname in ["Transformer", "LGATr", "LorentzTransformer"]:
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
        f"Fraction of quantized parameters: {num_params_quantized}/{num_params_total} ({num_params_quantized / num_params_total * 100:.2f}%)"
    )

    return param_groups


def init_param_groups_framesnet(framesnet):
    if isinstance(framesnet.equivectors, MLPVectors):
        params_framesnet = []
        params_framesnet_inout = []
        layers = framesnet.equivectors.block.mlp.mlp
        for i, layer in enumerate(layers):
            if i == 0 or i == len(layers) - 1:
                params_framesnet_inout += list(layer.parameters())
            else:
                params_framesnet += list(layer.parameters())
    elif isinstance(framesnet.equivectors, PELICANVectors):
        raise NotImplementedError("PELICANVectors not supported in PARQ yet")
    elif isinstance(framesnet.equivectors, LGATrVectors):
        raise NotImplementedError("LGATrVectors not supported in PARQ yet")
    else:
        raise ValueError("Unknown equivectors type in framesnet")
    return params_framesnet, params_framesnet_inout


def init_param_groups_transformer(model, cfg):
    # collect parameters in groups
    params_inout = list(model.net.linear_in.parameters()) + list(model.net.linear_out.parameters())
    params_attn = []
    params_mlp = []
    for block in model.net.blocks:
        params_attn += list(block.attention.parameters())
        params_mlp += list(block.mlp.parameters())

    params_noq = []
    params_framesnet, params_framesnet_inout = init_param_groups_framesnet(model.framesnet)
    return param_groups_transformer_helper(
        params_framesnet,
        params_framesnet_inout,
        params_inout,
        params_attn,
        params_mlp,
        params_noq,
        cfg,
    )


def init_param_groups_ParticleTransformer(model, cfg):
    # Note: be extra careful with distributing weight decay across parameter groups
    # See experiments/tagging/experiment.py in the _init_optimizer() function
    # Mindset: no weight decay for class token, layer normalization, c_attn, w_resid and biases
    # or in other words, weight_decay only for weight matrices in linear layers

    # collect parameters in groups
    params_inout = []
    params_attn = []
    params_mlp = []
    params_noq = [model.net.cls_token] + list(model.net.norm.parameters())

    # carefully seperate inout (inputs/outputs need high precision) and mlp
    for i, m in enumerate(model.net.embed.embed):
        if i <= 3:
            params_inout += list(m.parameters())
        else:
            params_mlp += list(m.parameters())
    params_inout += list(model.net.pair_embed.parameters())
    params_inout += list(model.net.fc.parameters())

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

    params_framesnet, params_framesnet_inout = init_param_groups_framesnet(model.framesnet)
    return param_groups_transformer_helper(
        params_framesnet,
        params_framesnet_inout,
        params_inout,
        params_attn,
        params_mlp,
        params_noq,
        cfg,
    )


def param_groups_transformer_helper(
    params_framesnet, params_framesnet_inout, params_inout, params_attn, params_mlp, params_noq, cfg
):
    def is_bias(param):
        return param.ndim == 1

    def sort_bias(in_list, out_nobias, out_bias):
        out_nobias += [p for p in in_list if not is_bias(p)]
        out_bias += [p for p in in_list if is_bias(p)]

    # divide backbone parameters
    params_q_wd = []
    params_noq_wd = []
    if cfg.weightquant.inout:
        sort_bias(params_inout, params_q_wd, params_noq)
    else:
        sort_bias(params_inout, params_noq_wd, params_noq)
    if cfg.weightquant.attn:
        sort_bias(params_attn, params_q_wd, params_noq)
    else:
        sort_bias(params_attn, params_noq_wd, params_noq)
    if cfg.weightquant.mlp:
        sort_bias(params_mlp, params_q_wd, params_noq)
    else:
        sort_bias(params_mlp, params_noq_wd, params_noq)

    param_groups = [
        {
            "params": params_q_wd,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "quant_bits": cfg.weightquant.bits,
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

    framesnet_params_q_wd = []
    framesnet_params_noq_wd = []
    framesnet_params_noq = []
    if cfg.weightquant.framesnet:
        if cfg.weightquant.inout:
            sort_bias(params_framesnet_inout, framesnet_params_q_wd, framesnet_params_noq)
        else:
            sort_bias(params_framesnet_inout, framesnet_params_noq_wd, framesnet_params_noq)
        sort_bias(params_framesnet, framesnet_params_q_wd, framesnet_params_noq)
    else:
        sort_bias(params_framesnet_inout, framesnet_params_noq_wd, framesnet_params_noq)
        sort_bias(params_framesnet, framesnet_params_noq_wd, framesnet_params_noq)

    param_groups += [
        {
            "params": framesnet_params_q_wd,
            "lr": cfg.training.lr_factor_framesnet * cfg.training.lr,
            "weight_decay": cfg.training.weight_decay_framesnet,
            "quant_bits": cfg.weightquant.bits,
        },
        {
            "params": framesnet_params_noq,
            "lr": cfg.training.lr_factor_framesnet * cfg.training.lr,
        },
        {
            "params": framesnet_params_noq_wd,
            "lr": cfg.training.lr_factor_framesnet * cfg.training.lr,
            "weight_decay": cfg.training.weight_decay_framesnet,
        },
    ]

    return param_groups
