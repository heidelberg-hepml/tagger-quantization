import torch
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


def init_param_groups_transformer(model, cfg):
    # collect parameters in groups
    params_inout = list(model.net.linear_in.parameters()) + list(model.net.linear_out.parameters())
    params_attn = []
    params_mlp = []
    for block in model.net.blocks:
        params_attn += list(block.attention.parameters())
        params_mlp += list(block.mlp.parameters())

    params_noq = []
    params_framesnet = list(model.framesnet.parameters())
    return param_groups_transformer_helper(
        params_framesnet, params_inout, params_attn, params_mlp, params_noq, cfg
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

    params_framesnet = list(model.framesnet.parameters())
    return param_groups_transformer_helper(
        params_framesnet, params_inout, params_attn, params_mlp, params_noq, cfg
    )


def param_groups_transformer_helper(
    params_framesnet, params_inout, params_attn, params_mlp, params_noq, cfg
):
    def is_bias(param):
        return param.ndim == 1

    def include_params(in_list, out_q_wd, out_noq):
        out_q_wd += [p for p in in_list if not is_bias(p)]
        out_noq += [p for p in in_list if is_bias(p)]

    # divide backbone parameters
    params_q_wd = []
    params_noq_wd = []
    if cfg.weightquant.inout:
        include_params(params_inout, params_q_wd, params_noq)
    else:
        params_noq_wd += [p for p in params_inout if not is_bias(p)]
        params_noq += [p for p in params_inout if is_bias(p)]
    if cfg.weightquant.attn:
        include_params(params_attn, params_q_wd, params_noq)
    else:
        params_noq_wd += [p for p in params_attn if not is_bias(p)]
        params_noq += [p for p in params_attn if is_bias(p)]
    if cfg.weightquant.mlp:
        include_params(params_mlp, params_q_wd, params_noq)
    else:
        params_noq_wd += [p for p in params_mlp if not is_bias(p)]
        params_noq += [p for p in params_mlp if is_bias(p)]

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
    framesnet_params_noq = [p for p in params_framesnet if is_bias(p)]
    weights = [p for p in params_framesnet if not is_bias(p)]
    if cfg.weightquant.framesnet:
        framesnet_params_q_wd += weights
    else:
        framesnet_params_noq_wd += weights

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


def compute_ternary_entropy(model, cfg, modelname):
    param_groups = init_parq_param_groups(model, cfg, modelname)
    num_params = 0
    num_pos_params = 0
    num_neg_params = 0
    num_zero_params = 0
    for group in param_groups:
        if "quant_bits" in group:
            assert group["quant_bits"] == 0
            for p in group["params"]:
                p_data = p.cpu()
                num_params += p_data.numel()
                num_pos_params += (p_data > 0).sum().item()
                num_neg_params += (p_data < 0).sum().item()
                num_zero_params += (p_data == 0).sum().item()
    prob_pos = num_pos_params / num_params
    prob_neg = num_neg_params / num_params
    prob_zero = num_zero_params / num_params
    entropy = 0.0
    for p in [prob_pos, prob_neg, prob_zero]:
        if p > 0:
            entropy -= p * torch.log2(torch.tensor(p)).item()
    return {"entropy": entropy, "prob_pos": prob_pos, "prob_neg": prob_neg, "prob_zero": prob_zero}