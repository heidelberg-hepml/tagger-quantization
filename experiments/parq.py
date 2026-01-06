from contextlib import contextmanager

from lloca.equivectors import MLPVectors
from lloca.framesnet.equi_frames import LearnedFrames
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

from experiments.floatquant import FloatQuantizer


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
    elif name == "float":
        assert bits in [4, 6, 8, 16], "Float quantizer only supports 4, 6, 8 or 16 bits"
        return FloatQuantizer(bits)
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
    if modelname in ["Transformer", "LGATr", "LGATrSlim"]:
        # Transformer and LGATr use the same high-level module syntax
        assert (
            hasattr(cfg, "finetune") or param_groups is None
        )  # not manually specified for these models
        param_groups = init_param_groups_transformer(model, cfg)
    elif modelname == "ParticleTransformer":
        # params_groups is given, but we have to do it again including the quantization split
        param_groups = init_param_groups_ParticleTransformer(model, cfg)
    else:
        raise NotImplementedError(f"PARQ not implemented for model {modelname}")
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
    else:
        # TODO: implement for other equivectors
        raise NotImplementedError(
            "Weight quantization for framesnet currently only implemented for MLPVectors"
        )
    return params_framesnet, params_framesnet_inout


def init_param_groups_transformer(model, cfg):
    # collect parameters in groups
    params_in = list(model.net.linear_in.parameters())
    params_out = list(model.net.linear_out.parameters())
    params_attn = []
    params_mlp = []
    for block in model.net.blocks:
        params_attn += list(block.attention.parameters())
        params_mlp += list(block.mlp.parameters())

    params_noq = []

    if isinstance(model.framesnet, LearnedFrames):
        params_framesnet, params_framesnet_inout = init_param_groups_framesnet(model.framesnet)
    else:
        params_framesnet = []
        params_framesnet_inout = []

    return param_groups_transformer_helper(
        params_framesnet,
        params_framesnet_inout,
        params_in,
        params_out,
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
    params_in = []
    params_attn = []
    params_mlp = []
    params_noq = [model.net.cls_token] + list(model.net.norm.parameters())

    # carefully seperate inout (inputs/outputs need high precision) and mlp

    for i, m in enumerate(model.net.embed.embed):
        if i <= 3:
            params_in += list(m.parameters())
        else:
            params_mlp += list(m.parameters())
    params_in += list(model.net.embed.input_bn.parameters())
    params_in += list(model.net.pair_embed.parameters())
    params_out = list(model.net.fc.parameters())

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

    if isinstance(model.framesnet, LearnedFrames):
        params_framesnet, params_framesnet_inout = init_param_groups_framesnet(model.framesnet)
    else:
        params_framesnet = []
        params_framesnet_inout = []

    return param_groups_transformer_helper(
        params_framesnet,
        params_framesnet_inout,
        params_in,
        params_out,
        params_attn,
        params_mlp,
        params_noq,
        cfg,
    )


def param_groups_transformer_helper(
    params_framesnet,
    params_framesnet_inout,
    params_in,
    params_out,
    params_attn,
    params_mlp,
    params_noq,
    cfg,
):
    def is_bias(param):
        return param.ndim == 1

    def sort_bias(in_list, out_nobias, out_bias):
        out_nobias += [p for p in in_list if not is_bias(p)]
        out_bias += [p for p in in_list if is_bias(p)]

    base_lr = cfg.training.lr if not hasattr(cfg, "finetune") else cfg.finetune.lr_backbone
    head_lr = cfg.training.lr if not hasattr(cfg, "finetune") else cfg.finetune.lr_head

    # divide backbone parameters
    params_q_wd = []
    params_noq_wd = []
    if cfg.weightquant.inout:
        sort_bias(params_in, params_q_wd, params_noq)
    else:
        sort_bias(params_in, params_noq_wd, params_noq)
    if cfg.weightquant.attn:
        sort_bias(params_attn, params_q_wd, params_noq)
    else:
        sort_bias(params_attn, params_noq_wd, params_noq)
    if cfg.weightquant.mlp:
        sort_bias(params_mlp, params_q_wd, params_noq)
    else:
        sort_bias(params_mlp, params_noq_wd, params_noq)

    # divide head parameters
    params_q_wd_head = []
    params_noq_wd_head = []
    params_noq_head = []
    if cfg.weightquant.inout:
        sort_bias(params_out, params_q_wd_head, params_noq_head)
    else:
        sort_bias(params_out, params_noq_wd_head, params_noq_head)
    param_groups = [
        {
            "params": params_q_wd,
            "lr": base_lr,
            "weight_decay": cfg.training.weight_decay,
            "quant_bits": cfg.weightquant.bits,
        },
        {
            "params": params_noq,
            "lr": base_lr,
        },
        {
            "params": params_noq_wd,
            "lr": base_lr,
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": params_q_wd_head,
            "lr": head_lr,
            "weight_decay": cfg.training.weight_decay,
            "quant_bits": cfg.weightquant.bits,
        },
        {
            "params": params_noq_head,
            "lr": head_lr,
        },
        {
            "params": params_noq_wd_head,
            "lr": head_lr,
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
            "lr": cfg.training.lr_factor_framesnet * base_lr,
            "weight_decay": cfg.training.weight_decay_framesnet,
            "quant_bits": cfg.weightquant.bits,
        },
        {
            "params": framesnet_params_noq,
            "lr": cfg.training.lr_factor_framesnet * base_lr,
        },
        {
            "params": framesnet_params_noq_wd,
            "lr": cfg.training.lr_factor_framesnet * base_lr,
            "weight_decay": cfg.training.weight_decay_framesnet,
        },
    ]

    return param_groups


def quantize_model(model, cfg):
    """
    Quantize model parameters according to config settings.

    Parameters
    ----------
    model: The PyTorch model to quantize
    cfg: Config containing weightquant settings

    Returns
    -------
    model: The PyTorch model with quantized weights
    original_params: Dictionary of original parameters indexed by parameter id
    """
    if cfg.weightquant.bits == 0:
        # Max uniform quantization with two bits preserves ternary quantization
        quantizer = get_quantizer("maxuniform", 2)
        bits = 2
    elif cfg.weightquant.quantizer in ["float", "maxuniform"]:
        quantizer = get_quantizer(cfg.weightquant.quantizer, cfg.weightquant.bits)
        bits = cfg.weightquant.bits
    else:
        raise ValueError(
            "Scale is not preserved over multiple quantizations for quantizer"
            f" {cfg.weightquant.quantizer} with {cfg.weightquant.bits} bits"
        )
    modelname = cfg.model.net._target_.rsplit(".", 1)[-1]
    param_groups = init_parq_param_groups(model, cfg, modelname)
    original_params = {}
    # Quantize parameters that have quant_bits specified
    for group in param_groups:
        if "quant_bits" in group:
            for p in group["params"]:
                # Store original
                param_id = id(p)
                original_params[param_id] = p.data.clone()
                # Quantize
                p.data, _ = quantizer.quantize(p=p.data, b=bits)
    return model, original_params


def restore_model(model, original_params):
    """
    Restore model parameters from original_params.

    Parameters
    ----------
    model: The PyTorch model to restore
    original_params: Dictionary of original parameters indexed by parameter id

    Returns
    -------
    model: The PyTorch model with parameters restored from original_params
    """
    for p in model.parameters():
        param_id = id(p)
        if param_id in original_params:
            p.data = original_params[param_id]
    return model


@contextmanager
def temporary_quantize(model, cfg):
    """
    Yields quantized model for evaluation, then restore .

    Parameters
    ----------
    model: The PyTorch model to quantize
    cfg: Config containing weightquant settings
    """

    model, original_params = quantize_model(model, cfg)
    try:
        yield
    finally:
        restore_model(model, original_params)
