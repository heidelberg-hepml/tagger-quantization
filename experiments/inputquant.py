from torch.nn import Linear
from lgatr.layers import EquiLinear

from .parq import get_quantizer


def input_quantize(model, modelname, cfg_inputs):
    # Replace linear layers by linear layers with input quantization inplace
    if modelname in ["Transformer", "LGATr"]:
        input_quantize_transformer(model, cfg_inputs)
    elif modelname == "ParticleTransformer":
        input_quantize_ParT(model, cfg_inputs)
    else:
        raise ValueError(f"Input quantization not implemented for {modelname}")


def input_quantize_transformer(model, cfg_inputs):
    for block in model.net.blocks:
        if cfg_inputs.attn:
            input_quantize_module(
                module=block.attention,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )
        if cfg_inputs.mlp:
            input_quantize_module(
                module=block.mlp,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )


def input_quantize_ParT(model, cfg_inputs):
    for block in model.net.blocks + model.net.cls_blocks:
        if cfg_inputs.attn:
            input_quantize_module(
                module=block.attn,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )
        if cfg_inputs.mlp:
            input_quantize_module(
                module=block.fc1,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )
            input_quantize_module(
                module=block.fc2,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )


def input_quantize_LGATr(model, cfg_inputs):
    for block in model.net.blocks:
        if cfg_inputs.attn:
            input_quantize_module(
                module=block.attention,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )
        if cfg_inputs.mlp:
            input_quantize_module(
                module=block.mlp,
                quantizer=cfg_inputs.quantizer,
                bits=cfg_inputs.bits,
            )


def input_quantize_module(module, quantizer, bits):
    for name, child in list(module.named_children()):
        if isinstance(child, Linear):
            new_layer = QuantLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                quantizer=quantizer,
                bits=bits,
            )
            module._modules[name] = new_layer
        elif isinstance(child, EquiLinear):
            new_layer = QuantEquiLinear(
                in_mv_channels=child.weight.shape[1],
                out_mv_channels=child.weight.shape[0],
                in_s_channels=child.s2mvs.weight.shape[1]
                if child.s2mvs is not None
                else 0,
                out_s_channels=child.mvs2s.weight.shape[0]
                if child.mvs2s is not None
                else 0,
                bias=(child.bias is not None),
                quantizer=quantizer,
                bits=bits,
            )
            module._modules[name] = new_layer
        else:
            input_quantize_module(child, quantizer, bits)

    return module


class QuantLinear(Linear):
    def __init__(self, *args, quantizer="uniform", bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits

    def forward(self, input):
        input_q, _ = self.quantizer.quantize(input, self.bits)
        return super().forward(input_q)


class QuantEquiLinear(EquiLinear):
    def __init__(self, *args, quantizer="uniform", bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits

    def forward(self, multivectors, scalars):
        # TODO: check multivector quantization
        # It should be applied for all multivector channels equally,
        # otherwise we violate equivariance
        multivectors_q, _ = self.quantizer.quantize(multivectors, self.bits)
        scalars_q, _ = self.quantizer.quantize(scalars, self.bits)
        return super().forward(multivectors_q, scalars_q)
