from lgatr.layers import EquiLinear
from torch import Tensor
from torch.nn import Linear

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
                cfg=cfg_inputs,
            )
        if cfg_inputs.mlp:
            input_quantize_module(
                module=block.mlp,
                cfg=cfg_inputs,
            )


def input_quantize_ParT(model, cfg_inputs):
    for block in model.net.blocks + model.net.cls_blocks:
        if cfg_inputs.attn:
            input_quantize_module(
                module=block.attn,
                cfg=cfg_inputs,
            )
        if cfg_inputs.mlp:
            input_quantize_module(
                module=block.fc1,
                cfg=cfg_inputs,
            )
            input_quantize_module(
                module=block.fc2,
                cfg=cfg_inputs,
            )


def input_quantize_LGATr(model, cfg_inputs):
    for block in model.net.blocks:
        if cfg_inputs.attn:
            input_quantize_module(
                module=block.attention,
                cfg=cfg_inputs,
            )
        if cfg_inputs.mlp:
            input_quantize_module(
                module=block.mlp,
                cfg=cfg_inputs,
            )


def input_quantize_module(module, cfg):
    for name, child in list(module.named_children()):
        if isinstance(child, Linear):
            new_layer = QuantLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                quantizer=cfg.quantizer,
                bits=cfg.bits,
                dim=cfg.dim,
                quantize_output=cfg.quantize_output,
            )
            module._modules[name] = new_layer
        elif isinstance(child, EquiLinear):
            new_layer = QuantEquiLinear(
                in_mv_channels=child.weight.shape[1],
                out_mv_channels=child.weight.shape[0],
                in_s_channels=(
                    child.s2mvs.weight.shape[1] if child.s2mvs is not None else 0
                ),
                out_s_channels=(
                    child.mvs2s.weight.shape[0] if child.mvs2s is not None else 0
                ),
                bias=(child.bias is not None),
                quantizer=cfg.quantizer,
                bits=cfg.bits,
                dim=cfg.dim,
                quantize_output=cfg.quantize_output,
            )
            module._modules[name] = new_layer
        else:
            input_quantize_module(child, cfg)

    return module


class QuantLayer:
    def __init__(
        self,
        quantizer: str = "uniform",
        bits: int = 8,
        dim: int | None = None,
        quantize_output: bool = False,
    ):
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits
        self.quantize_output = quantize_output
        self.dim = dim

    def ste_quantize(self, input: Tensor) -> Tensor:
        """
        Straight-Through Estimator to quantize activations
        """
        input_q, _ = self.quantizer.quantize(input, self.bits, self.dim)
        return input + (input_q - input).detach()


class QuantLinear(Linear, QuantLayer):
    def __init__(
        self,
        *args,
        quantizer: str = "uniform",
        bits: int = 8,
        dim: int | None = None,
        quantize_output: bool = True,
        **kwargs,
    ):
        Linear.__init__(self, *args, **kwargs)
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            dim=dim,
            quantize_output=quantize_output,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = self.ste_quantize(input)
        output = Linear.forward(self, input)
        if self.quantize_output:
            output = QuantLayer.ste_quantize(self, output)
        return output


class QuantEquiLinear(EquiLinear, QuantLayer):
    def __init__(
        self,
        *args,
        quantizer: str = "uniform",
        bits: int = 8,
        dim: int | None = None,
        quantize_output: bool = True,
        **kwargs,
    ):
        EquiLinear.__init__(self, *args, **kwargs)
        assert (
            dim is None
        ), "Quantization scale should be shared across channels to preserve equivariance"
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            dim=dim,
            quantize_output=quantize_output,
        )

    def forward(
        self, multivectors: Tensor, scalars: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        multivectors = QuantLayer.ste_quantize(self, multivectors)
        if scalars is not None:
            scalars = QuantLayer.ste_quantize(self, scalars)
        output = EquiLinear.forward(self, multivectors, scalars)
        if self.quantize_output:
            output_mv, output_s = output
            output_mv = QuantLayer.ste_quantize(self, output_mv)
            if output_s is not None:
                output_s = QuantLayer.ste_quantize(self, output_s)
            return output_mv, output_s
        else:
            return output
