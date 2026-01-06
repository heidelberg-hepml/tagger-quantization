import torch
from lgatr.layers import EquiLinear
from lgatr.nets.lgatr_slim import Linear as SlimEquiLinear
from lloca.equivectors import MLPVectors
from lloca.framesnet.equi_frames import LearnedFrames
from torch import Tensor
from torch.nn import Linear, Module

from .parq import get_quantizer


def input_quantize(model, modelname, cfg_inputs):
    # Replace linear layers by linear layers with input quantization inplace
    if modelname in ["Transformer", "LGATr", "LGATrSlim"]:
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
    if cfg_inputs.framesnet and isinstance(model.framesnet, LearnedFrames):
        if isinstance(model.framesnet.equivectors, MLPVectors):
            input_quantize_module(
                module=model.framesnet.equivectors,
                cfg=cfg_inputs,
            )
        else:
            # TODO: implement for other equivectors
            raise NotImplementedError(
                "Input quantization for framesnet currently only implemented for MLPVectors"
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


def input_quantize_module(module, cfg):
    quant_kwargs = dict(
        quantizer=cfg.quantizer,
        bits=cfg.bits,
        quant_per_channel=cfg.quant_per_channel,
        match_weightquant=cfg.match_weightquant,
        quantize_output=cfg.quantize_output,
        quantized_training=cfg.quantized_training,
    )
    for name, child in list(module.named_children()):
        if isinstance(child, Linear):
            new_layer = QuantLinear(
                child.in_features,
                child.out_features,
                bias=(child.bias is not None),
                **quant_kwargs,
            )
            module._modules[name] = new_layer
        elif isinstance(child, EquiLinear):
            new_layer = QuantEquiLinear(
                in_mv_channels=child._in_mv_channels,
                out_mv_channels=child._out_mv_channels,
                in_s_channels=(child._in_s_channels if child._in_s_channels is not None else 0),
                out_s_channels=(child._out_s_channels if child._out_s_channels is not None else 0),
                bias=child._bias,
                **quant_kwargs,
            )
            module._modules[name] = new_layer
        elif isinstance(child, SlimEquiLinear):
            new_layer = QuantSlimEquiLinear(
                in_v_channels=child._in_v_channels,
                out_v_channels=child._out_v_channels,
                in_s_channels=child._in_s_channels,
                out_s_channels=child._out_s_channels,
                bias=child._bias,
                **quant_kwargs,
            )
            module._modules[name] = new_layer
        else:
            input_quantize_module(child, cfg)

    return module


class QuantLayer(Module):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        quantize_output: bool = False,
        quantized_training: bool = True,
        **kwargs,
    ):
        if match_weightquant:
            # some quantizers do not preserve existing quantization
            # the scaling must be absmax to ensure that weights quantized during training
            # keep the same quantization when quantized on the fly
            if quantizer not in ["float", "maxuniform"]:
                raise NotImplementedError(
                    "STE quantization of the weights on the fly probably requires "
                    "a quantizer that preserves quantization (i.e. absmax scaling)"
                )
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits
        self.match_weightquant = match_weightquant
        self.quantize_output = quantize_output
        self.dim = 1 if quant_per_channel else None
        self.quantized_training = quantized_training
        super().__init__(*args, **kwargs)

    @property
    def quantize(self) -> bool:
        return self.quantized_training or (not self.training)

    def ste_quantize(self, input: Tensor) -> Tensor:
        """
        Straight-Through Estimator to quantize activations and weights
        """
        if input.numel() == 0:
            # handle empty parameter tensors
            return input
        shape = input.shape
        if input.dim() > 2:
            flat = input.view(input.size(0), -1)
        else:
            flat = input
        with torch.no_grad():
            flat_q, _ = self.quantizer.quantize(flat, self.bits, self.dim)
        input_q = flat_q.view(shape)
        output = input + (input_q - input).detach()
        return output


class QuantLinear(QuantLayer, Linear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        quantize_output: bool = True,
        quantized_training: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            quantize_output=quantize_output,
            quantized_training=quantized_training,
            **kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.quantize:
            input = QuantLayer.ste_quantize(self, input)
            if self.match_weightquant:
                for param in self.parameters():
                    param = QuantLayer.ste_quantize(self, param)
        output = Linear.forward(self, input)
        if self.quantize_output and self.quantize:
            output = QuantLayer.ste_quantize(self, output)
        return output


class QuantEquiLinear(QuantLayer, EquiLinear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        quantize_output: bool = True,
        quantized_training: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            quantize_output=quantize_output,
            quantized_training=quantized_training,
            **kwargs,
        )

    def forward(self, multivectors: Tensor, scalars: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.quantize:
            multivectors = QuantLayer.ste_quantize(self, multivectors)
            if scalars is not None:
                scalars = QuantLayer.ste_quantize(self, scalars)
            if self.match_weightquant:
                for param in self.parameters():
                    param = QuantLayer.ste_quantize(self, param)
        output_mv, output_s = EquiLinear.forward(self, multivectors, scalars)
        if self.quantize_output and self.quantize:
            output_mv = QuantLayer.ste_quantize(self, output_mv)
            if output_s is not None:
                output_s = QuantLayer.ste_quantize(self, output_s)
        return output_mv, output_s


class QuantSlimEquiLinear(QuantLayer, SlimEquiLinear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        quantize_output: bool = True,
        quantized_training: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            quantize_output=quantize_output,
            quantized_training=quantized_training,
            **kwargs,
        )

    def forward(self, vectors: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        if self.quantize:
            vectors = QuantLayer.ste_quantize(self, vectors)
            scalars = QuantLayer.ste_quantize(self, scalars)
            if self.match_weightquant:
                for param in self.parameters():
                    param = QuantLayer.ste_quantize(self, param)
        vectors_out, scalars_out = SlimEquiLinear.forward(self, vectors, scalars)
        if self.quantize_output and self.quantize:
            vectors_out = QuantLayer.ste_quantize(self, vectors_out)
            scalars_out = QuantLayer.ste_quantize(self, scalars_out)
        return vectors_out, scalars_out
