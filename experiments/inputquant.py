import math

import torch
from lgatr.layers import EquiLinear
from lloca.equivectors import MLPVectors
from lloca.framesnet.equi_frames import LearnedFrames
from torch import Tensor
from torch.nn import Linear, Parameter, init

from experiments.baselines.lorentztransformer import Linear as LorentzLinear

from .parq import get_quantizer


def input_quantize(model, modelname, cfg_inputs):
    # Replace linear layers by linear layers with input quantization inplace
    if modelname in ["Transformer", "LGATr", "LorentzTransformer"]:
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
            framesnet_inner_layers = model.framesnet.equivectors.block.mlp.mlp[1:-1]
            input_quantize_module(
                module=framesnet_inner_layers,
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
        quantize_output=cfg.quantize_output,
        scale_input=cfg.scale_input,
        scale_output=cfg.scale_output,
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
        elif isinstance(child, LorentzLinear):
            new_layer = QuantLorentzLinear(
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


class QuantLayer:
    def __init__(
        self,
        quantizer: str = "uniform",
        bits: int = 8,
        quant_per_channel: bool = False,
        quantize_output: bool = False,
        input_layer_dim: int | None = None,
        output_layer_dim: int | None = None,
    ):
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits
        self.quantize_output = quantize_output
        self.dim = 1 if quant_per_channel else None
        if input_layer_dim is not None:
            self.input_weight = Parameter(torch.empty(1, input_layer_dim))
            # Initialize scaling activations
            # TODO: adapt to quantization scheme
            init.kaiming_uniform_(self.input_weight, a=math.sqrt(5))
        else:
            self.input_weight = None
        if output_layer_dim is not None:
            self.output_weight = Parameter(torch.empty(1, output_layer_dim))
            # Initialize scaling weights
            # TODO: adapt to quantization scheme
            init.kaiming_uniform_(self.output_weight, a=math.sqrt(5))
        else:
            self.output_weight = None

    def ste_quantize(self, input: Tensor) -> Tensor:
        """
        Straight-Through Estimator to quantize activations
        """
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

    def scale_input(self, input: Tensor) -> Tensor:
        """
        Scale the input tensor by a learnable parameter if input_layer_dim is not None.
        This is equivalent to rescaling the activations of the network after quantization.
        """
        if self.input_weight is None:
            return input
        else:
            view_shape = (1,) * (input.dim() - 2) + self.input_weight.shape
            scale = self.input_weight.view(view_shape)
            return input * scale

    def scale_output(self, output: Tensor) -> Tensor:
        """
        Scale the output tensor by a learnable parameter if output_layer_dim is not None.
        This is equivalent to rescaling the weights of the network.
        """
        if self.output_weight is None:
            return output
        else:
            view_shape = (1,) * (output.dim() - 2) + self.output_weight.shape
            scale = self.output_weight.view(view_shape)
            return output * scale


class QuantLinear(Linear, QuantLayer):
    def __init__(
        self,
        *args,
        quantizer: str = "uniform",
        bits: int = 8,
        quant_per_channel: bool = False,
        quantize_output: bool = True,
        scale_input: bool = False,
        scale_output: bool = False,
        **kwargs,
    ):
        Linear.__init__(self, *args, **kwargs)
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            quantize_output=quantize_output,
            input_layer_dim=self.in_features if scale_input else None,
            output_layer_dim=self.out_features if scale_output else None,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = QuantLayer.ste_quantize(self, input)
        if self.scale_input:
            input = QuantLayer.scale_input(self, input)
        output = Linear.forward(self, input)
        if self.scale_output:
            output = QuantLayer.scale_output(self, output)
        if self.quantize_output:
            output = QuantLayer.ste_quantize(self, output)
        return output


class QuantEquiLinear(EquiLinear, QuantLayer):
    def __init__(
        self,
        *args,
        quantizer: str = "uniform",
        bits: int = 8,
        quant_per_channel: bool = False,
        quantize_output: bool = True,
        scale_input: bool = False,
        scale_output: bool = False,
        **kwargs,
    ):
        EquiLinear.__init__(self, *args, **kwargs)
        assert not quant_per_channel, (
            "Quantization scale should be shared across channels to preserve equivariance"
        )
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            quantize_output=quantize_output,
            input_layer_dim=1 if scale_input else None,  # shared scaling for equivariance
            output_layer_dim=1 if scale_output else None,
        )

    def forward(self, multivectors: Tensor, scalars: Tensor | None) -> tuple[Tensor, Tensor | None]:
        multivectors = QuantLayer.ste_quantize(self, multivectors)
        if self.scale_input:
            multivectors = QuantLayer.scale_input(self, multivectors)
        if scalars is not None:
            scalars = QuantLayer.ste_quantize(self, scalars)
            scalars = QuantLayer.scale_input(self, scalars)
        output_mv, output_s = EquiLinear.forward(self, multivectors, scalars)
        if self.scale_output:
            output_mv = QuantLayer.scale_output(self, output_mv)
            if output_s is not None:
                output_s = QuantLayer.scale_output(self, output_s)
        if self.quantize_output:
            output_mv = QuantLayer.ste_quantize(self, output_mv)
            if output_s is not None:
                output_s = QuantLayer.ste_quantize(self, output_s)
        return output_mv, output_s
        return output_mv, output_s


class QuantLorentzLinear(LorentzLinear, QuantLayer):
    def __init__(
        self,
        *args,
        quantizer: str = "uniform",
        bits: int = 8,
        quant_per_channel: bool = False,
        quantize_output: bool = True,
        scale_input: bool = False,
        scale_output: bool = False,
        **kwargs,
    ):
        LorentzLinear.__init__(self, *args, **kwargs)
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            quantize_output=quantize_output,
            input_layer_dim=1 if scale_input else None,  # shared scaling for equivariance
            output_layer_dim=1 if scale_output else None,
        )

    def forward(self, vectors: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        vectors = QuantLayer.ste_quantize(self, vectors)
        scalars = QuantLayer.ste_quantize(self, scalars)
        if self.scale_input:
            vectors = QuantLayer.scale_input(self, vectors)
            scalars = QuantLayer.scale_input(self, scalars)
        vectors_out, scalars_out = LorentzLinear.forward(self, vectors, scalars)
        if self.scale_output:
            vectors_out = QuantLayer.scale_output(self, vectors_out)
            scalars_out = QuantLayer.scale_output(self, scalars_out)
        if self.quantize_output:
            vectors_out = QuantLayer.ste_quantize(self, vectors_out)
            scalars_out = QuantLayer.ste_quantize(self, scalars_out)
        return vectors_out, scalars_out


def init_scaled_module(module, scale=1.0):
    for child in list(module.children()):
        if isinstance(child, (QuantEquiLinear, EquiLinear)):
            child.reset_parameters(initialization="default", gain=scale)
        elif isinstance(child, (QuantLorentzLinear, LorentzLinear)):
            child.reset_parameters(initialization="default", additional_factor=scale)
        elif isinstance(child, (QuantLinear, Linear)):
            init.kaiming_uniform_(child.weight)
            child.weight.data = scale * child.weight.data
        else:
            init_scaled_module(child, scale=scale)


def init_scaled_model(model, cfg_weights):
    for block in model.net.blocks:
        if cfg_weights.attn:
            init_scaled_module(
                module=block.attention,
                scale=cfg_weights.init_scale,
            )
        if cfg_weights.mlp:
            init_scaled_module(
                module=block.mlp,
                scale=cfg_weights.init_scale,
            )
    if cfg_weights.framesnet:
        init_scaled_module(
            module=model.framesnet,
            scale=cfg_weights.init_scale,
        )
