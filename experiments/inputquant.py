from contextlib import contextmanager

import torch
from lgatr.layers import EquiLinear
from lgatr.nets.lgatr_slim import Linear as SlimEquiLinear
from lloca.equivectors import MLPVectors
from lloca.framesnet.equi_frames import LearnedFrames
from torch import Tensor
from torch.nn import Conv1d, Linear, Module

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
    if cfg_inputs.mlp:
        for i, m in enumerate(model.net.embed.embed):
            if i > 3:
                input_quantize_module(module=m, cfg=cfg_inputs)
        for i, m in enumerate(model.net.pair_embed.embed):
            if i > 3:
                input_quantize_module(module=m, cfg=cfg_inputs)

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
        static=cfg.static,
        quant_per_channel=cfg.quant_per_channel,
        match_weightquant=cfg.match_weightquant,
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
        elif isinstance(child, Conv1d):
            new_layer = QuantConv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
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
        static: bool = True,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        **kwargs,
    ):
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits
        self.match_weightquant = match_weightquant
        self.dim = 1 if quant_per_channel else None
        super().__init__(*args, **kwargs)
        self.static = static
        if static:
            self.register_buffer("min_val", torch.tensor(float("inf")))
            self.register_buffer("max_val", torch.tensor(float("-inf")))
            self.register_buffer("scale", torch.tensor(1.0))
            self.register_buffer("zeropoint", torch.tensor(0.0))
            self.obs_count = 0
            self.obs_stop = 100000
            self.quantile = 1e-3

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
            if self.static:
                if self.training and self.obs_count < self.obs_stop:
                    self.observe(input)
                flat_q = quantize_int8(flat, self.scale, self.zeropoint)
            else:
                flat_q, _ = self.quantizer.quantize(flat, self.bits, self.dim)
        input_q = flat_q.view(shape)
        output = input + (input_q - input).detach()
        return output

    @contextmanager
    def quantize_params(self):
        original_params = []
        for param in self.parameters():
            original_params.append(param.data.clone())
            if self.match_weightquant:
                param.data = self.ste_quantize(param.data)
        try:
            yield
        finally:
            for param, original in zip(self.parameters(), original_params, strict=True):
                param.data = original

    def observe(self, input: Tensor):
        flat = input.detach().reshape(-1)
        q_min, q_max = torch.quantile(flat, self.quantile), torch.quantile(flat, 1 - self.quantile)
        if self.obs_count == 0:
            self.min_val = q_min
            self.max_val = q_max
        else:
            momentum = self.obs_count / (self.obs_count + 1)
            update_weight = 1 / (self.obs_count + 1)
            self.min_val = momentum * self.min_val + update_weight * q_min
            self.max_val = momentum * self.max_val + update_weight * q_max
        self.obs_count += 1
        range_val = torch.clamp(
            self.max_val - self.min_val,
            min=torch.tensor(torch.finfo(self.scale.dtype).eps, device=self.scale.device),
        )
        self.scale = range_val / 256
        self.zeropoint = -128 - torch.round(self.min_val / self.scale)


class QuantLinear(QuantLayer, Linear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: bool = True,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            static=static,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            **kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = QuantLayer.ste_quantize(self, input)
        with self.quantize_params():
            output = Linear.forward(self, input)
        return output


class QuantConv1d(QuantLayer, Conv1d):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            **kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        input = QuantLayer.ste_quantize(self, input)
        with self.quantize_params():
            output = Linear.forward(self, input)
        return output


class QuantEquiLinear(QuantLayer, EquiLinear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: bool = True,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            static=static,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            **kwargs,
        )

    def forward(self, multivectors: Tensor, scalars: Tensor | None) -> tuple[Tensor, Tensor | None]:
        multivectors = QuantLayer.ste_quantize(self, multivectors)
        if scalars is not None:
            scalars = QuantLayer.ste_quantize(self, scalars)
        with self.quantize_params():
            output_mv, output_s = EquiLinear.forward(self, multivectors, scalars)
        return output_mv, output_s


class QuantSlimEquiLinear(QuantLayer, SlimEquiLinear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: bool = True,
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            quantizer=quantizer,
            bits=bits,
            static=static,
            quant_per_channel=quant_per_channel,
            match_weightquant=match_weightquant,
            **kwargs,
        )

    def forward(self, vectors: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        vectors = QuantLayer.ste_quantize(self, vectors)
        scalars = QuantLayer.ste_quantize(self, scalars)
        with self.quantize_params():
            vectors_out, scalars_out = SlimEquiLinear.forward(self, vectors, scalars)
        return vectors_out, scalars_out


def quantize_int8(tensor: Tensor, scale: Tensor, zeropoint: Tensor) -> Tensor:
    qmin = -128
    qmax = 127
    tensor_q = torch.clamp(torch.round(tensor / scale) + zeropoint, qmin, qmax)
    tensor_q = (tensor_q - zeropoint) * scale
    return tensor_q
