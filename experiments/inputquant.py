from contextlib import contextmanager, nullcontext

import torch
from lgatr.layers import EquiLinear
from lgatr.nets.lgatr_slim import Linear as SlimEquiLinear
from lloca.equivectors import MLPVectors
from lloca.framesnet.equi_frames import LearnedFrames
from torch import Tensor
from torch.nn import Conv1d, Linear, Module

from experiments.parq import get_quantizer


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
        static: dict = {},
        quant_per_channel: bool = False,
        match_weightquant: bool = True,
        **kwargs,
    ):
        self.quantizer = get_quantizer(quantizer, bits)
        self.bits = bits
        self.match_weightquant = match_weightquant
        self.dim = 1 if quant_per_channel else None
        super().__init__(*args, **kwargs)
        self.static = static["use"]
        if self.static:
            self.min_val = -1
            self.max_val = 1
            self.scale = 1.0
            self.zeropoint = 0.0
            self.obs_count = 0
            self.obs_start = static["observer_start"]
            self.obs_stop = static["observer_stop"]
            self.beta = static["beta"]
            self.ema = static["ema"]
            q = static["quantile"]
            self.quantile = q

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
                if self.training:
                    if self.obs_count < self.obs_start:
                        # Dynamic quantization phase
                        flat_q = quantize_int8(flat)
                        self.obs_count += 1
                    elif self.obs_count < self.obs_stop:
                        # Observation + quantization with running statistics
                        self.observe(input)
                        self.obs_count += 1
                        flat_q = quantize_int8(flat, self.scale, self.zeropoint)
                    else:
                        # Quantization with fixed statistics
                        flat_q = quantize_int8(flat, self.scale, self.zeropoint)
                else:
                    # Quantization with fixed statistics
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
                if self.static:
                    param.data = quantize_int8(param.data)
                else:
                    param.data = self.ste_quantize(param.data)
        try:
            yield
        finally:
            for param, original in zip(self.parameters(), original_params, strict=True):
                param.data = original

    @torch.no_grad()
    def observe(self, input: Tensor):
        flat = input.detach().reshape(-1)
        q_min = torch.quantile(input=flat, q=self.quantile).item()
        q_max = torch.quantile(input=flat, q=1 - self.quantile).item()
        if self.obs_count == self.obs_start:
            self.min_val = q_min
            self.max_val = q_max
        else:
            if self.ema:
                self.min_val = self.beta * self.min_val + (1 - self.beta) * q_min
                self.max_val = self.beta * self.max_val + (1 - self.beta) * q_max
            else:
                self.min_val = min(self.min_val, q_min)
                self.max_val = max(self.max_val, q_max)
        range_val = max(self.max_val - self.min_val, torch.finfo(input.dtype).eps)
        self.scale = range_val / 255
        self.zeropoint = -128 - round(self.min_val / self.scale)


class QuantLinear(QuantLayer, Linear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: dict = {},
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
        with self.quantize_params() if self.match_weightquant else nullcontext():
            output = Linear.forward(self, input)
        return output


class QuantConv1d(QuantLayer, Conv1d):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: dict = {},
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
        with self.quantize_params() if self.match_weightquant else nullcontext():
            output = Conv1d.forward(self, input)
        return output


class QuantEquiLinear(QuantLayer, EquiLinear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: dict = {},
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
        with self.quantize_params() if self.match_weightquant else nullcontext():
            output_mv, output_s = EquiLinear.forward(self, multivectors, scalars)
        return output_mv, output_s


class QuantSlimEquiLinear(QuantLayer, SlimEquiLinear):
    def __init__(
        self,
        *args,
        quantizer: str = "float",
        bits: int = 8,
        static: dict = {},
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
        with self.quantize_params() if self.match_weightquant else nullcontext():
            vectors_out, scalars_out = SlimEquiLinear.forward(self, vectors, scalars)
        return vectors_out, scalars_out


def quantize_int8(
    tensor: Tensor, scale: Tensor | None = None, zeropoint: Tensor | None = None
) -> Tensor:
    if scale is None or zeropoint is None:
        min_val = tensor.min()
        max_val = tensor.max()
        scale = torch.clamp(max_val - min_val, min=torch.finfo(tensor.dtype).eps / 255.0)
        zeropoint = -128 - torch.round(min_val / scale)
    qmin = -128
    qmax = 127
    tensor_q = torch.clamp(torch.round(tensor / scale) + zeropoint, qmin, qmax)
    tensor_q = (tensor_q - zeropoint) * scale
    return tensor_q
