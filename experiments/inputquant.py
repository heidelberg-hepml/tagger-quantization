from contextlib import contextmanager, nullcontext

import torch
from lgatr.layers import EquiLinear
from lgatr.nets.lgatr_slim import Linear as SlimEquiLinear
from lloca.equivectors import MLPVectors
from lloca.framesnet.equi_frames import LearnedFrames
from torch import Tensor
from torch.nn import Conv1d, Linear, Module

from experiments.parq import get_quantizer
from experiments.quantizer import IntQuantizer


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
            assert isinstance(self.quantizer, IntQuantizer), (
                "Static quantization only supported for IntQuantizer"
            )
        self.observer = Observer(
            method=static["method"],
            dim=self.dim,
            quantile=static["quantile"],
            beta=static["beta"],
            obs_start=static["observer_start"],
            obs_stop=static["observer_stop"],
        )

    def ste_quantize(self, input: Tensor, observe: bool = True) -> Tensor:
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
            if self.static and observe:
                self.observer(flat)
                flat_q, _ = self.quantizer.quantize(
                    flat, self.bits, self.dim, self.observer.min_val, self.observer.max_val
                )
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
                param.data = self.ste_quantize(param.data, observe=False)
        try:
            yield
        finally:
            for param, original in zip(self.parameters(), original_params, strict=True):
                param.data = original


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


class Observer(torch.nn.Module):
    def __init__(
        self,
        method: str = "absolute",
        dim: int | None = None,
        quantile: float = 0,
        beta: float = 1.0,
        obs_start: int = 0,
        obs_stop: int = 0,
    ):
        super().__init__()
        self.method = method
        self.dim = dim
        self.quantile = quantile
        self.beta = beta
        self.observation_count = 0
        self.obs_start = obs_start
        self.obs_stop = obs_stop if obs_stop > 0 else float("inf")
        assert self.obs_start < self.obs_stop, "observer_start must be less than observer_stop"
        self.register_buffer("min_val", None)
        self.register_buffer("max_val", None)

    def observe(self, input: Tensor):
        q_min = torch.quantile(input, q=self.quantile, dim=self.dim, keepdim=True)
        q_max = torch.quantile(input, q=1 - self.quantile, dim=self.dim, keepdim=True)
        self.update(q_min, q_max)

    def init_vals(self, input: Tensor):
        if self.dim is None:
            self.min_val = torch.quantile(input, q=self.quantile)
            self.max_val = torch.quantile(input, q=1 - self.quantile)
        else:
            self.min_val = torch.quantile(input, q=self.quantile, dim=self.dim, keepdim=True)
            self.max_val = torch.quantile(input, q=1 - self.quantile, dim=self.dim, keepdim=True)

    def update(self, new_min: Tensor, new_max: Tensor):
        if self.method == "ema":
            self.min_val = self.beta * self.min_val + (1 - self.beta) * new_min
            self.max_val = self.beta * self.max_val + (1 - self.beta) * new_max
        elif self.method == "cma":
            n = self.observation_count - self.obs_start + 1
            self.min_val = (self.min_val * (n - 1) + new_min) / n
            self.max_val = (self.max_val * (n - 1) + new_max) / n
        elif self.method == "absolute":
            self.min_val = torch.min(
                self.min_val, torch.tensor(new_min, device=self.min_val.device)
            )
            self.max_val = torch.max(
                self.max_val, torch.tensor(new_max, device=self.max_val.device)
            )

    @torch.no_grad()
    def forward(self, input: Tensor):
        if self.observation_count == self.obs_start:
            self.init_vals(input)
        elif self.obs_start < self.observation_count < self.obs_stop:
            self.observe(input)
        self.observation_count += 1
