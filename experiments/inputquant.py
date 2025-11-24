from lgatr.layers import EquiLinear
from torch import Tensor
from torch.nn import Linear, init

from experiments.baselines.lorentztransformer import Linear as LorentzLinear
from experiments.logger import LOGGER

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
    if cfg_inputs.framesnet:
        input_quantize_module(
            module=model.framesnet,
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


def input_quantize_module(module, cfg):
    quant_kwargs = dict(
        quantizer=cfg.quantizer, bits=cfg.bits, dim=cfg.dim, quantize_output=cfg.quantize_output
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
        assert dim is None, (
            "Quantization scale should be shared across channels to preserve equivariance"
        )
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            dim=dim,
            quantize_output=quantize_output,
        )

    def forward(self, multivectors: Tensor, scalars: Tensor | None) -> tuple[Tensor, Tensor | None]:
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


class QuantLorentzLinear(LorentzLinear, QuantLayer):
    def __init__(
        self,
        *args,
        quantizer: str = "uniform",
        bits: int = 8,
        dim: int | None = None,
        quantize_output: bool = True,
        **kwargs,
    ):
        LorentzLinear.__init__(self, *args, **kwargs)
        QuantLayer.__init__(
            self,
            quantizer=quantizer,
            bits=bits,
            dim=dim,
            quantize_output=quantize_output,
        )

    def forward(self, vectors: Tensor, scalars: Tensor) -> Tensor:
        vectors = self.ste_quantize(vectors)
        scalars = self.ste_quantize(scalars)
        vectors_out, scalars_out = LorentzLinear.forward(self, vectors, scalars)
        if self.quantize_output:
            vectors_out = QuantLayer.ste_quantize(self, vectors_out)
            scalars_out = QuantLayer.ste_quantize(self, scalars_out)
        return vectors_out, scalars_out


def init_scaled_module(module, scale=1.0):
    LOGGER.info(f"Initializing module {module.__class__.__name__}")
    for name, child in list(module.named_children()):
        if isinstance(child, (QuantEquiLinear, EquiLinear)):
            LOGGER.info(f"Initializing EquiLinear with scale factor {scale}")
            child.reset_parameters(initialization="default", gain=scale)
            LOGGER.info(f"Weight std after scaling: {child.weight.std().item()}")
        elif isinstance(child, (QuantLorentzLinear, LorentzLinear)):
            LOGGER.info(f"Initializing LorentzLinear with scale factor {scale}")
            child.reset_parameters(initialization="default", additional_factor=scale)
            LOGGER.info(f"Weight std after scaling: {child.weight_v.std().item()}")
        elif isinstance(child, (QuantLinear, Linear)):
            LOGGER.info(f"Initializing Linear with scale factor {scale}")
            init.kaiming_uniform_(child.weight)
            child.weight.data = scale * child.weight.data
            LOGGER.info(f"Weight std after scaling: {child.weight.std().item()}")
        else:
            init_scaled_module(child, scale=scale)

def init_scaled_model(model, cfg_weights):
    LOGGER.info("Initializing model weights with scaling factor")
    for block in model.net.blocks:
        LOGGER.info(f"Initializing block {block.__class__.__name__}")
        if cfg_weights.attn:
            LOGGER.info(f"Initializing attention")
            init_scaled_module(
                module=block.attention,
                scale=cfg_weights.init_scale,
            )
        if cfg_weights.mlp:
            LOGGER.info(f"Initializing MLP")
            init_scaled_module(
                module=block.mlp,
                scale=cfg_weights.init_scale,
            )
    if cfg_weights.framesnet:
        init_scaled_module(
            module=model.framesnet,
            scale=cfg_weights.init_scale,
        )