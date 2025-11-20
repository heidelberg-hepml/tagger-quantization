# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
import hydra
import pytest
from lloca.reps import TensorReps
from torch.utils.flop_counter import FlopCounterMode
from torch_operation_counter import OperationsCounterMode

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment

from .estimate import estimate_flops


def execute(exp, architecture, arch_kwargs, seqlen):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    exp._init()
    exp.init_physics()
    try:
        exp.init_model()
    except Exception:
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    # generate fake data
    iterator = iter(exp.train_loader)
    data = next(iterator)
    while data.x.shape[0] < seqlen:
        data = next(iterator)
    data.x = data.x[:seqlen]
    data.scalars = data.scalars[:seqlen]
    data.batch = data.batch[:seqlen]
    data.ptr[-1] = seqlen

    with (
        FlopCounterMode(display=False) as flop_counter,
        OperationsCounterMode(exp.model) as op_counter,
    ):
        exp._get_ypred_and_label(data)
    flops_measured = flop_counter.get_total_flops()
    print(op_counter.total_operations)
    dicts = op_counter.operations_count

    def rec_open(asd, label):
        for k, v in asd.items():
            if isinstance(v, dict):
                rec_open(v, f"{label}.{k}")
            else:
                print(label, k, v)

    rec_open(dicts, "")

    flops_estimate = estimate_flops(
        architecture,
        arch_kwargs,
    )
    # print(flop_counter.get_table(depth=10))
    return flops_estimate, flops_measured


@pytest.mark.parametrize("seqlen", [50])
@pytest.mark.parametrize(
    "attn_reps,num_heads",
    [
        # ("4x0n+1x1n", 2),
        # ("4x0n+1x1n", 8),
        # ("4x0n+1x1n", 32),
        ("4x0n+1x1n", 64),
    ],
)
@pytest.mark.parametrize("mlp_ratio,attn_ratio", [(4, 1)])  # , (2, 2), (2, 1)])
@pytest.mark.parametrize("framesnet", ["identity"])  # "identity", "learnedpd"])
def test_transformer(framesnet, seqlen, attn_reps, num_heads, mlp_ratio, attn_ratio):
    rep_dim = TensorReps(attn_reps).dim
    channels = rep_dim * num_heads // attn_ratio
    arch_kwargs = {
        "blocks": 1,
        "seqlen": seqlen,
        "channels": channels,
        "mlp_ratio": mlp_ratio,
        "attn_ratio": attn_ratio,
    }

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            "model=tag_transformer_1k",
            f"model/framesnet={framesnet}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
            "model.net.num_blocks=1",
            f"model.net.mlp_factor={mlp_ratio}",
            f"model.net.attention_factor={attn_ratio}",
            f"model.net.attn_reps={attn_reps}",
            f"model.net.num_heads={num_heads}",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)

    architecture = "transformer" if framesnet == "identity" else "llocatransformer"
    flops_estimate, flops_measured = execute(exp, architecture, arch_kwargs, seqlen)

    ratio = (flops_estimate / flops_measured - 1) * 100
    # assert abs(ratio) < 5
    print(
        f"{framesnet} channels={channels:>4}; seqlen={seqlen}; attn_ratio={attn_ratio}; mlp_ratio={mlp_ratio}: "
        f"flops_est={flops_estimate:.2e} flops_meas={flops_measured:.2e}; ratio={ratio:.0f}%"
    )


@pytest.mark.parametrize("seqlen", [50])
@pytest.mark.parametrize(
    "hidden_mv_channels,hidden_s_channels",
    [
        (16, 32),
    ],
)
@pytest.mark.parametrize("mlp_ratio,attn_ratio", [(1, 1)])  # (4, 1), (2, 2), (2, 1)])
def test_lgatr(seqlen, hidden_mv_channels, hidden_s_channels, mlp_ratio, attn_ratio):
    arch_kwargs = {
        "blocks": 1,
        "seqlen": seqlen + 3,  # spurions
        "channels_mv": hidden_mv_channels,
        "channels_s": hidden_s_channels,
        "mlp_ratio": mlp_ratio,
        "attn_ratio": attn_ratio,
    }

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            "model=tag_lgatr",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
            "model.net.num_blocks=1",
            f"model.net.mlp.increase_hidden_channels={mlp_ratio}",
            f"model.net.attention.increase_hidden_channels={attn_ratio}",
            f"model.net.hidden_mv_channels={hidden_mv_channels}",
            f"model.net.hidden_s_channels={hidden_s_channels}",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)

    architecture = "lgatr"
    flops_estimate, flops_measured = execute(exp, architecture, arch_kwargs, seqlen)

    ratio = (flops_estimate / flops_measured - 1) * 100
    print(
        f"channels_mv={hidden_mv_channels:>2}; channels_s={hidden_s_channels:>3}; seqlen={seqlen}; attn_ratio={attn_ratio}; mlp_ratio={mlp_ratio}: "
        f"flops_est={flops_estimate:.2e} flops_meas={flops_measured:.2e}; ratio={ratio:.0f}%"
    )
