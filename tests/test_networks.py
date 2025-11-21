import hydra
import pytest

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment


@pytest.mark.parametrize(
    "framesnet",
    [
        "identity",
        "learnedpd",
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=tag_ParT"],
        ["model=tag_transformer"],
        ["model=tag_lgatr"],
        ["model=tag_lotr"],
    ],
)
def test_network_quantization(framesnet, model_list):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    try:
        exp.init_model()
    except Exception:
        # Continue when trying to construct LLoCa-LGATr
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp._init_optimizer()
    exp._init_scaler()

    data = next(iter(exp.train_loader))
    loss, _ = exp._batch_loss(data)
    exp.scaler.scale(loss).backward()  # populate gradients
    exp.optimizer.zero_grad()  # optimizer resets gradients

    # check that the optimizer sees all network parameters
    for name, param in exp.model.named_parameters():
        # print(name, param.shape)
        assert param.grad is None, f"Parameter {name} has gradient before backward."
