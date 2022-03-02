# In this file I want to test different supervised learning strategies.
import pytest
import torch
from imageflow.nets import CondNetWrapper
from imageflow.nets import CinnBasic
from imageflow.trainer import train_supervised
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18


subnet_width_values = [128, 256, 512, 1024]
subnet_depth_values = [1, 2, 3, 4, 5]
model_depth_values = [2, 10, 20]

@pytest.mark.parametrize("subnet_width", subnet_width_values)
@pytest.mark.parametrize("subnet_depth", subnet_depth_values)
@pytest.mark.parametrize("model_depth", model_depth_values)
def test_basic_cinn_architecture(subnet_width, subnet_depth, model_depth):
    run_dir = f"../evaluations/simple_model/supervised_{subnet_width}_{subnet_depth}_{model_depth}"
    data_dir = f"../data"
    train_supervised(subnet_width=subnet_width,
                     subnet_depth=subnet_depth,
                     model_depth=model_depth,
                     run_dir=run_dir,
                     data_dir=data_dir)

    # make heatmap with subnet width vs. depth

    # make plot with nr. parameters vs final loss for all 3 parameter axes (i.e. subnet params and model depth).

    # make plot with
    return True

# n_epochs_values = [20, 40, 60]
# lr_scheduler_values = [("MultiStep", ),
#                        ("Annealing", ),
#                        ("AnnealingWarmRestarts", )]
#
# @pytest.mark.parametrize("n_epochs", n_epochs_values)
# @pytest.mark.parametrize("scheduler, scheduler_config")
# def eval_basic_cinn_training_dynamics(n_epochs, scheduler, sheduler_config):
#     pass
#
# def eval_MultiResCINN_architecture():
#     pass