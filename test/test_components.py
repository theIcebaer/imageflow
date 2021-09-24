import pytest
import torch
from imageflow.nets import CondNetWrapper
from imageflow.nets import CinnBasic
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18


wrapper_test_data = [
    (mobilenet_v3_large(pretrained=True), "mobilenet", (64, 960, 1, 1)),
    (resnet18(pretrained=True), "resnet", (64, 512, 1, 1))
]


@pytest.mark.parametrize("cond_net, net_type, cond_shape", wrapper_test_data)
@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
def test_component_cond_net_wrapper(cond_net, net_type, cond_shape, device):
    wrapped_cond_net = CondNetWrapper(cond_net=cond_net, net_type=net_type)
    wrapped_cond_net.to(device)
    # test forward pass and shape of wrapped mobilenet output
    cond_data = torch.randn((64, 2, 28, 28), device=device)
    processed_cond = wrapped_cond_net(cond_data)
    assert processed_cond.shape == cond_shape
    # TODO: seed everything and check for exact values.


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@pytest.mark.parametrize("init_method", ["gaussian", "xavier"])
@pytest.mark.parametrize("cond_net, net_type, cond_shape", wrapper_test_data)
def test_component_basic_cinn(cond_net, net_type, cond_shape, init_method, device):
    wrapped_cond_net = CondNetWrapper(cond_net=cond_net, net_type=net_type).to(device)
    cinn = CinnBasic(device=device, cond_net=wrapped_cond_net, init_method=init_method)

    cond_data = torch.randn((64, 2, 28, 28), device=device)
    input_data = torch.randn((64, 2, 28, 28), device=device)

    # forward pass
    z, jac = cinn(input_data, c=cond_data)
    assert z.shape == (64, 1568)
    assert jac.shape == (64,)

