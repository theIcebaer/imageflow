import itertools
from imageflow.trainer import train_supervised
from multiprocessing import Process

# def test_basic_cinn_architecture(subnet_width, subnet_depth, model_depth):
#     run_dir = f"../evaluations/simple_model/supervised_{subnet_width}_{subnet_depth}_{model_depth}"
#     data_dir = f"../data"
#     train_supervised(subnet_width=subnet_width,
#                      subnet_depth=subnet_depth,
#                      model_depth=model_depth,
#                      run_dir=run_dir,
#                      data_dir=data_dir)
#     return None

from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18
from imageflow.nets import CondNetWrapper


wrapper_test_data = [
    ("mobilenet", (64, 960, 1, 1)),
    ("resnet", (64, 512, 1, 1))
]

pretrain = ['fixed', 'fine_tune_last', 'fine_tune_full', 'none']


# for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
for pretrain, (net_name, arch_params) in itertools.product(pretrain, wrapper_test_data):
    # cnet = config[0]
    # net_name = config[1]
    # arch_params = config[2]

    print("--------------------------------")
    print("running ")
    print(f"pretrain: {pretrain} \ncond_net: {net_name}, \nout_shape: {arch_params}")
    run_dir = f"supervised_{net_name}_{pretrain}"
    data_dir = f"../../../data"
    wrapped_cond_net = CondNetWrapper(net_type=net_name, pretraining=pretrain)
    # for param in wrapped_cond_net.parameters():
    #     print( param.requires_grad)

    kwargs = {

        # "init_params": {"gain": },
        "file_name": "mnist_rnd_distortions_10.hdf5",
        "run_dir": run_dir,
        "data_dir": data_dir,
        "init_method": "xavier",
        "model_type": "basic",
        "cond_net": wrapped_cond_net,
        "pretraining": pretrain,
        "n_epochs": 30
    }
    train_supervised(**kwargs)
    # p = Process(target=train_supervised, kwargs=kwargs)
    # p.start()
    # p.join()
