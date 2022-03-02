import itertools
import os.path

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


subnet_width_values = [128, 256, 512, 1024]
subnet_depth_values = [1, 2, 3, 4, 5]
model_depth_values = [2, 10, 20]

init_method = ['xavier', 'gaussian', 'kaiming']#, 'kaiming_uniform', 'xavier_uniform']
init_params = {key: {} for key in init_method}

gain_list = [1.0, 0.6, 0.3, 0.1, 0.01, 0.001]
means_list = [0.1, 0.01, 0.001, 0.0001]
stds_list = [0.1, 0.01, 0.001]

conf = {
    "xavier":  [{'gain':x} for x in gain_list],
    "kaiming":  [None],
    "gaussian":  [{"std":s} for s in stds_list],
    "kaiming_uniform": [None],
    "xavier_uniform": [{'gain':x} for x in gain_list]
}


for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
# for method in init_method:
#     for params in conf[method]:
        print("--------------------------------")
        print("running ")
        print(f"method: {method} \n params: {params}")
        run_dir = f"training_dynamics/supervised_{method}"
        if method == "gaussian":
            run_dir += f"_{params['std']}"
        elif method == 'xavier':
            run_dir += f"_{params['gain']}"
        else:
            pass
        if os.path.isdir(run_dir):
            print("skipping")
            continue
        data_dir = f"../../data"
        kwargs = {
            "n_epochs": 60,
            "init_params": params,
            "run_dir": run_dir,
            "data_dir": data_dir,
            "init_method": method
        }
        p = Process(target=train_supervised, kwargs=kwargs)
        p.start()
        p.join()
        # train_supervised(subnet_width=sw,
        #                  subnet_depth=sd,
        #                  model_depth=md,
        #                  run_dir=run_dir,
        #                  data_dir=data_dir)