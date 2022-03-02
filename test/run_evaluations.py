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


subnet_width_values = [128, 256, 512, 1024]
subnet_depth_values = [1, 2, 3, 4, 5]
model_depth_values = [2, 10, 20]

for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
    print("--------------------------------")
    print("running ")
    run_dir = f"../evaluations/simple_model/supervised_{sw}_{sd}_{md}"
    data_dir = f"../data"
    kwargs = {
        "subnet_width": sw,
        "subnet_depth": sd,
        "model_depth": md,
        "run_dir": run_dir,
        "data_dir": data_dir
    }
    p = Process(target=train_supervised, kwargs=kwargs)
    p.start()
    p.join()
    # train_supervised(subnet_width=sw,
    #                  subnet_depth=sd,
    #                  model_depth=md,
    #                  run_dir=run_dir,
    #                  data_dir=data_dir)