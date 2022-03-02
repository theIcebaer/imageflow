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


# subnet_width_values = [128, 256, 512, 1024]
# subnet_depth_values = [1, 2, 3, 4, 5]
# model_depth_values = [2, 10, 20]
#
# init_method = ['xavier', 'gaussian', 'kaiming']
# init_params = {key: {} for key in init_method}
#
# gain_list = [1.0, 0.6, 0.3, 0.1]
# means_list = [0.1, 0.01, 0.001, 0.0001]
# stds_list = [0.1, 0.01, 0.001]
#
# conf = {
#     "xavier":  [{'gain':x} for x in gain_list],
#     "kaiming":  [None],
#     "gaussian":  [{"std":s} for s in stds_list]
# }

block_depth = [4, ]
data_res = (28, 28)

cond_channels = [(32, 64, 128),
                 (64, 128, 128),
                 (64, 128, 256)]
                 # (64, 128, 128, 256)]
conv_channels = {
    3: [(32, 64, 128),
        (64, 64, 128),
        (64, 128, 256)],
    4: [(32, 64, 128, 128),
        (32, 64, 128, 256),
        (64, 64, 128, 128)]
}
downsampling = {
    3: [(1, 0)],
    4: [(1, 0, 0),
        (0, 0, 1),
        (0, 1, 0)]
}

splits = {
    3: [
        ((2, 6), (4, 4)),
        ((4, 4), (2, 6)),
        ((4, 4), (4, 4))
        ],
    4: [
        ((2, 6), (4, 4), (8, 8)),
        ((4, 4), (2, 6), (8, 8)),
        ((4, 4), (4, 4), (8, 8)),
        ((2, 6), (2, 6), (8, 8))
        ]
}

# for sw, sd, md in itertools.product(subnet_width_values, subnet_depth_values, model_depth_values):
for bd in block_depth:
    for cc in cond_channels:
        for d in downsampling[len(cc)]:
            for s in splits[len(cc)]:
                for conv_c in conv_channels[len(cc)]:
                    print("--------------------------------")
                    print("running ")
                    print(f"block depth: {bd}" )
                    print(f"conditioning channels: {cc}")
                    print(f"splits: {s}")
                    print(f"convolution subnet channels:{conv_c}")

                    run_dir = f"architecture/supervised_{bd}_{cc}_{s}_{conv_c}"

                    if os.path.isdir(run_dir):
                        print("skipping")
                        continue

                    data_dir = f"../../../../data"
                    kwargs = {
                        "n_epochs": 60,
                        "data": "MNIST",
                        "model_type": "multiRes",
                        "block_depth": bd,
                        "cond_channels": cc,
                        "splits": s,
                        "run_dir": run_dir,
                        "data_dir": data_dir,
                        "data_res": data_res,
                        "downsample": d,
                        "conv_channels": conv_c,
                    }
                    p = Process(target=train_supervised, kwargs=kwargs)
                    p.start()
                    p.join()
                    # train_supervised(**kwargs)
        #                  subnet_depth=sd,
        #                  model_depth=md,
        #                  run_dir=run_dir,
        #                  data_dir=data_dir)