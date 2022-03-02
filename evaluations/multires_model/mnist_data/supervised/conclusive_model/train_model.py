from imageflow.trainer import train_supervised


bd = 4
cc = (32, 64, 128)
conv_c = (64, 128, 256)
s = ((4, 4), (2, 6))
d = (1, 0)

data_res = (28, 28)
run_dir = f"supervised_{bd}_{cc}_{s}_{conv_c}"

data_dir = "/home/jens/thesis/imageflow/data"
kwargs = {
    "n_epochs": 40,
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
# p = Process(target=train_supervised, kwargs=kwargs)
# p.start()
# p.join()
train_supervised(**kwargs)

