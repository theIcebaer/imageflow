import os.path

from imageflow.trainer import train_unsupervised
from multiprocessing import Process


data_dir = f"../../../data"


multiplier = [1]
penalty = ["l2"]

resolutions = [512]

architectures = [
    (
        4,
        (4, 8, 16),
        (8, 16, 32),
        ((2, 6), (4, 4)),
        (1, 0)
    )
]



for scale in [5]:
    for res in resolutions:
        data_res = (res, res)
        for bd, cc, conv_c, s, d in architectures:
            for m in multiplier:
                for p in penalty:
                    run_dir = f"reduced_learning_rate_{scale}_{res}_{p}_{m}_{bd}_{cc}_{conv_c}_{s}_{d}"

                    if os.path.isdir(run_dir):
                        print(f"Model {run_dir} is already trained. Skipping to the next one")
                        continue

                    grad_loss_params = {
                        "multiplier": m,
                        "penalty": p
                    }

                    kwargs = {
                        "data": "Birl",
                        "model_type": "multiRes",
                        "run_dir": run_dir,
                        "data_dir": data_dir,
                    }
                    kwargs["block_depth"] = bd
                    kwargs["cond_channels"] = cc
                    kwargs["conv_channels"] = conv_c
                    kwargs["splits"] = s
                    kwargs["downsample"] = d
                    kwargs["data_res"] = data_res
                    kwargs["batch_size"] = 8
                    kwargs["n_batches"] = 100
                    kwargs["grad_loss"] = grad_loss_params
                    kwargs["n_epochs"] = 60
                    kwargs["birl_scale"] = scale
                    kwargs["show_running"] = False
                    kwargs["learning_rate"] = 1e-6

                    # train_unsupervised(**kwargs)
                    proc = Process(target=train_unsupervised, kwargs=kwargs)
                    proc.start()
                    proc.join()