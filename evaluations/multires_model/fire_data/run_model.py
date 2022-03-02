from multiprocessing import Process
from imageflow.trainer import train_unsupervised
import itertools
import os


regularizers = ['grad_loss']

params = {
    'grad_loss': {
        "penalty": ["l2"],
        "multiplier": [1]

    }
}

data_dir = "/home/jens/thesis/imageflow/data"
# run_dir = "unsupervised"

for r in regularizers:
    p = params[r]
    # print(p.keys())
    param_list =[p[k] for k in p.keys()]
    param_names = list(p.keys())
    # print(param_list)
    combs = list(itertools.product(*param_list))
    for comb in combs:






        reg_args = {param_names[i]: c for i, c in enumerate(comb)}
        # print(reg_args)
        kwargs = {}

        kwargs[r] = reg_args
        kwargs["data"] = "FIRE"
        kwargs["model_type"] = "multiRes"
        kwargs["data_dir"] = data_dir
        kwargs["n_epochs"] = 200
        kwargs["batch_size"] = 10
        kwargs["data_res"] = (32, 32)
        kwargs["image_res"] = (2912, 2912)
        kwargs["block_depth"] = 2
        kwargs["cond_channels"] = (4, 4, 4)
        kwargs["conv_channels"] = (8, 8, 8)
        kwargs["val_batch_size"] = 8
        print(kwargs)
        print("--")

        # param_string = "".join(f"_{c}" for c in comb)
        # run_dir = f"{r}{param_string}"
        run_dir = f"{kwargs['block_depth']}_{kwargs['cond_channels']}_{kwargs['conv_channels']}"
        if os.path.isdir(run_dir):
            print("* Model ist already trained. Skipping to the next one.")
            continue

        kwargs["run_dir"] = run_dir

        # p = Process(target=train_unsupervised, kwargs=kwargs)
        # p.start()
        # p.join()
        train_unsupervised(**kwargs)