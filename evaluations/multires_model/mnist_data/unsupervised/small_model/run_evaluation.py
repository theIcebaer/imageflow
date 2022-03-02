import os
import itertools
from imageflow.trainer import train_unsupervised
from multiprocessing import Process

data_dir = f"../../../../../data"

regularizers = ['grad_loss']

params = {
    'grad_loss': {
        "penalty": ["l2"],
        "multiplier": [ 1]

    },

    'curl': {
        "multiplier": []
    }
}
epochs = [60]

# architectures
block_depth = [2, 4]
cond_channels = [(32, 32, 32),
                 (16, 32, 64),
                 (32, 32, 64)]
# splits = []
# downsample = []
conv_channels = [(16, 16, 16),
                 (8, 8, 8),
                 (8, 16, 32),
                 (16, 32, 64)]

for n in epochs:
    for bd, cc, cv in itertools.product(block_depth, cond_channels, conv_channels):
        for r in regularizers:
            p = params[r]
            # print(p.keys())
            param_list =[p[k] for k in p.keys()]
            param_names = list(p.keys())
            # print(param_list)
            combs = list(itertools.product(*param_list))
            for comb in combs:


            # param_string = "".join(f"_{c}" for c in comb)
                run_dir = f"{bd}_{cc}_{cv}"
                if n != 10:
                    run_dir += f"_{n}"

                if os.path.isdir(run_dir):
                    print("* Model ist already trained. Skipping to the next one.")
                    continue

                reg_args = {param_names[i]: c for i, c in enumerate(comb)}
                print(reg_args)
                kwargs = {}

                kwargs[r] = reg_args
                kwargs["data"] = "MNIST"
                kwargs["model_type"] = "multiRes"
                kwargs["run_dir"] = run_dir
                kwargs["data_dir"] = data_dir
                kwargs["n_epochs"] = n
                kwargs["block_depth"] = bd
                kwargs["cond_channels"] = cc
                kwargs["conv_channels"] = cv
                # kwargs["dummy_run"] = True

                print(kwargs)
                print("--")
                p = Process(target=train_unsupervised, kwargs=kwargs)
                p.start()
                p.join()
                # train_unsupervised(**kwargs)