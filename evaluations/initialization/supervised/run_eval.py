import os

inits = ["kaiming", "kaiming_uniform", "xavier_uniform"]

for init in inits:
    data_dir = "/home/jens/thesis/imageflow/data"
    run_dir = f'{init}'

    # if not os.path.isdir('init'):
        # os.mkdir(init)
    if os.path.isdir(run_dir):
        print(f"already trained {init}. skipping to next one.")
        continue

    kwargs = {}
    kwargs['n_epochs'] = 60
    kwargs['init_method'] = init

