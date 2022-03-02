import os
import matplotlib.pyplot as plt
import numpy as np

def get_plot(path):
    with open(path) as f:
        loss_log = f.readlines()



    nll = [float(l.split()[2]) for l in  loss_log]
    val_nll = [float(l.split()[3]) for l in loss_log]

    reg_loss = [float(l.split()[-1]) for l in loss_log]

    run_avg_nll = [np.mean(nll[i:i+10]) for i, _ in enumerate(loss_log[:-10])]
    run_avg_nll = [*[np.mean(nll[0:i]) for i in range(1,11)], *run_avg_nll]

    run_avg_val = [np.mean(val_nll[i:i+10]) for i, _ in enumerate(loss_log[:-10])]
    run_avg_val = [*[np.mean(val_nll[0:i]) for i in range(1,11)], *run_avg_val]
    run_avg_val = [run_avg_val[i] + reg_loss[i] for i in range(len(reg_loss))]
# print(loss_log[0].split())

    n_datapoints = len(loss_log)
    ticks = [int(n_datapoints / 4), int(n_datapoints / 2), int(n_datapoints * 3 / 4), n_datapoints]
    print(ticks)
    ticks = [int(k) for k in range(1,n_datapoints+1) if k %80 == 0]
    print(ticks)
    epochs = [float(l.split()[0]) for l in  loss_log]
    labels = [int(epochs[x - 1] + 1) for x in ticks]

    print(len(run_avg_val))
    return run_avg_nll, run_avg_val, reg_loss, ticks, labels


run_avg_nll, run_avg_val, reg_loss, ticks, labels = get_plot("grad_loss_l2_1_50/loss.log")
run_avg_nll_0, run_avg_val_0, reg_loss_0, ticks_0, labels_0 = get_plot("grad_loss_l2_0_50/loss.log")


plt.plot(run_avg_nll_0, label="train $\lambda=0$")
plt.plot(run_avg_val_0, label="validation $\lambda=0$")

plt.plot(run_avg_nll, label="train $\lambda=1$")
plt.plot(run_avg_val, label="validation $\lambda=1$")

plt.plot(reg_loss, label="regularization")
plt.xticks(ticks, labels)
plt.ylim(-0.001, 0.5)
plt.xlabel("epoch")
plt.ylabel(r"$\mathcal{L}_{rec}$")
plt.legend()
plt.show()
