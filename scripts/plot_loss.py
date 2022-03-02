import csv
import matplotlib.pyplot as plt
import numpy as np
path = "runs/2021-09-27_16-47/loss.log"

rec_loss = []
prior_nll = []
log_term = []

with open(path, newline="") as loss_file:
    reader = csv.reader(loss_file, delimiter=' ')

    for row in reader:
        rec_loss.append(float(row[2]))
        prior_nll.append(float(row[4]))
        log_term.append(float(row[6]))
        print(type(rec_loss[-1]), prior_nll[-1], log_term[-1])


fig = plt.figure()
fig.add_subplot(131)
plt.plot(rec_loss)  # np.arange(len(rec_loss)),
fig.add_subplot(132)
plt.plot(prior_nll)  # np.arange(len(prior_nll))
fig.add_subplot(133)
plt.plot(log_term)  # np.arange(len(log_term)),
plt.show()
