import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_results(source, target, v_field, target_pred, v_field_pred, index='random'):

    # Shift everything to cpu since you can't plot from gpu directly.
    source = source.to(torch.device("cpu")).detach().numpy()
    target = target.to(torch.device("cpu")).detach().numpy()
    target_pred = target_pred.to(torch.device("cpu")).detach().numpy()
    v_field = v_field.to(torch.device("cpu")).detach().numpy()
    v_field_pred = v_field_pred.to(torch.device("cpu")).detach().numpy()

    # print(v_field.shape)
    # print(v_field_pred.shape)

    fig = plt.figure()

    indices = []
    if index == 'random':
        indices = [np.random.randint(source.shape[0])]
    elif index == 'all':
        indices = [np.arange(source.shape[0])]
        print(indices)
    elif type(index) == int:
        indices = [index]
    elif type(index) in [np.ndarray, list]:  # list or 1 dimensional array
        indices = index
    else:
        raise("Unknown index")

    for idx in indices:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_title("source img")
        ax1.imshow(source[idx, 0, :, :])

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("predictet target")
        ax2.imshow(target_pred[idx, 0, :, :])

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title("true target")
        ax3.imshow(target[idx, 0, :, :])
        # ax3.set_xlabel("mse: {}".format(round(reconstruction_err[-1], 2)))

        # plt.savefig(f"plots/unsupervised/prediction_{i}.pdf")

        fig2 = plt.figure()

        ax4 = fig2.add_subplot(1, 2, 1)
        ax4.set_title("predicted velocity field")
        # ax4.imshow(v_field_pred[idx, 0, :, :])
        u, v = v_field_pred[idx, 0, :, :], v_field_pred[idx, 1, :, :]
        x, y = np.meshgrid(np.linspace(0, 27, 28), np.linspace(0, 27, 28))
        plt.axis("equal")
        plt.streamplot(x, y, u, v)

        ax5 = fig2.add_subplot(1, 2, 2)
        ax5.set_title("true velocity field")
            # ax5.imshow(v_field[idx, 0, :, :])
        u, v = v_field[idx, 0], v_field[idx, 1]
        # print(u.shape)
        x, y = np.meshgrid(np.linspace(-13, 14, 28), np.linspace(-13, 14, 28))
        plt.axis("equal")
        plt.streamplot(x, y, u, v)
        # ax5.set_xlabel("mse: {}".format(round(field_err[-1], 2)))

        # plt.savefig(f"plots/unsupervised/field_{i}.pdf")

        plt.show()