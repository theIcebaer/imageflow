import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import imageflow
import tqdm

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import matplotlib.pyplot as plt
    import io
    from PIL import Image
    buf = io.BytesIO()
    # plt.tight_layout()
    fig.savefig(buf, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    # buf.close()
    plt.close()
    return img

def make_figures_mf(batch, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    idxs = np.arange(batch.shape[0])

    movings = []
    fixeds = []
    for idx in tqdm.tqdm(idxs):
        fig_m = plt.figure(1)
        ax_m = fig_m.add_subplot(111)
        ax_m.get_xaxis().set_visible(False)
        ax_m.get_yaxis().set_visible(False)
        plt.imshow(batch[idx, 0, ...])
        # plt.title(f"moving {idx}")
        im_1 = fig2img(fig_m)
        movings.append(im_1)
        fig_f = plt.figure(2)
        ax_f = fig_f.add_subplot(111)
        ax_f.get_xaxis().set_visible(False)
        ax_f.get_yaxis().set_visible(False)
        plt.imshow(batch[idx, 1, ...])
        # plt.title(f"fixed {idx}")
        im_2 = fig2img(fig_f)
        fixeds.append(im_2)
        # plt.show()
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            im_1.save(os.path.join(save_path, f"moving_{idx}.png"))
            im_2.save(os.path.join(save_path, f"fixed_{idx}.png"))
    return movings, fixeds

def make_fig_from_batch(batch, save_path, affix=None):
    import numpy as np
    import matplotlib.pyplot as plt
    idxs = np.arange(batch.shape[0])

    images = []
    for idx in tqdm.tqdm(idxs):
        fig_m = plt.figure(1)
        ax_m = fig_m.add_subplot(111)
        ax_m.get_xaxis().set_visible(False)
        ax_m.get_yaxis().set_visible(False)
        plt.imshow(batch[idx, 0, ...])
        im_1 = fig2img(fig_m)
        images.append(im_1)
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if affix:
                im_1.save(os.path.join(save_path, f"image_{idx}_{affix}.png"))
            else:
                im_1.save(os.path.join(save_path, f"image_{idx}.png"))
    return images


def streamplot_from_batch(batch, show_image=False, get_all=False, idxs=None, save_path=None, affix=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if get_all:
        idxs = np.arange(batch.shape[0])
    elif idxs is not None:
        if type(idxs) == int:
            idxs = [idxs]
        else:
            pass
    else:
        idxs = [np.random.randint(0, batch.shape[0]-1)]
    data_shape = batch.shape[2], batch.shape[3]

    images = []
    for idx in tqdm.tqdm(idxs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        u, v = batch[idx, 0, :, :], batch[idx, 1, :, :]
        x, y = np.meshgrid(np.linspace(0, data_shape[0]-1, data_shape[1]), np.linspace(0, data_shape[0]-1, data_shape[1]))
        vel = np.sqrt(u**2 + v**2)
        ax.set_aspect('equal', 'box')
        plt.streamplot(x, y, -v, u)
        # if len(idxs) > 10:
        #     print(idx)
        # ax.set_ylim((0, x.shape[1]))
        # ax.set_xlim((0, x.shape[0]))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.tight_layout()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        if show_image:
            plt.show()
        image = fig2img(fig)
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if affix:
                path = os.path.join(save_path, f"v_field_{idx}_{affix}.png")
            else:
                path = os.path.join(save_path, f"v_field_{idx}.png")
            image.save(path)

        images.append(image)

    return images

def quiver_from_batch(batch, show_image=False, get_all=False, idxs=None, save_path=None, affix=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if get_all:
        idxs = np.arange(batch.shape[0])
    elif idxs:
        pass
    else:
        idxs = [np.random.randint(0, batch.shape[0]-1)]
    data_shape = batch.shape[2], batch.shape[3]

    images = []
    for idx in tqdm.tqdm(idxs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        u, v = batch[idx, 0, :, :], batch[idx, 1, :, :]
        x, y = np.meshgrid(np.linspace(0, data_shape[0]-1, data_shape[1]), np.linspace(0, data_shape[0]-1, data_shape[1]))
        ax.set_aspect('equal', 'box')
        plt.quiver(x, y, -v, u)
        # if len(idxs) > 10:
        #     print(idx)
        # ax.set_ylim((0, x.shape[1]))
        # ax.set_xlim((0, x.shape[0]))
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.tight_layout()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        # plt.show()
        image = fig2img(fig)
        if save_path:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if affix:
                path = os.path.join(save_path, f"v_field_{idx}_{affix}.png")
            else:
                path = os.path.join(save_path, f"v_field_{idx}.png")
            image.save(path)

        if show_image:
            image.show()
        images.append(image)

    return images

def plot_mff_(batch, batch_pred, data_dir=None, idx=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not idx:
        # idx = np.random.randint(0, batch.shape[0] - 1)
        idx=0
    m = batch[idx, 0, ...]
    f = batch[idx, 1, ...]
    f_ = batch_pred[idx, 0, ...]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = plt.imshow(m)
    plt.title(r'$m$')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im1, cax=cax, orientation='vertical')

    ax2 = fig.add_subplot(1, 3, 2)
    im2 = plt.imshow(f)
    plt.title(r'$f$')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical')

    ax3 = fig.add_subplot(1, 3, 3)
    im3 = plt.imshow(f_)
    plt.title(r'$\hat{f}$')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im3, cax=cax, orientation='vertical')

    fig.tight_layout()

    if data_dir is not None:
        fig_dir = os.path.join(data_dir, "figures")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, "images_{}.pdf".format(idx)))
    plt.show()
    plt.close()

def plot_ff_(batch_target, batch_pred):
    import numpy as np
    import matplotlib.pyplot as plt
    idx = np.random.randint(0, batch_target.shape[0] - 1)
    f = batch_target[idx, 1, ...]
    f_pred = batch_pred[idx, 0, ...]

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(f)
    fig.add_subplot(1, 2, 2)
    plt.imshow(f_pred)
    plt.show()


def plot_running_fields(batch, true_batch=None, idx=None):

    if not idx:
        idx = 0

    data_res = (batch.shape[2], batch.shape[3])
    # fig = plt.figure()
    #
    # idx = np.random.randint(source.shape[1])
    #
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax1.set_title("source image")
    # ax1.imshow(source[idx, 0, :, :])
    #
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.set_title("predictet target")
    # ax2.imshow(target_pred[idx, 0, :, :])
    #
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax3.set_title("true target")
    # ax3.imshow(target[idx, 0, :, :])
    # ax3.set_xlabel("mse: {}".format(round(reconstruction_err[-1], 2)))
    #
    # fig_name = f"prediction_{i}.pdf"
    # plt.savefig(os.path.join(plot_dir, fig_name))

    fig2 = plt.figure()

    ax4 = fig2.add_subplot(1, 2, 1)
    ax4.set_title("predicted velocity field")
    # ax4.imshow(v_field_pred[idx, 0, :, :])
    u, v = batch[idx, 0, :, :], batch[idx, 1, :, :]
    x, y = np.meshgrid(np.arange(0, data_res[0]), np.arange(0, data_res[0]))
    plt.axis("equal")
    plt.streamplot(x, y, -v, u)

    if true_batch:
        ax5 = fig2.add_subplot(1, 2, 2)
        ax5.set_title("true velocity field")
        # ax5.imshow(v_field[idx, 0, :, :])
        u, v = true_batch[idx, 0], true_batch[idx, 1]
        # print(u.shape)
        x, y = np.meshgrid(np.linspace(-13, 14, 28), np.linspace(-13, 14, 28))
        plt.axis("equal")
        plt.streamplot(x, y, -v, u)
        # ax5.set_xlabel("mse: {}".format(round(field_err[-1], 2)))
        # fig_name = f"field_{i}.pdf"
    # plt.savefig(os.path.join(plot_dir, fig_name))

    plt.show()

def make_loss_plot(path, save_path=None):
    dir_, filename = os.path.split(path)
    if filename.split(".")[-1] == 'pt':
        loss_log = torch.load(path)


        plt.plot(loss_log['nll'], label='training loss')
        plt.plot(loss_log['val_nll'], label='validation loss')
        #make tiks
        n_datapoints = len(loss_log['nll'])
        ticks = [int(n_datapoints/4), int(n_datapoints/2), int(n_datapoints * 3/4), n_datapoints]
        labels = [loss_log['epoch'][x-1]+1 for x in ticks]

        plt.xticks(ticks, labels)
        plt.ylim(-2, -1)
        plt.xlabel("epoch")
        plt.ylabel(r"$\mathcal{L}_{nll}$")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def join_plots_(model_paths, dat='nll'):
    fig = plt.figure()

    logs = []
    for path in model_paths:
        dir_, filename = os.path.split(path)
        if filename.split(".")[-1] == 'pt':
            name = dir_.split('/')[-2].split('_')
            if name[1] == 'kaiming':
                type = f"{name[1]}"
            else:
                type = f"{name[1]} {name[2]}"
            loss_log = torch.load(path)
            logs.append(loss_log)
            plt.plot(loss_log[dat], label=f'{type}')
            # plt.plot(loss_log['val_nll'], label='{} validation loss')
            # make tiks



        else:
            raise AttributeError("loss log format not implemented yet.")

    n_datapoints = len(logs[0][dat])
    ticks = [int(n_datapoints / 4), int(n_datapoints / 2), int(n_datapoints * 3 / 4), n_datapoints]
    labels = [logs[0]['epoch'][x - 1] + 1 for x in ticks]
    plt.xticks(ticks, labels)
    plt.ylim(-2, 2)
    plt.xlabel("epoch")
    plt.ylabel(r"$\mathcal{L}_{nll}$")
    plt.legend()
    plt.savefig("loss_plot")
    plt.show()

def show_field(field):
    field = field.squeeze()
    data_shape = field.shape[1:]
    u, v = field[0, :, :], field[1, :, :]
    x, y = np.meshgrid(np.linspace(0, data_shape[0] - 1, data_shape[1]),
                       np.linspace(0, data_shape[0] - 1, data_shape[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    plt.streamplot(x, y, -v, u)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    # plt.tight_layout()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()

    plt.show()