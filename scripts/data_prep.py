import glob
import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from itertools import chain


# def flatten(a):
#     if isinstance(a, tuple):
#         a = list(a)
#     if a == []:
#         return a
#     if isinstance(a[0], (list, tuple)):
#         return flatten(a[0]) + flatten(a[1:])
#     return a[:1] + flatten(a[1:])
def divide_image(im, crop_res):
    res = im.size
    x = crop_res[0]
    y = crop_res[1]
    crops = []
    for upper in np.arange(0, res[1]-y, y):
        for left in np.arange(0, res[0]-x, x):

            right = left + x
            lower = upper + y
            crops.append(im.crop((left, upper, right, lower)))
    return crops

def get_landmarks(csv_path, crop_res, upper, left, lower, right):
    c_x = crop_res[0]
    c_y = crop_res[1]
    crop_csv = ""
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, x, y in reader:
            if left <= x < right and upper < y < lower:
                crop_csv += f"{i},{x % c_x},{y % c_y}\n"
    return crop_csv


def load_data(path, scale=5):
    """ Small script to load the birl dataset.
    path: path to the birl directory
    scale: data resolution to load
    """

    sets = os.listdir(path)
    lung_lesion_set = []
    for s in sets:
        if "lung-lesion" in s:
            set_path = os.path.join(path, s, f"scale-{scale}pc")
            images = glob.glob(set_path+"/*.jpg")
            csvs = glob.glob(set_path+"/*.csv")

            for k, im_path in enumerate(images):

                im = Image.open(im_path)
                csv_path = csvs[k]

                im_parts = divide_image(im, (64, 64))
                csv_parts = get_landmarks(csv_path, (64, 64))
                x_fig = int(im.size[0]//64) #13#int(np.ceil(np.sqrt(len(im_parts))))
                y_fig = int(im.size[1]//64) #13#int(np.ceil(np.sqrt(len(im_parts))))
                fig = plt.figure(figsize=(x_fig, y_fig))
                for i, part in enumerate(im_parts):
                    fig.add_subplot(y_fig, x_fig, i+1)
                    plt.imshow(part)
                    plt.axis('off')
                # plt.subplots_adjust(wspace=0, hspace=0)
                plt.show()

    return True


if __name__ == "__main__":
    data_dir = "../data/birl/"
    load_data(data_dir, scale=5)
