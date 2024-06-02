#%matplotlib widget
import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.dct import regroup as regroup_unnormalized
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.laplacian_pyramid import bpp, quantise

### Helper functions
def calc_quant_error(X, func_enc, func_dec, params, quant_step):
    Y = func_enc(X, *params)
    Yq = quantise(Y, quant_step)
    Z = func_dec(Yq, *params)
    return np.std(X - Z)

def regroup(Y, N):
    return regroup_unnormalized(Y, N)/N

def dctbpp(Yr, N):
    size = 256 // N
    entropy = 0
    for row in range(N):
        for col in range(N):
            Ys = Yr[row*size : (row+1)*size, col*size : (col+1)*size]
            entropy += bpp(Ys) * (size ** 2)
    return entropy


### DRAW
def make_plot(x, y, title, x_label, y_label, save_and_format = None):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if save_and_format is not None:
        save_fig(fig, save_and_format)
        


def draw_matrix(X, save_and_format = None):
    fig, ax = plt.subplots()
    plot_image(X, ax=ax)
    if not save_and_format is None:
        save_fig(fig, save_and_format)
        
def draw_matrices(imgs, titles, save_and_format = None):
    fig, axs = plt.subplots(1, len(imgs), figsize=(12, 3))
    for ax, img, title in zip(axs, imgs, titles):
        plot_image(img, ax=ax)
        ax.set(yticks=[], title=f'{title}')
    if not save_and_format is None:
        save_fig(fig, save_and_format)

def save_fig(fig, name):
    fig.savefig(name, format=name.split(".")[-1], dpi=1200, bbox_inches='tight', pad_inches = 0)



if __name__ == "__main__":
    print( )