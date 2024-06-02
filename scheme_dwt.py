import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.dct import regroup as regroup_unnormalized
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.dwt import dwt, idwt
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
    fig.show()
        


def draw_matrix(X, save_and_format = None):
    fig, ax = plt.subplots()
    plot_image(X, ax=ax)
    if not save_and_format is None:
        save_fig(fig, save_and_format)
    fig.show()
        
def draw_matrices(imgs, titles, save_and_format = None):
    fig, axs = plt.subplots(1, len(imgs), figsize=(12, 3))
    for ax, img, title in zip(axs, imgs, titles):
        plot_image(img, ax=ax)
        ax.set(yticks=[], title=f'{title}')
    if not save_and_format is None:
        save_fig(fig, save_and_format)
    fig.show()

def save_fig(fig, name):
    fig.savefig(name, format=name.split(".")[-1], dpi=1200, bbox_inches='tight', pad_inches = 0)

# DWT
def nlevdwt(X, n):
    assert(n >= 1)
    m=256
    Y=dwt(X)
    for i in range(1, n):
        m = m//2
        Y[:m, :m] = dwt(Y[:m, :m])
    return Y

def nlevidwt(Y, n):
    m = 256 // 2 ** n
    Z = Y.copy()
    for i in range(n):
        m = m*2
        Z[:m, :m] = idwt(Z[:m, :m])
    return Z

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtent: an array of shape `(3, n+1)` containing the entropies
    """
    def quantise_and_get_entropy(Y_sub, dwtstep, k, i):
        q = dwtstep[k, i]
        Yq_sub = quantise(Y_sub, q)
        ent = bpp(Yq_sub) * Yq_sub.shape[0] * Yq_sub.shape[1]
        return Yq_sub, ent

    n = dwtstep.shape[1] - 1
    Yq = Y.copy()
    ent_M = np.zeros(Y.shape)
    dwtent = np.zeros(dwtstep.shape)
    
    m = 512
    for i in range(n):
        m = m//2
        
        # Top right
        Yq[m//2:m, 0:m//2], dwtent[0, i] = quantise_and_get_entropy(Yq[m//2:m, 0:m//2], dwtstep, 0, i)
        ent_M[m//2:m, 0:m//2] = dwtent[0, i]

        # Bottom Left
        Yq[0:m//2, m//2:m], dwtent[1, i] = quantise_and_get_entropy(Yq[0:m//2, m//2:m], dwtstep, 1, i)
        ent_M[0:m//2, m//2:m] = dwtent[1, i]

        # Bottom right
        Yq[m//2:m, m//2:m], dwtent[2, i] = quantise_and_get_entropy(Yq[m//2:m, m//2:m], dwtstep, 2, i)
        ent_M[m//2:m, m//2:m] = dwtent[2, i]

    # Top left
    Yq[0:m//2, 0:m//2], dwtent[0, n] = quantise_and_get_entropy(Yq[0:m//2, 0:m//2], dwtstep, 0, n)
    ent_M[0:m//2, 0:m//2] = dwtent[0, n]
    
    return Yq, dwtent, ent_M

def calc_dwt_quant_error(X, N, dwtstep):
    Y = nlevdwt(X, N)
    Yq, _, _ = quantdwt(Y, dwtstep)
    Z = nlevidwt(Yq, N)
    err = np.std(Z-X)
    return err

def get_equivalent_dwt_error_quant(X, N, r = None):
    DIRECT_QUANT = 17
    EPSILON = 5e-3
    direct_error =  np.std(X - quantise(X, DIRECT_QUANT))
   

    dwt_quant = 300
    if r is not None:
        dwtstep = r * dwt_quant
    else:
        dwtstep = np.ones((3, N+1)) * dwt_quant
    step = dwt_quant/2
    i = 0 # Stop infinite loops
    dwt_error = calc_dwt_quant_error(X, N, dwtstep)
    while abs(dwt_error - direct_error) > EPSILON and i < 500:
        if dwt_error > direct_error:
            dwt_quant -= step
            if r is not None:
                dwtstep = r * dwt_quant
            else:
                dwtstep = np.ones((3, N+1)) * dwt_quant
        else:
            dwt_quant += step
            if r is not None:
                dwtstep = r * dwt_quant
            else:
                dwtstep = np.ones((3, N+1)) * dwt_quant
        step /= 2
        dwt_error = calc_dwt_quant_error(X, N, dwtstep)
        i += 1

    if i == 500:
        print("STEP OPTIMISATION REACHED ITERATION CAP")

    return dwtstep

def dwt_compression_analysis(X, N, r = None):
    dwtstep  = get_equivalent_dwt_error_quant(X, N, r)
    Y = nlevdwt(X, N)
    Yq, dwtent, _ = quantdwt(Y, dwtstep)
    bits = dwtent.sum()
    Xq = quantise(X, 17)
    direct_bits = bpp(Xq) * (256 ** 2)

    comp = direct_bits / bits
    Z = nlevidwt(Yq, N)

    return comp, Z


def get_step_ratios(layers):
    """Get equal MSE DWT step ratios."""
    image_size = 256
    ratios = np.zeros((3, layers + 1))
    impulse_amplitude = 100

    # Iterate to find impulse centres
    edge = image_size  # initialise the right edge value
    d = image_size // 4  # initialise the distance of centre to the right edge
    for layer in range(layers):
        # Initialize Yt and Zt
        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        # Implement impulse
        Yt[edge - d, d] = impulse_amplitude # Top right
        Z = nlevidwt(Yt, layers)
        ratios[0][layer] =  1/ np.std(Z)

        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        Yt[d, edge - d] = impulse_amplitude # Bottom left
        Z = nlevidwt(Yt, layers)
        ratios[1][layer] =  1/ np.std(Z)

        Yt = np.zeros((image_size, image_size)) # Create test compressed image with zeros
        Yt[edge - d, edge - d] = impulse_amplitude # Bottom right
        Z = nlevidwt(Yt, layers)
        ratios[2][layer] = 1/ np.std(Z)

        # Update right edge and d
        edge = edge // 2
        d = d // 2
    
    # Top left
    d = d * 2
    Yt = np.zeros((image_size, image_size))
    Yt[d, d] = impulse_amplitude
    Z = nlevidwt(Yt, layers)
    ratios[0, layers] = 1/ np.std(Z)
    ratios = ratios / ratios.max()
    ratios[1, layers] = None
    ratios[2, layers] = None

    return ratios


# -----------

def load_image(path):
    """Load image as mean-zero matrix and return."""
    X, _ = load_mat_img(img=path, img_info='X', cmap_info={'map', 'map2'})
    mean = np.mean(X) # TODO USING MEAN INSTEAD OF 128 - Is this better?
    X = X - mean
    return X

def test_nlevdwt():
    X = load_image("lighthouse.mat")
    Y = nlevdwt(X, 4)
    Z = nlevidwt(Y, 4)
    err = np.std(Z-X)
    assert(err == 0)

# MAIN

def dwt_analysis(X, equal_mse=True):
    Zs = []
    comps = []
    for i in range(1, 6):
        if equal_mse:
            ratios = get_step_ratios(i)
        else:
            ratios = None
        comp, Z = dwt_compression_analysis(X, i, ratios)
        comps.append(comp)
        Zs.append(Z)

    names = []
    for i, comp in enumerate(comps):
        names.append(f"n={i+1}, comp: {round(comp, 3)}")

    draw_matrices(Zs, names)
    with np.printoptions(precision=3):
        print(comps)
    Xq = quantise(X, 17)
    draw_matrix(Xq)

if __name__ == "__main__":
    # h1 = np.array([-1, 2, 6, 2, -1])/8
    # h2 = np.array([-1, 2, -1])/4
    # g1 = np.array([1, 2, 1])/2
    # g2 = np.array([-1, -2, 6, -2, -1])/4

    X = load_image("lighthouse.mat")
    dwt_analysis(X)
    input() # Await input from terminal before closing