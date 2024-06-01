import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from scipy.optimize import minimize_scalar
from cued_sf2_lab.dct import dct_ii, colxfm, regroup

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

def dctbpp(Yr, N):
    # Your code here
    h, w = Yr.shape
    total_bits = 0
    d = h//N
    for i in range(0, h, d):
        for j in range(0, w, d):
            Ys = Yr[i: i+d, j: j+d]
            total_bits += bpp(Ys) * (d ** 2)
    return total_bits

C8 = dct_ii(8)
Yl = colxfm(colxfm(Xl, C8).T, C8).T
step = 17
rise1 = step * 1.5
Yq = quantise(Yl, step, rise1)
N = 8
Yr = regroup(Yq, N)

# Total number of bits using dctbpp
def dct_total_bits(step, X, N, n):
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    rise1 = step * n
    Yq = quantise(Y, step, rise1)
    Yr = regroup(Yq, N)
    dctbpp_bits = dctbpp(Yr, N)
    return dctbpp_bits

# Calculate rms error
def dct_rms_error(step, X, N, n):
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    rise1 = step * n
    Yq = quantise(Y, step, rise1)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    dct_rms_error = np.std(X - Z)
    return dct_rms_error

# Reconstruct the image
def dct_reconstruct(step, X, N, n):
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    rise1 = step * n
    Yq = quantise(Y, step, rise1)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    return Z

# Gap between total bits to target bits
def dct_total_bits_gap(step, X, N, n):
    """
    N is dct transform size.
    n is coefficient for rise1.
    """
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    rise1 = step * n
    Yq = quantise(Y, step, rise1)
    Yr = regroup(Yq, N)
    dctbpp_bits = dctbpp(Yr, N)
    return abs(dctbpp_bits - 40960.0)

# Find opt step size
def optimize_dct_step_size(X, N, n):
    result = minimize_scalar(
        dct_total_bits_gap, 
        args=(X, N, n),
        bounds=(0.1, 50), 
        method='bounded'
        )
    return result.x

step_opt = optimize_dct_step_size(Xb, 8, 1.5)
print(step_opt)
print(dct_rms_error(step_opt, Xb, 8, 1.5))
print(dct_total_bits(step_opt, Xb, 8, 1.5))

fig, ax = plt.subplots()
plot_image(dct_reconstruct(step_opt, Xb, 8, 1.5), ax=ax)
plt.show()
