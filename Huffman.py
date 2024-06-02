import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from scipy.optimize import minimize_scalar
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.jpeg import (
    jpegenc, jpegdec, quant1, quant2, huffenc, huffdflt, huffdes, huffgen)
from jpeg_modi import jpegdec_lbt, jpegenc_lbt

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

def huffman_bits_gap(step, X, N, M):
    qstep = step
    vlc = jpegenc(X, qstep, N, M, opthuff=True)[0]
    total_bits = sum(vlc[:, 1])
    return abs(total_bits - 40960.0 + 1424.0)

def optimize_huffman_step_size(X, N, M):
    result = minimize_scalar(
        huffman_bits_gap, 
        args=(X, N, M),
        bounds=(0.1, 80), 
        method='bounded'
        )
    return result.x

X_test = Xf
n = 16
m = 32
step_opt = optimize_huffman_step_size(X_test, n, m)
# print(step_opt)
# step_opt = 0

vlc_m, hufftab_m = jpegenc_lbt(X_test, step_opt, n, m, opthuff=True)
Z_lbt = jpegdec_lbt(vlc_m, step_opt, n, m, hufftab=hufftab_m)
jpeg_rms_error = np.std(X_test - Z_lbt)
print(f'MSE for jpeg is: {jpeg_rms_error}')
fig, ax = plt.subplots()
plot_image(Z_lbt, ax=ax)
plt.show()

# vlc, hufftab = jpegenc(Xf, step_opt, n, m, opthuff=True)
# Z_lbt = jpegdec(vlc, step_opt, n, m, hufftab=hufftab)
# # vlc, hufftab = jpegenc(Xl, step_opt, 4, 8)
# # Z_lbt = jpegdec(vlc, step_opt, 4, 8)
# jpeg_rms_error = np.std(Xf - Z_lbt)
# print(f'MSE for jpeg is: {jpeg_rms_error}')
# fig, ax = plt.subplots()
# plot_image(Z_lbt, ax=ax)
# plt.show()