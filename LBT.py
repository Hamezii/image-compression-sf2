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
from cued_sf2_lab.lbt import pot_ii

# Initialise three images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

Xl = lighthouse - 128.0
Xb = bridge - 128.0
Xf = flamingo - 128.0

def LBT(X: np.ndarray, N: int, s:float) -> np.ndarray:
    Pf, Pr = pot_ii(N, s)
    C = dct_ii(N)
    t = np.s_[N//2:-N//2]
    Xp = X.copy()
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T
    return Y

def ILBT(Y: np.ndarray, N: int, s: float) -> np.ndarray:
    Pf, Pr = pot_ii(N, s)
    C = dct_ii(N)
    t = np.s_[N//2:-N//2]
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    Zp = Z.copy()
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Zp

# Yl = LBT(Xl, 8, s=np.sqrt(2))
# Zl = ILBT(Yl, 8, s=np.sqrt(2))
# fig, ax = plt.subplots()
# plot_image(Zl, ax=ax)
# plt.show()