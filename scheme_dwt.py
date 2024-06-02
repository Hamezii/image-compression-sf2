import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.dct import colxfm, regroup
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.laplacian_pyramid import bpp, quantise
from cued_sf2_lab.jpeg import (diagscan, runampl, HuffmanTable,
    jpegenc, jpegdec, quant1, quant2, huffenc, huffdflt, huffdes, huffgen, dwtgroup)
from typing import Tuple, NamedTuple, Optional
from cued_sf2_lab.bitword import bitword


### Helper functions
def calc_quant_error(X, func_enc, func_dec, params, quant_step):
    Y = func_enc(X, *params)
    Yq = quantise(Y, quant_step)
    Z = func_dec(Yq, *params)
    return np.std(X - Z)


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
    X = X - 128.0
    return X

def test_nlevdwt():
    X = load_image("lighthouse.mat")
    Y = nlevdwt(X, 4)
    Z = nlevidwt(Y, 4)
    err = np.std(Z-X)
    assert(err == 0)

# ENCODING

def jpegenc_dwt(X: np.ndarray, qstep: float, N: int = 4,
        opthuff: bool = False, dcbits: int = 8, log: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Encodes the image in X to generate a variable length bit stream.

    Parameters:
        X: the input greyscale image
        qstep: the quantisation step to use in encoding
        N: depth of DWT compression (defaults to 4)
        opthuff: if true, the Huffman table is optimised based on the data in X
        dcbits: the number of bits to use to encode the DC coefficients
            of the DCT.

    Returns:
        vlc: variable length output codes, where ``vlc[:,0]`` are the codes and
            ``vlc[:,1]`` the number of corresponding valid bits, so that
            ``sum(vlc[:,1])`` gives the total number of bits in the image
        hufftab: optional outputs containing the Huffman encoding
            used in compression when `opthuff` is ``True``.
    '''

    M = 2 ** N

    # DCT on input image X.
    if log:
        print(f"Forward {N} depth DWT.")

    Y = nlevdwt(X, N)

    # Quantise to integers.
    if log:
        print('Quantising to step size of {}'.format(qstep))
    Yq = quant1(Y, qstep, qstep).astype('int')

    # Grouping
    Yq = dwtgroup(Yq, N)

    # Generate zig-zag scan of AC coefs.
    scan = diagscan(M)

    # On the first pass use default huffman tables.
    if log:
        print('Generating huffcode and ehuf using default tables')
    dhufftab = huffdflt(1)  # Default tables.
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows')
    sy = Yq.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M,c:c+M]
            # Possibly regroup
            # if M > N:
            #     yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            if dccoef not in range(2**dcbits):
                raise ValueError(
                    'DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    # Return here if the default tables are sufficient, otherwise repeat the
    # encoding process using the custom designed huffman tables.
    if not opthuff:
        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        return vlc, dhufftab

    # Design custom huffman tables.
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M, c:c+M]
            # Possibly regroup
            # if M > N:
            #     yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan])
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhufftab.huffval.shape))*8))

    return vlc, dhufftab


def jpegdec_dwt(vlc: np.ndarray, qstep: float, N: int = 4,
        hufftab: Optional[HuffmanTable] = None,
        dcbits: int = 8, W: int = 256, H: int = 256, log: bool = True
        ) -> np.ndarray:
    '''
    Decodes a (simplified) JPEG bit stream to an image

    Parameters:

        vlc: variable length output code from jpegenc
        qstep: quantisation step to use in decoding
        N: depth of DWT compression (defaults to 4)
        M: width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        hufftab: if supplied, these will be used in Huffman decoding
            of the data, otherwise default tables are used
        dcbits: the number of bits to use to decode the DC coefficients
            of the DCT
        W, H: the size of the image (defaults to 256 x 256)

    Returns:

        Z: the output greyscale image
    '''

    opthuff = (hufftab is not None)

    M = 2 ** N

    # Set up standard scan sequence
    scan = diagscan(M)

    if opthuff:
        if len(hufftab.bits.shape) != 1:
            raise ValueError('bits.shape must be (len(bits),)')
        if log:
            print('Generating huffcode and ehuf using custom tables')
    else:
        if log:
            print('Generating huffcode and ehuf using default tables')
        hufftab = huffdflt(1)
    # Define starting addresses of each new code length in huffcode.
    # 0-based indexing instead of 1
    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    # Set up huffman coding arrays.
    huffcode, ehuf = huffgen(hufftab)

    # Define array of powers of 2 from 1 to 2^16.
    k = 2 ** np.arange(17)

    # For each block in the image:

    # Decode the dc coef (a fixed-length word)
    # Look for any 15/0 code words.
    # Choose alternate code words to be decoded (excluding 15/0 ones).
    # and mark these with vector t until the next 0/0 EOB code is found.
    # Decode all the t huffman codes, and the t+1 amplitude codes.

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    if log:
        print('Decoding rows')
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M**2)

            # Decode DC coef - assume no of bits is correctly given in vlc table.
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError(
                    'The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
            i += 1

            # Loop for each non-zero AC coef.
            while np.any(vlc[i] != eob):
                run = 0

                # Decode any runs of 16 zeros first.
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1

                # Decode run and size (in bits) of AC coef.
                start = huffstart[vlc[i, 1] - 1]
                res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1

                # Decode amplitude of AC coef.
                if vlc[i, 1] != si:
                    raise ValueError(
                        'Problem with decoding .. you might be using the wrong hufftab table')
                ampl = vlc[i, 0]

                # Adjust ampl for negative coef (i.e. MSB = 0).
                thr = k[si - 1]
                yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                i += 1

            # End-of-block detected, save block.
            i += 1

            yq = yq.reshape((M, M)).T

            # Possibly regroup yq
            # if M > N:
            #     yq = regroup(yq, M//N)
            Zq[r:r+M, c:c+M] = yq

    if log:
        print('Inverse quantising to step size of {}'.format(qstep))

    # Undo grouping
    Zi = dwtgroup(Zq, -N)

    # Undo quant
    Zi = quant2(Zi, qstep, qstep)

    if log:
        print('Inverse {} x {} DCT\n'.format(N, N))

    Z = nlevidwt(Zi, N)

    return Z



# ANALYSIS

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

def dwt_huff_analysis(X):
    N = 2
    quant = 10
    vlc, hufftab = jpegenc_dwt(X, quant, N, opthuff=True)
    Z = jpegdec_dwt(vlc, quant, N, hufftab=hufftab)
    draw_matrix(Z)

    print(np.std(Z- X))


if __name__ == "__main__":
    # h1 = np.array([-1, 2, 6, 2, -1])/8
    # h2 = np.array([-1, 2, -1])/4
    # g1 = np.array([1, 2, 1])/2
    # g2 = np.array([-1, -2, 6, -2, -1])/4

    X = load_image("lighthouse.mat")
    dwt_huff_analysis(X)
    input() # Await input from terminal before closing