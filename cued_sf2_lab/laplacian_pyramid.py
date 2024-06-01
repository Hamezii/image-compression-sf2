import scipy.io
import numpy as np

from .familiarisation import plot_image
from .encoder import Encoder

__all__ = [
    "rowdec",
    "rowdec2",
    "plot_laplacian_pyramid",
    "beside",
    "rowint",
    "rowint2",
    "quant1",
    "quant2",
    "quantise",
    "bpp",
]


def rowdec(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Filter rows of image X with h and then decimate by a factor of 2.

    Parameters:
        X: Image matrix (Usually 256x256)
        h: Filter coefficients
    Returns:
        Y: Image with filtered and decimated rows

    If len(H) is odd, each output sample is aligned with the first of
    each pair of input samples.
    If len(H) is even, each output sample is aligned with the mid point
    of each pair of input samples.
    """
    r, c = X.shape
    m = len(h)
    m2 = m // 2
    if m % 2:
        X = np.pad(X, [(0, 0), (m2, m2)], mode='reflect')
    else:
        X = np.pad(X, [(0, 0), (m2-1, m2-1)], mode='symmetric')

    Y = np.zeros((r, (c+1)//2))
    # Loop for each term in h.
    for i in range(m):
        Y = Y + h[i] * X[:, i:i+c:2]
    return Y


# TODO: FIX this - breaks for even filters (like MATLAB function)
def rowdec2(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Filter rows of image X with h and then decimate by a factor of 2.

    Parameters:
        X: Image matrix (Usually 256x256)
        h: Filter coefficients
    Returns:
        Y: Image with filtered and decimated rows

    If len(H) is odd, each output sample is aligned with the second of
    each pair of input samples.
    If len(H) is even, each output sample is aligned with the mid point
    of each pair of input samples.
    """
    r, c = X.shape
    m = len(h)
    m2 = m // 2
    if m % 2:
        X = np.pad(X, [(0, 0), (m2, m2)], mode='reflect')
    else:
        X = np.pad(X, [(0, 0), (m2-1, m2-1)], mode='symmetric')

    Y = np.zeros((r, c // 2))
    # Loop for each term in h.
    for i in range(m):
        Y = Y + h[i] * X[:, i+1:i+c:2]
    return Y


# Something like `axs = plt.subplots(5, sharex=True, sharey=True)`
# TODO: Use beside function several times
def plot_laplacian_pyramid(X, decimated_list):
    """
    Plot laplacian pyramid images side by side.

    Parameters:
    X (numpy.ndarray): Original image matrix (Usually 256x256)
    decimated_list (list): List of X1, X2 etc
    """
    plot_list = [X]
    for X_dec in decimated_list:
        X_dec_padded = np.zeros_like(X)
        X_dec_padded[:X_dec.shape[0], :X_dec.shape[1]] = X_dec
        plot_list.append(X_dec_padded)

    plot_image(np.hstack(tuple(plot_list)))


# TODO: Fixup
def beside(X1, X2):
    """
    Arrange two images beside eachother.

    Parameters:
    X1, X2 (numpy.ndarray): Original image matrices (Usually 256x256)

    Returns:
    Y (numpy.ndarray): Padded with zeros as necessary and the images are
    separated by a blank column
    """
    [m1, n1] = X1.shape
    [m2, n2] = X2.shape
    # print(m1,n1,m2,n2)
    m = max(m1, m2)
    Y = np.zeros((m, n1+n2+1))
    # print(Y.shape)
    # print(((m-m1)/2)+1)
    # print(type(n1))

    # index slicing must use integers
    Y[int(((m-m1)/2)):int(((m-m1)/2)+m1), :n1] = X1
    Y[int(((m-m2)/2)):int(((m-m2)/2)+m2), n1+1:n1+1+n2] = X2

    return Y


def rowint(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Interpolates the rows of image X by 2 using h.

    Parameters:
        X: Image matrix (Usually 256x256)
        h: Filter coefficients
    Returns:
        Y: Image with interpolated rows

    If len(h) is odd, each input sample is aligned with the first of
    each pair of output samples.
    If len(h) is even, each input sample is aligned with the mid point
    of each pair of output samples.
    """
    r, c = X.shape
    m = len(h)
    m2 = m // 2
    c2 = 2 * c

    # Generate X2 as X interleaved with columns of zeros.
    X2 = np.zeros((r, c2), dtype=X.dtype)
    X2[:, ::2] = X

    X2 = np.pad(X2, [(0, 0), (m2, m2)], mode='reflect' if m % 2 else 'symmetric')

    Y = np.zeros((r, c2))
    # Loop for each term in h.
    for i in range(m):
        Y = Y + h[i] * X2[:, i:i+c2]
    return Y


def rowint2(X, h):
    r, c = X.shape
    m = len(h)
    m2 = m // 2
    c2 = 2 * c

    # Generate X2 as X interleaved with columns of zeros.
    X2 = np.zeros((r, c2), dtype=X.dtype)
    X2[:, 1::2] = X

    if m % 2:
        X2 = np.pad(X2, [(0, 0), (m2, m2)], mode='reflect')
    else:
        raise NotImplementedError("It's not clear what this should do")

    Y = np.zeros((r, c2))
    # Loop for each term in h.
    for i in range(m):
        Y = Y + h[i] * X2[:, i:i+c2]
    return Y


def quant1(x, step, rise1=None):
    """
    Quantise the matrix x using steps of width step.

    The result is the quantised integers Q. If rise1 is defined,
    the first step rises at rise1, otherwise it rises at step/2 to
    give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        q = x.copy()
        return q
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1
    # Quantise abs(x) to integer values, and incorporate sign(x)..
    temp = np.ceil((np.abs(x) - rise)/step)
    q = temp*(temp > 0)*np.sign(x)
    return q


def quant2(q, step, rise1=None):
    """
    Reconstruct matrix Y from quantised values q using steps of width step.

    The result is the reconstructed values. If rise1 is defined, the first
    step rises at rise1, otherwise it rises at step/2 to give a uniform
    quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = q.copy()
        return y
    if rise1 is None:
        rise = step/2.0
        return q * step
    else:
        rise = rise1
        # Reconstruct quantised values and incorporate sign(q).
        y = q * step + np.sign(q) * (rise - step/2.0)
        return y


class QuantizingEncoder(Encoder):
    def __init__(self, step, rise1=None):
        if rise1 is None:
            rise1 = step/2
        self.step = step
        self.rise1 = rise1

    def encode(self, X):
        return quant1(X, self.step, self.rise1)

    def decode(self, Y):
        return quant2(Y, self.step, self.rise1)


def quantise(x, step, rise1=None):
    """
    Quantise matrix x in one go with step width of step using quant1 and quant2

    If rise1 is defined, the first step rises at rise1, otherwise it rises at
    step/2 to give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = x.copy()
        return y
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1
    # Perform both quantisation steps
    y = quant2(quant1(x, step, rise), step, rise)
    return y


def bpp(x):
    """
    Calculate the entropy in bits per element (or pixel) for matrix x

    The entropy represents the number of bits per element to encode x
    assuming an ideal first-order entropy code.
    """
    minx = np.min(x, axis=None)
    maxx = np.max(x, axis=None)
    # Calculate histogram of x in bins defined by bins.
    bins = list(range(int(np.floor(minx)), int(np.ceil(maxx)+1)))
    if len(bins) < 2:
        # in this case there is no information, as all the values are identical
        return 0

    h, s = np.histogram(x, bins)

    # Convert bin counts to probabilities, and remove zeros.
    p = h / np.sum(h)
    p = p[p > 0]

    # Calculate the entropy of the histogram using base 2 logs.
    return -np.sum(p * np.log(p)) / np.log(2)
