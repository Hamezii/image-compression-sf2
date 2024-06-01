import numpy as np
import operator

__all__ = ["dct_ii", "dct_iv", "colxfm", "regroup"]

def dct_ii(N: int) -> np.ndarray:
    """
    Generate the 1D DCT transform matrix of size N.

    Parameters:
    N (int): Size of DCT matrix required

    Returns:
    C (2D np array): 1D DCT transform matrix

    Uses an orthogonal Type-II DCT.
    Y = C * X tranforms N-vector X into Y.
    """
    C = np.ones((N, N)) / np.sqrt(N)
    theta = (np.arange(N) + 0.5) * (np.pi/N)
    g = np.sqrt(2/N)
    for i in range(1, N):
        C[i, :] = g * np.cos(theta*i)

    return C


def dct_iv(N: int) -> np.ndarray:
    """
    Generate the 1D DCT transform matrix of size N.

    Parameters:
    N (int): Size of DCT matrix required

    Returns:
    C (2D np array): 1D DCT transform matrix

    Uses an orthogonal Type-IV DCT.
    Y = C * X tranforms N-vector X into Y.
    """
    C = np.ones((N, N)) / np.sqrt(N)
    theta = (np.arange(N) + 0.5) * (np.pi/N)
    g = np.sqrt(2/N)
    for i in range(N):
        C[i, :] = g * np.cos(theta*(i+0.5))

    return C


def colxfm(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Transforms the columns of X using the tranformation in C.

    Parameters:
        X: Image whose columns are to be transformed
        C: N-size 1D DCT coefficients obtained using dct_ii(N)
    Returns:
        Y: Image with transformed columns

    PS: The height of X must be a multiple of the size of C (N).
    """
    N = len(C)
    m, n = X.shape

    # catch mismatch in size of X
    if m % N != 0:
        raise ValueError('colxfm error: height of X not multiple of size of C')

    Y = np.zeros((m, n))
    # transform columns of each horizontal stripe of pixels, N*n
    for i in range(0, m, N):
        Y[i:i+N, :] = C @ X[i:i+N, :]

    return Y


def regroup(X, N):
    """
    Regroup the rows and columns in X.
    Rows/Columns that are N apart in X are adjacent in Y.

    Parameters:
    X (np.ndarray): Image to be regrouped
    N (list): Size of 1D DCT performed (could give int)

    Returns:
    Y (np.ndarray): Regoruped image
    """
    # if N is a 2-element list, N[0] is used for columns and N[1] for rows.
    # if a single value is given, a square matrix is assumed
    try:
        N_m = N_n = operator.index(N)
    except TypeError:
        N_m, N_n = N

    m, n = X.shape

    if m % N_m != 0 or n % N_n != 0:
        raise ValueError('regroup error: X dimensions not multiples of N')

    X = X.reshape(m // N_m, N_m, n // N_n, N_n)  # subdivide the axes
    X = X.transpose((1, 0, 3, 2))                # permute them
    return X.reshape(m, n)                       # and recombine
