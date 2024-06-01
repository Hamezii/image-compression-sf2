"""Module providing basic functions for familiarisation phase."""


import scipy.io
import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib import colors


def load_mat_img(img, img_info, cmap_info={}):
    """
    Load a .mat image into python.

    Parameters:
    img (str): .mat file path
    img_info (str): name under which the image matrix is stored
    cmap_info (set of strings): a set of strings indicating names of colormaps

    Returns:
    X (numpy.ndarray): image stored in a matrix
    cmaps_dict (dict): Dictionary of numpy.ndarray's of colormaps
    """
    # check that a .mat filename is provided
    if not img.endswith('.mat'):
        raise ValueError('Please provide a .mat image name.')
    img_contents = scipy.io.loadmat(img)
    X = img_contents[img_info]
    cmaps_dict = {}
    for cmap_array in cmap_info:
        cmaps_dict[cmap_array] = prep_cmap_array_plt(
            img_contents[cmap_array], cmap_array)

    return X, cmaps_dict


def prep_cmap_array_plt(cmap_array, cmap_name, N=256):
    """
    Convert colormaps from numpy arrays to matplotlib format.

    Parameters:
    cmap_array (numpy.ndarray): Array containing a colormap
    cmap_name (str): name to store the colormap with
    N (int): Length of colormap with a default of 256

    Returns:
    cmap_plt (matplotlib.colors.LinearSegmentedColormap)
    """
    # assumes array is Nx3 for RGB
    # appends 1's for the alpha channel making the array Nx4
    cmap_array = np.c_[cmap_array, np.ones(len(cmap_array))]
    # TODO: Find out what the gamma parameter below is
    cmap_plt = colors.LinearSegmentedColormap.from_list(cmap_name,
                                                        cmap_array, N)
    return cmap_plt


class PowerTwoTickLocator(matplotlib.ticker.Locator):
    def __call__(self):
        """Return the locations of the ticks."""
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        nticks = max(self.axis.get_tick_space(), 2)
        return multiples_pow2_between(vmin, vmax, nticks)


def multiples_pow2_between(vmin, vmax, n: int) -> np.ndarray:
    """
    Get all multiples between `vmin` and `vmax` of the largest power of 2 such
    that there are at most `n` such multiples.

    The powers of 2 can be negative.

    This is not part of the lab content, and is simply used to choose nice ticks
    for plots.
    """
    if vmax < vmin:
        vmin, vmax = vmax, vmin

    diff = vmax - vmin
    dmant, dexp = np.frexp(diff)

    def possible_sizes():
        step = dexp
        min_i = np.ceil(vmin / (2.0 ** step))
        max_i = np.floor(vmax / (2.0 ** step))
        while min_i > max_i:
            step -= 1
            min_i = np.ceil(vmin / (2.0 ** step))
            max_i = np.floor(vmax / (2.0 ** step))
        yield (min_i, max_i, step)
        while max_i - min_i + 1 <= n:
            yield (min_i, max_i, step)
            step -= 1
            min_i = np.ceil(vmin / (2.0 ** step))
            max_i = np.floor(vmax / (2.0 ** step))

    # find the largest possible size, and use it
    for min_i, max_i, step in possible_sizes():
        pass
    return np.arange(min_i, max_i + 1)*(2.0**step)


def plot_image(X, *, ax=None, **kwargs):
    """
    A wrapper around `plt.imshow` that uses the `gray` colormap by default,
    and chooses suitable axis ticks for this lab.
    """
    m, n = X.shape
    kwargs.setdefault('extent', (0, n, m, 0))
    kwargs.setdefault('cmap', 'gray')
    if ax is None:
        ax = plt.gca()
    ret = ax.imshow(X, **kwargs)
    ax.xaxis.set_major_locator(PowerTwoTickLocator())
    ax.yaxis.set_major_locator(PowerTwoTickLocator())
    return ret


if __name__ == "__main__":
    # to run this python -m cued_sf2_lab.familiarisation
    img = 'lighthouse.mat'
    img_info = 'X'
    cmap_info = {'map', 'map2'}
    X, cmaps_dict = load_mat_img(img, img_info, cmap_info)
    # print('Loaded X of shape: ', X.shape)
    # print('Loaded color_map_1 of shape:', cmaps_dict['map'].shape)
    # print(type(X))

    cmap_array = cmaps_dict['map2']
    cmap_plt = prep_cmap_array_plt(cmap_array, 'map2')
    # plot_image(X, cmap_plt='gray')
    plot_image(X, cmap_plt)
