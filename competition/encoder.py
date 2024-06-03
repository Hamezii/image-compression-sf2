""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.jpeg import jpegenc

from .common import HeaderType, optimize_huffman_step_size, jpegenc_lbt, get_equivalent_lbt_error_quant

from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.jpeg import (HuffmanTable,
    jpegenc, jpegdec, quant1, quant2, huffenc, huffdflt, huffdes, huffgen, diagscan, runampl, HuffmanTable)
from typing import Tuple, NamedTuple, Optional
from cued_sf2_lab.bitword import bitword


def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    # Size for default implementation, plus 8 bytes for float representing step size
    return (len(header[0].bits) + len(header[0].huffval)) * 8 + 8*8


def encode(X: np.ndarray) -> Tuple[np.ndarray, HeaderType]:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen encoding scheme. If you do not use a header,
    # then `return vlc, None`.
    # vlc, hufftab = jpegenc(X, jpeg_quant_size, opthuff=True, log=False)

    # jpeg_rms_error = []
    # size = [[8, 8], [8, 16], [16, 16]]
    # for i in range(3):
    #     n, m = size[i][0], size[i][1]
    #     step_opt = optimize_huffman_step_size(X, n, m)
    #     vlc, hufftab = jpegenc_lbt(X, step_opt, n, m, opthuff=True, log=False)
    #     Z_lbt = jpegdec_lbt(vlc, step_opt, n, m, hufftab=hufftab)
    #     jpeg_rms_error.append(np.std(X_test - Z_lbt))
    X = X - 128.0
    jpeg_quant_size = optimize_huffman_step_size(X, 8, 8) #* 1.1
    vlc, hufftab = jpegenc_lbt(X, jpeg_quant_size, 8, 8, opthuff=True, log=False)
    return vlc, [hufftab, jpeg_quant_size]