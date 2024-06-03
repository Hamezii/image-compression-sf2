""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec
from .common import HeaderType, jpegdec_lbt

def decode(vlc: np.ndarray, header: HeaderType) -> np.ndarray:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen decoding scheme
    # return jpegdec(vlc, jpeg_quant_size, hufftab=header, log=False)
    jpeg_quant_size = header[1]
    return jpegdec_lbt(vlc, jpeg_quant_size, hufftab=header[0], log=False) + 128.0
