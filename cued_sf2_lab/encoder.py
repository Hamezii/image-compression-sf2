"""
This file contains an experiment attempt at providing an object-oriented
interface for building multi-step encoders.

The notebooks do not currently describe using these, but you are free to use
them if you want to. One implementation is provided,
`cued.laplacian_pyramid.QuantizationEncoder`.
"""

from typing import List
from functools import reduce

__all__ = ["Encoder", "SequentialEncoder"]

class Encoder:
    """ Base class for a stage in an image processing pipeline """
    __slots__ = ()

    def encode(self, X):
        raise NotImplementedError

    def decode(self, Y):
        raise NotImplementedError


class SequentialEncoder(Encoder):
    """ Perform encoders sequentially """

    __slots__ = ('seq',)

    def __init__(self, encs: List[Encoder]):
        self.seq = encs

    def encode(self, X):
        return reduce(lambda X, enc: enc.encode(X), self.seq, X)

    def decode(self, Y):
        # note: decode in reverse order
        return reduce(lambda Y, enc: enc.decode(Y), self.seq[::-1], Y)
