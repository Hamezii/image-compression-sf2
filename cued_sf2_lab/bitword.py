"""
This file contains an experimental attempt at providing a custom numpy dtype
for storing variable-length words.

The main advantages of using `bitword` over a two-column array are:
* Better printing, codewords are shown in binary.
* Marginally improved spelling; ``vlc[i].val`` instead of ``vlc[i,0]``.
* Built-in checking for illegal values at construction time.

Conversion back and forth between this and simple two-column arrays can be done
with `numpy.lib.recfunctions.structured_to_unstructured` and
`numpy.lib.recfunctions.unstructured_to_structured` (with
``dtype=bitword.dtype``).
"""

import numpy as np
import operator
import warnings
import functools

__all__ = ["bitword"]


class bitword(np.record):
    """ Helper numpy structured type for storing variable-width codewords.

    This has two fields:
    * ``'val'``, which stores the codeword itself
    * ``'bits'``, which counts the number of bits in the codeword

    The constructor enforces that things like ``bitword(0b100, 2)`` are illegal,
    as `0b100` needs 3 bits not two to encode.
    """
    def __new__(cls, val, bits=None):
        if bits is None:
            bits = operator.index(val).bit_length()
        elif val not in range(1 << bits):
            raise OverflowError(f'0b{val:b} does not fit in {bits} bits')
        return np.array((val, bits), dtype=cls.dtype)[()]

    def __str__(self) -> str:
        return f"0b{self.val:0{self.bits}b}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)}, {self.bits})"

    @classmethod
    def verify(cls, arr) -> None:
        if arr.dtype != cls.dtype:
            raise TypeError(f"Input array must be of type {cls.__name__}")
        arr_max = np.array(1, dtype=object) << arr['bits']
        bad = arr['val'] >= arr_max
        bad_inds, = bad.nonzero()
        if len(bad_inds):
            raise ValueError(
                f"word lengths are inconsistent at indices:\n"
                f"{bad_inds}\n"
                f"with values:\n"
                f"{arr[bad_inds]}")


# Adding this attribute this makes `dtype=bitword` work in a handful of places.
# The numpy maintainers haven't decided whether this behavior is intentional!
bitword.dtype = np.dtype((bitword, [('val', np.uint32), ('bits', np.uint8)]))


def _patch(obj, attr):
    """ decorator for replacing `obj.attr` """
    def decorator(val):
        old = getattr(obj, attr)
        setattr(obj, attr, val(old))
    return decorator


# Patch the numpy printing to make bitwords look nice! Using implementation
# details like this is fragile, but
# - there's no other way to do it besides not doing it at all.
# - the demonstrator responsible for writing this code was involved in writing
#   the original code in numpy that it patches.
try:
    from numpy.core.arrayprint import StructuredVoidFormat, _get_format_function
    old_from_data = StructuredVoidFormat.from_data
except Exception:
    warnings.warn("unable to patch repr of `bitword`")
else:
    @_patch(StructuredVoidFormat, 'from_data')
    def _(old_from_data):
        @functools.wraps(old_from_data)
        def from_data(data, **options):
            if data.dtype == bitword.dtype:
                bit_format = _get_format_function(data['bits'], **options)
                max_bits = np.max(data['bits'])
                def formatter(x):
                    return f"bitword({str(x):>{max_bits+2}s}, {bit_format(x.bits)})"
                return formatter
            else:
                return old_from_data(data, **options)
        return from_data

    @_patch(np.core.arrayprint, 'dtype_is_implied')
    def _(old_dtype_is_implied):
        @functools.wraps(old_dtype_is_implied)
        def dtype_is_implied(dtype):
            return dtype == bitword.dtype or old_dtype_is_implied(dtype)
        return dtype_is_implied
