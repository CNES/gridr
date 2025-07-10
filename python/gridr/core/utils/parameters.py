# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Parameters operations utils module
"""
from typing import Union, Tuple, Any

def tuplify(p: Union[Any, Tuple], ndim: int):
    """Utility method to convert a single parameter to a tuple.

    If the parameter `p` is already a sequence (list or tuple), it is returned
    as is. Otherwise, the output tuple corresponds to the repeated couple
    `(p, p)` along each dimension.

    For example:
    ::

        tuplify('a', 3)  # Returns (('a', 'a'), ('a', 'a'), ('a', 'a'))

    Parameters
    ----------
    p : Any or tuple
        The parameter to tuplify. It can be any single value or an existing
        sequence.
    ndim : int
        The number of dimensions, which determines how many times `(p, p)`
        is repeated if `p` is not already a sequence.

    Returns
    -------
    Any or tuple
        The tuplified parameter. If `p` was already a sequence, it returns `p`
        itself. Otherwise, it returns a tuple of `ndim` pairs, where each pair
        is `(p, p)`.

    """
    out = p
    try:
        p[0]
    except TypeError:
        out = ((p, p), ) * ndim
    return out
