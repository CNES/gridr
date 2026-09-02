# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://github.com/CNES/gridr).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Parameters operations utils module
"""
import numpy as np
from typing import Any, Tuple, Union


def tuplify(
    p: Any, ndim: int, fill: Any, strict: bool = True
) -> Tuple[Tuple[Any, Any], ...]:
    """Utility method to convert a single parameter to a tuple of pairs.

    If the parameter `p` is already a list or tuple, its entries are used
    as-is, one per dimension. Otherwise, `p` is treated as a scalar and
    repeated as the pair `(p, p)` for each of the `ndim` dimensions.
    
    For example:
    ::

        tuplify('a', 3)  # Returns (('a', 'a'), ('a', 'a'), ('a', 'a'))
        tuplify((('a', 'b'), ), ndim=2, fill='c', strict=False)  # Returns (('c', 'c'), ('a', 'b'))
        tuplify((('a', 'b'), ('a', 'b')), ndim=1, fill='c')  # Raises an exception

    Parameters
    ----------
    p : Any or tuple/list of pairs
        The parameter to tuplify. A scalar value, or an existing tuple/list
        of `(value, value)` pairs, at most `ndim` long.
    
    ndim : int
        The number of dimensions expected in the output, i.e. the number of
        pairs.
    
    fill : Any
        The value used to left-pad the output when `p` is given as a tuple/list
        shorter than `ndim` and `strict` is False.
    
    strict : bool, default True
        If `p` is given as a tuple/list with fewer than `ndim` entries: raise
        an exception when True; left-pad with `(fill, fill)` pairs to reach
        `ndim` when False.

    Returns
    -------
    tuple of tuple
        A tuple of `ndim` pairs.
    
    Raises
    ------
    ValueError
        If `p` is given as a tuple/list with more entries than `ndim`, or with
        fewer entries than `ndim` while `strict` is True.

    """
    if isinstance(p, (tuple, list)):
        pairs = tuple(p)
    else:
        return ((p, p),) * ndim
        
    n = len(pairs)
    if n > ndim:
        raise ValueError(
            f"`p` has {n} entries but `ndim`={ndim}: too many dimensions "
            "provided."
        )
    if n < ndim:
        if strict:
            raise ValueError(
                f"`p` has {n} entries but `ndim`={ndim}: not enough dimensions"
                f" provided (use strict=False to auto-fill)."
            )
        pairs = ((fill, fill),) * (ndim - n) + pairs
    
    return pairs
