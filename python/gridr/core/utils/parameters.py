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
from typing import Any, Tuple, Union


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
        out = ((p, p),) * ndim
    return out
