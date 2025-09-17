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
Chunk definition computation module
"""
import itertools
from typing import List, Tuple

import numpy as np


def get_chunk_boundaries(
    nsize: int,
    chunk_size: int,
    merge_last: bool = False,
) -> List[Tuple[int, int]]:
    """Compute chunks from a total number of elements and a chunk size.

    This method divides a total number of elements (`nsize`) into smaller
    segments (chunks) based on a specified `chunk_size`. Each chunk is
    represented by a tuple `(start_index, end_index)`.

    The `merge_last` argument provides an option to merge the final chunk with
    the preceding one if its size is less than the `chunk_size`. This prevents
    very small chunks at the end of the sequence.

    Parameters
    ----------
    nsize : int
        The total number of elements to be divided into chunks.

    chunk_size : int
        The desired maximum size for each chunk. Must be a positive integer.

    merge_last : bool, default False
        If ``True``, the last chunk will be merged with the second-to-last
        chunk if its size is smaller than `chunk_size`.

    Returns
    -------
    list[tuple[int, int]]
        A list of tuples, where each tuple `(start, end)` represents the
        inclusive start and exclusive end indices of a chunk.

    """
    # Set default fallback in case chunk_size equals 0
    intervals = [
        (0, nsize),
    ]
    if chunk_size > 0 and chunk_size < nsize:
        limits = np.unique(np.concatenate((np.arange(0, nsize + 1, chunk_size), [nsize])))
        intervals = np.asarray(list(zip(limits[0:-1], limits[1:], strict=True)))
        if merge_last and (intervals[-1][1] - intervals[-1][0]) < chunk_size:
            # change second last interval upper limit to correspond to last interval
            # upper limit.
            intervals[-2][1] = intervals[-1][1]
            # do not consider last interval
            intervals = intervals[0:-1]
    return intervals


def get_chunk_shapes(
    shape: Tuple, chunk_shape: Tuple, merge_last=False
) -> List[Tuple[Tuple[int, int]]]:
    """Compute chunks for an N-dimensional shape.

    This method calculates the tensor product of chunks for each axis of an
    N-dimensional array based on a given `chunk_shape`.

    Parameters
    ----------
    shape : tuple of int
        The N-dimensional shape to be chunked, e.g., `(rows, cols, depth)`. Each
        element must be a non-negative integer.

    chunk_shape : tuple of int
        The desired shape of a single chunk, e.g., `(chunk_rows, chunk_cols)`.
        Its length must match the `shape`'s length, and each element must be a
        positive integer.

    merge_last : bool, default False
        If `True`, the last chunk along each dimension will be merged with the
        previous one if its size is smaller than the corresponding `chunk_shape`
        dimension.

    Returns
    -------
    list[tuple[tuple[int, int], ...]]
        A list where each element is an N-dimensional chunk definition. Each
        N-dimensional chunk is represented as a tuple of N 2-element tuples,
        where each 2-element tuple `(start, end)` defines the inclusive start
        and exclusive end indices for that dimension.

    """
    chunks = [
        get_chunk_boundaries(nsize, chunk_size, merge_last)
        for nsize, chunk_size in zip(shape, chunk_shape, strict=True)
    ]
    return list(itertools.product(*chunks))
