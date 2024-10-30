# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Chunk definition computation module
"""
from typing import List, Tuple

import numpy as np

def get_chunk_boundaries(
        nsize: int,
        chunk_size: int,
        merge_last: bool = False,
        ) -> List[Tuple[int, int]]:
    """
    Compute chunks from a total number of elements and a chunk size.
    The merge_last optional argument can be set to true to merge last chunk
    with the previous one, if its size is lower than the chunk size.
    
    The chunks are returned as a list of boundaries index.
    
    Args:
        nsize: total number of elements
        chunk_size: target chunk size
        merge_last: boolean option to enable merge of the last chunk if its size
                is lower than the chunk size.
    
    Returns:
        the chunk's intervals
    """
    # Set default fallback in case chunk_size equals 0
    intervals = [(0, nsize),]
    if chunk_size > 0:   
        limits = np.unique(np.concatenate(
                (np.arange(0, nsize+1, chunk_size), [nsize])))
        intervals = np.asarray(list(zip(limits[0:-1], limits[1:])))
        if merge_last and (intervals[-1][1] - intervals[-1][0]) < chunk_size:
            # change second last interval upper limit to correspond to last interval
            # upper limit.
            intervals[-2][1] = intervals[-1][1]
            # do not consider last interval
            intervals = intervals[0:-1]
    return intervals