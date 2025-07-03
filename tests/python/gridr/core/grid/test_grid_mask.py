# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.grid.grid_mask module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_mask.py
"""
import os
import numpy as np
import pytest

import shapely

from gridr.core.grid.grid_rasterize import (
        GridRasterizeAlg,
        ShapelyPredicate,
        )
from gridr.core.grid.grid_mask import build_mask, Validity

#List of test parameteres for test_build_mask
# Input parameters are given in the following order :
# 
# resolution: Tuple[int, int],
# out: np.ndarray,
# geometry_origin: Tuple[float, float],
# geometry: Optional[Union[shapely.geometry.Polygon,
        # List[shapely.geometry.Polygon], shapely.geometry.MultiPolygon]],
# mask_in: Optional[np.ndarray],
# mask_in_target_win: np.ndarray,
# mask_in_resolution: Optional[Tuple[int, int]],
# mask_in_binary_threshold: float = 1e-3,
# rasterize_kwargs: Optional[Dict] = {
        # 'alg': GridRasterizeAlg.SHAPELY,
        # 'kwargs_alg': {'shapely_predicate': ShapelyPredicate.COVERS}},
        

ALG_RASTERIZE_SHAPELY_COVERS = {'alg': GridRasterizeAlg.SHAPELY,
        'kwargs_alg': {'shapely_predicate': ShapelyPredicate.COVERS}}
ALG_RASTERIZE_RASTERIO_RASTERIZE = {'alg': GridRasterizeAlg.RASTERIO_RASTERIZE,
        'kwargs_alg': {}}
DEFAULT_ALG = ALG_RASTERIZE_RASTERIO_RASTERIZE

INPUT_CHECK_001 = (
            (
                None, # shape
                None, # resolution 
                None, # out
                None, # geometry_origin
                (None, None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError # expected
        )

INPUT_CHECK_002= (
            (
                (2,3), # shape
                None, # resolution 
                None, # out
                None, # geometry_origin
                (None, None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError # expected
        )

INPUT_CHECK_003= (
            (
                (2,3), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                (None, None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.full((2,3), Validity.VALID, dtype=np.uint8) # expected
        )

INPUT_CHECK_003b= (
            (
                (2,3), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                None, # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.full((2,3), Validity.VALID, dtype=np.uint8) # expected
        )

# 004 : check col resolution cannot be different from 1
INPUT_CHECK_004= (
            (
                (2,3), # shape
                (1,2), # resolution 
                None, # out
                None, # geometry_origin
                (None, None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError # expected
        )
        
# 005 : check row resolution cannot be different from 1
INPUT_CHECK_005= (
            (
                (2,3), # shape
                (2,1), # resolution 
                None, # out
                None, # geometry_origin
                (None, None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError # expected
        )

# 006 : test out shape isgood
INPUT_CHECK_006= (
            (
                (2,3), # shape
                (1,1), # resolution 
                np.full((2,3), Validity.VALID, dtype=np.uint8), # out
                None, # geometry_origin
                None, # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.full((2,3), Validity.VALID, dtype=np.uint8) # expected
        )

# 007 : test out shape is not good
INPUT_CHECK_007= (
            (
                (2,3), # shape
                (1,1), # resolution 
                np.full((3,2), Validity.VALID, dtype=np.uint8), # out
                None, # geometry_origin
                None, # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError # expected
        )

# 008 : test geometry_origin must be provided
INPUT_CHECK_008 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError # expected
        )

# 008b : test rasterize_kwargs is given and alg is known
INPUT_CHECK_008b = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            KeyError # expected KeyError when trying to get 'alg' key
        )
        
INPUT_CHECK_008c = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                {'dummy': 0}, # rasterize_kwargs
            ),
            ValueError # expected ValueError 'alg' unknown
        )
        
INPUT_CHECK_008d = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                {'alg': 0}, # rasterize_kwargs
            ),
            ValueError # expected ValueError 'alg' unknown
        )
        
# test geometry_origin must be provided - give invalid geometry
V = Validity.VALID
I = Validity.INVALID

INPUT_CHECK_009 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (None, shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)])), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V, V],
                      [V, I, I, I, V, V],
                      [V, I, I, I, V, V],
                      [V, I, I, I, V, V],
                      [V, I, I, I, V, V]])
        )

# variant : give geometry as valid
INPUT_CHECK_009b = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[I, I, I, I, I, I],
                      [I, V, V, V, I, I],
                      [I, V, V, V, I, I],
                      [I, V, V, V, I, I],
                      [I, V, V, V, I, I]])
        )

# variant : give same geometry as valid and invalid => should result in full invalid
INPUT_CHECK_009c = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]),
                    shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)])), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.full((5,6), Validity.INVALID, dtype=np.uint8)
        )

# variant : give different geometry as valid and invalid
INPUT_CHECK_009d = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]),
                    shapely.geometry.Polygon([(2.5,2.5), (2.5,3.5), (3.5,3.5), (3.5,2.5)])), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[I, I, I, I, I, I],
                      [I, V, V, V, I, I],
                      [I, V, I, I, I, I],
                      [I, V, I, I, I, I],
                      [I, V, V, V, I, I]])
        )

# test geometry_origin must be provided - no pixel should be valid
INPUT_CHECK_010 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (5.5, 6.5), # geometry_origin
                (None, shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)])), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.full((5,6), Validity.VALID, dtype=np.uint8)
        )

# test geometry_origin must be provided - all pixel should be invalid
INPUT_CHECK_010b = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (5.5, 6.5), # geometry_origin
                (shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.full((5,6), Validity.INVALID, dtype=np.uint8)
        )

                  
# test geometry_origin must be provided - no pixel should be invalid except the first line
INPUT_CHECK_011 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (4.5, 0.5), # geometry_origin
                (None, shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)])), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[I, I, I, I, I, I],
                      [V, V, V, V, V, V],
                      [V, V, V, V, V, V],
                      [V, V, V, V, V, V],
                      [V, V, V, V, V, V]])
        )

# test geometry_origin must be provided - no pixel should be valid except the first line
INPUT_CHECK_011b = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (4.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V, V],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I]])
        )

#            np.array([[V, V, V, V, V, V],
#                      [V, V, V, V, V, V],
#                      [V, V, V, V, V, V],
#                      [V, V, V, V, V, V],
#                      [V, V, V, V, V, V]])
#            np.array([[I, I, I, I, I, I],
#                      [I, I, I, I, I, I],
#                      [I, I, I, I, I, I],
#                      [I, I, I, I, I, I],
#                      [I, I, I, I, I, I]])


# test geometry_origin must be provided - no pixel should be invalid except the first col
INPUT_CHECK_012 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 5.5), # geometry_origin
                (None, shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)])), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[I, V, V, V, V, V],
                      [I, V, V, V, V, V],
                      [I, V, V, V, V, V],
                      [I, V, V, V, V, V],
                      [I, V, V, V, V, V]])
        )

# test geometry_origin must be provided - no pixel should be valid except the first col
INPUT_CHECK_012b = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 5.5), # geometry_origin
                (shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None), # geometry_pair
                None, # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, I, I, I, I, I],
                      [V, I, I, I, I, I],
                      [V, I, I, I, I, I],
                      [V, I, I, I, I, I],
                      [V, I, I, I, I, I]])
        )
        
                  
# test mask_in_resolution must be provided - mask_in is given
INPUT_CHECK_013 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                None, # geometry_pair
                np.full((5,6), Validity.VALID, dtype=np.uint8), # mask_in
                None, # mask_in_target_win
                None, # mask_in_resolution
                None, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            ValueError
        )
        
# test mask_in_resolution must be provided - mask_in is given => output zero
INPUT_CHECK_014 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                None, # geometry_pair
                np.full((5,6), Validity.VALID, dtype=np.uint8), # mask_in
                None, # mask_in_target_win
                (1,1), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.full((5,6), Validity.VALID, dtype=np.uint8)
        )
        
# test mask_in : check binarization
INPUT_CHECK_015 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                None, # geometry_pair
                np.array([[V, V-0.1, V-0.01, V-0.001, V-0.0001, V],
                          [I, I, I, I, I, I],
                          [I, I, I, I, I, I],
                          [I, I+0.9999, I+0.99, I+0.999, I, I],
                          [I, I, I, I, I, I]]), # mask_in
                None, # mask_in_target_win
                (1,1), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.array([[V, I, I, V, V, V],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I],
                      [I, V, I, V, I, I],
                      [I, I, I, I, I, I]])
        )


# test mask_in_resolution must be provided - mask_in is given => 
# check that shape and value match (the binary threshold is set to 0.999
# Non strict V value will be set to I)
INPUT_CHECK_016 = (
            (
                (10,5), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                None, # geometry_pair
                np.array([[V, V, V],
                          [V, I, V],
                          [V, I, I],
                          [V, V, V]]), # mask_in
                None, # mask_in_target_win
                (3,2), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V],
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, V, V, V, V]])
        )

# same as 16 but we set a window
INPUT_CHECK_017 = (
            (
                (7,2), # shape
                (1,1), # resolution 
                None, # out
                None, # geometry_origin
                None, # geometry_pair
                np.array([[V, V, V],
                          [V, I, V],
                          [V, I, I],
                          [V, V, V]]), # mask_in
                [(3,9),(3,4)], # mask_in_target_win
                (3,2), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                None, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V],
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, V, V, V, V]])[3:10, 3:5]
        )

# mix test : from INPUT_CHECK_011 : add a mask_in with only one masked pixel
INPUT_CHECK_018 = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (4.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None), # geometry_pair
                np.array([[V, I, V, V, V, V],
                          [V, V, V, V, V, V],
                          [V, V, V, V, V, V],
                          [V, V, I, V, V, V],
                          [V, V, V, V, V, V]]), # mask_in
                None, # mask_in_target_win
                (1,1), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, I, V, V, V, V],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I],
                      [I, I, I, I, I, I]])
        )
        
INPUT_CHECK_018b = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (4.5, 0.5), # geometry_origin
                (None, shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)])), # geometry_pair
                np.array([[V, I, V, V, V, V],
                          [V, V, V, V, V, V],
                          [V, V, V, V, V, V],
                          [V, V, I, V, V, V],
                          [V, V, V, V, V, V]]), # mask_in
                None, # mask_in_target_win
                (1,1), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[I, I, I, I, I, I],
                      [V, V, V, V, V, V],
                      [V, V, V, V, V, V],
                      [V, V, I, V, V, V],
                      [V, V, V, V, V, V]])
        )

INPUT_CHECK_018c = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (4.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]),
                    shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)])), # geometry_pair
                np.array([[V, I, V, V, V, V],
                          [V, V, V, V, V, V],
                          [V, V, V, V, V, V],
                          [V, V, I, V, V, V],
                          [V, V, V, V, V, V]]), # mask_in
                None, # mask_in_target_win
                (1,1), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.full((5,6), Validity.INVALID, dtype=np.uint8)
        )
        
        
INPUT_CHECK_018d = (
            (
                (5,6), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]),
                    shapely.geometry.Polygon([(2.5,2.5), (2.5,3.5), (3.5,3.5), (3.5,2.5)])), # geometry_pair
                np.array([[V, V, V, V, V, V],
                          [V, V, V, I, V, V],
                          [V, V, V, V, V, V],
                          [V, V, I, V, V, V],
                          [V, I, V, V, V, V]]), # mask_in
                None, # mask_in_target_win
                (1,1), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[I, I, I, I, I, I],
                      [I, V, V, I, I, I],
                      [I, V, I, I, I, I],
                      [I, V, I, I, I, I],
                      [I, I, V, V, I, I]])
        )



# In this test we want to check everything
# We inspire from 017 (oversample grid mask) but change the input mask and add geometry        
INPUT_CHECK_019 = (
            (
                (7,2), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (None, None),
                np.array([[V, V, V],
                          [V, I, V],
                          [V, V, I],
                          [V, V, V]]), # mask_in
                [(3,9),(3,4)], # mask_in_target_win
                (3,2), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V],
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, V], # first line taken ; the geometry mask invalidates the last col here.
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, V, V]])[3:10, 3:5]
        )

# Add an invalid geometry - notice that the geometry is applied on the output geometry (after windowing of the
# raster mask)
INPUT_CHECK_019b = (
            (
                (7,2), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (None, shapely.geometry.Polygon([(1.5,0.5), (1.5,1.5), (2.5,1.5), (2.5,0.5)])),
                np.array([[V, V, V],
                          [V, I, V],
                          [V, V, I],
                          [V, V, V]]), # mask_in
                [(3,9),(3,4)], # mask_in_target_win
                (3,2), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V],
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, I], # first line taken ; the geometry mask invalidates the last col here.
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, V, V]])[3:10, 3:5]
        )


# Add a valid geometry - here we cover all the masked area that is invalidated by the raster
# that mask should not change anything
INPUT_CHECK_019c = (
            (
                (7,2), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(-10.5,-10.5), (-10.5,10.5), (10.5,10.5), (10.5,-10.5)]),
                    None),
                np.array([[V, V, V],
                          [V, I, V],
                          [V, V, I],
                          [V, V, V]]), # mask_in
                [(3,9),(3,4)], # mask_in_target_win
                (3,2), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V],
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, V],
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, V, V]])[3:10, 3:5]
        )



# Add a valid geometry - here we cover all the masked area that is invalidated by the raster
# that mask should not change anything
INPUT_CHECK_019d = (
            (
                (7,2), # shape
                (1,1), # resolution 
                None, # out
                (0.5, 0.5), # geometry_origin
                (shapely.geometry.Polygon([(-10.5,-10.5), (-10.5,10.5), (10.5,10.5), (10.5,-10.5)]),
                    shapely.geometry.Polygon([(1.5,0.5), (1.5,1.5), (2.5,1.5), (2.5,0.5)])),
                np.array([[V, V, V],
                          [V, I, V],
                          [V, V, I],
                          [V, V, V]]), # mask_in
                [(3,9),(3,4)], # mask_in_target_win
                (3,2), # mask_in_resolution
                0.999, # mask_in_binary_threshold
                DEFAULT_ALG, # rasterize_kwargs
            ),
            np.array([[V, V, V, V, V],
                      [V, I, I, I, V], 
                      [V, I, I, I, V], 
                      [V, I, I, I, I], # first line taken ; the geometry mask invalidates the last col here.
                      [V, I, I, I, I],
                      [V, I, I, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, I, I],
                      [V, V, V, V, V]])[3:10, 3:5]
        )

        
class TestGridMask:
    """Test class"""
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (INPUT_CHECK_001[0], INPUT_CHECK_001[1], 6),
            (INPUT_CHECK_002[0], INPUT_CHECK_002[1], 6),
            (INPUT_CHECK_003[0], INPUT_CHECK_003[1], 6),
            (INPUT_CHECK_003b[0], INPUT_CHECK_003b[1], 6),
            (INPUT_CHECK_004[0], INPUT_CHECK_004[1], 6),
            (INPUT_CHECK_005[0], INPUT_CHECK_005[1], 6),
            (INPUT_CHECK_006[0], INPUT_CHECK_006[1], 6),
            (INPUT_CHECK_007[0], INPUT_CHECK_007[1], 6),
            (INPUT_CHECK_008b[0], INPUT_CHECK_008[1], 6),
            (INPUT_CHECK_008c[0], INPUT_CHECK_008b[1], 6),
            (INPUT_CHECK_008d[0], INPUT_CHECK_008c[1], 6),
            (INPUT_CHECK_008[0], INPUT_CHECK_008d[1], 6),
            (INPUT_CHECK_009[0], INPUT_CHECK_009[1], 6),
            (INPUT_CHECK_009b[0], INPUT_CHECK_009b[1], 6),
            (INPUT_CHECK_009c[0], INPUT_CHECK_009c[1], 6),
            (INPUT_CHECK_009d[0], INPUT_CHECK_009d[1], 6),
            (INPUT_CHECK_010[0], INPUT_CHECK_010[1], 6),
            (INPUT_CHECK_010b[0], INPUT_CHECK_010b[1], 6),
            (INPUT_CHECK_011[0], INPUT_CHECK_011[1], 6),
            (INPUT_CHECK_011b[0], INPUT_CHECK_011b[1], 6),
            (INPUT_CHECK_012[0], INPUT_CHECK_012[1], 6),
            (INPUT_CHECK_012b[0], INPUT_CHECK_012b[1], 6),
            (INPUT_CHECK_013[0], INPUT_CHECK_013[1], 6),
            (INPUT_CHECK_014[0], INPUT_CHECK_014[1], 6),
            (INPUT_CHECK_015[0], INPUT_CHECK_015[1], 6),
            (INPUT_CHECK_016[0], INPUT_CHECK_016[1], 6),
            (INPUT_CHECK_017[0], INPUT_CHECK_017[1], 6),
            (INPUT_CHECK_018[0], INPUT_CHECK_018[1], 6),
            (INPUT_CHECK_018b[0], INPUT_CHECK_018b[1], 6),
            (INPUT_CHECK_018c[0], INPUT_CHECK_018c[1], 6),
            (INPUT_CHECK_018d[0], INPUT_CHECK_018d[1], 6),
            (INPUT_CHECK_019[0], INPUT_CHECK_019[1], 6),
            (INPUT_CHECK_019b[0], INPUT_CHECK_019b[1], 6),
            (INPUT_CHECK_019c[0], INPUT_CHECK_019c[1], 6),
            (INPUT_CHECK_019d[0], INPUT_CHECK_019d[1], 6),
            ])
    def test_build_mask(self, data, expected, testing_decimal):
        """Test the build_mask
        
        Args:
            data : input data as a tuple containing all the arguments
            expected: expected data 
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        shape,\
                resolution,\
                out,\
                geometry_origin,\
                geometry_pair,\
                mask_in,\
                mask_in_target_win,\
                mask_in_resolution,\
                mask_in_binary_threshold,\
                rasterize_kwargs = data
        #expected_mask = expected
        dtype = np.uint8
        expected_out_id = None
        if out is not None:
            expected_dtype = out.dtype
            expected_out_id = id(out.data)
        else:
            expected_dtype = dtype
        try:
            mask = build_mask(shape=shape, resolution=resolution, out=out,
                    geometry_origin=geometry_origin, geometry_pair=geometry_pair,
                    mask_in=mask_in, mask_in_target_win=mask_in_target_win,
                    mask_in_resolution=mask_in_resolution,
                    mask_in_binary_threshold=mask_in_binary_threshold,
                    rasterize_kwargs=rasterize_kwargs,
                    oversampling_dtype=np.float32)
            if out is not None:
                mask = out
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except TypeError:
                pass
            # Check
            np.testing.assert_array_almost_equal(mask, expected, decimal=testing_decimal)
            assert(mask.dtype == expected_dtype)
            if expected_out_id is not None:
                assert(id(mask.data) == expected_out_id)