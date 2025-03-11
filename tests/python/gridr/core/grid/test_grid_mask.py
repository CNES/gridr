# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
from gridr.core.grid.grid_mask import build_mask

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
        (None, None, None, None, None, None, None, None, None, None),
        ValueError)
INPUT_CHECK_002 = (
        ((2,3), None, None, None, None, None, None, None, None, None),
        ValueError)
INPUT_CHECK_003 = (
        ((2,3), (1,1), None, None, None, None, None, None, None, None),
        np.zeros((2,3)))
INPUT_CHECK_004 = (
        ((2,3), (1,2), None, None, None, None, None, None, None, None),
        ValueError)
INPUT_CHECK_005 = (
        ((2,3), (2,1), None, None, None, None, None, None, None, None),
        ValueError)
# test out shape isgood
INPUT_CHECK_006 = (
        ((2,3), (1,1), np.zeros((2,3)), None, None, None, None, None, None, None),
        np.zeros((2,3)))
# test out shape is not good
INPUT_CHECK_007 = (
        ((2,3), (1,1), np.zeros((3,2)), None, None, None, None, None, None, None),
        ValueError)
# test geometry_origin must be provided
INPUT_CHECK_008 = (
        ((5,6), (1,1), None, None, shapely.geometry.Polygon([(1.5,1.5), (3.5,1.5), (3.5,4.5), (1.5,4.5)]), None, None, None, None, None),
        ValueError)
# test geometry_origin must be provided - no pixel should be masked
INPUT_CHECK_009 = (
        ((5,6), (1,1), None, (0.5, 0.5), shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None, None, None, None, DEFAULT_ALG),
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]))
# test geometry_origin must be provided - all pixels should be masked
INPUT_CHECK_010 = (
        ((5,6), (1,1), None, (5.5, 6.5), shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None, None, None, None, DEFAULT_ALG),
        np.array([[1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1]]))
# test geometry_origin must be provided - all pixels should be masked except the first line
INPUT_CHECK_011 = (
        ((5,6), (1,1), None, (4.5, 0.5), shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None, None, None, None, DEFAULT_ALG),
        np.array([[0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1]]))
# test geometry_origin must be provided - all pixels should be masked except the first column
INPUT_CHECK_012 = (
        ((5,6), (1,1), None, (0.5, 5.5), shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), None, None, None, None, DEFAULT_ALG),
        np.array([[0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1, 1]]))
# test mask_in_resolution must be provided - mask_in is given
INPUT_CHECK_013 = (
        ((5,6), (1,1), None, None, None, np.array([[0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0]]),
                None, None, None, None),
        ValueError)
# test mask_in_resolution must be provided - mask_in is given => output zero
INPUT_CHECK_014 = (
        ((5,6), (1,1), None, None, None, np.array([[0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0]]),
                None, (1,1), 1e-3, None),
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]))
# test mask_in : check binarization
INPUT_CHECK_015 = (
        ((5,6), (1,1), None, None, None, np.array([[0, 0, 0.1999, 0, 0, 0],
                                                   [0, 0, 0.2, 0, 0, 0],
                                                   [0, 0, 0.3, 0, 0, 0],
                                                   [0, 0, 0, 0, 0.4, 0],
                                                   [0, 0, 0, 0, 0, 0]]),
                None, (1,1), 0.2, None),
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0]]))
# test mask_in_resolution must be provided - mask_in is given => output zero
# mask at full resolution : 
        # np.array([[0, 0, 0, 0, 0],
                  # [0, 1, 1, 1, 0], 
                  # [0, 1, 1, 1, 0], 
                  # [0, 1, 1, 1, 0], 
                  # [0, 1, 1, 1, 1],
                  # [0, 1, 1, 1, 1],
                  # [0, 1, 1, 1, 1],
                  # [0, 1, 1, 1, 1],
                  # [0, 1, 1, 1, 1],
                  # [0, 0, 0, 0, 0]]))
# check that shape and value match (the binary threshold is set to 1e-3
# Non strict 0 value will be set to 1)
INPUT_CHECK_016 = (
        ((10,5), (1,1), None, None, None, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 1, 1],
                                                   [0, 0, 0]]),
                None, (3,2), 1e-3, None),
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0]]))
# same as 16 but we set a window
INPUT_CHECK_017 = (
        ((7,2), (1,1), None, None, None, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 1, 1],
                                                   [0, 0, 0]]),
                [(3,9),(3,4)], (3,2), 1e-3, None),
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0]])[3:10, 3:5])
# mix test : from INPUT_CHECK_011 : add a mask_in with only one masked pixel
INPUT_CHECK_018 = (
        ((5,6), (1,1), None, (4.5, 0.5), shapely.geometry.Polygon([(0.,0.), (6.,0.), (6.,5.), (0.,5.)]), np.array([[0, 1, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0, 0, 0]]),
                None, (1,1), 1e-3, DEFAULT_ALG),
        np.array([[0, 1, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1]]))
# mix everything
# take 17 and add a geometry polygon that covers all but the first column
# the result should be the same as 17 except last column => should be masked
# Note : a pixel is contained if its center is contained
INPUT_CHECK_019 = (
        ((7,2), (1,1), None, (0.5, 0.5), shapely.geometry.Polygon([(0.,0.), (1.,0.), (1.,7.), (0.,7.)]), np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 1, 1],
                                                   [0, 0, 0]]),
                [(3,9),(3,4)], (3,2), 1e-3, DEFAULT_ALG),
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 0], 
                  [0, 1, 1, 1, 1], 
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1]])[3:10, 3:5])

        
class TestGridMask:
    """Test class"""
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (INPUT_CHECK_001[0], INPUT_CHECK_001[1], 6),
            (INPUT_CHECK_002[0], INPUT_CHECK_002[1], 6),
            (INPUT_CHECK_003[0], INPUT_CHECK_003[1], 6),
            (INPUT_CHECK_004[0], INPUT_CHECK_004[1], 6),
            (INPUT_CHECK_005[0], INPUT_CHECK_005[1], 6),
            (INPUT_CHECK_006[0], INPUT_CHECK_006[1], 6),
            (INPUT_CHECK_007[0], INPUT_CHECK_007[1], 6),
            (INPUT_CHECK_008[0], INPUT_CHECK_008[1], 6),
            (INPUT_CHECK_009[0], INPUT_CHECK_009[1], 6),
            (INPUT_CHECK_010[0], INPUT_CHECK_010[1], 6),
            (INPUT_CHECK_011[0], INPUT_CHECK_011[1], 6),
            (INPUT_CHECK_012[0], INPUT_CHECK_012[1], 6),
            (INPUT_CHECK_013[0], INPUT_CHECK_013[1], 6),
            (INPUT_CHECK_014[0], INPUT_CHECK_014[1], 6),
            (INPUT_CHECK_015[0], INPUT_CHECK_015[1], 6),
            (INPUT_CHECK_016[0], INPUT_CHECK_016[1], 6),
            (INPUT_CHECK_017[0], INPUT_CHECK_017[1], 6),
            (INPUT_CHECK_018[0], INPUT_CHECK_018[1], 6),
            (INPUT_CHECK_019[0], INPUT_CHECK_019[1], 6),
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
                geometry,\
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
                    geometry_origin=geometry_origin, geometry=geometry,
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