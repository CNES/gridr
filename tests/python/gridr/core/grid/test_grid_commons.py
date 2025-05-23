# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.grid.grid_utils module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_commons.py
"""
import os
import numpy as np
import pytest

from gridr.core.grid.grid_commons import (
        grid_regular_coords_1d,
        grid_regular_coords_2d,
        regular_grid_shape_origin_resolution,
        window_apply_grid_coords,
        window_apply_shape_origin_resolution,
        check_grid_coords_definition,
        grid_resolution_window,
        grid_resolution_window_safe,
        )

# Define test data
MAKEGRID1Dx00 = (((4,3), (0,0.5), (3,1)),
    (np.array([0.5, 1.5, 2.5]),
     np.array([0., 3., 6., 9.])))
        
MAKEGRID2Dx00 = (((4,3), (0,0.5), (3,1)),
    (np.array([[0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5]]),
     np.array([[0., 0., 0.],
        [3., 3., 3.],
        [6., 6., 6.],
        [9., 9., 9.]])))

MAKEGRID2Dx00_WINDOW = (((4,3), (0,0.5), (3,1), np.array([(2,3),(1,2)])),
    ((2,2), (6,1.5), (3,1)))

MAKEGRID1Dx00_COORDS_WINDOW = (
    ((np.array([0.5, 1.5, 2.5]), np.array([0., 3., 6., 9.])), np.array([(2,3),(1,2)])),
    (np.array([1.5, 2.5]), np.array([6., 9.])))

MAKEGRID2Dx00_COORDS_WINDOW = (
    ((np.array([[0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5]]),
     np.array([[0., 0., 0.],
        [3., 3., 3.],
        [6., 6., 6.],
        [9., 9., 9.]])), np.array([(2,3),(1,2)])),
    (np.array([
        [1.5, 2.5],
        [1.5, 2.5]]),
     np.array([
        [6., 6.],
        [9., 9.]])))

MAKEGRID3Dx00_COORDS_WINDOW = (
    (np.array([[[0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5],
        [0.5, 1.5, 2.5]],
       [[0., 0., 0.],
        [3., 3., 3.],
        [6., 6., 6.],
        [9., 9., 9.]]]), np.array([(2,3),(1,2)])),
    (np.array([
        [1.5, 2.5],
        [1.5, 2.5]]),
     np.array([
        [6., 6.],
        [9., 9.]])))

class TestGridCommons:
    """Test class"""    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (MAKEGRID1Dx00[0], MAKEGRID1Dx00[1], 6),
            ])
    def test_grid_regular_coords_1d(self, data, expected, testing_decimal):
        """Test a regular grid interpolation with mask
        
        Args:
            data : input data as a tuple containing the shape, origin and resolution tuples
            expected: expected data 
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        shape, origin, resolution = data
        expected_grid_col, expected_grid_row = expected
        
        grid_col, grid_row = grid_regular_coords_1d(shape, origin, resolution)
        
        # Check
        np.testing.assert_array_almost_equal(grid_col, expected_grid_col, decimal=testing_decimal)
        np.testing.assert_array_almost_equal(grid_row, expected_grid_row, decimal=testing_decimal)
    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (MAKEGRID2Dx00[0], MAKEGRID2Dx00[1], 6),
            ])
    def test_grid_regular_coords_2d(self, data, expected, testing_decimal):
        """Test a regular grid interpolation with mask
        
        Args:
            data : input data as a tuple containing the shape, origin and resolution tuples
            expected: expected data 
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        shape, origin, resolution = data
        expected_grid_col, expected_grid_row = expected
        
        grid_col, grid_row = grid_regular_coords_2d(shape, origin, resolution, sparse=False)
        
        # Check
        np.testing.assert_array_almost_equal(grid_col, expected_grid_col, decimal=testing_decimal)
        np.testing.assert_array_almost_equal(grid_row, expected_grid_row, decimal=testing_decimal)
    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (MAKEGRID2Dx00[1], MAKEGRID2Dx00[0], 6),
            (MAKEGRID2Dx00[1], np.asarray(MAKEGRID2Dx00[0]), 6),
            (MAKEGRID1Dx00[1], MAKEGRID1Dx00[0], 6),
            ])
    def test_regular_grid_shape_origin_resolution(self, data, expected, testing_decimal):
        """Test a regular grid interpolation with mask
        
        Args:
            data : input data as a tuple containing the grid coordinates
            expected: expected data as a tuple containing the shape, origin and resolution tuples
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_coords = data
        expected_shape, expected_origin, expected_resolution = expected
        
        shape, origin, resolution = regular_grid_shape_origin_resolution(grid_coords)
        
        # Check
        np.testing.assert_array_equal(shape, expected_shape)
        np.testing.assert_array_almost_equal(origin, expected_origin, decimal=testing_decimal)
        np.testing.assert_array_almost_equal(resolution, expected_resolution, decimal=testing_decimal)
    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (MAKEGRID1Dx00_COORDS_WINDOW[0], MAKEGRID1Dx00_COORDS_WINDOW[1], 6),
            (MAKEGRID2Dx00_COORDS_WINDOW[0], MAKEGRID2Dx00_COORDS_WINDOW[1], 6),
            (MAKEGRID3Dx00_COORDS_WINDOW[0], MAKEGRID3Dx00_COORDS_WINDOW[1], 6),
            ])
    @pytest.mark.parametrize("check", [True, False])
    def test_window_apply_grid_coords(self, data, expected, check, testing_decimal):
        """Test a window application to a grid_coords
        
        Args:
            data : input data as a tuple containing the shape, origin, resolution and window array
            expected: expected data as a tuple containing the shape, origin and resolution tuples
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_coords, win = data
        expected_grid_coords = expected
        
        grid_coords = window_apply_grid_coords(grid_coords, win, check)
        
        # Check
        np.testing.assert_array_equal(grid_coords, expected_grid_coords)
    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (MAKEGRID2Dx00_WINDOW[0], MAKEGRID2Dx00_WINDOW[1], 6),
            ])
    def test_window_apply_shape_origin_resolution(self, data, expected, testing_decimal):
        """Test a window application to a shape, origin, resolution grid definition
        
        Args:
            data : input data as a tuple containing the shape, origin, resolution and window array
            expected: expected data as a tuple containing the shape, origin and resolution tuples
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        shape, origin, resolution, win = data
        expected_shape, expected_origin, expected_resolution = expected
        
        shape, origin, resolution = window_apply_shape_origin_resolution(shape, origin, resolution, win)
        
        # Check
        np.testing.assert_array_equal(shape, expected_shape)
        np.testing.assert_array_almost_equal(origin, expected_origin, decimal=testing_decimal)
        np.testing.assert_array_almost_equal(resolution, expected_resolution, decimal=testing_decimal)
        
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            ((None, None, None, None), ValueError, 6),
            ((None, (2,2), None, None), ValueError, 6),
            ((None, (2,2), (0,0), None), ValueError, 6),
            ((None, (2,2), (0,0), (2,3)), None, 6),
            ((MAKEGRID1Dx00[1], (2,2), (0,0), (2,3)), ValueError, 6),
            ((MAKEGRID1Dx00[1], None, None, None), MAKEGRID1Dx00[1], 6),
            ((MAKEGRID3Dx00_COORDS_WINDOW[0][0], None, None, None), (MAKEGRID3Dx00_COORDS_WINDOW[0][0][0], MAKEGRID3Dx00_COORDS_WINDOW[0][0][1]), 6),
            ((MAKEGRID2Dx00_COORDS_WINDOW[0][0], None, None, None), MAKEGRID2Dx00_COORDS_WINDOW[0][0], 6),
            ((MAKEGRID1Dx00[1][0], None, None, None), TypeError, 6),
            ])
    def test_check_grid_coords_definition(self, data, expected, testing_decimal):
        """Test a regular grid interpolation with mask
        
        Args:
            data : input data as a tuple containing the shape, origin and resolution tuples
            expected: expected data 
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_coords, shape, origin, resolution = data
        expected_out = expected

        try:
            grid_coords_out = check_grid_coords_definition(grid_coords, shape, origin, resolution)
        except Exception as e:
            if isinstance(e, expected_out):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected_out, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected_out}")
            except TypeError:
                pass
            # Check
            if expected_out is not None:
                assert(isinstance(grid_coords_out, tuple))
                np.testing.assert_array_almost_equal(grid_coords_out[0], expected_out[0], decimal=testing_decimal)
                np.testing.assert_array_almost_equal(grid_coords_out[1], expected_out[1], decimal=testing_decimal)
    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (((1,), np.array([[2,10],])), (np.array([[2,10],]), np.array([[0,8],])), 6),
            (((1,1), np.array([[2,10], [3, 5]])), (np.array([[2,10], [3, 5]]), np.array([[0,8], [0, 2]])), 6),
            (((3,1), np.array([[2,10], [3, 5]])), (np.array([[0,4], [3, 5]]), np.array([[2,10], [0, 2]])), 6),
            (((3,1), np.array([[5,6], [3, 5]])), (np.array([[1,2], [3, 5]]), np.array([[2,3], [0, 2]])), 6),
            ])
    def test_grid_resolution_window(self, data, expected, testing_decimal):
        """Test the grid_resolution_window method
        
        Args:
            data : input data as a tuple containing the resolution and the target win at full resolution
            expected: expected data as a tuple containing the window in the grid and the relative window
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_resolution, win, = data
        expected_grid_win, expected_rel_win = expected

        try:
            grid_win, rel_win = grid_resolution_window(grid_resolution, win)
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
            np.testing.assert_array_almost_equal(expected_grid_win, grid_win, decimal=testing_decimal)
            np.testing.assert_array_almost_equal(expected_rel_win, rel_win, decimal=testing_decimal)
    
    
    @pytest.mark.parametrize("data, expected, testing_decimal", [
            (((1,), np.array([[2,10],]), (20,)), (np.array([[2,10],]), np.array([[0,8],])), 6),
            (((1,), np.array([[2,10],]), (5,)), (np.array([[2,4],]), np.array([[0,2],])), 6),
            #(((1,), np.array([[12,14],]), (5,)), (np.array([[12,4],]), np.array([[0,0],])), 6), => shoud raise an exception but not yet managed
            
            (((1,1), np.array([[2,10], [3, 5]]), (20, 20)), (np.array([[2,10], [3, 5]]), np.array([[0,8], [0, 2]])), 6),
            (((1,1), np.array([[2,10], [3, 5]]), (5, 20)), (np.array([[2,4], [3, 5]]), np.array([[0,2], [0, 2]])), 6),
            (((1,1), np.array([[2,10], [3, 5]]), (20, 4)), (np.array([[2,10], [3, 3]]), np.array([[0,8], [0, 0]])), 6),
            
            (((3,1), np.array([[2,10], [3, 5]]), (20, 20)), (np.array([[0,4], [3, 5]]), np.array([[2,10], [0, 2]])), 6),
            (((3,1), np.array([[2,10], [3, 5]]), (2, 20)), (np.array([[0,1], [3, 5]]), np.array([[2,3], [0, 2]])), 6),
            (((3,1), np.array([[3,3], [3, 5]]), (20, 20)), (np.array([[1,2], [3, 5]]), np.array([[0,0], [0, 2]])), 6), # test the grid_win > 1
            (((3,1), np.array([[3,3], [3, 5]]), (2, 20)), (np.array([[0,1], [3, 5]]), np.array([[3,3], [0, 2]])), 6), # test backward shift
            
            (((3,1), np.array([[5,6], [3, 5]]), (20, 20)), (np.array([[1,2], [3, 5]]), np.array([[2,3], [0, 2]])), 6),
            ])
    def test_grid_resolution_window_safe(self, data, expected, testing_decimal):
        """Test the grid_resolution_window_safe method
        
        Args:
            data : input data as a tuple containing the resolution and the target win at full resolution
            expected: expected data as a tuple containing the window in the grid and the relative window
            testing_decimal: decimal precision used for np.testing.assert_array_almost_equal
        """
        # Create intial coordinates so that new coordinates match an integer sequence
        grid_resolution, win, grid_shape = data
        expected_grid_win, expected_rel_win = expected

        try:
            grid_win, rel_win = grid_resolution_window_safe(grid_resolution, win, grid_shape)
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
            np.testing.assert_array_almost_equal(expected_grid_win, grid_win, decimal=testing_decimal)
            np.testing.assert_array_almost_equal(expected_rel_win, rel_win, decimal=testing_decimal)
           