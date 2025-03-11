# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.utils.array_window module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/utils/test_array_window.py
"""
import os
import numpy as np
import pytest
import rasterio

from gridr.core.utils.array_utils import (
        array_replace)

ARRAY_I8_00 = np.zeros(4*7, dtype=np.int8).reshape(4,7)
ARRAY_I8_00[1,:] = 1

ARRAY_U8_00 = np.zeros(4*7, dtype=np.uint8).reshape(4,7)
ARRAY_U8_00[1,:] = 1

ARRAY_I32_00 = np.zeros(4*7, dtype=np.int32).reshape(4,7)
ARRAY_I32_00[1,:] = 1



class TestArrayWindow:
    """Class for test
    """
    @pytest.mark.parametrize("data, expected", [
            ((ARRAY_I8_00, 0, 0, 1, None), ARRAY_I8_00),
            ((ARRAY_I8_00, 0, 1, 0, None), np.where(ARRAY_I8_00 == 0, 1, 0)),
            # Testing slices : here take all array through slice => it should be OK
            ((ARRAY_I8_00, 0, 1, 0, (slice(0,4), slice(0,7))), np.where(ARRAY_I8_00 == 0, 1, 0)),
            # Testing slices : here take all lines but select only some columns => the view is not
            # a contiguous view => an AssertionError must be raised
            ((ARRAY_I8_00, 0, 1, 0, (slice(0,4), slice(2,4))), AssertionError),
            # Testing slices : here take somes lines but select all columns => the view is still
            # a contiguous view => it should be OK
            ((ARRAY_I8_00, 0, 1, 0, (slice(1,3), slice(0,7))), np.where(ARRAY_I8_00 == 0, 1, 0)),
            
            ((ARRAY_U8_00, 0, 0, 1, None), ARRAY_U8_00),
            ((ARRAY_I32_00, 0, 0, 1, None), AssertionError),
            ((ARRAY_I8_00, 1, 3, 2, None), np.where(ARRAY_I8_00 == 1, 3, 2)),
            ])
    def test_array_replace(self, data, expected):
        """Test array_replace method
        """
        array, val_cond, val_true, val_false, slices = data
        expected_array = expected
        # Copy input array in order to not affect the global variable
        array_copy = np.copy(array)
        
        try:
            if slices is None:
                array_replace(array_copy, val_cond, val_true, val_false)
            else:
                array_replace(array_copy[slices], val_cond, val_true, val_false)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            if slices is None:
                assert(np.all(array_copy == expected_array))
            else:
                assert(np.all(array_copy[slices] == expected_array[slices]))
                
