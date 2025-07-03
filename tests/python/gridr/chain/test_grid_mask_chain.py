# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.chain.grid_mask_chain

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/chain/test_grid_mask_chain.py
"""
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
import shapely
import rasterio
from scipy.interpolate import RegularGridInterpolator

from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.io.common import GridRIOMode
from gridr.chain.grid_mask_chain import (
        build_mask_chain,
        build_grid_mask_chain,
        )

MASK_IN_UNMASKED_VALUE = 0 # Non conventional for test
MASK_OUT_VALUES = (0, 1) # non conventional for test

# Define test data
DTYPE_00 = np.float32
RESOLUTION_00 = (7,4)
SHAPE_00 = (35,20)
SHAPE_OUT_00 = ((SHAPE_00[0]-1)*RESOLUTION_00[0]+1,
        (SHAPE_00[1]-1)*RESOLUTION_00[1]+1)

# Define GRID
Y_00 = np.linspace(0, (SHAPE_00[0]-1) * RESOLUTION_00[0], SHAPE_00[0], dtype=DTYPE_00)
X_00 = np.linspace(0, (SHAPE_00[1]-1) * RESOLUTION_00[1], SHAPE_00[1], dtype=DTYPE_00)
# Target grid coordinates
Y_OUT_00 = np.linspace(0, (SHAPE_OUT_00[0]-1), SHAPE_OUT_00[0], dtype=DTYPE_00)
X_OUT_00 = np.linspace(0, (SHAPE_OUT_00[1]-1), SHAPE_OUT_00[1], dtype=DTYPE_00)


GRID_IN_ARRAY_ROW_00 = np.arange(np.prod(SHAPE_00), dtype=np.float32).reshape(SHAPE_00)
GRID_IN_ARRAY_COL_00 = 10 + 10. * np.arange(np.prod(SHAPE_00), dtype=np.float32).reshape(SHAPE_00)
GRID_IN_ARRAY_00 = np.stack((GRID_IN_ARRAY_ROW_00, GRID_IN_ARRAY_COL_00))
# Compute expected output

# Create the "sparse" coordinates grid in order to preserve memory
x_new_sparse_00, y_new_sparse_00 = np.meshgrid(X_OUT_00, Y_OUT_00, indexing='xy',
            sparse=True)
#rows
GRID_OUT_ARRAY_ROW_00 = np.empty((SHAPE_OUT_00[0], SHAPE_OUT_00[1]), dtype=DTYPE_00)
GRID_OUT_ARRAY_COL_00 = np.empty((SHAPE_OUT_00[0], SHAPE_OUT_00[1]), dtype=DTYPE_00)
interpolator_row_00 = RegularGridInterpolator((Y_00, X_00), GRID_IN_ARRAY_ROW_00,
                    method='linear', bounds_error=False, fill_value=np.nan)
interpolator_col_00 = RegularGridInterpolator((Y_00, X_00), GRID_IN_ARRAY_COL_00,
                    method='linear', bounds_error=False, fill_value=np.nan)
GRID_OUT_ARRAY_ROW_00[:,:] = interpolator_row_00((y_new_sparse_00, x_new_sparse_00))
GRID_OUT_ARRAY_COL_00[:,:] = interpolator_col_00((y_new_sparse_00, x_new_sparse_00))
GRID_OUT_ARRAY_00 = np.stack((GRID_OUT_ARRAY_ROW_00, GRID_OUT_ARRAY_COL_00))
assert(np.all(GRID_OUT_ARRAY_00[:,::RESOLUTION_00[0],::RESOLUTION_00[1]].shape == GRID_IN_ARRAY_00.shape))
np.testing.assert_array_equal(GRID_OUT_ARRAY_00[:,::RESOLUTION_00[0],::RESOLUTION_00[1]], GRID_IN_ARRAY_00)


# Define mask the input mask supposed undersampled
MASK_IN_ARRAY_00_RESOLUTION = RESOLUTION_00
MASK_IN_ARRAY_00_SHAPE = SHAPE_00
MASK_IN_ARRAY_00 = np.ones(np.prod(MASK_IN_ARRAY_00_SHAPE), dtype=np.uint8).reshape(MASK_IN_ARRAY_00_SHAPE)
# Mask a window in the input array
MASK_IN_ARRAY_00_MASK_WINDOW = [(20,30), (11,15)]
MASK_IN_ARRAY_00[
        slice(*MASK_IN_ARRAY_00_MASK_WINDOW[0]),
        slice(*MASK_IN_ARRAY_00_MASK_WINDOW[1])] = 0

# Expected output given the resolution
MASK_OUT_ARRAY_00_SHAPE = (
        (MASK_IN_ARRAY_00_SHAPE[0]-1) * MASK_IN_ARRAY_00_RESOLUTION[0] + 1,
        (MASK_IN_ARRAY_00_SHAPE[1]-1) * MASK_IN_ARRAY_00_RESOLUTION[1] + 1,)

MASK_OUT_ARRAY_00 = np.ones(np.prod(MASK_OUT_ARRAY_00_SHAPE), dtype=np.uint8).reshape(MASK_OUT_ARRAY_00_SHAPE)
# Mask the window (from the input definition) - 
# The method apply a strict interpolation : interpolated unmasked value can only be achieve
# if all control points are unmasked :
# Therefore the sup limit of slice here are given by :
# (undersampled_sup_limit - 1) * resolution + 1 
MASK_OUT_ARRAY_00_MASK_WINDOW = [
         (MASK_IN_ARRAY_00_MASK_WINDOW[0][0] * MASK_IN_ARRAY_00_RESOLUTION[0],
          (MASK_IN_ARRAY_00_MASK_WINDOW[0][1]-1) * MASK_IN_ARRAY_00_RESOLUTION[0]+1),
         (MASK_IN_ARRAY_00_MASK_WINDOW[1][0] * MASK_IN_ARRAY_00_RESOLUTION[1],
          (MASK_IN_ARRAY_00_MASK_WINDOW[1][1]-1) * MASK_IN_ARRAY_00_RESOLUTION[1]+1)]
MASK_OUT_ARRAY_00[
        slice(*MASK_OUT_ARRAY_00_MASK_WINDOW[0]),
        slice(*MASK_OUT_ARRAY_00_MASK_WINDOW[1])] = 0

# Create the grid out integrating the mask
GRID_MASK_VALUE_00 = 99999
GRID_OUT_ARRAY_W_MASK_00 = np.copy(GRID_OUT_ARRAY_00)
GRID_OUT_ARRAY_W_MASK_00[:,MASK_OUT_ARRAY_00 == MASK_OUT_VALUES[1]] = GRID_MASK_VALUE_00

# Data vector :
# - shape
# - resolution
# - mask in array (will be written as tif)
BUILD_MASK_DATA_01 = (
        MASK_IN_ARRAY_00_SHAPE,
        MASK_IN_ARRAY_00_RESOLUTION,
        MASK_IN_ARRAY_00
        )
BUILD_MASK_EXPECTED_01 = MASK_OUT_ARRAY_00


# Data vector :
# - shape
# - resolution
# - mask in array (will be written as tif)
BUILD_GRID_MASK_DATA_01 = (
        SHAPE_00,
        RESOLUTION_00,
        GRID_IN_ARRAY_00,
        MASK_IN_ARRAY_00,
        )
BUILD_GRID_MASK_EXPECTED_01 = (
        GRID_OUT_ARRAY_00,
        MASK_OUT_ARRAY_00,
        GRID_OUT_ARRAY_W_MASK_00,
        )

class TestGridMaskChain:
    """Test class"""
    
    @pytest.mark.parametrize("data, expected", [
            (BUILD_MASK_DATA_01, BUILD_MASK_EXPECTED_01),
            ])
    @pytest.mark.parametrize("io_strip_size", [100, 1000, 10000])
    @pytest.mark.parametrize("io_strip_size_target", [GridRIOMode.INPUT, GridRIOMode.OUTPUT])
    @pytest.mark.parametrize("ncpu", [1, 2])
    @pytest.mark.parametrize("cpu_tile_shape", [(1000,1000),])
    @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    def test_build_mask_chain(self,
            request,
            data,
            expected,
            io_strip_size,
            io_strip_size_target,
            ncpu,
            cpu_tile_shape,
            computation_dtype,
            ):
        """Test the grid_mask_chain
        
        Args:
            data : input data as a tuple containing all the arguments
            expected: expected data
        """
        test_id = request.node.nodeid.split('::')[-1].replace('[','-').replace(']','')
        shape, resolution, mask_data_in = data
        mask_data_out_expected = expected

        mask_in_ds = None
        mask_in_tmp = None
        output_dir = None
        mask_out = None
        try:
            output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
            mask_out = Path(output_dir) / "mask_out.tif"
            
            # First write input mask on disk
            if mask_data_in is not None:
                mask_in_tmp = Path(output_dir) / "mask_in_tmp.tif"
                with rasterio.open(mask_in_tmp, "w",
                        driver="GTiff",
                        dtype=mask_data_in.dtype,
                        height=mask_data_in.shape[0],
                        width=mask_data_in.shape[1],
                        count=1,
                        #nbits=1, => not working for int8 (only for uint8)
                        ) as mask_in_ds:
                    mask_in_ds.write(mask_data_in, 1)
                mask_in_ds = None
            
            shape_out = grid_full_resolution_shape(shape=shape, resolution=resolution)
            
            with rasterio.open(mask_in_tmp, "r") as mask_in_ds:
                with rasterio.open(mask_out, "w",
                        driver="GTiff",
                        dtype=np.uint8,
                        height=shape_out[0],
                        width=shape_out[1],
                        count=1,
                        nbits=1) as mask_out_ds:
                    # Create output raster ds
                    build_mask_chain(shape=shape, resolution=resolution,
                            mask_out_ds=mask_out_ds, mask_out_dtype=np.uint8,
                            mask_in_ds=mask_in_ds, mask_in_unmasked_value=MASK_IN_UNMASKED_VALUE,
                            mask_in_band=1, geometry_origin=(0.5,0.5),
                            geometry_pair=None, rasterize_kwargs=None,
                            mask_out_values = MASK_OUT_VALUES, io_strip_size=io_strip_size,
                            io_strip_size_target=io_strip_size_target,
                            ncpu=ncpu, cpu_tile_shape = cpu_tile_shape,
                            computation_dtype=computation_dtype)
            
        except Exception as e:
            raise
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
            
            # check
            with rasterio.open(mask_out, "r") as mask_out_ds:
                mask_out_data = mask_out_ds.read(1)
                #test = mask_out_data == mask_data_out_expected
                #where_fail = np.where(test == False)
                #raise Exception(where_fail)
                np.testing.assert_array_equal(mask_out_data, mask_data_out_expected)
            
        finally:
            if mask_in_tmp:
                os.unlink(mask_in_tmp)
            if mask_out:
                os.unlink(mask_out)
            if output_dir:
                os.rmdir(output_dir)
            # Check
    
    
    
    @pytest.mark.parametrize("data, expected", [
            (BUILD_MASK_DATA_01, BUILD_MASK_EXPECTED_01),
            ])
    @pytest.mark.parametrize("io_strip_size", [100, 1000, 10000])
    @pytest.mark.parametrize("io_strip_size_target", [GridRIOMode.INPUT, GridRIOMode.OUTPUT])
    @pytest.mark.parametrize("ncpu", [1, 2])
    @pytest.mark.parametrize("cpu_tile_shape", [(1000,1000),])
    @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    def test_build_mask_chain_no_mask_in(self,
            request,
            data,
            expected,
            io_strip_size,
            io_strip_size_target,
            ncpu,
            cpu_tile_shape,
            computation_dtype,
            ):
        """Test the grid_mask_chain
        
        Args:
            data : input data as a tuple containing all the arguments
            expected: expected data
        """
        test_id = request.node.nodeid.split('::')[-1].replace('[','-').replace(']','')
        shape, resolution, _ = data
        mask_data_out_expected = expected

        mask_in_ds = None
        output_dir = None
        mask_out = None
        try:
            output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
            mask_out = Path(output_dir) / "mask_out.tif"
            shape_out = grid_full_resolution_shape(shape=shape, resolution=resolution)
            
            with rasterio.open(mask_out, "w",
                    driver="GTiff",
                    dtype=np.uint8,
                    height=shape_out[0],
                    width=shape_out[1],
                    count=1,
                    nbits=1) as mask_out_ds:
                # Create output raster ds
                build_mask_chain(shape=shape, resolution=resolution,
                            mask_out_ds=mask_out_ds, mask_out_dtype=np.uint8,
                            mask_in_ds=None, mask_in_unmasked_value=None,
                            mask_in_band=None, geometry_origin=(0.5,0.5),
                            geometry_pair=None, rasterize_kwargs=None,
                            mask_out_values = MASK_OUT_VALUES, io_strip_size=io_strip_size,
                            io_strip_size_target=io_strip_size_target,
                            ncpu=ncpu, cpu_tile_shape = cpu_tile_shape,
                            computation_dtype=computation_dtype)
            
        except Exception as e:
            raise
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
            
            # check
            with rasterio.open(mask_out, "r") as mask_out_ds:
                mask_out_data = mask_out_ds.read(1)
                # We gave no input mask : all should be unmasked
                np.testing.assert_array_equal(mask_out_data, MASK_OUT_VALUES[0])
            
        finally:
            if mask_out:
                os.unlink(mask_out)
            if output_dir:
                os.rmdir(output_dir)


    
    @pytest.mark.parametrize("data, expected", [
            (BUILD_GRID_MASK_DATA_01, BUILD_GRID_MASK_EXPECTED_01),
            ])
    @pytest.mark.parametrize("io_strip_size", [100, 1000, 10000])
    @pytest.mark.parametrize("io_strip_size_target", [GridRIOMode.INPUT, GridRIOMode.OUTPUT])
    @pytest.mark.parametrize("ncpu", [1, 2])
    @pytest.mark.parametrize("cpu_tile_shape", [(10,10), (1000,1000),])
    @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    @pytest.mark.parametrize("merge_mask_grid", [None, GRID_MASK_VALUE_00])
    def test_build_grid_mask_chain(self,
            request,
            data,
            expected,
            io_strip_size,
            io_strip_size_target,
            ncpu,
            cpu_tile_shape,
            computation_dtype,
            merge_mask_grid,
            ):
        """Test the grid_mask_chain
        
        Args:
            data : input data as a tuple containing all the arguments
            expected: expected data
        """
        test_id = request.node.nodeid.split('::')[-1].replace('[','-').replace(']','')
        shape, resolution, grid_data_in, mask_data_in = data
        grid_data_out_expected, mask_data_out_expected, grid_data_out_expected_w_mask = expected
        
        if merge_mask_grid is not None:
            grid_data_out_expected = grid_data_out_expected_w_mask
        #mask_data_out_expected = expected
        #assert(mask_data_out_expected.ndim == 2)
        grid_in_ds = None
        mask_in_ds = None
        mask_in_tmp = None
        output_dir = None
        mask_out = None
        grid_out= None
        try:
            output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
            grid_out = Path(output_dir) / "grid_out.tif"
            mask_out = Path(output_dir) / "mask_out.tif"
            
            # First write inputs on disk
            #-------------------------------------------------------------------
            grid_in_tmp = Path(output_dir) / "grid_in_tmp.tif"
            with rasterio.open(grid_in_tmp, "w",
                    driver="GTiff", dtype=grid_data_in.dtype,
                    height=grid_data_in.shape[1], width=grid_data_in.shape[2],
                    count=2) as grid_in_ds:
                grid_in_ds.write(grid_data_in[0], 1)
                grid_in_ds.write(grid_data_in[1], 2)
            grid_in_ds = None
            
            # Write input mask
            if mask_data_in is not None:
                mask_in_tmp = Path(output_dir) / "mask_in_tmp.tif"
                with rasterio.open(mask_in_tmp, "w",
                        driver="GTiff",
                        dtype=mask_data_in.dtype,
                        height=mask_data_in.shape[0],
                        width=mask_data_in.shape[1],
                        count=1,
                        #nbits=1, => not working for int8 (only for uint8)
                        ) as mask_in_ds:
                    mask_in_ds.write(mask_data_in, 1)
                mask_in_ds = None
            
            shape_out = grid_full_resolution_shape(shape=shape, resolution=resolution)
            
            grid_in_ds = rasterio.open(grid_in_tmp, "r")
            mask_in_ds = rasterio.open(mask_in_tmp, "r")
            
            open_kwargs = {"driver": "GTiff", "height":shape_out[0], "width":shape_out[1]}
            with rasterio.open(grid_in_tmp, "r") as grid_in_ds, \
                    rasterio.open(mask_in_tmp, "r") as mask_in_ds, \
                    rasterio.open(grid_out, "w", dtype=computation_dtype, count=2, **open_kwargs) as grid_out_ds, \
                    rasterio.open(mask_out, "w", dtype=np.uint8, count=1, nbits=1, **open_kwargs) as mask_out_ds:
                
                # Create output raster ds
                build_grid_mask_chain(
                    resolution=resolution, 
                    grid_in_ds=grid_in_ds,
                    grid_in_col_ds=None,
                    grid_in_row_coords_band=1,
                    grid_in_col_coords_band=2,
                    grid_out_ds=grid_out_ds,
                    grid_out_col_ds=None,
                    grid_out_row_coords_band=1,
                    grid_out_col_coords_band=2,
                    mask_out_ds=mask_out_ds,
                    mask_out_dtype=np.uint8,
                    mask_in_ds=mask_in_ds, mask_in_unmasked_value=MASK_IN_UNMASKED_VALUE,
                    mask_in_band=1, geometry_origin=(0.5,0.5),
                    geometry_pair=None, rasterize_kwargs=None,
                    mask_out_values = MASK_OUT_VALUES,
                    merge_mask_grid = merge_mask_grid,
                    io_strip_size=io_strip_size,
                    io_strip_size_target=io_strip_size_target,
                    ncpu=ncpu, cpu_tile_shape = cpu_tile_shape,
                    computation_dtype=computation_dtype)
                    
        except Exception as e:
            raise
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
            
            with rasterio.open(grid_out, "r") as grid_out_ds:
                grid_out_data_row = grid_out_ds.read(1)
                grid_out_data_col = grid_out_ds.read(2)
                np.testing.assert_array_equal(grid_out_data_col, grid_data_out_expected[1])
                np.testing.assert_array_equal(grid_out_data_row, grid_data_out_expected[0])
            
            # check
            with rasterio.open(mask_out, "r") as mask_out_ds:
                mask_out_data = mask_out_ds.read(1)
                #test = mask_out_data == mask_data_out_expected
                #where_fail = np.where(test == False)
                #raise Exception(where_fail)
                np.testing.assert_array_equal(mask_out_data, mask_data_out_expected)
            
        finally:
            if grid_in_tmp:
                os.unlink(grid_in_tmp)
            if mask_in_tmp:
                os.unlink(mask_in_tmp)
            if grid_out:
                os.unlink(grid_out)
            if mask_out:
                os.unlink(mask_out)
            if output_dir:
                os.rmdir(output_dir)
            # Check

