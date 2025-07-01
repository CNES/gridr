# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.chain.grid_resampling_chain

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/chain/test_grid_resampling_chain.py
"""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import pytest
import shapely
import rasterio

from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.io.common import GridRIOMode
from gridr.core.grid.grid_resampling import array_grid_resampling
from gridr.chain.grid_resampling_chain import basic_grid_resampling_chain

from gridr.misc.mandrill import mandrill

UNMASKED_VALUE=1
MASKED_VALUE=0


def create_grid(nrow, ncol, origin_pos, origin_node, v_row_y, v_row_x, v_col_y, v_col_x, grid_dtype):
    """
    """
    x = np.arange(0, ncol, dtype=grid_dtype)
    y = np.arange(0, nrow, dtype=grid_dtype)
    xx, yy = np.meshgrid(x, y)

    xx -= origin_pos[0]
    yy -= origin_pos[1]

    yyy = origin_node[0] + yy * v_row_y + xx * v_col_y
    xxx = origin_node[1] + yy * v_row_x + xx * v_col_x

    return yyy, xxx


def write_array(array, dtype, fileout):
    """
    """
    if array.ndim == 3:
        with rasterio.open(fileout, "w", driver="GTiff", dtype=dtype,
                height=array.shape[1], width=array.shape[2], count=array.shape[0],
                ) as array_in_ds:
            for band in range(array.shape[0]):
                array_in_ds.write(array[band].astype(dtype), band+1)
            array_in_ds = None
    elif array.ndim == 2:
        with rasterio.open(fileout, "w", driver="GTiff", dtype=dtype,
                height=array.shape[0], width=array.shape[1], count=1,
                ) as array_in_ds:
            array_in_ds.write(array.astype(dtype), 1)
            array_in_ds = None

def shape2(array):
    if array.ndim == 3:
        return (array.shape[1], array.shape[2])
    else:
        return array.shape
        

class TestGridResamplingChain:
    """Test class"""
    
    @pytest.mark.parametrize("array_in, array_in_mask_positions, array_in_bands", [
            (mandrill[0], None, [1,]),
            (mandrill[0], [(60, 160),], [1,])])
    @pytest.mark.parametrize("array_in_dtype", [(np.float64)])
    @pytest.mark.parametrize("grid_shape, grid_vec_row, grid_vec_col, grid_origin_pos, grid_origin_node, grid_dtype, grid_mask", [
            ((50, 40), (5.2, 1.2), (-2.7, 7.1), (0.3, 0.2), (0., 0.), np.float64, True),
            ((50, 40), (5.2, 1.2), (-2.7, 7.1), (0.3, 0.2), (0., 0.), np.float64, False)])
    @pytest.mark.parametrize("grid_resolution", [(10, 10),])
    #@pytest.mark.parametrize("grid_resolution", [(3, 4),])
    #@pytest.mark.parametrize("io_strip_size", [100, 1000, 10000])
    #@pytest.mark.parametrize("io_strip_size_target", [GridRIOMode.INPUT, GridRIOMode.OUTPUT])
    #@pytest.mark.parametrize("ncpu", [1, 2])
    #@pytest.mark.parametrize("cpu_tile_shape", [(1000,1000),])
    #@pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    def test_build_mask_chain(self,
            request,
            array_in, array_in_mask_positions, array_in_bands, array_in_dtype,
            grid_shape, grid_vec_row, grid_vec_col, grid_origin_pos, grid_origin_node, grid_dtype, grid_mask,
            grid_resolution,
            ):
        """Test the grid_resampling_chain - compare results with results in the core method
        """
        test_id = request.node.nodeid.split('::')[-1].replace('[','-').replace(']','')
        test_id = hashlib.md5(test_id.encode('utf-8')).hexdigest()[:8]
        
        output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
        array_in_path = Path(output_dir) / "array_in.tif"
        grid_in_path = Path(output_dir) / "grid_in.tif"
        grid_mask_in_path = None
        array_mask_in_path = None
        
        
        try:
            # Write input array
            write_array(array_in, array_in_dtype, array_in_path)
            
            # Write input grid
            grid_row, grid_col = create_grid(
                    nrow=grid_shape[0], ncol=grid_shape[1],
                    origin_pos=np.asarray(grid_origin_pos),
                    origin_node=np.asarray(grid_origin_node),
                    v_row_y=grid_vec_row[0], v_row_x=grid_vec_row[1],
                    v_col_y=grid_vec_col[0], v_col_x=grid_vec_col[1],
                    grid_dtype=grid_dtype)
            write_array(np.array([grid_row, grid_col]), grid_dtype, grid_in_path)
            
            # Create a grid mask if True
            if grid_mask:
                grid_mask_in_path = Path(output_dir) / "grid_mask_in.tif"
                grid_mask_array = np.ones(grid_row.shape, dtype=np.uint8) * UNMASKED_VALUE
                grid_mask_array[grid_row.shape[0]//8: grid_row.shape[0]//8+3,
                                grid_row.shape[1]//10: grid_row.shape[1]//10+2] = MASKED_VALUE
                write_array(grid_mask_array, np.uint8, grid_mask_in_path)
            
            # Create array mask
            if array_in_mask_positions is not None:
                array_mask_in_path = Path(output_dir) / "array_mask_in.tif"
                array_mask = np.ones(shape2(array_in), dtype=np.uint8) * UNMASKED_VALUE
                for pos in array_in_mask_positions:
                    array_mask[*pos] = MASKED_VALUE
                write_array(array_mask, np.uint8, array_mask_in_path)
            
            full_output_shape = grid_full_resolution_shape(shape=grid_row.shape, resolution=grid_resolution)
            array_out_path = Path(output_dir) / "array_out.tif"
            computation_dtype = np.float64
            output_dtype = np.float64
            
            output_shape = full_output_shape
            window = None
            if window is None:
                window = np.array(((0, full_output_shape[0]-1), (0, full_output_shape[1]-1)))
            output_shape = window[:,1] - window[:,0] + 1
            
            F=1
            with rasterio.open(grid_in_path, 'r') as grid_in_ds, \
                    rasterio.open(array_in_path, 'r') as array_in_ds, \
                    rasterio.open(array_out_path, "w",  driver="GTiff", dtype=output_dtype,
                            height=output_shape[0], width=output_shape[1],
                            count=len(array_in_bands)) as array_out_ds:
                basic_grid_resampling_chain(
                    grid_ds = grid_in_ds,
                    grid_col_ds = grid_in_ds,
                    grid_row_coords_band = 1,
                    grid_col_coords_band = 2,
                    grid_resolution = grid_resolution,
                    array_src_ds = array_in_ds,
                    array_src_bands = array_in_bands,
                    
                    array_src_mask_ds = None,
                    array_src_mask_band = 1,
                    array_out_ds = array_out_ds,
                    interp = "bicubic",
                    nodata_out = 0,
                    window = window,
                    mask_out_ds = None,
                    grid_mask_in_ds = None,
                    grid_mask_in_unmasked_value = UNMASKED_VALUE,
                    grid_mask_in_band = 1,
                    computation_dtype = computation_dtype,
                    geometry_origin = (0, 0),
                    #geometry = None,
                    io_strip_size = 75*F, #10000,
                    io_strip_size_target = GridRIOMode.OUTPUT,
                    #tile_shape = (50, 180),
                    tile_shape = None,
                )

        finally:
            # Clear
            os.unlink(array_in_path)
            os.unlink(grid_in_path)
            os.unlink(array_out_path)
            if grid_mask_in_path:
                os.unlink(grid_mask_in_path)
            if array_mask_in_path:
                os.unlink(array_mask_in_path)
            os.rmdir(output_dir)
        
        # mask_in_ds = None
        # mask_in_tmp = None
        # output_dir = None
        # mask_out = None
        # try:
            # output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
            # mask_out = Path(output_dir) / "mask_out.tif"
            
            # # First write input mask on disk
            # if mask_data_in is not None:
                # mask_in_tmp = Path(output_dir) / "mask_in_tmp.tif"
                # with rasterio.open(mask_in_tmp, "w",
                        # driver="GTiff",
                        # dtype=mask_data_in.dtype,
                        # height=mask_data_in.shape[0],
                        # width=mask_data_in.shape[1],
                        # count=1,
                        # #nbits=1, => not working for int8 (only for uint8)
                        # ) as mask_in_ds:
                    # mask_in_ds.write(mask_data_in, 1)
                # mask_in_ds = None
            
            # shape_out = grid_full_resolution_shape(shape=shape, resolution=resolution)
            
            # with rasterio.open(mask_in_tmp, "r") as mask_in_ds:
                # with rasterio.open(mask_out, "w",
                        # driver="GTiff",
                        # dtype=np.uint8,
                        # height=shape_out[0],
                        # width=shape_out[1],
                        # count=1,
                        # nbits=1) as mask_out_ds:
                    # # Create output raster ds
                    # build_mask_chain(shape=shape, resolution=resolution,
                            # mask_out_ds=mask_out_ds, mask_out_dtype=np.uint8,
                            # mask_in_ds=mask_in_ds, mask_in_unmasked_value=MASK_IN_UNMASKED_VALUE,
                            # mask_in_band=1, geometry_origin=(0.5,0.5),
                            # geometry=None, rasterize_kwargs=None,
                            # mask_out_values = MASK_OUT_VALUES, io_strip_size=io_strip_size,
                            # io_strip_size_target=io_strip_size_target,
                            # ncpu=ncpu, cpu_tile_shape = cpu_tile_shape,
                            # computation_dtype=computation_dtype)
            
        # except Exception as e:
            # raise
            # if isinstance(e, expected):
                # pass
            # else:
                # raise
        # else:
            # try:
                # if issubclass(expected, BaseException):
                    # raise Exception(f"The test should have raised an exceptionof type {expected}")
            # except TypeError:
                # pass
            
            # # check
            # with rasterio.open(mask_out, "r") as mask_out_ds:
                # mask_out_data = mask_out_ds.read(1)
                # #test = mask_out_data == mask_data_out_expected
                # #where_fail = np.where(test == False)
                # #raise Exception(where_fail)
                # np.testing.assert_array_equal(mask_out_data, mask_data_out_expected)
            
        # finally:
            # if mask_in_tmp:
                # os.unlink(mask_in_tmp)
            # if mask_out:
                # os.unlink(mask_out)
            # if output_dir:
                # os.rmdir(output_dir)
            # # Check
    
    
    
    # @pytest.mark.parametrize("data, expected", [
            # (BUILD_MASK_DATA_01, BUILD_MASK_EXPECTED_01),
            # ])
    # @pytest.mark.parametrize("io_strip_size", [100, 1000, 10000])
    # @pytest.mark.parametrize("io_strip_size_target", [GridRIOMode.INPUT, GridRIOMode.OUTPUT])
    # @pytest.mark.parametrize("ncpu", [1, 2])
    # @pytest.mark.parametrize("cpu_tile_shape", [(1000,1000),])
    # @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    # def test_build_mask_chain_no_mask_in(self,
            # request,
            # data,
            # expected,
            # io_strip_size,
            # io_strip_size_target,
            # ncpu,
            # cpu_tile_shape,
            # computation_dtype,
            # ):
        # """Test the grid_mask_chain
        
        # Args:
            # data : input data as a tuple containing all the arguments
            # expected: expected data
        # """
        # test_id = request.node.nodeid.split('::')[-1].replace('[','-').replace(']','')
        # shape, resolution, _ = data
        # mask_data_out_expected = expected

        # mask_in_ds = None
        # output_dir = None
        # mask_out = None
        # try:
            # output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
            # mask_out = Path(output_dir) / "mask_out.tif"
            # shape_out = grid_full_resolution_shape(shape=shape, resolution=resolution)
            
            # with rasterio.open(mask_out, "w",
                    # driver="GTiff",
                    # dtype=np.uint8,
                    # height=shape_out[0],
                    # width=shape_out[1],
                    # count=1,
                    # nbits=1) as mask_out_ds:
                # # Create output raster ds
                # build_mask_chain(shape=shape, resolution=resolution,
                            # mask_out_ds=mask_out_ds, mask_out_dtype=np.uint8,
                            # mask_in_ds=None, mask_in_unmasked_value=None,
                            # mask_in_band=None, geometry_origin=(0.5,0.5),
                            # geometry=None, rasterize_kwargs=None,
                            # mask_out_values = MASK_OUT_VALUES, io_strip_size=io_strip_size,
                            # io_strip_size_target=io_strip_size_target,
                            # ncpu=ncpu, cpu_tile_shape = cpu_tile_shape,
                            # computation_dtype=computation_dtype)
            
        # except Exception as e:
            # raise
            # if isinstance(e, expected):
                # pass
            # else:
                # raise
        # else:
            # try:
                # if issubclass(expected, BaseException):
                    # raise Exception(f"The test should have raised an exceptionof type {expected}")
            # except TypeError:
                # pass
            
            # # check
            # with rasterio.open(mask_out, "r") as mask_out_ds:
                # mask_out_data = mask_out_ds.read(1)
                # # We gave no input mask : all should be unmasked
                # np.testing.assert_array_equal(mask_out_data, MASK_OUT_VALUES[0])
            
        # finally:
            # if mask_out:
                # os.unlink(mask_out)
            # if output_dir:
                # os.rmdir(output_dir)


    
    # @pytest.mark.parametrize("data, expected", [
            # (BUILD_GRID_MASK_DATA_01, BUILD_GRID_MASK_EXPECTED_01),
            # ])
    # @pytest.mark.parametrize("io_strip_size", [100, 1000, 10000])
    # @pytest.mark.parametrize("io_strip_size_target", [GridRIOMode.INPUT, GridRIOMode.OUTPUT])
    # @pytest.mark.parametrize("ncpu", [1, 2])
    # @pytest.mark.parametrize("cpu_tile_shape", [(10,10), (1000,1000),])
    # @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    # @pytest.mark.parametrize("merge_mask_grid", [None, GRID_MASK_VALUE_00])
    # def test_build_grid_mask_chain(self,
            # request,
            # data,
            # expected,
            # io_strip_size,
            # io_strip_size_target,
            # ncpu,
            # cpu_tile_shape,
            # computation_dtype,
            # merge_mask_grid,
            # ):
        # """Test the grid_mask_chain
        
        # Args:
            # data : input data as a tuple containing all the arguments
            # expected: expected data
        # """
        # test_id = request.node.nodeid.split('::')[-1].replace('[','-').replace(']','')
        # shape, resolution, grid_data_in, mask_data_in = data
        # grid_data_out_expected, mask_data_out_expected, grid_data_out_expected_w_mask = expected
        
        # if merge_mask_grid is not None:
            # grid_data_out_expected = grid_data_out_expected_w_mask
        # #mask_data_out_expected = expected
        # #assert(mask_data_out_expected.ndim == 2)
        # grid_in_ds = None
        # mask_in_ds = None
        # mask_in_tmp = None
        # output_dir = None
        # mask_out = None
        # grid_out= None
        # try:
            # output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
            # grid_out = Path(output_dir) / "grid_out.tif"
            # mask_out = Path(output_dir) / "mask_out.tif"
            
            # # First write inputs on disk
            # #-------------------------------------------------------------------
            # grid_in_tmp = Path(output_dir) / "grid_in_tmp.tif"
            # with rasterio.open(grid_in_tmp, "w",
                    # driver="GTiff", dtype=grid_data_in.dtype,
                    # height=grid_data_in.shape[1], width=grid_data_in.shape[2],
                    # count=2) as grid_in_ds:
                # grid_in_ds.write(grid_data_in[0], 1)
                # grid_in_ds.write(grid_data_in[1], 2)
            # grid_in_ds = None
            
            # # Write input mask
            # if mask_data_in is not None:
                # mask_in_tmp = Path(output_dir) / "mask_in_tmp.tif"
                # with rasterio.open(mask_in_tmp, "w",
                        # driver="GTiff",
                        # dtype=mask_data_in.dtype,
                        # height=mask_data_in.shape[0],
                        # width=mask_data_in.shape[1],
                        # count=1,
                        # #nbits=1, => not working for int8 (only for uint8)
                        # ) as mask_in_ds:
                    # mask_in_ds.write(mask_data_in, 1)
                # mask_in_ds = None
            
            # shape_out = grid_full_resolution_shape(shape=shape, resolution=resolution)
            
            # grid_in_ds = rasterio.open(grid_in_tmp, "r")
            # mask_in_ds = rasterio.open(mask_in_tmp, "r")
            
            # open_kwargs = {"driver": "GTiff", "height":shape_out[0], "width":shape_out[1]}
            # with rasterio.open(grid_in_tmp, "r") as grid_in_ds, \
                    # rasterio.open(mask_in_tmp, "r") as mask_in_ds, \
                    # rasterio.open(grid_out, "w", dtype=computation_dtype, count=2, **open_kwargs) as grid_out_ds, \
                    # rasterio.open(mask_out, "w", dtype=np.uint8, count=1, nbits=1, **open_kwargs) as mask_out_ds:
                
                # # Create output raster ds
                # build_grid_mask_chain(
                    # resolution=resolution, 
                    # grid_in_ds=grid_in_ds,
                    # grid_in_col_ds=None,
                    # grid_in_row_coords_band=1,
                    # grid_in_col_coords_band=2,
                    # grid_out_ds=grid_out_ds,
                    # grid_out_col_ds=None,
                    # grid_out_row_coords_band=1,
                    # grid_out_col_coords_band=2,
                    # mask_out_ds=mask_out_ds,
                    # mask_out_dtype=np.uint8,
                    # mask_in_ds=mask_in_ds, mask_in_unmasked_value=MASK_IN_UNMASKED_VALUE,
                    # mask_in_band=1, geometry_origin=(0.5,0.5),
                    # geometry=None, rasterize_kwargs=None,
                    # mask_out_values = MASK_OUT_VALUES,
                    # merge_mask_grid = merge_mask_grid,
                    # io_strip_size=io_strip_size,
                    # io_strip_size_target=io_strip_size_target,
                    # ncpu=ncpu, cpu_tile_shape = cpu_tile_shape,
                    # computation_dtype=computation_dtype)
                    
        # except Exception as e:
            # raise
            # if isinstance(e, expected):
                # pass
            # else:
                # raise
        # else:
            # try:
                # if issubclass(expected, BaseException):
                    # raise Exception(f"The test should have raised an exceptionof type {expected}")
            # except TypeError:
                # pass
            
            # with rasterio.open(grid_out, "r") as grid_out_ds:
                # grid_out_data_row = grid_out_ds.read(1)
                # grid_out_data_col = grid_out_ds.read(2)
                # np.testing.assert_array_equal(grid_out_data_col, grid_data_out_expected[1])
                # np.testing.assert_array_equal(grid_out_data_row, grid_data_out_expected[0])
            
            # # check
            # with rasterio.open(mask_out, "r") as mask_out_ds:
                # mask_out_data = mask_out_ds.read(1)
                # #test = mask_out_data == mask_data_out_expected
                # #where_fail = np.where(test == False)
                # #raise Exception(where_fail)
                # np.testing.assert_array_equal(mask_out_data, mask_data_out_expected)
            
        # finally:
            # if grid_in_tmp:
                # os.unlink(grid_in_tmp)
            # if mask_in_tmp:
                # os.unlink(mask_in_tmp)
            # if grid_out:
                # os.unlink(grid_out)
            # if mask_out:
                # os.unlink(mask_out)
            # if output_dir:
                # os.rmdir(output_dir)
            # # Check

