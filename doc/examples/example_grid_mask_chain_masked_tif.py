# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
This is an example to use the build_grid_mask_chain to merge a grid and a binary
mask. 

Command to run :
PYTHONPATH=${PWD}/../../python python example_grid_mask_chain_masked_tif.py
"""
import os
import logging
from pathlib import Path
import tempfile
import sys

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.chain.grid_mask_chain import (
        GridRIOMode,
        build_mask_chain,
        build_grid_mask_chain,
        )


ALOGGER = logging.getLogger(__name__)
ALOGGER.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
log_rio = rasterio.logging.getLogger()
log_rio.setLevel(logging.ERROR)

# Set variables to define the data 
MASK_IN_UNMASKED_VALUE = 0
MASK_OUT_VALUES = (0, 1)
DTYPE = np.float32
RESOLUTION = (1,1)
SHAPE = (20000,2000)
# Input grid
GRID_IN_ARRAY_ROW = np.arange(np.prod(SHAPE), dtype=DTYPE).reshape(SHAPE)
GRID_IN_ARRAY_COL = 10 + 10. * np.arange(np.prod(SHAPE), dtype=DTYPE).reshape(SHAPE)
GRID_IN_ARRAY = np.stack((GRID_IN_ARRAY_ROW, GRID_IN_ARRAY_COL))
# Input mask
MASK_IN_ARRAY = np.ones(np.prod(SHAPE), dtype=np.uint8).reshape(SHAPE)
# Mask a window in the input array
MASK_IN_ARRAY_MASK_WINDOW = [(20,30), (11,15)]
MASK_IN_ARRAY[
        slice(*MASK_IN_ARRAY_MASK_WINDOW[0]),
        slice(*MASK_IN_ARRAY_MASK_WINDOW[1])] = MASK_IN_UNMASKED_VALUE

# Create the grid out integrating the mask
MERGE_GRID_MASK_VALUE = 99999

def create_data(path_grid, path_mask):
    """
    """
    with rasterio.open(path_grid, "w",
            driver="GTiff", dtype=GRID_IN_ARRAY.dtype,
            height=GRID_IN_ARRAY.shape[1], width=GRID_IN_ARRAY.shape[2],
            count=2) as grid_ds:
        grid_ds.write(GRID_IN_ARRAY[0], 1)
        grid_ds.write(GRID_IN_ARRAY[1], 2)
    # Write input mask
    with rasterio.open(path_mask, "w",
            driver="GTiff",
            dtype=MASK_IN_ARRAY.dtype,
            height=MASK_IN_ARRAY.shape[0],
            width=MASK_IN_ARRAY.shape[1],
            count=1,
            nbits=1) as mask_ds:
        mask_ds.write(MASK_IN_ARRAY, 1)
            
def merge_mask_grid(path_grid_in, path_mask_in, path_grid_out):
    """
    """
    open_kwargs = {'driver':"GTiff",
            'height':GRID_IN_ARRAY.shape[1], 'width':GRID_IN_ARRAY.shape[2],
            }
    with rasterio.open(path_grid_in, "r") as grid_in_ds, \
            rasterio.open(path_mask_in, "r") as mask_in_ds, \
            rasterio.open(path_grid_out, "w", dtype=DTYPE, count=2, **open_kwargs) as grid_out_ds:
        
        # Create output raster ds
        build_grid_mask_chain(
                resolution=RESOLUTION, # target resolution - set it the same as input resolution just to merge 
                grid_in_ds=grid_in_ds,
                grid_in_col_ds=None, # same dataset reader as rows (same file)
                grid_in_row_coords_band=1,
                grid_in_col_coords_band=2,
                grid_out_ds=grid_out_ds,
                grid_out_col_ds=None, # same output file
                grid_out_row_coords_band=1,
                grid_out_col_coords_band=2,
                mask_out_ds=None, # set to None to not write the mask
                mask_out_dtype=np.uint8,
                mask_in_ds=mask_in_ds,
                mask_in_unmasked_value=MASK_IN_UNMASKED_VALUE,
                mask_in_band=1,
                geometry_origin=(0.5,0.5),
                geometry=None,
                rasterize_kwargs=None,
                mask_out_values = MASK_OUT_VALUES,
                merge_mask_grid = MERGE_GRID_MASK_VALUE,
                io_strip_size=5000,
                io_strip_size_target=GridRIOMode.INPUT, # here does not change with OUTPUT ; resolutions are the same
                ncpu=2,
                cpu_tile_shape = (5000,1000), # will not be used if ncpu=1
                computation_dtype=DTYPE, # no interpolation are done => should not affect results
                logger=ALOGGER,
                )

if __name__ == '__main__':
    
    basename = Path(sys.argv[0]).name.replace('.py','')
    path_grid_in = f'{basename}_grid_in.tif'
    path_mask_in = f'{basename}_mask_in.tif'
    path_grid_out = f'{basename}_grid_out.tif'
    create_data(path_grid_in, path_mask_in)
    merge_mask_grid(path_grid_in, path_mask_in, path_grid_out)
    print(f"output : {path_grid_out}")
    
    with rasterio.open(path_grid_out) as ds:
        print(ds.profile)
        row = ds.read(1)
        col = ds.read(2)
        
        indices = (slice(MASK_IN_ARRAY_MASK_WINDOW[0][0]-5, MASK_IN_ARRAY_MASK_WINDOW[0][1]+5),
                slice(MASK_IN_ARRAY_MASK_WINDOW[1][0]-5, MASK_IN_ARRAY_MASK_WINDOW[1][1]+5))
        print(f"print indices : {indices}")
        np.set_printoptions(formatter={'all': '{: 0.0f}'.format})
        np.set_printoptions(linewidth=400)
        print("\n\nrows\n", row[indices])
        print("\n\ncols\n", col[indices])