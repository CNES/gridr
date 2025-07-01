# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Module for a Grid and Mask creation chain
# @doc
"""
from functools import partial
import logging
from pathlib import Path
from typing import Union, NoReturn, Optional, Tuple, List, Dict

import multiprocessing
from multiprocessing import shared_memory, Pool

import numpy as np
import rasterio
from rasterio.windows import Window
import shapely

from gridr.core.utils import chunks
from gridr.core.utils.array_utils import array_replace, ArrayProfile
from gridr.core.utils.array_window import (window_shape, window_check,
        window_indices, as_rio_window, window_from_chunk, window_shift)
from gridr.core.grid.grid_commons import (grid_full_resolution_shape,
        grid_resolution_window)
from gridr.core.grid.grid_utils import oversample_regular_grid
from gridr.core.grid.grid_rasterize import (grid_rasterize,
        GridRasterizeAlg, ShapelyPredicate)
from gridr.core.grid.grid_mask import build_mask
from gridr.core.grid.grid_utils import build_grid
from gridr.io.common import GridRIOMode
from gridr.scaling.shmutils import (SharedMemoryArray, shmarray_wrap,
        create_and_register_sma)


DEFAULT_IO_STRIP_SIZE = 1000
DEFAULT_CPU_TILE_SHAPE = (1000, 1000)
DEFAULT_NCPU = 1


def build_mask_tile_worker(arg):
    """A worker that calls the build_mask method
    This method is passed to the multiprocessing Pool.map function
    
    The implementation uses directly the shmarray_wrap decorator in order to
    wrap the gridr core buid_mask function in order to conserve the same
    signature and to manage arguments typed as arrays that are passed as
    SharedMemoryArray in the multiprocessing process.    
    """
    shmarray_wrap(build_mask)(**arg)

def build_grid_mask_tile_worker(arg):
    """A worker that calls the build_mask and build_grid method
    This method is passed to the multiprocessing Pool.map function
    
    The implementation uses directly the shmarray_wrap decorator in order to
    wrap the gridr core buid_mask function in order to conserve the same
    signature and to manage arguments typed as arrays that are passed as
    SharedMemoryArray in the multiprocessing process.    
    """
    build_mask_arg, build_grid_arg = arg
    if len(build_mask_arg) > 0:
        shmarray_wrap(build_mask)(**build_mask_arg)
    shmarray_wrap(build_grid)(**build_grid_arg)


def build_mask_chain(
        shape: Tuple[int, int],
        resolution: Tuple[int, int],
        mask_out_ds: rasterio.io.DatasetWriter,
        mask_out_dtype: Union[np.dtypes.Int8DType, np.dtypes.UInt8DType],
        mask_in_ds: Optional[rasterio.io.DatasetReader],
        mask_in_unmasked_value: Optional[int],
        mask_in_band: Optional[int],
        computation_dtype: np.dtype,
        geometry_origin: Tuple[float, float],
        geometry: Optional[Union[shapely.geometry.Polygon,
                List[shapely.geometry.Polygon], shapely.geometry.MultiPolygon]],
        rasterize_kwargs: Optional[Dict] = None,
        mask_out_values: Optional[Tuple[int, int]] = (0,1),
        io_strip_size: int = DEFAULT_IO_STRIP_SIZE,
        io_strip_size_target: GridRIOMode = GridRIOMode.INPUT,
        ncpu: int = DEFAULT_NCPU,
        cpu_tile_shape: Optional[Tuple[int, int]] = DEFAULT_CPU_TILE_SHAPE,
        logger: Optional[logging.Logger] = None,
        ) -> int:
    """
    @doc
    Grid mask computation chain.
    
    This method wraps the call to the build_mask core method with I/O 
    resources management and a parallel computation capacity.
    
    The 'build_mask' method's goal is to compute a full resoltuion binary mask
    by merging an optional undersampled raster mask with a polygonized vector
    geometry. 
    
    Should you wish further details on the 'build_mask' method please read its 
    own documentation.
    
    Masked values
    -------------
    The 'build_mask' core method adopts a convention where the value '0' is 
    considered unmasked (i.e. valid) and the value '1' is considered masked.
    This method allows user to define independantly :
    - 'mask_out_values' : the values to use as output for unmasked and masked
    data.
    - 'mask_in_unmasked_value' : the valeur to consider as valid in the optional
    input mask.
    If the 'mask_in_unmasked_value' differs from the core method convention, the
    method converts the optional input mask to be complient with it directly
    afer read instructions.
    If the 'mask_out_umasked_value' differs from the core method convention, the
    method converts the output mask to match the user's input.
    
    Read/Write operations:
    ---------------------
    I/O for read and write are performed by strip chunks sequentially.
    A strip is defined as a window whose :
    - number of columns is the same as the read raster
    - number of rows is defined by the strip's size.

    There for a first sequential loop is performed on independantly on each
    strip consisting in the chaining of the 3 steps :
    'read' > 'process' > 'write'
  
    The strip's size can either be set to adress the read raster size or the
    written raster size (i.e. the computed raster size at each strip).
    The choice is defined through the definition of the 'io_strip_size_target'
    argument :
    - set it to 'GridRIOMode.INPUT' to adress the read strip's buffer size
    - set it to 'GridRIOMode.OUTPUT' to adress the write strip's buffer size
    
    Parallel processing
    -------------------
    This methods offers parallelization inside each strip through a 
    multiprocessing Pool.
    In order to activate parallel computing you have to define :
    - argument 'ncpu' to be greater than 1.
    - argument 'cpu_tile_shape' to be different from None and smaller than the
    output shape.
    
    Shared Memory
    -------------
    The read and the output buffers are set at once and used for each strip.
    They are allocated as Shared Memory in order to be efficiently shared among
    multiple parallel/concurrent processes.
    Please note that no lock is set on written memory ; this is justified by the
    fact that a strict tiling computation ensures that no overlap occurs across
    different processes.
    
    Computation Data Type
    ---------------------
    The user needs to provide the data type to use for the interpolation of the
    mask. The precision of the computation may differ between float32 and
    float64.
    
    Args:
        shape: the corresponding grid shape (nrows, ncols) at the grid's
                resolution.
        resolution: the grid's resolution tuple (oversampling row, oversampling
                col).
        mask_out_ds : the output mask as a DatasetWriter. The target size should
                corresponds to 'shape' x 'resolution'.
        mask_out_values : the tuple (<unmasked_value>, <masked_value) to use for
                output. If not given the convention used by the 'build_mask'
                method is preserved.
        mask_out_dtype : numpy type used to encode output mask. Should be either
                unsigned or signed 8 bits integer.
        mask_in_ds : the input mask as a DatasetReader. Its shape and resolution
                must correspond to the 'shape' and 'resolution' arguments.
                This argument is optional.
        mask_in_unmasked_value: the integer value to consider as valid.
        mask_in_band: the index of the mask band to use in order to read the 
                mask raster from 'mask_ds'.
        computation_dtype: data type to use for computation (interpolation).
        geometry_origin: geometric coordinates that are mapped to the output
                first pixel indexed by (0,0) in the array. This argument is
                mandatory if geometries is set.
        geometry: Definition of non masked geometries as a polygon or a list
                of polygons.
        rasterize_kwargs: dictionnary of parameters for the rasterize process.
                egg. {'alg': GridRasterizeAlg.SHAPELY,
                'kwargs_alg': {'shapely_predicate': ShapelyPredicate.COVERS}
                This argument is direclty passed to the build_mask method.
        io_strip_size : size (in number of rows) of a strip used for IO 
                operations.
        io_strip_size_target: definition of the mode to be used to consider the 
                strip size.
                If the target is set to 'GridRIOMode.INPUT', the 'io_strip_size'
                is used as is to direclty read buffers of 'io_strip_size' rows.
                It the target is set to 'GridRIOmode.OUTPUT', the
                'io_strip_size' corresponds to output full resolution strip's 
                size, thus limiting the read operation to fewer rows.
        ncpu: the number of workers set to the multiprocessing pool.
        cpu_tile_shape: the dimensions of the tiles adressed by each worker in
                case of multiprocessing.
                Please note that this argument has to be set in order to
                activate multiprocessing computation.
        logger: python logger object to use for logging. If None a logger is
                initialized internally.
    """
    # Init a list to register SharedMemoryArray buffers
    sma_buffer_name_list = []
    register_sma = partial(create_and_register_sma,
            register=sma_buffer_name_list)
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    nrow_in, ncol_in = shape
    logger.debug(f"Grid shape : {nrow_in} rows x {ncol_in} columns")
    
    if mask_out_values is None:
        mask_out_values = (0, 1)
    
    if mask_in_ds is not None:
        mask_nrow_in, mask_ncol_in = mask_in_ds.height, mask_in_ds.width
        logger.debug(f"Mask shape : {mask_nrow_in} rows x {mask_ncol_in} "
                "columns")
        assert(nrow_in == mask_nrow_in)
        assert(ncol_in == mask_ncol_in)
    else:
        logger.debug(f"Mask : no input mask")
    
    # Cut in strip chunks
    if io_strip_size not in [0, None]:
        if io_strip_size_target == GridRIOMode.INPUT:
            # We have to take into account the grid's resolution along rows
            io_strip_size = io_strip_size * resolution[0]
        elif io_strip_size_target == GridRIOMode.OUTPUT:
            # The strip_size is directly given for the target output
            pass
        else:
            raise ValueError(f"Not recognized value {io_strip_size_target} for "
                    "the 'io_strip_size_target' argument")
    
    # Compute the output shape
    shape_out = grid_full_resolution_shape(shape=(nrow_in, ncol_in),
            resolution=resolution)
    logger.debug(f"Computed output shape : {shape_out[0]} rows x {shape_out[1]}"
            " columns")
    
    # Compute strips definitions
    chunk_boundaries = chunks.get_chunk_boundaries(
                nsize=shape_out[0], chunk_size=io_strip_size, merge_last=True)
    
    # Allocate a dest buffer for the rasterio read operation
    # We have to compute read window for each chunk and take the max size
    
    # First convert chunks to target windows
    # Please note that the last coordinate in each chunk is not contained
    # in the chunk (it corresponds to the index), whereas the window 
    # definition contains index that are in the window
    chunk_windows = [ np.array([[c0, c1-1], [0, shape_out[1]-1]])
            for c0, c1 in chunk_boundaries]
    
    # Compute the window to read for each chunk window.
    # This will returns both the window to read, and the relative window
    # corresponding to the target chunk window.
    chunk_windows_read = [grid_resolution_window(resolution=resolution,
            win=chunk_win) for chunk_win in chunk_windows]
    
    # Determine the read buffer shape    
    read_buffer_shape = np.max(np.asarray([window_shape(read_win)
            for read_win, rel_win in chunk_windows_read]), axis=0)
    
    # Determine the write buffer shape
    buffer_shape = np.max(np.asarray([window_shape(chunk_win)
            for chunk_win in chunk_windows]), axis=0)
    
    sma_read_buffer = None
    if mask_in_ds is not None:
        # Create shared memory array for read
        sma_read_buffer = register_sma(read_buffer_shape, 
                mask_in_ds.dtypes[mask_in_band-1])
    
    # Create shared memory array for output
    sma_write_buffer = register_sma(buffer_shape, mask_out_dtype)
    
    try:
        for chunk_idx, (chunk_win, (win_read, win_rel)) in enumerate(
                zip(chunk_windows, chunk_windows_read)):
            
            logger.debug(f"Chunk {chunk_idx} - chunk_win: {chunk_win}")
            
            # Compute current strip chunk parameters to pass to the build_mask
            # method :
            # - cshape : current strip output buffer shape
            # - cmask_shape : current strip read buffer shape (input mask)
            # - cmask_arr : current strip array containing the read data (input
            #       mask)
            # - cslices : current strip slices to adress the output buffer whose
            #       origin corresponds to the origin of the current strip
            # - cslices_write : current strip slice to adress the whole output 
            #       dataset for IO write operation
            # - cgeometry_origin : the shifted geometry origin corresponding to
            #       the strip.
            cshape = window_shape(chunk_win)
            cslices = window_indices(chunk_win, reset_origin=True)
            cslices_write = window_indices(chunk_win, reset_origin=False)
            # read the data
            cmask_shape = window_shape(win_read)
            cmask_arr = None
            
            if sma_read_buffer is not None:
                cmask_arr = mask_in_ds.read(mask_in_band,
                        window = as_rio_window(win_read),
                        out=sma_read_buffer.array[0:cmask_shape[0],
                                0:cmask_shape[1]])
            
            cgeometry_origin = (geometry_origin[0] + chunk_win[0][0],
                    geometry_origin[1] + chunk_win[1][0])
            
            logger.debug(f"Chunk {chunk_idx} - shape : {cshape}")
            logger.debug(f"Chunk {chunk_idx} - buffer slices : {cslices}")
            logger.debug(f"Chunk {chunk_idx} - target slices : {cslices_write}")
            logger.debug(f"Chunk {chunk_idx} - build mask starts...")
            
            # Check valid/masked convention and ensure that the input buffer
            # is complient with the core convention, i.e. 0 for valid data. 
            if cmask_arr is not None and mask_in_unmasked_value != 0:
                array_replace(cmask_arr, mask_in_unmasked_value, 0, 1)
            
            # Choose between the tiled multiprocessing branch and the
            # processing branch
            if ncpu > 1 and cpu_tile_shape is not None:
                # Run on multiple cores
                # Cut strip shape into tiled chunks.
                logger.debug(f"Chunk {chunk_idx} - Tiled multiprocessing on "
                        f"{ncpu} workers with tiles of {cpu_tile_shape[0]} x "
                        f"{cpu_tile_shape[1]}")
                        
                chunk_tiles = chunks.get_chunk_shapes(cshape, cpu_tile_shape,
                        merge_last=True)
                
                logger.debug(f"Chunk {chunk_idx} - Number of tiles to process :"
                        f" {len(chunk_tiles)}")
                
                # Init the list of process arguments as 'tasks'
                tasks = []
                for tile in chunk_tiles:
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} "
                            "- preparing args...")
                    
                    # Compute current strip chunk parameters to pass to the
                    # 'build_mask_tile_worker' ('build_mask' wrapper) method
                    # - tile_origin : the tile origin corresponds here to the
                    #        coordinates relative to the current strip, ie the
                    #        first element of each window
                    # - tile_win : the window corresponding to the tile (convert
                    #        the chunk index convention to the window index
                    #        convention ; with no origin shift)
                    # - tile_geometry_origin : the shifted geometry origin
                    #        corresponding to the current tile.
                    # - tile_shape : current tile output shape
                    # - tile_slice : current tile slices to adress the output
                    #        buffer whose origin corresponds to the origin of
                    #        the current strip
                    # - tile_mask_in_target_win : 'win_rel' variable contains
                    #        the windowing to apply at full resolution of the
                    #        mask. It has to be shifted of the current tile's
                    #        upper left corner.
                    tile_origin = chunk_win[...,0]
                    tile_win = window_from_chunk(chunk=tile, origin=None)
                    tile_geometry_origin = (cgeometry_origin[0] + tile_win[0][0],
                            cgeometry_origin[1] + tile_win[1][0])
                            
                    tile_shape = window_shape(tile_win)
                    tile_slice = window_indices(tile_win, reset_origin=False)
                    tile_mask_in_target_win = window_shift(tile_win, win_rel[:,0])
                    
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - "
                            f"tile's chunk origin : {tile_origin}")
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - "
                            f"tile window : {tile_win}")
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - "
                            f"mask_in_target_win : {tile_mask_in_target_win} "
                            f"from strip target win : {win_rel}")
                    
                    # Manage shared memory to pass to the process
                    # - tile_smb_out : a SharedMemoryBuffer object to pass for
                    #        output buffer by cloning the caracteristics of the
                    #        main output buffer except the slice
                    # - tile_smb_maks_in : a SharedMemoryBuffer object to pass for
                    #        read buffer by cloning the caracteristics of the
                    #        main output buffer. The shift for the current tile
                    #        is applied through the 'mask_in_target_win' arg.
                    tile_smb_out = SharedMemoryArray.clone(
                            sma=sma_write_buffer, array_slice=tile_slice)

                    tile_smb_mask_in = None
                    if sma_read_buffer is not None:
                        tile_smb_mask_in = SharedMemoryArray.clone(
                                sma=sma_read_buffer, array_slice=None)
                    
                    # Append process parameters to 'tasks'
                    tasks.append({'shape': tile_shape,
                            'resolution':(1,1),
                            'out':tile_smb_out,
                            'geometry_origin':tile_geometry_origin,
                            'geometry':geometry,
                            'mask_in':tile_smb_mask_in,
                            'mask_in_target_win':tile_mask_in_target_win,
                            'mask_in_resolution':resolution,
                            'oversampling_dtype':computation_dtype,
                            'mask_in_binary_threshold':1e-3,
                            'rasterize_kwargs':rasterize_kwargs})

                with Pool(processes=ncpu) as pool:
                    pool.map(build_mask_tile_worker, tasks)
            
            else:
                # Build mask on full strip - no multiprocessing 
                logger.debug(f"Chunk {chunk_idx} - Full strip computation "
                        "(no tiling)")
                        
                _ = build_mask(
                        shape=cshape ,
                        resolution=(1,1),
                        out=sma_write_buffer.array[cslices],
                        geometry_origin=cgeometry_origin,#: Tuple[float, float],
                        geometry=geometry,
                        mask_in=cmask_arr,
                        mask_in_target_win=win_rel,
                        mask_in_resolution=resolution,
                        oversampling_dtype=computation_dtype,
                        mask_in_binary_threshold=1e-3,
                        rasterize_kwargs=rasterize_kwargs,)
                    
            logger.debug(f"Chunk {chunk_idx} - build mask ends.")
            
            # Check masked/unmasked convention and ensure that the output buffer
            # is complient with the user's given convention.
            if not np.all(mask_out_values==(0,1)):
                val_cond = 0 # considered true in build_mask method
                val_true, val_false = mask_out_values 
                array_replace(sma_write_buffer.array[cslices],
                        val_cond, val_true, val_false)
                #sma_write_buffer.array[*cslices] = np.where(
                #        sma_write_buffer.array[*cslices]==0,
                #        mask_out_valid_value, ~mask_out_valid_value)
            
            # Write the data
            if mask_out_ds is not None:
                logger.debug(f"Chunk {chunk_idx} - write mask...")
                
                mask_out_ds.write(sma_write_buffer.array[cslices], 1,
                        window=as_rio_window(chunk_win))
                        
                logger.debug(f"Chunk {chunk_idx} - write ends.")
    except:
        raise
    finally:
        # Release the Shared memory buffer
        SharedMemoryArray.clear_buffers(sma_buffer_name_list)    
    return 1


def build_grid_mask_chain(
        resolution: Tuple[int, int],
        grid_in_ds: rasterio.io.DatasetReader,
        grid_in_col_ds: Union[rasterio.io.DatasetReader, None],
        grid_in_row_coords_band: int,
        grid_in_col_coords_band: int,
        grid_out_ds: rasterio.io.DatasetWriter,
        grid_out_col_ds: Union[rasterio.io.DatasetWriter, None],
        grid_out_row_coords_band: int,
        grid_out_col_coords_band: int,
        mask_out_ds: rasterio.io.DatasetWriter,
        mask_out_dtype: Union[np.dtypes.Int8DType, np.dtypes.UInt8DType],
        mask_in_ds: Optional[rasterio.io.DatasetReader],
        mask_in_unmasked_value: Optional[int],
        mask_in_band: Optional[int],
        computation_dtype: np.dtype,
        geometry_origin: Tuple[float, float],
        geometry: Optional[Union[shapely.geometry.Polygon,
                List[shapely.geometry.Polygon], shapely.geometry.MultiPolygon]],
        rasterize_kwargs: Optional[Dict] = None,
        mask_out_values: Optional[Tuple[int, int]] = (0,1),
        merge_mask_grid: Optional[Union[int, float]] = None,
        io_strip_size: int = DEFAULT_IO_STRIP_SIZE,
        io_strip_size_target: GridRIOMode = GridRIOMode.INPUT,
        ncpu: int = DEFAULT_NCPU,
        cpu_tile_shape: Optional[Tuple[int, int]] = DEFAULT_CPU_TILE_SHAPE,
        logger: Optional[logging.Logger] = None,
        ) -> int:
    """
    @doc
    Grid and mask computation chain.
    
    This method wraps both the call to the build_grid method and the call to the
    call to the build_mask core method with I/O resources management and a
    parallel computation capacity.
    
    The 'build_grid' method's goal is to compute a full resoltuion grid. 
    
    The 'build_mask' method's goal is to compute a full resoltuion binary mask
    by merging an optional undersampled raster mask with a polygonized vector
    geometry. 
    
    Should you wish further details on the 'build_grid' and 'build_mask' method
    please read their own documentations.
    
    Masked values
    -------------
    The 'build_mask' core method adopts a convention where the value '0' is 
    considered unmasked (i.e. valid) and the value '1' is considered masked.
    This method allows user to define independantly :
    - 'mask_out_values' : the values to use as output for unmasked and masked
    data.
    - 'mask_in_unmasked_value' : the valeur to consider as valid in the optional
    input mask.
    If the 'mask_in_unmasked_value' differs from the core method convention, the
    method converts the optional input mask to be complient with it directly
    afer read instructions.
    If the 'mask_out_umasked_value' differs from the core method convention, the
    method converts the output mask to match the user's input.
    
    Merging the mask in the grid
    ----------------------------
    This method provides an option to insert the mask in the grid by affecting
    a special value to mask pixel. In order to do so, just set a not None value
    to the 'merge_mask_grid' parameter. This value will be used to fill the
    grid at masked coordinates.
    
    
    Read/Write operations:
    ---------------------
    I/O for read and write are performed by strip chunks sequentially.
    A strip is defined as a window whose :
    - number of columns is the same as the read raster
    - number of rows is defined by the strip's size.

    There for a first sequential loop is performed on independantly on each
    strip consisting in the chaining of the 3 steps :
    'read' > 'process' > 'write'
  
    The strip's size can either be set to adress the read raster size or the
    written raster size (i.e. the computed raster size at each strip).
    The choice is defined through the definition of the 'io_strip_size_target'
    argument :
    - set it to 'GridRIOMode.INPUT' to adress the read strip's buffer size
    - set it to 'GridRIOMode.OUTPUT' to adress the write strip's buffer size
    
    Parallel processing
    -------------------
    This methods offers parallelization inside each strip through a 
    multiprocessing Pool.
    In order to activate parallel computing you have to define :
    - argument 'ncpu' to be greater than 1.
    - argument 'cpu_tile_shape' to be different from None and smaller than the
    output shape.
    
    Shared Memory
    -------------
    The read and the output buffers are set at once and used for each strip.
    They are allocated as Shared Memory in order to be efficiently shared among
    multiple parallel/concurrent processes.
    Please note that no lock is set on written memory ; this is justified by the
    fact that a strict tiling computation ensures that no overlap occurs across
    different processes.
    
    Computation Data Type
    ---------------------
    The user needs to provide the data type to use for the interpolation of the
    mask. The precision of the computation may differ between float32 and
    float64.
    
    Args:
        resolution: the input grid and mask resolution tuple (oversampling row,
                oversampling col).
        grid_in_ds: the input grid as a DatasetReader. If 'grid_in_col_ds' is
                provided, this argument is only used to read the rows
                coordinates. Otherwise it is considered for both the rows and
                the columns coordinates. The argument 'grid_in_row_coords_band'
                and 'grid_in_col_coords_band' are use to respectively read the
                rows and columns grids.
        grid_in_col_ds: An optional DatasetReader to read the columns 
                coordinates. If set to None, the 'grid_in_ds' is used for both
                the rows and the columns coordinates.
        grid_in_row_coords_band: 1-based index of the rows band in the
                corresponding DatasetReader.
        grid_in_col_coords_band: 1-based index of the columns band in the
                corresponding DatasetReader.
        grid_out_ds: the output grid as a DatasetWriter. If 'grid_out_col_ds' is
                provided, this argument is only used to write the rows
                coordinates. Otherwise it is considered for both the rows and
                the columns coordinates. The argument 'grid_out_row_coords_band'
                and 'grid_out_col_coords_band' are use to respectively write the
                rows and columns grids.
        grid_out_col_ds: An optional DatasetWriter to read the columns 
                coordinates. If set to None, the 'grid_out_ds' is used for both
                the rows and the columns coordinates.
        grid_out_row_coords_band: 1-based index of the rows band in the
                corresponding DatasetWriter.
        grid_out_col_coords_band: 1-based index of the columns band in the
                corresponding DatasetWriter.
        mask_out_ds : the output mask as a DatasetWriter. The target size should
                corresponds to 'shape' x 'resolution'.
        mask_out_dtype : numpy type used to encode output mask. Should be either
                unsigned or signed 8 bits integer.
        mask_in_ds : the input mask as a DatasetReader. Its shape and resolution
                must correspond to the 'shape' and 'resolution' arguments.
                This argument is optional.
        mask_in_unmasked_value: the integer value to consider as valid.
        mask_in_band: the index of the mask band to use in order to read the 
                mask raster from 'mask_ds'.
        computation_dtype: data type to use for computation (interpolation).
        geometry_origin: geometric coordinates that are mapped to the output
                first pixel indexed by (0,0) in the array. This argument is
                mandatory if geometries is set.
        geometry: Definition of non masked geometries as a polygon or a list
                of polygons.
        rasterize_kwargs: dictionnary of parameters for the rasterize process.
                egg. {'alg': GridRasterizeAlg.SHAPELY,
                'kwargs_alg': {'shapely_predicate': ShapelyPredicate.COVERS}
                This argument is direclty passed to the build_mask method.
        mask_out_values : the tuple (<unmasked_value>, <masked_value) to use for
                output. If not given the convention used by the 'build_mask'
                method is preserved.        
        merge_mask_grid: value to fill in grid output for masked data
        io_strip_size : size (in number of rows) of a strip used for IO 
                operations.
        io_strip_size_target: definition of the mode to be used to consider the 
                strip size.
                If the target is set to 'GridRIOMode.INPUT', the 'io_strip_size'
                is used as is to direclty read buffers of 'io_strip_size' rows.
                It the target is set to 'GridRIOmode.OUTPUT', the
                'io_strip_size' corresponds to output full resolution strip's 
                size, thus limiting the read operation to fewer rows.
        ncpu: the number of workers set to the multiprocessing pool.
        cpu_tile_shape: the dimensions of the tiles adressed by each worker in
                case of multiprocessing.
                Please note that this argument has to be set in order to
                activate multiprocessing computation.
        logger: python logger object to use for logging. If None a logger is
                initialized internally.
    """
    # Init a list to register SharedMemoryArray buffers
    sma_buffer_name_list = []
    register_sma = partial(create_and_register_sma,
            register=sma_buffer_name_list)
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Get input shape from input grid
    nrow_in, ncol_in = grid_in_ds.height, grid_in_ds.width
    logger.debug(f"Grid shape : {nrow_in} rows x {ncol_in} columns")
    
    grid_in_row_ds = grid_in_ds
    if grid_in_col_ds is None:
        grid_in_col_ds = grid_in_ds
        assert(grid_in_row_coords_band != grid_in_col_coords_band)
    else:
        # Test shapes are the same
        assert(nrow_in == grid_in_col_ds.height)
        assert(ncol_in == grid_in_col_ds.width)
    
    # Check output DatasetWriter definitions
    grid_out_row_ds = grid_out_ds
    if grid_out_col_ds is None:
        grid_out_col_ds = grid_out_ds
        assert(grid_out_row_coords_band != grid_out_col_coords_band)
    else:
        assert(grid_out_col_ds.height == grid_out_col_ds.height)
        assert(grid_out_row_ds.width == grid_out_row_ds.width)
    
    
    # Check mask shape
    if mask_in_ds is not None:
        mask_nrow_in, mask_ncol_in = mask_in_ds.height, mask_in_ds.width
        logger.debug(f"Mask shape : {mask_nrow_in} rows x {mask_ncol_in} "
                "columns")
        assert(nrow_in == mask_nrow_in)
        assert(ncol_in == mask_ncol_in)
    else:
        logger.debug(f"Mask : no input mask")
    
    if mask_out_values is None:
        mask_out_values = (0, 1)
    
    # Cut in strip chunks
    # io_strip_size is computed for the output grid but it can either be piloted
    # by setting a target size for the read buffer.
    if io_strip_size not in [0, None]:
        if io_strip_size_target == GridRIOMode.INPUT:
            # We have to take into account the grid's resolution along rows
            io_strip_size = io_strip_size * resolution[0]
        elif io_strip_size_target == GridRIOMode.OUTPUT:
            # The strip_size is directly given for the target output
            pass
        else:
            raise ValueError(f"Not recognized value {io_strip_size_target} for "
                    "the 'io_strip_size_target' argument")
    
    # Compute the output shape
    shape_out = grid_full_resolution_shape(shape=(nrow_in, ncol_in),
            resolution=resolution)
    logger.debug(f"Computed output shape : {shape_out[0]} rows x {shape_out[1]}"
            " columns")
    
    # Compute strips definitions
    chunk_boundaries = chunks.get_chunk_boundaries(
                nsize=shape_out[0], chunk_size=io_strip_size, merge_last=True)
    
    # Allocate a dest buffer for the rasterio read operation
    # We have to compute read window for each chunk and take the max size
    
    # First convert chunks to target windows
    # Please note that the last coordinate in each chunk is not contained
    # in the chunk (it corresponds to the index), whereas the window 
    # definition contains index that are in the window
    chunk_windows = [ np.array([[c0, c1-1], [0, shape_out[1]-1]])
            for c0, c1 in chunk_boundaries]
    
    # Compute the window to read for each chunk window.
    # This will returns both the window to read, and the relative window
    # corresponding to the target chunk window.
    chunk_windows_read = [grid_resolution_window(resolution=resolution,
            win=chunk_win) for chunk_win in chunk_windows]
    
    # Determine the read buffer shape    
    read_buffer_shape = np.max(np.asarray([window_shape(read_win)
            for read_win, rel_win in chunk_windows_read]), axis=0)
    read_buffer_shape3 = np.insert(read_buffer_shape, 0, 2) 
    
    # Create shared memory array for output
    buffer_shape = np.max(np.asarray([window_shape(chunk_win)
            for chunk_win in chunk_windows]), axis=0)
    buffer_shape3 = np.insert(buffer_shape, 0, 2) 
    
    # Create shared memory array for read
    # input grid
    sma_r_buffer_grid = register_sma(read_buffer_shape3,
            grid_in_row_ds.dtypes[grid_in_row_coords_band-1])
            
    # - mask in
    sma_r_buffer_mask = None
    if mask_in_ds is not None:
        sma_r_buffer_mask = register_sma(read_buffer_shape,
                mask_in_ds.dtypes[mask_in_band-1])
    
    # Create shared memory array for write
    sma_w_buffer_grid = register_sma(buffer_shape3,
            grid_out_row_ds.dtypes[grid_out_row_coords_band-1])
            
    # - mask out
    compute_mask = False
    sma_w_buffer_mask = None
    # Determine if mask must be computed
    if (mask_out_ds is not None) \
            or (mask_in_ds is not None) \
            or (geometry is not None) :
        compute_mask = True
    
    #if mask_out_ds is not None or mask_out_dtype is not None:
    if compute_mask:
        sma_w_buffer_mask = register_sma(buffer_shape, mask_out_dtype)
    
    try:
        for chunk_idx, (chunk_win, (win_read, win_rel)) in enumerate(
                zip(chunk_windows, chunk_windows_read)):
            
            logger.debug(f"Chunk {chunk_idx} - chunk_win: {chunk_win}")
            
            # Compute current strip chunk parameters to pass to the build_mask
            # method :
            # - cshape : current strip output buffer shape
            # - cread_shape : current strip read buffer shape (input mask)
            # - cread_rows_arr : current strip array containing the read data 
            #           for the grid rows 
            # - cread_cols_arr : current strip array containing the read data 
            #           for the grid columns 
            # - cread_mask_arr : current strip array containing the read data
            #           for the input mask if given
            # - cslices : current strip slices to adress the output buffer whose
            #       origin corresponds to the origin of the current strip
            # - cslices_write : current strip slice to adress the whole output 
            #       dataset for IO write operation
            # - cgeometry_origin : the shifted geometry origin corresponding to
            #       the strip.
            cshape = window_shape(chunk_win)
            cslices = window_indices(chunk_win, reset_origin=True)
            cslices3 = (slice(None, None),) + cslices
            cslices_write = window_indices(chunk_win, reset_origin=False)
            # read the data
            cread_shape = window_shape(win_read)
            
            # First row and col grids
            _ = grid_in_row_ds.read(grid_in_row_coords_band,
                    window = as_rio_window(win_read),
                    out = sma_r_buffer_grid.array[0, 0:cread_shape[0], 0:cread_shape[1]])
            
            _ = grid_in_col_ds.read(grid_in_col_coords_band,
                    window = as_rio_window(win_read),
                    out = sma_r_buffer_grid.array[1, 0:cread_shape[0], 0:cread_shape[1]])
            
            cread_grid_arr = sma_r_buffer_grid.array[:, 0:cread_shape[0], 0:cread_shape[1]]
            
            # Input mask if given
            cread_mask_arr = None
            if compute_mask:
                if sma_r_buffer_mask is not None:
                    cread_mask_arr = mask_in_ds.read(mask_in_band,
                            window = as_rio_window(win_read),
                            out=sma_r_buffer_mask.array[0:cread_shape[0], 0:cread_shape[1]])
            
            # Shift geometry origin for the strip
            cgeometry_origin = (geometry_origin[0] + chunk_win[0][0],
                    geometry_origin[1] + chunk_win[1][0])
            
            logger.debug(f"Chunk {chunk_idx} - shape : {cshape}")
            logger.debug(f"Chunk {chunk_idx} - buffer slices : {cslices}")
            logger.debug(f"Chunk {chunk_idx} - target slices : {cslices_write}")
            logger.debug(f"Chunk {chunk_idx} - build starts...")
            
            # Mask management
            if cread_mask_arr is not None:
                # Check valid/masked convention and ensure that the input buffer
                # is complient with the core convention, i.e. 0 for valid data. 
                if mask_in_unmasked_value != 0:
                    array_replace(cread_mask_arr, mask_in_unmasked_value, 0, 1)
            
            # Choose between the tiled multiprocessing branch and the
            # processing branch
            if ncpu > 1 and cpu_tile_shape is not None:
                # Run on multiple cores
                # Cut strip shape into tiled chunks.
                logger.debug(f"Chunk {chunk_idx} - Tiled multiprocessing on "
                        f"{ncpu} workers with tiles of {cpu_tile_shape[0]} x "
                        f"{cpu_tile_shape[1]}")
                        
                chunk_tiles = chunks.get_chunk_shapes(cshape, cpu_tile_shape,
                        merge_last=True)
                
                logger.debug(f"Chunk {chunk_idx} - Number of tiles to process :"
                        f" {len(chunk_tiles)}")
                
                # Init the list of process arguments as 'tasks'
                tasks = []
                for tile in chunk_tiles:
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} "
                            "- preparing args...")
                    
                    # Compute current strip chunk parameters to pass to the
                    # 'build_mask_tile_worker' ('build_mask' wrapper) method
                    # - tile_origin : the tile origin corresponds here to the
                    #        coordinates relative to the current strip, ie the
                    #        first element of each window
                    # - tile_win : the window corresponding to the tile (convert
                    #        the chunk index convention to the window index
                    #        convention ; with no origin shift)
                    # - tile_geometry_origin : the shifted geometry origin
                    #        corresponding to the current tile.
                    # - tile_shape : current tile output shape
                    # - tile_slice : current tile slices to adress the output
                    #        buffer whose origin corresponds to the origin of
                    #        the current strip
                    # - tile_mask_in_target_win : 'win_rel' variable contains
                    #        the windowing to apply at full resolution of the
                    #        mask. It has to be shifted of the current tile's
                    #        upper left corner.
                    tile_origin = chunk_win[...,0]
                    tile_win = window_from_chunk(chunk=tile, origin=None)
                    tile_geometry_origin = (cgeometry_origin[0] + tile_win[0][0],
                            cgeometry_origin[1] + tile_win[1][0])
                            
                    tile_shape = window_shape(tile_win)
                    tile_slice = window_indices(tile_win, reset_origin=False)
                    tile_slices3 = (slice(None, None),) + tile_slice
                    tile_target_win = window_shift(tile_win, win_rel[:,0])
                    
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - "
                            f"tile's chunk origin : {tile_origin}")
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - "
                            f"tile window : {tile_win}")
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - "
                            f"target_win : {tile_target_win} "
                            f"from strip target win : {win_rel}")
                    
                    # Manage shared memory to pass to the process
                    # - tile_smb_grid_out : a SharedMemoryBuffer object to pass 
                    #        for output buffer by cloning the caracteristics of 
                    #        the grid output buffer except the slice
                    # - tile_smb_grid_in : a SharedMemoryBuffer object to pass 
                    #        for read buffer by cloning the caracteristics of 
                    #        the grid input buffer. The shift for the current 
                    #        tile is applied through the 'mask_in_target_win' 
                    #        arg.
                    # - tile_smb_mask_out : a SharedMemoryBuffer object to pass 
                    #        for output buffer by cloning the caracteristics of 
                    #        the mask output buffer except the slice
                    # - tile_smb_mask_in : a SharedMemoryBuffer object to pass 
                    #        for read buffer by cloning the caracteristics of 
                    #        the mask input buffer. The shift for the current 
                    #        tile is applied through the 'mask_in_target_win' 
                    #        arg.
                    # grid
                    tile_smb_grid_out = SharedMemoryArray.clone(
                            sma=sma_w_buffer_grid, array_slice=tile_slices3)

                    tile_smb_grid_in = SharedMemoryArray.clone(
                            sma=sma_r_buffer_grid, array_slice=None)
                    
                    build_grid_args = {'resolution': (1,1),
                            'grid': tile_smb_grid_in,
                            'grid_target_win': tile_target_win,
                            'grid_resolution': resolution,
                            'computation_dtype': computation_dtype,
                            'out': tile_smb_grid_out}
                    
                    build_mask_args = {}
                    if compute_mask:
                        # mask
                        tile_smb_mask_out = SharedMemoryArray.clone(
                                sma=sma_w_buffer_mask, array_slice=tile_slice)

                        tile_smb_mask_in = None
                        if sma_r_buffer_mask is not None:
                            tile_smb_mask_in = SharedMemoryArray.clone(
                                    sma=sma_r_buffer_mask, array_slice=None)
                    
                        # Append process parameters to 'tasks'
                        build_mask_args = {'shape': tile_shape,
                                'resolution': (1,1),
                                'out': tile_smb_mask_out,
                                'geometry_origin': tile_geometry_origin,
                                'geometry': geometry,
                                'mask_in': tile_smb_mask_in,
                                'mask_in_target_win': tile_target_win,
                                'mask_in_resolution': resolution,
                                'mask_in_binary_threshold': 1e-3,
                                'rasterize_kwargs': rasterize_kwargs,
                                'oversampling_dtype': computation_dtype}
                    
                    tasks.append((build_mask_args, build_grid_args))

                with Pool(processes=ncpu) as pool:
                    pool.map(build_grid_mask_tile_worker, tasks)
                    
            # Mono processing for now
            else:
                # Build mask on full strip - no multiprocessing 
                logger.debug(f"Chunk {chunk_idx} - Full strip computation "
                        "(no tiling)")
                
                if compute_mask:
                    _ = build_mask(
                            shape=cshape ,
                            resolution=(1,1),
                            out=sma_w_buffer_mask.array[cslices],
                            geometry_origin=cgeometry_origin,
                            geometry=geometry,
                            mask_in=cread_mask_arr,
                            mask_in_target_win=win_rel,
                            mask_in_resolution=resolution,
                            mask_in_binary_threshold=1e-3,
                            rasterize_kwargs=rasterize_kwargs,
                            oversampling_dtype=computation_dtype,)

                # Process grid
                _ = build_grid(
                        resolution=(1,1),
                        grid=cread_grid_arr,
                        grid_target_win=win_rel,
                        grid_resolution=resolution,
                        computation_dtype=computation_dtype,
                        out=sma_w_buffer_grid.array[cslices3])
                
            
            
            logger.debug(f"Chunk {chunk_idx} - build mask ends.")

            # Here we merge
            if compute_mask and merge_mask_grid is not None:
                #mask_indices = sma_w_buffer_mask.array[cslices] == 1 
                #sma_w_buffer_grid.array[cslices3][:, mask_indices] = merge_mask_grid
                #array_replace(array=sma_w_buffer_grid.array[cslices3],
                #        val_cond=0, val_true=merge_mask_grid, val_false=None,
                #        array_cond=sma_w_buffer_mask.array[cslices],
                #        array_cond_val=1, win=None)
                array_replace(array=sma_w_buffer_grid.array[0][cslices],
                        val_cond=0, val_true=merge_mask_grid, val_false=None,
                        array_cond=sma_w_buffer_mask.array[cslices],
                        array_cond_val=1, win=None)
                array_replace(array=sma_w_buffer_grid.array[1][cslices],
                        val_cond=0, val_true=merge_mask_grid, val_false=None,
                        array_cond=sma_w_buffer_mask.array[cslices],
                        array_cond_val=1, win=None)
            
            # Check masked/unmasked convention and ensure that the output buffer
            # is complient with the user's given convention.
            if compute_mask and not np.all(mask_out_values==(0,1)):
                val_cond = 0 # considered true in build_mask method
                val_true, val_false = mask_out_values 
                array_replace(sma_w_buffer_mask.array[cslices],
                        val_cond, val_true, val_false)
                #sma_write_buffer.array[*cslices] = np.where(
                #        sma_write_buffer.array[*cslices]==0,
                #        mask_out_valid_value, ~mask_out_valid_value)

            # Write the data
            # Write grid rows
            logger.debug(f"Chunk {chunk_idx} - write grid rows...")
            
            grid_out_row_ds.write(sma_w_buffer_grid.array[0][cslices],
                    grid_out_row_coords_band, window=as_rio_window(chunk_win))
            
            logger.debug(f"Chunk {chunk_idx} - write ends.")
            
            # Write grid columns
            logger.debug(f"Chunk {chunk_idx} - write grid columns...")
            
            grid_out_col_ds.write(sma_w_buffer_grid.array[1][cslices],
                    grid_out_col_coords_band, window=as_rio_window(chunk_win))
            
            logger.debug(f"Chunk {chunk_idx} - write ends.")
            
            if compute_mask and mask_out_ds is not None:
                logger.debug(f"Chunk {chunk_idx} - write mask...")
                
                mask_out_ds.write(sma_w_buffer_mask.array[cslices], 1,
                        window=as_rio_window(chunk_win))
                        
                logger.debug(f"Chunk {chunk_idx} - write ends.")

    except:
        raise
    
    finally:
        # Release the Shared memory buffer
        SharedMemoryArray.clear_buffers(sma_buffer_name_list)   
    
    return 1

