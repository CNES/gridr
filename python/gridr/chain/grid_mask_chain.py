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
Module for a Grid and Mask creation chain
# @doc
"""
import logging
from functools import partial
from multiprocessing import Pool
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio

from gridr.core.grid.grid_commons import grid_full_resolution_shape, grid_resolution_window
from gridr.core.grid.grid_mask import Validity, build_mask
from gridr.core.grid.grid_rasterize import GeometryType
from gridr.core.grid.grid_utils import build_grid
from gridr.core.utils import chunks
from gridr.core.utils.array_utils import array_replace
from gridr.core.utils.array_window import (
    as_rio_window,
    window_from_chunk,
    window_indices,
    window_shape,
    window_shift,
)
from gridr.io.common import GridRIOMode
from gridr.scaling.shmutils import SharedMemoryArray, create_and_register_sma, shmarray_wrap

MASK_BINARY_THRESHOLD = 0.999
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
    geometry_pair: Optional[Tuple[Optional[GeometryType], Optional[GeometryType]]],
    rasterize_kwargs: Optional[Dict] = None,
    mask_out_values: Optional[Tuple[int, int]] = (Validity.VALID, Validity.INVALID),
    io_strip_size: int = DEFAULT_IO_STRIP_SIZE,
    io_strip_size_target: GridRIOMode = GridRIOMode.INPUT,
    ncpu: int = DEFAULT_NCPU,
    cpu_tile_shape: Optional[Tuple[int, int]] = DEFAULT_CPU_TILE_SHAPE,
    logger: Optional[logging.Logger] = None,
) -> int:
    """@doc
    Grid mask computation chain.

    This method wraps the call to the `build_mask` core method with I/O
    resources management and a parallel computation capacity.

    The `build_mask` method's goal is to compute a full-resolution binary mask
    by merging an optional undersampled raster mask with a polygonized vector
    geometries pair.

    Parameters
    ----------
    shape : tuple of int
        The corresponding grid shape ``(nrows, ncols)`` at the grid's
        resolution.

    resolution : tuple of int
        The grid's resolution as an oversampling factor ``(rows, columns)``.

    mask_out_ds : rasterio.io.DatasetWriter
        The output mask as a `DatasetWriter`. Its target size should correspond
        to ``shape`` x ``resolution``.

    mask_out_dtype : numpy.dtypes.Int8DType or numpy.dtypes.UInt8DType
        The NumPy data type to use for encoding the output mask. This should be
        an 8-bit signed or unsigned integer type.

    mask_in_ds : rasterio.io.DatasetReader or None, optional
        An optional input mask as a `DatasetReader`. Its shape and resolution
        must align with the `shape` and `resolution` arguments. Defaults to None.

    mask_in_unmasked_value : int or None, optional
        The integer value within the `mask_in_ds` to consider as **valid**.
        Defaults to None.

    mask_in_band : int or None, optional
        The index of the band from `mask_in_ds` to use as the input mask raster.
        Defaults to None.

    computation_dtype : numpy.dtype
        The data type to use for internal computations, specifically for mask
        interpolation.

    geometry_origin : tuple of float
        The geometric coordinates that map to the output array's ``(0,0)`` pixel.
        This argument is required if `geometry_pair` is provided.

    geometry_pair : tuple of (GeometryType or None), optional
        A tuple containing two optional `GeometryType` elements:
          - The first element: Represents the **valid** geometries.
          - The second element: Represents the **invalid** geometries.

        Defaults to None.

    rasterize_kwargs : dict or None, optional
        A dictionary of parameters to pass directly to the rasterization
        process, e.g., ``{'alg': GridRasterizeAlg.SHAPELY, 'kwargs_alg':
        {'shapely_predicate': ShapelyPredicate.COVERS}}``. These arguments are
        forwarded to the `build_mask` method. Defaults to None.

    mask_out_values : tuple of int or None, optional
        A tuple ``(<unmasked_value>, <masked_value>)`` to use for output.
        If `None`, the default convention used by the `build_mask` method
        (i.e., ``(Validity.VALID, Validity.INVALID)``) is applied.
        Defaults to ``(Validity.VALID, Validity.INVALID)``.

    io_strip_size : int, optional
        The size (in number of rows) of a strip used for I/O operations.
        Defaults to `DEFAULT_IO_STRIP_SIZE`.

    io_strip_size_target : GridRIOMode, optional
        Defines the mode for interpreting the `io_strip_size`.
          - If set to `GridRIOMode.INPUT`, `io_strip_size` directly corresponds
            to the number of rows read from the input buffer.
          - If set to `GridRIOMode.OUTPUT`, `io_strip_size` represents the size
            of the full-resolution output strip, which may result in reading
            fewer rows from the input due to resolution differences.

        Defaults to `GridRIOMode.INPUT`.

    ncpu : int, optional
        The number of worker processes to allocate to the multiprocessing pool.
        Defaults to `DEFAULT_NCPU`.

    cpu_tile_shape : tuple of int or None, optional
        The dimensions of tiles processed by each worker when multiprocessing
        is enabled. This argument must be set and smaller than the output
        shape to activate parallel computation. Defaults to
        `DEFAULT_CPU_TILE_SHAPE`.

    logger : logging.Logger or None, optional
        A Python logger object for logging messages. If `None`, a logger is
        initialized internally. Defaults to None.

    Returns
    -------
    int
        Returns 1 upon successful completion of the mask computation.

    Notes
    -----
    Should you wish further details on the `build_mask` method, please read
    its own documentation.

    **Masked values**
    The `build_mask` core method designates a pixel as invalid (value
    `Validity.INVALID`) if it is masked by the input raster, falls outside the
    **valid** geometry, or lies within the **invalid** geometry.

    This method allows the user to define independently:

        - `mask_out_values`: the values to use as output for unmasked (i.e.,
          valid) and masked (i.e., invalid) data.
        - `mask_in_unmasked_value`: the value to consider as valid in the
          optional input mask.

    If the `mask_in_unmasked_value` differs from the core method convention,
    the method converts the optional input mask to be compliant with it
    directly after read instructions.
    If the `mask_out_umasked_value` differs from the core method convention,
    the method converts the output mask to match the user's input.

    **Read/Write operations**
    I/O for read and write are performed by strip chunks sequentially.
    A strip is defined as a window whose:

        - number of columns is the same as the read raster
        - number of rows is defined by the strip's size.

    Therefore, a first sequential loop is performed independently on each
    strip, consisting in the chaining of the 3 steps: 'read' > 'process' > 'write'.

    The strip's size can either be set to address the read raster size or the
    written raster size (i.e., the computed raster size at each strip).
    The choice is defined through the `io_strip_size_target` argument:

        - set it to `GridRIOMode.INPUT` to address the read strip's buffer size.
        - set it to `GridRIOMode.OUTPUT` to address the write strip's buffer
          size.

    **Parallel processing**
    This method offers parallelization inside each strip through a
    multiprocessing Pool. To activate parallel computing, you have to define:

        - argument `ncpu` to be greater than 1.
        - argument `cpu_tile_shape` to be different from None and smaller than
          the output shape.

    **Shared Memory**
    The read and the output buffers are set at once and used for each strip.
    They are allocated as Shared Memory to be efficiently shared among
    multiple parallel/concurrent processes. Please note that no lock is set
    on written memory; this is justified by the fact that a strict tiling
    computation ensures that no overlap occurs across different processes.

    **Computation Data Type**
    The user needs to provide the data type to use for the interpolation of the
    mask. The precision of the computation may differ between float32 and
    float64.
    """
    # Init a list to register SharedMemoryArray buffers
    sma_buffer_name_list = []
    register_sma = partial(create_and_register_sma, register=sma_buffer_name_list)

    if logger is None:
        logger = logging.getLogger(__name__)

    nrow_in, ncol_in = shape
    logger.debug(f"Grid shape : {nrow_in} rows x {ncol_in} columns")

    if mask_out_values is None:
        mask_out_values = (Validity.VALID, Validity.INVALID)

    if mask_in_ds is not None:
        mask_nrow_in, mask_ncol_in = mask_in_ds.height, mask_in_ds.width
        logger.debug(f"Mask shape : {mask_nrow_in} rows x {mask_ncol_in} " "columns")
        assert nrow_in == mask_nrow_in
        assert ncol_in == mask_ncol_in
    else:
        logger.debug("Mask : no input mask")

    # Cut in strip chunks
    if io_strip_size not in [0, None]:
        if io_strip_size_target == GridRIOMode.INPUT:
            # We have to take into account the grid's resolution along rows
            io_strip_size = io_strip_size * resolution[0]
        elif io_strip_size_target == GridRIOMode.OUTPUT:
            # The strip_size is directly given for the target output
            pass
        else:
            raise ValueError(
                f"Not recognized value {io_strip_size_target} for "
                "the 'io_strip_size_target' argument"
            )

    # Compute the output shape
    shape_out = grid_full_resolution_shape(shape=(nrow_in, ncol_in), resolution=resolution)
    logger.debug(f"Computed output shape : {shape_out[0]} rows x {shape_out[1]}" " columns")

    # Compute strips definitions
    chunk_boundaries = chunks.get_chunk_boundaries(
        nsize=shape_out[0], chunk_size=io_strip_size, merge_last=True
    )

    # Allocate a dest buffer for the rasterio read operation
    # We have to compute read window for each chunk and take the max size

    # First convert chunks to target windows
    # Please note that the last coordinate in each chunk is not contained
    # in the chunk (it corresponds to the index), whereas the window
    # definition contains index that are in the window
    chunk_windows = [np.array([[c0, c1 - 1], [0, shape_out[1] - 1]]) for c0, c1 in chunk_boundaries]

    # Compute the window to read for each chunk window.
    # This will returns both the window to read, and the relative window
    # corresponding to the target chunk window.
    chunk_windows_read = [
        grid_resolution_window(resolution=resolution, win=chunk_win) for chunk_win in chunk_windows
    ]

    # Determine the read buffer shape
    read_buffer_shape = np.max(
        np.asarray([window_shape(read_win) for read_win, rel_win in chunk_windows_read]), axis=0
    )

    # Determine the write buffer shape
    buffer_shape = np.max(
        np.asarray([window_shape(chunk_win) for chunk_win in chunk_windows]), axis=0
    )

    sma_read_buffer = None
    if mask_in_ds is not None:
        # Create shared memory array for read
        sma_read_buffer = register_sma(read_buffer_shape, mask_in_ds.dtypes[mask_in_band - 1])

    # Create shared memory array for output
    sma_write_buffer = register_sma(buffer_shape, mask_out_dtype)

    has_geometry = (geometry_pair is not None) and (geometry_pair != (None, None))
    init_out = not ((sma_read_buffer is not None) or has_geometry)

    try:
        for chunk_idx, (chunk_win, (win_read, win_rel)) in enumerate(
            zip(chunk_windows, chunk_windows_read, strict=True)
        ):

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
                cmask_arr = mask_in_ds.read(
                    mask_in_band,
                    window=as_rio_window(win_read),
                    out=sma_read_buffer.array[0 : cmask_shape[0], 0 : cmask_shape[1]],
                )

            cgeometry_origin = (
                geometry_origin[0] + chunk_win[0][0],
                geometry_origin[1] + chunk_win[1][0],
            )

            logger.debug(f"Chunk {chunk_idx} - shape : {cshape}")
            logger.debug(f"Chunk {chunk_idx} - buffer slices : {cslices}")
            logger.debug(f"Chunk {chunk_idx} - target slices : {cslices_write}")
            logger.debug(f"Chunk {chunk_idx} - build mask starts...")

            # Check valid/masked convention and ensure that the input buffer
            # is complient with the core convention, i.e. Validity.VALID (1) for
            # valid data.
            if cmask_arr is not None and mask_in_unmasked_value != Validity.VALID:
                array_replace(cmask_arr, mask_in_unmasked_value, Validity.VALID, Validity.INVALID)

            # Choose between the tiled multiprocessing branch and the
            # processing branch
            if ncpu > 1 and cpu_tile_shape is not None:
                # Run on multiple cores
                # Cut strip shape into tiled chunks.
                logger.debug(
                    f"Chunk {chunk_idx} - Tiled multiprocessing on "
                    f"{ncpu} workers with tiles of {cpu_tile_shape[0]} x "
                    f"{cpu_tile_shape[1]}"
                )

                chunk_tiles = chunks.get_chunk_shapes(cshape, cpu_tile_shape, merge_last=True)

                logger.debug(
                    f"Chunk {chunk_idx} - Number of tiles to process :" f" {len(chunk_tiles)}"
                )

                # Init the list of process arguments as 'tasks'
                tasks = []
                for tile in chunk_tiles:
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} " "- preparing args...")

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
                    tile_origin = chunk_win[..., 0]
                    tile_win = window_from_chunk(chunk=tile, origin=None)
                    tile_geometry_origin = (
                        cgeometry_origin[0] + tile_win[0][0],
                        cgeometry_origin[1] + tile_win[1][0],
                    )

                    tile_shape = window_shape(tile_win)
                    tile_slice = window_indices(tile_win, reset_origin=False)
                    tile_mask_in_target_win = window_shift(tile_win, win_rel[:, 0])

                    logger.debug(
                        f"Chunk {chunk_idx} - tile {tile} - " f"tile's chunk origin : {tile_origin}"
                    )
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - " f"tile window : {tile_win}")
                    logger.debug(
                        f"Chunk {chunk_idx} - tile {tile} - "
                        f"mask_in_target_win : {tile_mask_in_target_win} "
                        f"from strip target win : {win_rel}"
                    )

                    # Manage shared memory to pass to the process
                    # - tile_smb_out : a SharedMemoryBuffer object to pass for
                    #        output buffer by cloning the caracteristics of the
                    #        main output buffer except the slice
                    # - tile_smb_maks_in : a SharedMemoryBuffer object to pass for
                    #        read buffer by cloning the caracteristics of the
                    #        main output buffer. The shift for the current tile
                    #        is applied through the 'mask_in_target_win' arg.
                    tile_smb_out = SharedMemoryArray.clone(
                        sma=sma_write_buffer, array_slice=tile_slice
                    )

                    tile_smb_mask_in = None
                    if sma_read_buffer is not None:
                        tile_smb_mask_in = SharedMemoryArray.clone(
                            sma=sma_read_buffer, array_slice=None
                        )

                    # Append process parameters to 'tasks'
                    tasks.append(
                        {
                            "shape": tile_shape,
                            "resolution": (1, 1),
                            "out": tile_smb_out,
                            "geometry_origin": tile_geometry_origin,
                            "geometry_pair": geometry_pair,
                            "mask_in": tile_smb_mask_in,
                            "mask_in_target_win": tile_mask_in_target_win,
                            "mask_in_resolution": resolution,
                            "oversampling_dtype": computation_dtype,
                            "mask_in_binary_threshold": MASK_BINARY_THRESHOLD,
                            "rasterize_kwargs": rasterize_kwargs,
                            "init_out": init_out,
                        }
                    )

                with Pool(processes=ncpu) as pool:
                    pool.map(build_mask_tile_worker, tasks)

            else:
                # Build mask on full strip - no multiprocessing
                logger.debug(f"Chunk {chunk_idx} - Full strip computation " "(no tiling)")

                _ = build_mask(
                    shape=cshape,
                    resolution=(1, 1),
                    out=sma_write_buffer.array[cslices],
                    geometry_origin=cgeometry_origin,  #: Tuple[float, float],
                    geometry_pair=geometry_pair,
                    mask_in=cmask_arr,
                    mask_in_target_win=win_rel,
                    mask_in_resolution=resolution,
                    oversampling_dtype=computation_dtype,
                    mask_in_binary_threshold=MASK_BINARY_THRESHOLD,
                    rasterize_kwargs=rasterize_kwargs,
                    init_out=init_out,
                )

            logger.debug(f"Chunk {chunk_idx} - build mask ends.")

            # Check masked/unmasked convention and ensure that the output buffer
            # is complient with the user's given convention.
            if not np.all(mask_out_values == (Validity.VALID, Validity.INVALID)):
                val_cond = Validity.VALID  # considered true in build_mask method
                val_true, val_false = mask_out_values
                array_replace(sma_write_buffer.array[cslices], val_cond, val_true, val_false)
                # sma_write_buffer.array[*cslices] = np.where(
                #        sma_write_buffer.array[*cslices]==0,
                #        mask_out_valid_value, ~mask_out_valid_value)

            # Write the data
            if mask_out_ds is not None:
                logger.debug(f"Chunk {chunk_idx} - write mask...")

                mask_out_ds.write(
                    sma_write_buffer.array[cslices], 1, window=as_rio_window(chunk_win)
                )

                logger.debug(f"Chunk {chunk_idx} - write ends.")
    except Exception:
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
    geometry_pair: Optional[Tuple[Optional[GeometryType], Optional[GeometryType]]],
    rasterize_kwargs: Optional[Dict] = None,
    mask_out_values: Optional[Tuple[int, int]] = (Validity.VALID, Validity.INVALID),
    merge_mask_grid: Optional[Union[int, float]] = None,
    io_strip_size: int = DEFAULT_IO_STRIP_SIZE,
    io_strip_size_target: GridRIOMode = GridRIOMode.INPUT,
    ncpu: int = DEFAULT_NCPU,
    cpu_tile_shape: Optional[Tuple[int, int]] = DEFAULT_CPU_TILE_SHAPE,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Grid and mask computation chain.

    This method orchestrates both grid generation and mask computation,
    wrapping calls to the `build_grid` and `build_mask` core methods.
    It includes comprehensive I/O resource management and parallel
    computation capabilities.

    Parameters
    ----------
    resolution : tuple of int
        The grid's resolution as an oversampling factor ``(rows, columns)``.

    grid_in_ds : rasterio.io.DatasetReader
        The input dataset for reading row coordinates of the grid. If
        `grid_in_col_ds` is provided, this argument is solely used for
        reading row coordinates. Otherwise, it's used for both row and
        column coordinates. Bands are specified by `grid_in_row_coords_band`
        and `grid_in_col_coords_band`.

    grid_in_col_ds : rasterio.io.DatasetReader or None, optional
        An optional separate `DatasetReader` for reading grid column
        coordinates. If `None`, `grid_in_ds` is used for both. Defaults to None.

    grid_in_row_coords_band : int
        The 1-based index of the row coordinates band within the corresponding
        input `DatasetReader`.

    grid_in_col_coords_band : int
        The 1-based index of the column coordinates band within the
        corresponding input `DatasetReader`.

    grid_out_ds : rasterio.io.DatasetWriter
        The output dataset for writing row coordinates of the grid. If
        `grid_out_col_ds` is provided, this argument is solely used for
        writing row coordinates. Otherwise, it's used for both. Bands are
        specified by `grid_out_row_coords_band` and `grid_out_col_coords_band`.

    grid_out_col_ds : rasterio.io.DatasetWriter or None, optional
        An optional separate `DatasetWriter` for writing grid column
        coordinates. If `None`, `grid_out_ds` is used for both. Defaults to None.

    grid_out_row_coords_band : int
        The 1-based index of the row coordinates band within the corresponding
        output `DatasetWriter`.

    grid_out_col_coords_band : int
        The 1-based index of the column coordinates band within the
        corresponding output `DatasetWriter`.

    mask_out_ds : rasterio.io.DatasetWriter
        The output mask as a `DatasetWriter`. Its target size should correspond
        to ``shape`` x ``resolution`` (where ``shape`` is derived from `grid_in_ds`).

    mask_out_dtype : numpy.dtypes.Int8DType or numpy.dtypes.UInt8DType
        The NumPy data type to use for encoding the output mask. This should be
        an 8-bit signed or unsigned integer type.

    mask_in_ds : rasterio.io.DatasetReader or None, optional
        An optional input mask as a `DatasetReader`. Its shape and resolution
        must align with the `grid_in_ds` and `resolution` arguments.
        Defaults to None.

    mask_in_unmasked_value : int or None, optional
        The integer value within the `mask_in_ds` to consider as **valid**.
        Defaults to None.

    mask_in_band : int or None, optional
        The index of the band from `mask_in_ds` to use as the input mask raster.
        Defaults to None.

    computation_dtype : numpy.dtype
        The data type to use for internal computations, specifically for mask
        interpolation.

    geometry_origin : tuple of float
        The geometric coordinates that map to the output array's ``(0,0)`` pixel.
        This argument is required if `geometry_pair` is provided.

    geometry_pair : tuple of (GeometryType or None), optional
        A tuple containing two optional `GeometryType` elements:
          - The first element: Represents the **valid** geometries.
          - The second element: Represents the **invalid** geometries.

        Defaults to None.

    rasterize_kwargs : dict or None, optional
        A dictionary of parameters to pass directly to the rasterization
        process, e.g., ``{'alg': GridRasterizeAlg.SHAPELY, 'kwargs_alg':
        {'shapely_predicate': ShapelyPredicate.COVERS}}``. These arguments are
        forwarded to the `build_mask` method. Defaults to None.

    mask_out_values : tuple of int or None, optional
        A tuple ``(<unmasked_value>, <masked_value>)`` to use for output.
        If `None`, the default convention used by the `build_mask` method
        (i.e., ``(Validity.VALID, Validity.INVALID)``) is applied.
        Defaults to ``(Validity.VALID, Validity.INVALID)``.

    merge_mask_grid : int or float or None, optional
        The value to use for filling in masked data within the output grid.
        If `None`, the mask is not merged into the grid. Defaults to None.

    io_strip_size : int, optional
        The size (in number of rows) of a strip used for I/O operations.
        Defaults to `DEFAULT_IO_STRIP_SIZE`.

    io_strip_size_target : GridRIOMode, optional
        Defines the mode for interpreting the `io_strip_size`.
          - If set to `GridRIOMode.INPUT`, `io_strip_size` directly corresponds
            to the number of rows read from the input buffer.
          - If set to `GridRIOMode.OUTPUT`, `io_strip_size` represents the size
            of the full-resolution output strip, which may result in reading
            fewer rows from the input due to resolution differences.

        Defaults to `GridRIOMode.INPUT`.

    ncpu : int, optional
        The number of worker processes to allocate to the multiprocessing pool.
        Defaults to `DEFAULT_NCPU`.

    cpu_tile_shape : tuple of int or None, optional
        The dimensions of tiles processed by each worker when multiprocessing
        is enabled. This argument must be set and smaller than the output
        shape to activate parallel computation. Defaults to
        `DEFAULT_CPU_TILE_SHAPE`.

    logger : logging.Logger or None, optional
        A Python logger object for logging messages. If `None`, a logger is
        initialized internally. Defaults to None.

    Returns
    -------
    int
        Returns 1 upon successful completion of the grid and mask computation.

    Notes
    -----
    This method serves as a high-level orchestrator for both grid generation
    (via `build_grid`) and mask computation (via `build_mask`). For detailed
    information on the underlying `build_grid` and `build_mask` functionalities,
    please refer to their respective documentations.

    **Masked values**
    The `build_mask` core method designates a pixel as invalid (value
    `Validity.INVALID`) if it is masked by the input raster, falls outside the
    **valid** geometry, or lies within the **invalid** geometry.

    This method allows the user to define independently:

        - `mask_out_values`: the values to use as output for unmasked (i.e.,
          valid) and masked (i.e., invalid) data.
        - `mask_in_unmasked_value`: the value to consider as valid in the
          optional input mask.

    If the `mask_in_unmasked_value` differs from the core method convention,
    the method converts the optional input mask to be compliant with it
    directly after read instructions.
    If the `mask_out_umasked_value` differs from the core method convention,
    the method converts the output mask to match the user's input.

    **Merging the mask in the grid**
    This method provides an option to insert the computed mask directly into
    the output grid by assigning a special value to masked pixels. To activate
    this feature, set the `merge_mask_grid` parameter to a non-None value.
    This value will then be used to fill the grid at all masked coordinates.

    **Read/Write operations**
    I/O for read and write are performed by strip chunks sequentially.
    A strip is defined as a window whose:
    - number of columns is the same as the read raster
    - number of rows is defined by the strip's size.

    Therefore, a first sequential loop is performed independently on each
    strip, consisting in the chaining of the 3 steps: 'read' > 'process' > 'write'.

    The strip's size can either be set to address the read raster size or the
    written raster size (i.e., the computed raster size at each strip).
    The choice is defined through the `io_strip_size_target` argument:

        - set it to `GridRIOMode.INPUT` to address the read strip's buffer size.
        - set it to `GridRIOMode.OUTPUT` to address the write strip's buffer size.

    **Parallel processing**
    This method offers parallelization inside each strip through a
    multiprocessing Pool. To activate parallel computing, you have to define:

        - argument `ncpu` to be greater than 1.
        - argument `cpu_tile_shape` to be different from None and smaller than
          the output shape.

    **Shared Memory**
    The read and the output buffers are set at once and used for each strip.
    They are allocated as Shared Memory to be efficiently shared among
    multiple parallel/concurrent processes. Please note that no lock is set
    on written memory; this is justified by the fact that a strict tiling
    computation ensures that no overlap occurs across different processes.

    **Computation Data Type**
    The user needs to provide the data type to use for the interpolation of the
    mask. The precision of the computation may differ between float32 and
    float64.
    """
    # Init a list to register SharedMemoryArray buffers
    sma_buffer_name_list = []
    register_sma = partial(create_and_register_sma, register=sma_buffer_name_list)

    if logger is None:
        logger = logging.getLogger(__name__)

    # Get input shape from input grid
    nrow_in, ncol_in = grid_in_ds.height, grid_in_ds.width
    logger.debug(f"Grid shape : {nrow_in} rows x {ncol_in} columns")

    grid_in_row_ds = grid_in_ds
    if grid_in_col_ds is None:
        grid_in_col_ds = grid_in_ds
        assert grid_in_row_coords_band != grid_in_col_coords_band
    else:
        # Test shapes are the same
        assert nrow_in == grid_in_col_ds.height
        assert ncol_in == grid_in_col_ds.width

    # Check output DatasetWriter definitions
    grid_out_row_ds = grid_out_ds
    if grid_out_col_ds is None:
        grid_out_col_ds = grid_out_ds
        assert grid_out_row_coords_band != grid_out_col_coords_band
    else:
        assert grid_out_col_ds.height == grid_out_col_ds.height
        assert grid_out_row_ds.width == grid_out_row_ds.width

    # Check mask shape
    if mask_in_ds is not None:
        mask_nrow_in, mask_ncol_in = mask_in_ds.height, mask_in_ds.width
        logger.debug(f"Mask shape : {mask_nrow_in} rows x {mask_ncol_in} " "columns")
        assert nrow_in == mask_nrow_in
        assert ncol_in == mask_ncol_in
    else:
        logger.debug("Mask : no input mask")

    if mask_out_values is None:
        mask_out_values = (Validity.VALID, Validity.INVALID)

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
            raise ValueError(
                f"Not recognized value {io_strip_size_target} for "
                "the 'io_strip_size_target' argument"
            )

    # Compute the output shape
    shape_out = grid_full_resolution_shape(shape=(nrow_in, ncol_in), resolution=resolution)
    logger.debug(f"Computed output shape : {shape_out[0]} rows x {shape_out[1]}" " columns")

    # Compute strips definitions
    chunk_boundaries = chunks.get_chunk_boundaries(
        nsize=shape_out[0], chunk_size=io_strip_size, merge_last=True
    )

    # Allocate a dest buffer for the rasterio read operation
    # We have to compute read window for each chunk and take the max size

    # First convert chunks to target windows
    # Please note that the last coordinate in each chunk is not contained
    # in the chunk (it corresponds to the index), whereas the window
    # definition contains index that are in the window
    chunk_windows = [np.array([[c0, c1 - 1], [0, shape_out[1] - 1]]) for c0, c1 in chunk_boundaries]

    # Compute the window to read for each chunk window.
    # This will returns both the window to read, and the relative window
    # corresponding to the target chunk window.
    chunk_windows_read = [
        grid_resolution_window(resolution=resolution, win=chunk_win) for chunk_win in chunk_windows
    ]

    # Determine the read buffer shape
    read_buffer_shape = np.max(
        np.asarray([window_shape(read_win) for read_win, rel_win in chunk_windows_read]), axis=0
    )
    read_buffer_shape3 = np.insert(read_buffer_shape, 0, 2)

    # Create shared memory array for output
    buffer_shape = np.max(
        np.asarray([window_shape(chunk_win) for chunk_win in chunk_windows]), axis=0
    )
    buffer_shape3 = np.insert(buffer_shape, 0, 2)

    # Create shared memory array for read
    # input grid
    sma_r_buffer_grid = register_sma(
        read_buffer_shape3, grid_in_row_ds.dtypes[grid_in_row_coords_band - 1]
    )

    # - mask in
    sma_r_buffer_mask = None
    if mask_in_ds is not None:
        sma_r_buffer_mask = register_sma(read_buffer_shape, mask_in_ds.dtypes[mask_in_band - 1])

    # Create shared memory array for write
    sma_w_buffer_grid = register_sma(
        buffer_shape3, grid_out_row_ds.dtypes[grid_out_row_coords_band - 1]
    )

    # - mask out
    compute_mask = False
    sma_w_buffer_mask = None

    has_geometry = (geometry_pair is not None) and (geometry_pair != (None, None))
    # Determine if mask must be computed
    if (mask_out_ds is not None) or (mask_in_ds is not None) or has_geometry:
        compute_mask = True

    # Determine if output buffer has to be initialized
    init_out = not (compute_mask or has_geometry)

    # if mask_out_ds is not None or mask_out_dtype is not None:
    if compute_mask:
        sma_w_buffer_mask = register_sma(buffer_shape, mask_out_dtype)

    try:
        for chunk_idx, (chunk_win, (win_read, win_rel)) in enumerate(
            zip(chunk_windows, chunk_windows_read, strict=True)
        ):

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
            _ = grid_in_row_ds.read(
                grid_in_row_coords_band,
                window=as_rio_window(win_read),
                out=sma_r_buffer_grid.array[0, 0 : cread_shape[0], 0 : cread_shape[1]],
            )

            _ = grid_in_col_ds.read(
                grid_in_col_coords_band,
                window=as_rio_window(win_read),
                out=sma_r_buffer_grid.array[1, 0 : cread_shape[0], 0 : cread_shape[1]],
            )

            cread_grid_arr = sma_r_buffer_grid.array[:, 0 : cread_shape[0], 0 : cread_shape[1]]

            # Input mask if given
            cread_mask_arr = None
            if compute_mask:
                if sma_r_buffer_mask is not None:
                    cread_mask_arr = mask_in_ds.read(
                        mask_in_band,
                        window=as_rio_window(win_read),
                        out=sma_r_buffer_mask.array[0 : cread_shape[0], 0 : cread_shape[1]],
                    )

            # Shift geometry origin for the strip
            cgeometry_origin = (
                geometry_origin[0] + chunk_win[0][0],
                geometry_origin[1] + chunk_win[1][0],
            )

            logger.debug(f"Chunk {chunk_idx} - shape : {cshape}")
            logger.debug(f"Chunk {chunk_idx} - buffer slices : {cslices}")
            logger.debug(f"Chunk {chunk_idx} - target slices : {cslices_write}")
            logger.debug(f"Chunk {chunk_idx} - build starts...")

            # Mask management
            if cread_mask_arr is not None:
                # Check valid/masked convention and ensure that the input buffer
                # is complient with the core convention, i.e. Validity.VALID (1)
                # for valid data.
                if mask_in_unmasked_value != Validity.VALID:
                    array_replace(
                        cread_mask_arr, mask_in_unmasked_value, Validity.VALID, Validity.INVALID
                    )

            # Choose between the tiled multiprocessing branch and the
            # processing branch
            if ncpu > 1 and cpu_tile_shape is not None:
                # Run on multiple cores
                # Cut strip shape into tiled chunks.
                logger.debug(
                    f"Chunk {chunk_idx} - Tiled multiprocessing on "
                    f"{ncpu} workers with tiles of {cpu_tile_shape[0]} x "
                    f"{cpu_tile_shape[1]}"
                )

                chunk_tiles = chunks.get_chunk_shapes(cshape, cpu_tile_shape, merge_last=True)

                logger.debug(
                    f"Chunk {chunk_idx} - Number of tiles to process :" f" {len(chunk_tiles)}"
                )

                # Init the list of process arguments as 'tasks'
                tasks = []
                for tile in chunk_tiles:
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} " "- preparing args...")

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
                    tile_origin = chunk_win[..., 0]
                    tile_win = window_from_chunk(chunk=tile, origin=None)
                    tile_geometry_origin = (
                        cgeometry_origin[0] + tile_win[0][0],
                        cgeometry_origin[1] + tile_win[1][0],
                    )

                    tile_shape = window_shape(tile_win)
                    tile_slice = window_indices(tile_win, reset_origin=False)
                    tile_slices3 = (slice(None, None),) + tile_slice
                    tile_target_win = window_shift(tile_win, win_rel[:, 0])

                    logger.debug(
                        f"Chunk {chunk_idx} - tile {tile} - " f"tile's chunk origin : {tile_origin}"
                    )
                    logger.debug(f"Chunk {chunk_idx} - tile {tile} - " f"tile window : {tile_win}")
                    logger.debug(
                        f"Chunk {chunk_idx} - tile {tile} - "
                        f"target_win : {tile_target_win} "
                        f"from strip target win : {win_rel}"
                    )

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
                        sma=sma_w_buffer_grid, array_slice=tile_slices3
                    )

                    tile_smb_grid_in = SharedMemoryArray.clone(
                        sma=sma_r_buffer_grid, array_slice=None
                    )

                    build_grid_args = {
                        "resolution": (1, 1),
                        "grid": tile_smb_grid_in,
                        "grid_target_win": tile_target_win,
                        "grid_resolution": resolution,
                        "computation_dtype": computation_dtype,
                        "out": tile_smb_grid_out,
                    }

                    build_mask_args = {}
                    if compute_mask:
                        # mask
                        tile_smb_mask_out = SharedMemoryArray.clone(
                            sma=sma_w_buffer_mask, array_slice=tile_slice
                        )

                        tile_smb_mask_in = None
                        if sma_r_buffer_mask is not None:
                            tile_smb_mask_in = SharedMemoryArray.clone(
                                sma=sma_r_buffer_mask, array_slice=None
                            )

                        # Append process parameters to 'tasks'
                        build_mask_args = {
                            "shape": tile_shape,
                            "resolution": (1, 1),
                            "out": tile_smb_mask_out,
                            "geometry_origin": tile_geometry_origin,
                            "geometry_pair": geometry_pair,
                            "mask_in": tile_smb_mask_in,
                            "mask_in_target_win": tile_target_win,
                            "mask_in_resolution": resolution,
                            "mask_in_binary_threshold": MASK_BINARY_THRESHOLD,
                            "rasterize_kwargs": rasterize_kwargs,
                            "oversampling_dtype": computation_dtype,
                            "init_out": init_out,
                        }

                    tasks.append((build_mask_args, build_grid_args))

                with Pool(processes=ncpu) as pool:
                    pool.map(build_grid_mask_tile_worker, tasks)

            # Mono processing for now
            else:
                # Build mask on full strip - no multiprocessing
                logger.debug(f"Chunk {chunk_idx} - Full strip computation " "(no tiling)")

                if compute_mask:
                    _ = build_mask(
                        shape=cshape,
                        resolution=(1, 1),
                        out=sma_w_buffer_mask.array[cslices],
                        geometry_origin=cgeometry_origin,
                        geometry_pair=geometry_pair,
                        mask_in=cread_mask_arr,
                        mask_in_target_win=win_rel,
                        mask_in_resolution=resolution,
                        mask_in_binary_threshold=MASK_BINARY_THRESHOLD,
                        rasterize_kwargs=rasterize_kwargs,
                        oversampling_dtype=computation_dtype,
                        init_out=init_out,
                    )

                # Process grid
                _ = build_grid(
                    resolution=(1, 1),
                    grid=cread_grid_arr,
                    grid_target_win=win_rel,
                    grid_resolution=resolution,
                    computation_dtype=computation_dtype,
                    out=sma_w_buffer_grid.array[cslices3],
                )

            logger.debug(f"Chunk {chunk_idx} - build mask ends.")

            # Here we merge
            if compute_mask and merge_mask_grid is not None:
                # mask_indices = sma_w_buffer_mask.array[cslices] == 1
                # sma_w_buffer_grid.array[cslices3][:, mask_indices] \
                #        = merge_mask_grid
                # array_replace(array=sma_w_buffer_grid.array[cslices3],
                #        val_cond=0, val_true=merge_mask_grid, val_false=None,
                #        array_cond=sma_w_buffer_mask.array[cslices],
                #        array_cond_val=1, win=None)
                array_replace(
                    array=sma_w_buffer_grid.array[0][cslices],
                    val_cond=0,
                    val_true=merge_mask_grid,
                    val_false=None,
                    array_cond=sma_w_buffer_mask.array[cslices],
                    array_cond_val=Validity.INVALID,
                    win=None,
                )
                array_replace(
                    array=sma_w_buffer_grid.array[1][cslices],
                    val_cond=0,
                    val_true=merge_mask_grid,
                    val_false=None,
                    array_cond=sma_w_buffer_mask.array[cslices],
                    array_cond_val=Validity.INVALID,
                    win=None,
                )

            # Check masked/unmasked convention and ensure that the output buffer
            # is complient with the user's given convention.
            if compute_mask and not np.all(mask_out_values == (Validity.VALID, Validity.INVALID)):
                val_cond = Validity.VALID  # considered true in build_mask method
                val_true, val_false = mask_out_values
                array_replace(sma_w_buffer_mask.array[cslices], val_cond, val_true, val_false)

            # Write the data
            # Write grid rows
            logger.debug(f"Chunk {chunk_idx} - write grid rows...")

            grid_out_row_ds.write(
                sma_w_buffer_grid.array[0][cslices],
                grid_out_row_coords_band,
                window=as_rio_window(chunk_win),
            )

            logger.debug(f"Chunk {chunk_idx} - write ends.")

            # Write grid columns
            logger.debug(f"Chunk {chunk_idx} - write grid columns...")

            grid_out_col_ds.write(
                sma_w_buffer_grid.array[1][cslices],
                grid_out_col_coords_band,
                window=as_rio_window(chunk_win),
            )

            logger.debug(f"Chunk {chunk_idx} - write ends.")

            if compute_mask and mask_out_ds is not None:
                logger.debug(f"Chunk {chunk_idx} - write mask...")

                mask_out_ds.write(
                    sma_w_buffer_mask.array[cslices], 1, window=as_rio_window(chunk_win)
                )

                logger.debug(f"Chunk {chunk_idx} - write ends.")

    except Exception:
        raise

    finally:
        # Release the Shared memory buffer
        SharedMemoryArray.clear_buffers(sma_buffer_name_list)

    return 1
