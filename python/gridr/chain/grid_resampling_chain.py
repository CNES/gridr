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
Module for a Grid Resampling Chain
# @doc
"""
import logging
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio

from gridr.cdylib import PyGridGeometriesMetricsF64
from gridr.core.grid.grid_commons import grid_full_resolution_shape, grid_resolution_window_safe
from gridr.core.grid.grid_mask import Validity, build_mask
from gridr.core.grid.grid_rasterize import GeometryType, GridRasterizeAlg
from gridr.core.grid.grid_resampling import array_grid_resampling
from gridr.core.grid.grid_utils import array_compute_resampling_grid_geometries
from gridr.core.utils import chunks
from gridr.core.utils.array_utils import ArrayProfile, array_convert, array_replace
from gridr.core.utils.array_window import (
    as_rio_window,
    window_check,
    window_extend,
    window_from_chunk,
    window_from_indices,
    window_indices,
    window_overflow,
    window_shape,
    window_shift,
)
from gridr.io.common import GridRIOMode
from gridr.scaling.shmutils import SharedMemoryArray, create_and_register_sma

DEFAULT_IO_STRIP_SIZE = 1000
DEFAULT_TILE_SHAPE = (1000, 1000)
DEFAULT_NCPU = 1

READ_TILE_MIN_SIZE = (1000, 1000)

GEOMETRY_RASTERIZE_KWARGS = {"alg": GridRasterizeAlg.RASTERIO_RASTERIZE}


def read_win_from_grid_metrics(
    grid_metrics: PyGridGeometriesMetricsF64,
    array_src_profile_2d: ArrayProfile,
    margins: np.ndarray,
    logger: logging.Logger,
    logger_msg_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the source read window from grid metrics.

    This function determines the read window (`src_win_read`) from the
    source array based on the destination and source bounds contained in
    `grid_metrics`. It applies the necessary margins to ensure sufficient
    neighborhood data is available for operations such as interpolation,
    while handling cases where the window exceeds the array boundaries.

    Parameters
    ----------
    grid_metrics : PyGridGeometriesMetricsF64
        Object containing geometric metrics of the grid, including source and
        destination bounds.

    array_src_profile_2d : ArrayProfile
        Metadata/profile of the 2D source array, including shape and number of
        dimensions information.

    margins : numpy.ndarray of shape (2, 2)
        Margins to apply to the read window, formatted as
        ``[[top_margin, bottom_margin], [left_margin, right_margin]]``.

    logger : logging.Logger
        Logger instance used for debugging messages.

    logger_msg_prefix : str, optional
        Optional prefix to include in log messages for better traceability.
        Defaults to ''.

    Returns
    -------
    src_win_read : numpy.ndarray of shape (2, 2)
        Final source read window adjusted for margins and boundary constraints.
        Format: ``[[row_start, row_end], [col_start, col_end]]``. This is the
        window that should be used for the raster read IO.

    src_win_marged : numpy.ndarray of shape (2, 2)
        Desired read window with margins applied, before boundary correction
        (padding).

    pad : numpy.ndarray of shape (2, 2)
        Amount of padding required if the marged window extends outside the
        source array. Format: ``[[top_pad, bottom_pad], [left_pad, right_pad]]``.

    Notes
    -----
    This function is critical for operations requiring contextual data beyond
    the immediate processing area, such as filtering or interpolation, ensuring
    that no data loss or edge artifacts occur due to insufficient neighborhood
    information.
    """

    def DEBUG(msg):
        logger.debug(f"{logger_msg_prefix} - {msg}")

    # The metrics have been computed, ie there is at least one
    # valid point in the grid regarding the masking options.
    # We can get the dst and src window :
    # - the dst bounds are given relative to the current strip
    #   low resolute shape.
    # - the src bounds are absolute coordinates value in the
    #   input array.
    dst_lowres_bounds = grid_metrics.dst_bounds
    src_bounds = grid_metrics.src_bounds

    DEBUG(f"dst low res bounds : {dst_lowres_bounds} ")
    DEBUG(f"src bounds : {src_bounds} ")

    # Define the strip low res computing window (upper limit included)
    dst_lowres_win = np.array(
        (
            (dst_lowres_bounds.ymin, dst_lowres_bounds.ymax),
            (dst_lowres_bounds.xmin, dst_lowres_bounds.xmax),
        )
    )
    DEBUG(f"dst win : {dst_lowres_win}")

    src_win_read = None
    src_win_marged = None
    pad = None

    if (
        (src_bounds.ymin < 0 and src_bounds.ymax < 0)
        or (src_bounds.xmin < 0 and src_bounds.xmax < 0)
        or (
            src_bounds.ymin > array_src_profile_2d.shape[0] - 1
            and src_bounds.ymax > array_src_profile_2d.shape[0] - 1
        )
        or (
            src_bounds.xmin > array_src_profile_2d.shape[1] - 1
            and src_bounds.xmax > array_src_profile_2d.shape[1] - 1
        )
    ):
        # The source is not readable from raster - fully outside for at least
        # one direction.
        DEBUG(
            "The source read window is not available ; it goes fully "
            "outside of raster in one direction at least"
        )

    else:
        # Define the input read window
        src_win = np.array(
            (
                (int(np.floor(src_bounds.ymin)), int(np.ceil(src_bounds.ymax))),
                (int(np.floor(src_bounds.xmin)), int(np.ceil(src_bounds.xmax))),
            )
        )
        DEBUG(f"src read win (preliminary) : {src_win}")

        # Here we got a preliminary read window, but :
        # - That window may overflow (ie. adress coordinates outside
        #   of the raster
        # - We have to consider some margins :
        #   -- margin required for the interpolation kernel
        #   -- margin required for spline interpolation preprocessing.
        #   -- margin that may be required for other processing (egg.
        #      antialiasing filtering)
        #   A marged window may also overflow but we may have to
        #   manage edges here : the passed array has to cover for
        #   margins.
        #
        # Strategy :
        # 1. compute overflow of the preliminary window and limit it
        #    to the available region
        # 2. Add margins
        # 3. Compute overflow of the marged window
        # 4. Read the valid window and perform edge management if
        #    needed

        # 1. compute overflow
        src_win_overflow = window_overflow(array_src_profile_2d, src_win)
        DEBUG(f"src read win (preliminary) overflow : {src_win_overflow}")

        # 2. compute marged window
        # 2.1 first crop the overflow if any
        src_win_marged = window_extend(src_win, src_win_overflow, reverse=True)
        DEBUG(f"src read win before margin : {src_win_marged}")

        # 2.2 apply the margin
        src_win_marged = window_extend(src_win_marged, margins, reverse=False)
        DEBUG(f"src read win after margin : {src_win_marged}")

        # 3. Compute overflow of the marged window
        src_win_marged_overflow = window_overflow(array_src_profile_2d, src_win_marged)
        DEBUG(f"src read win required pad : {src_win_marged_overflow}")

        # `cstrip_read_win` corresponds to the window to read from src array
        src_win_read = window_extend(src_win_marged, src_win_marged_overflow, reverse=True)
        src_win_read_shape = window_shape(src_win_read)
        DEBUG(f"src read win read : {src_win_read} with shape {src_win_read_shape}")

        pad = np.array([[0, 0], [0, 0]])
        if not np.all(src_win_marged_overflow == 0):
            pad = src_win_marged_overflow

    return src_win_read, src_win_marged, pad


def basic_grid_resampling_array(
    interp: str,
    grid_arr: np.ndarray,
    grid_arr_shape: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    grid_nodata: Optional[Union[int, float]],
    grid_mask_arr: np.ndarray,
    grid_mask_in_unmasked_value: np.uint8,
    array_src_ds: rasterio.io.DatasetReader,
    array_src_bands: Union[int, List[int]],
    array_src_mask_ds: Optional[rasterio.io.DatasetReader],
    array_src_mask_band: Optional[int],
    array_src_mask_validity_pair: Optional[Tuple[int, int]],
    array_src_geometry_origin: Optional[Tuple[float, float]],
    array_src_geometry_pair: Optional[Tuple[Optional[GeometryType], Optional[GeometryType]]],
    oversampled_grid_win: np.ndarray,
    margin: np.ndarray,
    sma_out_buffer: SharedMemoryArray,
    out_win: np.ndarray,
    nodata_out: Optional[Union[int, float]],
    sma_out_mask_buffer: Optional[SharedMemoryArray],
    logger_msg_prefix: str,
    logger: logging.Logger,
):
    """Resamples source data into a target oversampled window on a grid.

    This method processes a 3D raster grid (`grid_arr`) that includes row and
    column coordinates. The `grid_arr` may represent a sub-region of a larger,
    non-oversampled grid. Additionally, since `grid_arr` might correspond to a
    uniquely allocated buffer used for multiple sub-regions, we cannot rely on
    its `shape` attribute to determine the dimensions of the sub-region.
    Therefore, the `grid_arr_shape` argument is required.

    Parameters
    ----------
    interp : str
        The interpolator.

    grid_arr : numpy.ndarray
        A 3D array of shape ``(2, rows, cols)``, containing the raster grid's
        row and column coordinates. It may represent a sub-region of a larger
        (non-oversampled) grid. The local origin ``(0, 0)`` corresponds to
        ``grid_arr[:, 0, 0]``.

    grid_arr_shape : tuple of int
        Shape of the active sub-region in `grid_arr`, given as ``(rows, cols)``.
        Required because `grid_arr` may be a larger buffer reused across
        tiles or subregions.

    grid_resolution : tuple of int
        Resolution of the coarse grid, typically in pixels or map units per
        pixel (e.g., ``(10, 10)``).

    grid_nodata : scalar or None
        The NoData value associated with `grid_arr`, marking invalid or
        missing data points.

    grid_mask_arr : numpy.ndarray, optional
        Optional 2D `uint8` or `int8` mask array aligned with `grid_arr`
        (shape: ``(rows, cols)``).
        Indicates valid (unmasked) and invalid (masked) data. Defaults to None.

    grid_mask_in_unmasked_value : numpy.uint8
        Value in `grid_mask_arr` that represents a valid/unmasked data point.

    array_src_ds : rasterio.io.DatasetReader
        The source dataset (e.g., a GDAL or Rasterio object) from which
        raster data will be read and resampled.

    array_src_bands : int or list of int
        List of band indices to read from `array_src_ds`. If a single band is
        provided, it can be an integer.

    array_src_mask_ds : rasterio.io.DatasetReader or None, optional
        Optional dataset representing the mask associated with `array_src_ds`.
        Defaults to None.

    array_src_mask_band : int or None, optional
        Band index to read from `array_src_mask_ds` for the source mask.
        Defaults to None.

    array_src_mask_validity_pair : tuple of int, optional
        A tuple containing two integer :
          - The first integer corresponds to the value to consider as valid in
            the mask array.
          - The second integer corresponds to the value to consider as invalid
            in the mask array.

        If the tuple differs from (`Validity.VALID`, `Validity.INVALID`) a
        replace operation will be performed in order to make the mask compliant
        with the core resampling method.

    array_src_geometry_origin : tuple of float or None, optional
        This optional parameter specifies the origin convention for the
        `array_src_geometry_pair` definition. GridR uses a ``(0, 0)`` image
        coordinate system to address the first pixel of the `array_src`
        raster. This parameter allows you to align the `array_src_geometry_pair`
        definition with GridR's convention, ensuring proper spatial
        referencing. Please note, its internal usage is solely for modifying
        `array_src_geometry_pair`. Defaults to None.

    array_src_geometry_pair : tuple of (GeometryType or None), optional
        A tuple containing two optional `GeometryType` elements:
          - The first element: Represents the **valid** geometries.
          - The second element: Represents the **invalid** geometries.

        If provided, a rasterization of those geometries is
        performed locally on the current `array_src` raster window. This
        generated mask is then merged with any additional raster mask supplied
        via the `array_src_mask_ds` dataset. The rasterization itself is
        delegated to the `build_mask` gridr's core method. Defaults to None.

    oversampled_grid_win : numpy.ndarray
        Target window for resampling, defined in full-resolution coordinates,
        relative to the local origin of `grid_arr`.

    margin : numpy.ndarray
        Pixel margin to apply when computing the minimal read window from
        `array_src_ds`, ensuring context for resampling (e.g., for kernels).
        Format: ``[[top_margin, bottom_margin], [left_margin, right_margin]]``.

    sma_out_buffer : SharedMemoryArray or numpy.ndarray
        Output array (or shared memory buffer) where resampled values will
        be written.

    out_win : numpy.ndarray
        Window within `sma_out_buffer` specifying where to write the output
        data. Format: ``[[row_start, row_end], [col_start, col_end]]``.

    nodata_out : scalar or None
        NoData value to fill the output if the grid metrics are invalid or if
        no valid data points can be found.

    sma_out_mask_buffer : SharedMemoryArray or numpy.ndarray or None, optional
        Optional output array (or shared memory buffer) where output mask
        will be written. Defaults to None.

    logger_msg_prefix : str
        Prefix to prepend to all logger messages, useful for debugging and
        tracing within logs.

    logger : logging.Logger
        Logger instance used for debug and informational messages.

    Notes
    -----
    The goal is to generate data within a specific target window
    (`oversampled_grid_win`), which is defined in a full-resolution geometry.
    The coordinates of this target window are relative to the local origin
    of `grid_arr`.

    To optimize the loading of only the necessary extent from the source image
    (`array_src_ds`), this method calls `array_compute_resampling_grid_geometries`
    on the minimal low-resolution grid window that completely contains the
    target full-resolution window. The provided `margin` parameter is also
    incorporated into this calculation to ensure sufficient data coverage.

    The method writes the resampled output to a shared memory array
    (`sma_out_buffer`), using `out_win` to specify the target writing window
    within this buffer.

    Optional masks for both the grid and the source array may be passed.

    Masks can be passed as either `int8` or `uint8` arrays, but the values must
    be positive and within the uint8 range of [0-255]. If you provide an `int8`
    mask, the method will internally shadow it with a `uint8` view.

    If the grid metrics are not valid (i.e., there was not sufficient valid data
    to determine the grid and source boundaries), the method fills the windowed
    output with the `nodata_out` value.
    """

    def DEBUG(msg):
        logger.debug(f"{logger_msg_prefix} - {msg}")

    # Determine the minimal coarse-grid window containing the oversampled window
    # `oversamped_grid_win`.
    grid_arr_win, _ = grid_resolution_window_safe(
        resolution=grid_resolution, win=oversampled_grid_win, grid_shape=grid_arr_shape
    )

    # Check grid_mask_arr - no change done here but the grid_mask_arr
    # may be shadowed with an uint8 view if passed as int8
    if grid_mask_arr is not None:

        # Check array_src_mask dtype is an 8 bit integer
        # We will ensure it is passed as uint8 later
        if grid_mask_arr.dtype not in (np.int8, np.uint8):
            raise TypeError("The grid mask array must be an 8 bit integer " "raster.")
        # Check validity value in range of uint8
        if grid_mask_in_unmasked_value < 0 or grid_mask_in_unmasked_value > 255:
            raise ValueError(
                "The `grid_mask_in_unmasked_value` must be in the "
                "[0-255] range."
                f"The current value is {grid_mask_in_unmasked_value}"
            )
        if grid_mask_arr.dtype == np.int8:
            grid_mask_arr = grid_mask_arr.view(np.uint8)

    # Compute strip grid metrics for current tile
    DEBUG("Computing grid metrics... ")
    grid_metrics = array_compute_resampling_grid_geometries(
        grid_row=grid_arr[0],
        grid_col=grid_arr[1],
        grid_resolution=grid_resolution,
        win=grid_arr_win,
        grid_mask=grid_mask_arr,
        grid_mask_valid_value=grid_mask_in_unmasked_value,
        grid_nodata=None,  # TODO : manage grid_nodata input
    )

    full_nodata = True

    if grid_metrics:

        array_src_profile_2d = ArrayProfile(
            shape=(array_src_ds.height, array_src_ds.width),
            ndim=2,
            dtype=np.dtype(array_src_ds.dtypes[array_src_bands[0] - 1]),
        )

        array_src_win_read, array_src_win_marged, pad = read_win_from_grid_metrics(
            grid_metrics=grid_metrics,
            array_src_profile_2d=array_src_profile_2d,
            margins=margin,
            logger=logger,
            logger_msg_prefix=logger_msg_prefix,
        )

        if array_src_win_read is not None:

            # Read data is available
            full_nodata = False

            array_src_win_read_shape = window_shape(array_src_win_read)

            # memory required - for array_src
            DEBUG("Computing required memory for array source read")
            array_src_win_memory = 0

            for band in array_src_bands:
                array_src_win_band_memory = np.dtype("float64").itemsize * np.prod(
                    np.diff(array_src_win_read, axis=1)
                )
                array_src_win_memory += array_src_win_band_memory
                DEBUG(
                    f"memory needed for array_src_win + margin band {band}: "
                    f"{array_src_win_band_memory} bytes"
                )

            # memory required - for array_src_mask

            DEBUG("Computing required memory for array source mask read")

            array_src_mask_win_memory = 0
            array_src_mask_dtype = np.uint8
            array_src_mask_read_dtype = np.uint8
            if array_src_mask_ds is not None:
                array_src_mask_read_dtype = np.dtype(
                    array_src_mask_ds.dtypes[array_src_mask_band - 1]
                )

                # Check array_src_mask dtype is an 8 bit integer
                # We will ensure it is passed as uint8 later
                if array_src_mask_read_dtype not in (np.int8, np.uint8):
                    raise TypeError("The mask array must be an 8 bit integer " "raster.")

                array_src_mask_win_memory = array_src_mask_read_dtype.itemsize * np.prod(
                    np.diff(array_src_win_read, axis=1)
                )
                DEBUG(
                    f"Memory required for array_src_mask_win + margin : "
                    f"{array_src_mask_win_memory} bytes"
                )

                array_src_win_memory += array_src_mask_win_memory

            else:
                DEBUG("No available mask for array source\n")

            DEBUG(
                "Memory required for array_src_mask + margin : "
                f"{array_src_mask_win_memory} bytes"
            )

            DEBUG(
                f"Total memory required for array_src_win and mask : "
                f"{array_src_win_memory} bytes"
            )

            # The `tile_read_buffer_shape` corresponds to the shape of the buffer
            # that will hold the read data. It can be larger than the actual read
            # window set in the `tile_src_sin_read` variable in order to also hold
            # "virtual" margins.
            array_src_read_buffer_shape = window_shape(array_src_win_marged)
            array_src_read_buffer_shape = np.insert(
                array_src_read_buffer_shape, 0, len(array_src_bands)
            )
            DEBUG(f"tile read buffer shape : {array_src_read_buffer_shape}")

            # TODO : do not force float 64 here => requires bound core function to handle other
            # types
            # cstrip_read_buffer = np.zeros(cstrip_read_buffer_shape, dtype=array_src_profile.dtype)
            array_src_read_buffer = np.zeros(
                array_src_read_buffer_shape, dtype=np.float64, order="C"
            )

            # Manage the mask assuming here the same shape as image
            # Default value to 0 in order to init as not valid
            # Here we init in all cases (array_src_mask_ds given or not)
            array_src_mask_read_buffer_shape = window_shape(array_src_win_marged)
            array_src_mask_read_buffer = np.full(
                array_src_mask_read_buffer_shape,
                Validity.INVALID,
                dtype=array_src_mask_dtype,
                order="C",
            )

            # Read the source array.
            # - Due to "virtual" margins we have to compute the correct indices in
            #   the tile_read_buffer that will hold the
            # read array for each band.
            DEBUG("Reading tiles...")

            def get_read_buffer_indices(b, p, s):
                return (
                    b,
                    slice(p[0][0], p[0][0] + s[0], None),
                    slice(p[1][0], p[1][0] + s[1], None),
                )

            def get_mask_read_buffer_indices(p, s):
                return (
                    slice(p[0][0], p[0][0] + s[0], None),
                    slice(p[1][0], p[1][0] + s[1], None),
                )

            for band_read, band_in in enumerate(array_src_bands):
                indices = get_read_buffer_indices(band_read, pad, array_src_win_read_shape)

                # TODO : the bellow commented line should be decommented when f64 is not forced.
                # array_src_ds.read(band+1, window = as_rio_window(ctile_src_win_read),
                #        out=ctile_read_buffer[indices])
                # Save read data in the buffer at `indices`.

                DEBUG(
                    f"Reading source window for source band {band_in} "
                    f"- source window : {array_src_win_read} "
                    f"- target indices : {indices} ..."
                )

                array_src_read_buffer[indices] = array_src_ds.read(
                    band_in, window=as_rio_window(array_src_win_read)
                ).astype(np.float64)

                DEBUG(
                    f"Reading source window for source band {band_in} "
                    f"- source window : {array_src_win_read} "
                    f"- target indices : {indices} [DONE]"
                )

            # Manage raster mask
            if array_src_mask_ds:
                indices = get_mask_read_buffer_indices(pad, array_src_win_read_shape)

                DEBUG(
                    f"Reading source window for source mask "
                    f"- source window : {array_src_win_read} "
                    f"- target indices : {indices} ..."
                )

                mask_buffer_tmp = array_src_mask_ds.read(
                    array_src_mask_band, window=as_rio_window(array_src_win_read)
                )

                # We do not use the second element of `array_src_mask_validity_pair`
                # It must be given in order to check if we have to force replacement
                # even if the input valid value corresponds to `Validity.VALID`
                if array_src_mask_validity_pair != (Validity.VALID, Validity.INVALID):
                    DEBUG("Replacing read mask values to comply with internal convention")
                    array_replace(
                        mask_buffer_tmp,
                        array_src_mask_validity_pair[0],
                        Validity.VALID,
                        Validity.INVALID,
                    )

                if mask_buffer_tmp.dtype == np.uint8:
                    array_src_mask_read_buffer[indices] = mask_buffer_tmp

                else:
                    # Due to previous test it is int8
                    # In that case we just get a view
                    array_src_mask_read_buffer[indices] = mask_buffer_tmp.view(np.uint8)

                DEBUG(
                    f"Reading source window for source mask "
                    f"- source window : {array_src_win_read} "
                    f"- target indices : {indices} [DONE]"
                )

            else:
                # TODO : should we activate this code only if there is pad ?

                # The mask was not given but we can still fill the mask buffer
                # to be valid on indices where we can read data VS virtual margins.
                indices = get_mask_read_buffer_indices(pad, array_src_win_read_shape)

                array_src_mask_read_buffer[indices] = Validity.VALID

            # TODO : implement edge management. For now we leave it
            # as it is considering the init zero value.

            # The grid stores absolute source coordinates. However, when operating
            # on a localized sub-region of the raster, we must compensate for the
            # relative shift of its origin.
            # This calculated offset is then provided to the resampling function,
            # which will apply it during the target coordinate to interpolate.
            # We avoid modifying the grid in place to prevent unintended side
            # effects if it's used concurrently by other processes.
            #
            # Specifically, 'tile_src_origin' is derived by adjusting for any
            # 'tile_pad' (virtual padding) relative to the absolute upper-left
            # corner of the read window ('tile_src_win_read'). This value precisely
            # defines the origin of the 'tile_read_buffer' in the context of the
            # overall 'array_src' coordinate system.
            array_src_origin = (
                pad[0][0] - array_src_win_read[0][0],
                pad[1][0] - array_src_win_read[1][0],
            )

            # Manage geometry mask
            if array_src_geometry_pair is not None:

                # We have to define the rasterization mesh so that is will be
                # aligned with the current raster mask.
                # Here the rasterize grid is given by :
                # - its origin : `array_src_origin`
                # - its shape : `array_src_mask_read_buffer` shape
                #
                # The `array_src_mask_read_buffer` is defined in all cases
                # (with or without `array_src_mask_ds` defined)
                # The goal here is to use build_mask in order to merge the
                # existing `array_src_mask_read_buffer` with the geometry
                # masks

                # `array_src_origin` defines the origin of the current sub-array
                # in the full array using GridR's internal convention, ie using
                # (0, 0) as origin.
                #
                # ┌────────────────────────────────────────────────────────────┐
                # │ WARN :                                                     │
                # │                                                            │
                # │ `array_src_origin` is a negative shift (see definition)    │
                # │ We need to take its opposite here                          │
                # └────────────────────────────────────────────────────────────┘
                # In order to rasterize we have to take care of the
                # `array_src_geometry_origin`

                cgeometry_origin = -np.asarray(array_src_origin).astype(np.float64)
                
                if array_src_geometry_origin is not None:
                    cgeometry_origin += array_src_geometry_origin

                # ┌────────────────────────────────────────────────────────────┐
                # │ NOTE :                                                     │
                # │                                                            │
                # │ The build_mask core method adopts the same convention :    │
                # │ - Validity.VALID (1) is considered valid                   │
                # │ - Validity.INVALID (0) is considered invalid               │
                # └────────────────────────────────────────────────────────────┘

                # We do not pass the mask_in here as it will currently result in
                # non necessary thresholding
                # In that case we cant pass the out with the current code as
                # it will result in a overwrite of the buffer.
                geom_mask = build_mask(
                    shape=array_src_mask_read_buffer.shape,
                    resolution=(1, 1),
                    out=None,
                    geometry_origin=cgeometry_origin,
                    geometry_pair=array_src_geometry_pair,
                    mask_in=None,
                    mask_in_target_win=None,
                    mask_in_resolution=None,
                    oversampling_dtype=None,
                    mask_in_binary_threshold=None,
                    rasterize_kwargs=GEOMETRY_RASTERIZE_KWARGS,
                    init_out=False,
                )

                array_src_mask_read_buffer &= geom_mask

            # TODO/TOCHECK We may reset the sma_w_array_buffer.array if the cslices
            # is limited (not the case for now)

            array_in_mask = array_src_mask_read_buffer

            # array_out_mask
            array_out_mask = sma_out_mask_buffer.array if sma_out_mask_buffer is not None else None

            # Call the resampling method - this method returns a tuple containing
            # the output array and the output mask.
            # Here both are returned as None as the buffer are given as input

            # min_row, max_row = grid_arr[0].min(), grid_arr[0].max()
            # min_col, max_col = grid_arr[1].min(), grid_arr[1].max()

            # min_row += array_src_origin[0]
            # max_row += array_src_origin[0]
            # min_col += array_src_origin[1]
            # max_col += array_src_origin[1]

            check_boundaries = False
            if np.any(pad != 0):
                check_boundaries = True
            # if min_row < 0 \
            #        or max_row > array_src_read_buffer.shape[0] - 1 \
            #        or min_col < 0 \
            #        or max_col > array_src_read_buffer.shape[1] - 1:
            #    check_boundaries = True

            _ = array_grid_resampling(
                interp=interp,
                array_in=array_src_read_buffer,  # thats the previously read buffer
                grid_row=grid_arr[0],  # the grid rows
                grid_col=grid_arr[1],  # the grid columns
                grid_resolution=grid_resolution,
                array_out=sma_out_buffer.array,
                array_out_win=out_win,  # dst window in the array_out
                nodata_out=nodata_out,
                array_in_origin=array_src_origin,
                win=oversampled_grid_win,  # the production window
                array_in_mask=array_in_mask,  # TODO : Optional[np.ndarray] = None,
                grid_mask=grid_mask_arr,  # TO CHECK: Optional[np.ndarray] = None,
                grid_mask_valid_value=grid_mask_in_unmasked_value,  #: Optional[int] = 1,
                grid_nodata=None,  # TODO : manage grid_nodata input
                array_out_mask=array_out_mask,  # TODO: Optional[np.ndarray] = None,
                check_boundaries=check_boundaries,  # We are sure we pass all the required data
            )

    if full_nodata:
        # Write NODATA
        win_slice = window_indices(out_win, reset_origin=False)
        # Add the band axis.
        win_slice3 = (slice(None, None),) + win_slice
        sma_out_buffer.array[win_slice3] = nodata_out

        # If mask out - set all to no valid (0)
        if sma_out_mask_buffer is not None:
            sma_out_mask_buffer.array[win_slice] = 0


def basic_grid_resampling_chain(
    grid_ds: rasterio.io.DatasetReader,
    grid_row_coords_band: int,
    grid_col_coords_band: int,
    grid_resolution: Tuple[int, int],
    array_src_ds: rasterio.io.DatasetReader,
    array_src_bands: Union[int, List[int]],
    array_out_ds: rasterio.io.DatasetWriter,
    interp: str,
    nodata_out: Union[int, float],
    grid_col_ds: Union[rasterio.io.DatasetReader, None] = None,
    win: Optional[np.ndarray] = None,
    array_src_mask_ds: Optional[rasterio.io.DatasetReader] = None,
    array_src_mask_band: Optional[int] = None,
    array_src_mask_validity_pair: Optional[Tuple[int, int]] = None,
    mask_out_ds: Optional[rasterio.io.DatasetWriter] = None,
    grid_mask_in_ds: Optional[rasterio.io.DatasetReader] = None,
    grid_mask_in_unmasked_value: Optional[int] = None,
    grid_mask_in_band: Optional[int] = None,
    array_src_geometry_origin: Optional[Tuple[float, float]] = None,
    array_src_geometry_pair: Optional[Tuple[Optional[GeometryType], Optional[GeometryType]]] = None,
    io_strip_size: int = DEFAULT_IO_STRIP_SIZE,
    io_strip_size_target: GridRIOMode = GridRIOMode.INPUT,
    ncpu: int = DEFAULT_NCPU,
    tile_shape: Optional[Tuple[int, int]] = DEFAULT_TILE_SHAPE,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Performs a comprehensive grid-based resampling operation.

    This function orchestrates the entire resampling process from input grid
    and source raster datasets to an output raster dataset, handling various
    masking, interpolation, and I/O strategies. It leverages the
    `basic_grid_resampling_array` core method for the actual array processing.

    Parameters
    ----------
    grid_ds : rasterio.io.DatasetReader
        Input dataset containing the grid coordinates. This dataset provides
        the destination geometry for resampling.

    grid_row_coords_band : int
        Band index in `grid_ds` corresponding to the row coordinates of the
        grid.

    grid_col_coords_band : int
        Band index in `grid_ds` (or `grid_col_ds`) corresponding to the
        column coordinates of the grid.

    grid_resolution : tuple of int
        Resolution of the coarse grid, typically in pixels or map units per
        pixel (e.g., ``(10, 10)``).

    array_src_ds : rasterio.io.DatasetReader
        The source dataset from which raster data will be read and resampled.

    array_src_bands : int or list of int
        Band index or list of band indices to read from `array_src_ds`.

    array_out_ds : rasterio.io.DatasetWriter
        Output dataset where the resampled raster data will be written.

    interp : str
        Interpolation method to use (e.g., 'nearest', 'linear', 'cubic').
        Specific interpolation options are delegated to the underlying
        resampling engine.

    nodata_out : scalar
        NoData value to fill the output raster when no valid data points
        can be found for a given output pixel.

    grid_col_ds : rasterio.io.DatasetReader or None, optional
        Optional separate dataset for grid column coordinates if they are not
        in `grid_ds`. Defaults to None.

    win : numpy.ndarray, optional
        Optional output window of the `grid_ds` to process, defined as
        ``[[row_start, row_end], [col_start, col_end]]``. This defines the
        region of interest for the resampling. If None, the full grid extent
        is considered.
        Defaults to None.

    array_src_mask_ds : rasterio.io.DatasetReader or None, optional
        Optional dataset representing the mask associated with `array_src_ds`.
        Defaults to None.

    array_src_mask_band : int or None, optional
        Band index to read from `array_src_mask_ds` for the source mask.
        Defaults to None.

    array_src_mask_validity_pair : tuple of int, optional
        A tuple containing two integer :
          - The first integer corresponds to the value to consider as valid in
            the mask array.
          - The second integer corresponds to the value to consider as invalid
            in the mask array.

        If the tuple differs from (`Validity.VALID`, `Validity.INVALID`) a
        replace operation will be performed in order to make the mask compliant
        with the core resampling method.

    mask_out_ds : rasterio.io.DatasetWriter
        Output dataset where the resampled validity mask will be written.
        This mask indicates which output pixels contain valid resampled data.

    grid_mask_in_ds : rasterio.io.DatasetReader or None, optional
        Optional input dataset for the grid mask. This mask can define valid
        areas within the grid itself. Defaults to None.

    grid_mask_in_unmasked_value : int or None, optional
        Value in `grid_mask_in_ds` that represents a valid/unmasked data point.
        Defaults to None.

    grid_mask_in_band : int or None, optional
        Band index to read from `grid_mask_in_ds` for the input grid mask.
        Defaults to None.

    array_src_geometry_origin : tuple of float or None, optional
        Specifies the origin convention for `array_src_geometry_pair` definition.
        GridR uses a ``(0, 0)`` image coordinate system to address the first
        pixel of the source raster. This parameter aligns the geometry
        definition with GridR's convention. Defaults to None.

    array_src_geometry_pair : tuple of (GeometryType or None), optional
        A tuple containing two optional `GeometryType` elements:
          - The first element: Represents the **valid** geometries.
          - The second element: Represents the **invalid** geometries.

        If provided, a rasterization of those geometries is
        performed locally on the current `array_src` raster window. This
        generated mask is then merged with any additional raster mask supplied
        via the `array_src_mask_ds` dataset. The rasterization itself is
        delegated to the `build_mask` gridr's core method. Defaults to None.

    io_strip_size : int, optional
        The number of rows per chunk for I/O operations. This parameter
        optimizes memory usage and processing speed by dividing the input
        and output operations into manageable strips. Defaults to
        `DEFAULT_IO_STRIP_SIZE`.

    io_strip_size_target : GridRIOMode, optional
        Defines how `io_strip_size` is applied, e.g., to input or output
        strips. Defaults to `GridRIOMode.INPUT`.

    ncpu : int, optional
        Number of CPU cores to use for parallel processing. Defaults to
        `DEFAULT_NCPU`.

    tile_shape : tuple of int or None, optional
        Shape ``(rows, cols)`` for internal processing tiles within strips,
        optimizing cache usage. Defaults to `DEFAULT_TILE_SHAPE`.

    logger : logging.Logger or None, optional
        Logger instance for debugging and informational messages. If None,
        a default logger is initialized internally. Defaults to None.

    Returns
    -------
    int
        Returns 1 upon successful completion of the resampling process.
        A return value other than 1 indicates an error.

    Notes
    -----
    This function manages the reading of input data in chunks (strips),
    calls the `basic_grid_resampling_array` method for processing each chunk,
    and then writes the results to the output datasets.

    The method handles grid data from one or two separate datasets for row and
    column coordinates. It also incorporates masking capabilities for both
    the input grid and the source array, allowing for flexible data validity
    management.

    The `win` parameter is crucial for defining the specific output region
    to be processed, enabling partial grid resampling without loading the
    entire dataset into memory.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Init a list to register SharedMemoryArray buffers
    sma_buffer_name_list = []
    register_sma = partial(create_and_register_sma, register=sma_buffer_name_list)

    # Get input shape from input grid
    grid_nrow, grid_ncol = grid_ds.height, grid_ds.width
    logger.debug(f"Grid shape : {grid_nrow} rows x {grid_ncol} columns")

    grid_row_ds = grid_ds
    if grid_col_ds is None:
        grid_col_ds = grid_ds
        assert grid_row_coords_band != grid_col_coords_band
    else:
        # Test shapes are the same
        assert grid_nrow == grid_col_ds.height
        assert grid_ncol == grid_col_ds.width

    if grid_mask_in_ds is not None:
        grid_mask_nrow, grid_mask_ncol = grid_mask_in_ds.height, grid_mask_in_ds.width
        logger.debug(f"Grid mask shape : {grid_mask_nrow} rows x " f"{grid_mask_ncol} columns")
        logger.debug(f"Grid mask unmasked value : {grid_mask_in_unmasked_value}")
        assert grid_nrow == grid_mask_nrow
        assert grid_ncol == grid_mask_ncol
        assert grid_mask_in_unmasked_value is not None
        assert grid_mask_in_unmasked_value >= 0
        logger.debug(f"Grid mask dtype : {np.dtype(grid_mask_in_ds.dtypes[grid_mask_in_band-1])}")
        # assert(np.dtype(grid_mask_in_ds.dtypes[grid_mask_in_band-1]) == np.dtype('uint8'))
    else:
        logger.debug("Grid mask : no input grid mask")

    # Cut in strip chunks
    # io_strip_size is computed for the output grid but it can either be piloted
    # by setting a target size for the read buffer.
    if io_strip_size not in [0, None]:
        if io_strip_size_target == GridRIOMode.INPUT:
            # We have to take into account the grid's resolution along rows
            io_strip_size = io_strip_size * grid_resolution[0]
        elif io_strip_size_target == GridRIOMode.OUTPUT:
            # The strip_size is directly given for the target output
            pass
        else:
            raise ValueError(
                f"Not recognized value {io_strip_size_target} for "
                "the 'io_strip_size_target' argument"
            )
        logger.debug(f"Computed strip size (number of rows) : {io_strip_size}")

    # Compute the full output shape - used if no window
    full_shape_out = grid_full_resolution_shape(
        shape=(grid_nrow, grid_ncol), resolution=grid_resolution
    )
    logger.debug(
        f"Computed full output shape : {full_shape_out[0]} rows x {full_shape_out[1]}" " columns"
    )
    # Create an array profile for full output in order to use window utils
    full_profile_out = ArrayProfile(
        shape=full_shape_out, ndim=2, dtype=np.dtype(array_out_ds.dtypes[0])
    )

    # The window is given in target geometry - if None we set it to full grid
    # The window contains the first and last index for each axis
    if win is None:
        logger.debug("No window given : the full grid is considered as window")
        win = np.array(((0, full_shape_out[0] - 1), (0, full_shape_out[1] - 1)))

    # If the window is given we have to check that it lies in the grid
    elif not window_check(full_profile_out, win):
        raise Exception("The given 'window' is outside the grid domain of " "definition.")

    logger.debug(f"Window : {win}")

    shape_out = win[:, 1] - win[:, 0] + 1
    logger.debug(f"Shape out : {shape_out}")

    # Compute strips definitions
    # TODO : MAKE SURE HERE THAT GRID CHUNKS COVER AT LEAST 2 ROWS IF RESOLUTION > 1
    chunk_boundaries = chunks.get_chunk_boundaries(
        nsize=shape_out[0], chunk_size=io_strip_size, merge_last=False
    )

    # First convert chunks to target windows
    # Please note that the last coordinate in each chunk is not contained
    # in the chunk (it corresponds to the index), whereas the window
    # definition contains index that are in the window
    # Notes :
    # - the shape_out corresponds to the production window shape
    # - we shift the chunk_windows of the window origin
    chunk_windows = [
        np.array([[c0, c1 - 1], [0, shape_out[1] - 1]]) + win[:, 0].reshape(-1, 1)
        for c0, c1 in chunk_boundaries
    ]

    # Compute the window to read for each chunk window.
    # This will returns both the window to read, and the relative window
    # corresponding to the target chunk window.
    # The `grid_resolution_window_safe` ensures for each dimension where
    # resolution > 1, that stop - start + 1 >= 2 for the read window.
    # This is required by the resampling method in order to interpolate the
    # grid values.
    chunk_windows_read = [
        grid_resolution_window_safe(
            resolution=grid_resolution, win=chunk_win, grid_shape=(grid_nrow, grid_ncol)
        )
        for chunk_win in chunk_windows
    ]

    # Determine the read buffer shape for the grid
    read_grid_buffer_shape = np.max(
        np.asarray([window_shape(read_win) for read_win, rel_win in chunk_windows_read]), axis=0
    )
    read_grid_buffer_shape3 = np.insert(read_grid_buffer_shape, 0, 2)
    logger.debug(f"Read grid buffer shape : {read_grid_buffer_shape}")
    logger.debug(f"Read grid buffer shape 3 : {read_grid_buffer_shape3}")

    # Create shared memory array for read input grid.
    sma_r_buffer_grid = register_sma(
        read_grid_buffer_shape3, grid_row_ds.dtypes[grid_row_coords_band - 1]
    )

    # TO_CHECK
    # Create shared memory array for grid mask
    sma_in_buffer_grid_mask = None
    if grid_mask_in_ds is not None:
        sma_in_buffer_grid_mask = register_sma(
            read_grid_buffer_shape, grid_mask_in_ds.dtypes[grid_mask_in_band - 1]
        )

    # Get array profile for input array
    try:
        array_src_bands[0]
    except TypeError:
        array_src_bands = [array_src_bands]

    # array_src_profile = ArrayProfile(
    #        shape=(array_src_ds.height, array_src_ds.width),
    #        ndim=array_src_ds.count,
    #        dtype=np.dtype(array_src_ds.dtypes[array_src_bands[0]]))

    # Determine the write buffer shape
    # The computation is performed using the chunk_windows.
    # - nrow x ncol : from max full res grid strip size
    # - ndim : from the number of bands to process
    write_buffer_shape2 = np.max(
        np.asarray([window_shape(chunk_win) for chunk_win in chunk_windows]), axis=0
    )
    write_buffer_shape = np.insert(write_buffer_shape2, 0, len(array_src_bands))

    # Create shared memory array for output
    # The core resampling method currently supports float64 buffers only.
    # If the output dataset has a type other than float64, we create two buffers:
    # - The first buffer holds the computation in float64.
    # - The second buffer maintains the target dtype and is written to disk.
    array_out_ds_dtype = np.dtype(array_out_ds.dtypes[0])

    # `sma_w_array_buffer_convert` determines if conversion is needed
    # - If the buffer is None, no conversion is required
    # - If the buffer is defined, conversion is required and it will contain the converted data
    #   ready for writing
    sma_w_array_buffer_convert = None

    logger.debug(f"Create float64 computation array buffer with shape {write_buffer_shape}")
    sma_w_array_buffer = register_sma(write_buffer_shape, np.float64)
    logger.debug(f"Create float64 computation array buffer with shape {write_buffer_shape} DONE")

    if array_out_ds_dtype != np.dtype("float64"):
        logger.debug(f"Create write array buffer with shape {write_buffer_shape}")
        sma_w_array_buffer_convert = register_sma(write_buffer_shape, array_out_ds_dtype)
        logger.debug(f"Create write array buffer with shape {write_buffer_shape} DONE")

    # Manage output mask
    sma_w_mask_buffer = None
    if mask_out_ds is not None:
        mask_out_dtype = np.dtype(mask_out_ds.dtypes[0])
        # Create shared memory array for output mask
        logger.debug(f"Create write mask buffer with shape {write_buffer_shape2}")
        sma_w_mask_buffer = register_sma(write_buffer_shape2, mask_out_dtype)
        logger.debug(f"Create write mask buffer with shape {write_buffer_shape2} DONE")

    # Define here the margins required by interpolation
    # TODO : make it generic / for now set the 2 margin for bicubic interp
    # margin = np.array(((2, 2), (2, 2)))
    margin = None
    match interp:
        case "nearest":
            margin = np.array(((1, 1), (1, 1)))
        case "linear":
            margin = np.array(((1, 1), (1, 1)))
        case "cubic":
            margin = np.array(((2, 2), (2, 2)))
        case _:
            raise ValueError(f"Unknown interpolator {interp}")

    try:
        for chunk_idx, (chunk_win, (win_read, win_rel)) in enumerate(
            zip(chunk_windows, chunk_windows_read, strict=True)
        ):

            logger.debug(f"Chunk {chunk_idx} - chunk_win: {chunk_win}")

            # Compute current strip chunk parameters
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
            # - cslices_as_win : current strip slices converted to a window
            # - cslices_write : current strip slice to adress the whole output
            #       dataset for IO write operation
            cshape = window_shape(chunk_win)
            cslices = window_indices(chunk_win, reset_origin=True)
            cslices_as_win = window_from_indices(cslices, cshape)
            cslices3 = (slice(None, None),) + cslices
            # Define the target positioning window `cstrip_target_win` to write to
            # disk. We have to revert back the shift of the production window.
            cstrip_target_win = chunk_win - win[:, 0].reshape(-1, 1)
            # read the grid data
            cread_shape = window_shape(win_read)

            # First row and col grids
            _ = grid_row_ds.read(
                grid_row_coords_band,
                window=as_rio_window(win_read),
                out=sma_r_buffer_grid.array[0, 0 : cread_shape[0], 0 : cread_shape[1]],
            )

            _ = grid_col_ds.read(
                grid_col_coords_band,
                window=as_rio_window(win_read),
                out=sma_r_buffer_grid.array[1, 0 : cread_shape[0], 0 : cread_shape[1]],
            )

            cread_grid_arr = sma_r_buffer_grid.array[:, 0 : cread_shape[0], 0 : cread_shape[1]]
            assert cread_grid_arr[0].flags.c_contiguous
            assert cread_grid_arr[1].flags.c_contiguous

            # Read the grid mask data if given
            cread_grid_mask_arr = None
            if sma_in_buffer_grid_mask is not None:
                _ = grid_mask_in_ds.read(
                    grid_mask_in_band,
                    window=as_rio_window(win_read),
                    out=sma_in_buffer_grid_mask.array[0 : cread_shape[0], 0 : cread_shape[1]],
                )
                cread_grid_mask_arr = sma_in_buffer_grid_mask.array[
                    0 : cread_shape[0], 0 : cread_shape[1]
                ]

            logger.debug(f"Chunk {chunk_idx} - shape : {cshape}")
            logger.debug(f"Chunk {chunk_idx} - grid read shape : {cread_shape}")
            logger.debug(f"Chunk {chunk_idx} - buffer slices : {cslices}")
            logger.debug(f"Chunk {chunk_idx} - buffer slices as win : {cslices_as_win}")
            logger.debug(f"Chunk {chunk_idx} - target write window : {cstrip_target_win}")
            logger.debug(f"Chunk {chunk_idx} - resampling starts...")

            # TO_CHECK
            # cin_grid_mask = sma_in_buffer_grid_mask.array[0:cread_shape[0], 0:cread_shape[1]]

            # If no mask is given : set it to 1
            # cin_grid_mask = None
            # cin_grid_mask[:,:] = 1
            # Check against image in dimension - here we will have to take care of origin convention
            # For now let keep it simple and assume it that 0 is the pixel center (and not 0.5)
            # TODO : replace by a less memory consuming code : each test expression generate a
            # temporary array

            # No mask provided – we must at least ensure that addressed
            # coordinates lie within the domain bounds.
            # Oversampled grids pose a challenge: masking a single point may
            # invalidate surrounding interpolated values, which can
            # unintentionally exclude otherwise valid points.
            # A better approach is to invalidate only those points outside the
            # domain that have no valid neighbors within it.

            # TODO: Implement masking for points outside the domain, ensuring
            # consistency with the interpolation strategy described above.
            # if cin_grid_mask is None:
            #    cin_grid_mask = sma_in_buffer_grid_mask.array[0:cread_shape[0], 0:cread_shape[1]]
            #    cin_grid_mask[:,:] = 1
            #    cin_grid_mask[ np.logical_or(
            #        np.logical_or(
            #            cread_grid_arr[0] < 0., cread_grid_arr[0] > array_src_ds.height - 1.
            #        ),
            #        np.logical_or(
            #            cread_grid_arr[1] < 0., cread_grid_arr[1] > array_src_ds.width - 1.
            #        )
            #    )] = 0
            #    grid_mask_in_unmasked_value = 1

            if tile_shape is not None:
                # Cut strip shape into tiled chunks.
                logger.debug(
                    f"Chunk {chunk_idx} - Tiled processing with tiles of {tile_shape[0]} x "
                    f"{tile_shape[1]}"
                )

                chunk_tiles = chunks.get_chunk_shapes(cshape, tile_shape, merge_last=False)

                logger.debug(
                    f"Chunk {chunk_idx} - Number of tiles to process :" f" {len(chunk_tiles)}"
                )

                for ctile in chunk_tiles:
                    logger.debug(f"Chunk {chunk_idx} - tile {ctile} " "- preparing args...")
                    # Compute current strip chunk parameters to pass to the
                    # 'build_mask_tile_worker' ('build_mask' wrapper) method
                    # - ctile_origin : the tile origin corresponds here to the
                    #        coordinates relative to the current strip, ie the
                    #        first element of each window
                    # - ctile_win : the window corresponding to the tile (convert
                    #        the chunk index convention to the window index
                    #        convention ; with no origin shift) relative to
                    #        the current chunk
                    # - ctile_grid_win : the full-resolution window within the
                    #       current chunk's grid that corresponds to the active
                    #       tile.
                    ctile_origin = chunk_win[..., 0]
                    ctile_win = window_from_chunk(chunk=ctile, origin=None)

                    # Calculate the full-resolution window within the current
                    # chunk's grid that corresponds to the active tile. The tile
                    # window (`ctile_win`) is relative to the current chunk;
                    # therefore, we apply a relative shift (`win_rel`) to
                    # `ctile_win` to obtain the window aligned with the chunk's
                    # grid array, suitable for `basic_grid_resampling_array`.
                    ctile_grid_win = window_shift(ctile_win, np.asarray(win_rel[:, 0]))

                    logger.debug(
                        f"Chunk {chunk_idx} - tile {ctile} - "
                        f"tile's chunk origin : {ctile_origin}"
                    )
                    logger.debug(
                        f"Chunk {chunk_idx} - tile {ctile} - " f"tile window : {ctile_win}"
                    )
                    logger.debug(
                        f"Chunk {chunk_idx} - tile {ctile} - "
                        f"tile full-resolution window within the chunk's "
                        f"grid : {ctile_grid_win}"
                    )

                    basic_grid_resampling_array(
                        interp=interp,
                        grid_arr=cread_grid_arr,
                        grid_arr_shape=cread_shape,
                        grid_resolution=grid_resolution,
                        grid_nodata=None,  # TODO
                        grid_mask_arr=cread_grid_mask_arr,
                        grid_mask_in_unmasked_value=grid_mask_in_unmasked_value,
                        array_src_ds=array_src_ds,
                        array_src_bands=array_src_bands,
                        array_src_mask_ds=array_src_mask_ds,
                        array_src_mask_band=array_src_mask_band,
                        array_src_mask_validity_pair=array_src_mask_validity_pair,
                        array_src_geometry_origin=array_src_geometry_origin,
                        array_src_geometry_pair=array_src_geometry_pair,
                        oversampled_grid_win=ctile_grid_win,
                        margin=margin,
                        sma_out_buffer=sma_w_array_buffer,
                        out_win=ctile_win,
                        nodata_out=nodata_out,
                        sma_out_mask_buffer=sma_w_mask_buffer,
                        logger_msg_prefix=f"Chunk {chunk_idx} - tile {ctile} - ",
                        logger=logger,
                    )

            else:
                # Resampling on full strip - no tiling
                logger.debug(f"Chunk {chunk_idx} - Full strip computation " "(no tiling)")

                basic_grid_resampling_array(
                    interp=interp,
                    grid_arr=cread_grid_arr,
                    grid_arr_shape=cread_shape,
                    grid_resolution=grid_resolution,
                    grid_nodata=None,  # TODO
                    grid_mask_arr=cread_grid_mask_arr,
                    grid_mask_in_unmasked_value=grid_mask_in_unmasked_value,
                    array_src_ds=array_src_ds,
                    array_src_bands=array_src_bands,
                    array_src_mask_ds=array_src_mask_ds,  # TODO
                    array_src_mask_band=array_src_mask_band,  # TODO
                    array_src_mask_validity_pair=array_src_mask_validity_pair,
                    array_src_geometry_origin=array_src_geometry_origin,
                    array_src_geometry_pair=array_src_geometry_pair,
                    oversampled_grid_win=win_rel,
                    margin=margin,
                    sma_out_buffer=sma_w_array_buffer,
                    out_win=cslices_as_win,
                    nodata_out=nodata_out,
                    sma_out_mask_buffer=sma_w_mask_buffer,
                    logger_msg_prefix=f"Chunk {chunk_idx} - ",
                    logger=logger,
                )

            # Write full chunk at once
            logger.debug(f"Chunk {chunk_idx} - write for full strip...")

            # Manage output data type conversion
            if sma_w_array_buffer_convert is None:
                array_out_ds.write(
                    sma_w_array_buffer.array[cslices3], window=as_rio_window(cstrip_target_win)
                )
            else:
                # We have to convert the data from float64 (computation type) to output type
                logger.debug(f"Chunk {chunk_idx} - convert data type for full strip")

                rounding_method = "round" if array_out_ds_dtype.kind in "iu" else None

                # TODO : limit data dtype conversion to cslices3
                array_convert(
                    sma_w_array_buffer.array,
                    sma_w_array_buffer_convert.array,
                    clip="auto",
                    rounding_method=rounding_method,
                )

                logger.debug(f"Chunk {chunk_idx} - write converted data for full strip...")
                array_out_ds.write(
                    sma_w_array_buffer_convert.array[cslices3],
                    window=as_rio_window(cstrip_target_win),
                )

            logger.debug(f"Chunk {chunk_idx} - write ends.")

            # Set mask to no valid
            if mask_out_ds is not None:
                logger.debug(f"Chunk {chunk_idx} - write mask for full strip...")

                mask_out_ds.write(
                    sma_w_mask_buffer.array[cslices], 1, window=as_rio_window(cstrip_target_win)
                )

                logger.debug(f"Chunk {chunk_idx} - write mask ends.")
    except Exception:
        raise

    finally:
        # Release the Shared memory buffer
        SharedMemoryArray.clear_buffers(sma_buffer_name_list)

    return 1
