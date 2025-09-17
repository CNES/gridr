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
Module for a FFT Filtering chain with overlap-add strip management
"""

import logging
from typing import NoReturn, Union

import numpy as np
import rasterio
from rasterio.windows import Window

from gridr.core.convolution.fft_filtering import (
    BoundaryPad,
    ConvolutionOutputMode,
    fft_array_filter,
    fft_array_filter_output_shape,
    fft_odd_filter,
)
from gridr.core.utils import chunks
from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.parameters import tuplify


def check_oa_strip_size(
    arr: Union[np.ndarray, ArrayProfile],
    fil: np.ndarray,
    strip_size: int,
) -> int:
    """
    Check the strip size against array and filter dimension.

    The method returns 0 if either :
      - the half number of lines in the array is lesser than the strip size
      - the number of rows in the filter is greater than or equal to the half
        number of lines in the input array.

    Otherwise, it returns the given strip_size.

    Parameters
    ----------
    arr : Union[np.ndarray, ArrayProfile]
        The input image array.

    fil : np.ndarray
        The filter given as an array in the spatial domain.

    strip_size : int
        The chunk target number of rows.

    Returns
    -------
    int
        The original strip_size or 0.
    """
    if arr.shape[0] / 2 < strip_size:
        strip_size = 0
    if fil.shape[0] >= arr.shape[0] / 2:
        # There is no use of overlap
        # Set strip_size to 0 in order to get only 1 chunk
        strip_size = 0
    return strip_size


def fft_array_filter_fallback(
    ds_in: rasterio.io.DatasetReader,
    ds_out: rasterio.io.DatasetWriter,
    band: int,
    fil: np.ndarray,
    boundary: BoundaryPad,
    out_mode: ConvolutionOutputMode,
    binary: bool = False,
    binary_threshold: float = 1e-3,
    zoom: int = 1,
    round_out: bool = True,
) -> NoReturn:
    """Wrapper to the fft_array_filter core method in case of no strip.

    Parameters
    ----------
    ds_in : rasterio.io.DatasetReader
        Opened input image dataset.

    ds_out : rasterio.io.DatasetWriter
        Opened output dataset.

    band : int
        Band to consider in the input dataset.

    fil : numpy.ndarray
        The filter given as an array in the spatial domain.

    boundary : BoundaryPad
        The edge management rule as a single value (similar for each side) or
        a tuple ``((top, bottom), (left, right))``. The rule is defined by the
        :class:`~gridr.chain.enums.BoundaryPad` enum.

    out_mode : ConvolutionOutputMode
        The output mode for the returned array.

    binary : bool, optional
        Option to save output as binary (0 or 1). Defaults to False.

    binary_threshold : float, optional
        In case the `binary` option is activated, all values greater or equal
        to `binary_threshold` are set to 1, 0 otherwise. Defaults to 1e-3.

    zoom : int or tuple, optional
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q and given as (P, Q).

        Defaults to 1.

    round_out : bool, optional
        Option to round the written output to the nearest integer.
        Defaults to True.

    Returns
    -------
    NoReturn
        This function performs an in-place operation on `ds_out` and
        does not return any value.

    Notes
    -----
    This function acts as a fallback when strip processing is not required or
    not applicable, directly calling the core `fft_array_filter` method.
    """
    # Full read
    buffer = np.zeros((ds_in.height, ds_in.width), dtype=ds_in.profile["dtype"], order="C")
    arr = ds_in.read(band, out=buffer)
    arr_out, shift_same = fft_array_filter(
        arr=arr,
        fil=fil,
        win=None,  # full array
        boundary=boundary,
        out_mode=out_mode,
        zoom=zoom,
        axes=None,
    )
    if binary:
        arr_out = (np.abs(arr_out) >= binary_threshold).astype(np.uint8)
    elif round_out:
        arr_out = np.round(arr_out)
    ds_out.write(arr_out, 1)


def fft_filtering_oa_strip_chain(
    ds_in: rasterio.io.DatasetReader,
    ds_out: rasterio.io.DatasetWriter,
    band: int,
    fil: np.ndarray,
    boundary: BoundaryPad,
    out_mode: ConvolutionOutputMode,
    strip_size: int = 512,
    binary: bool = False,
    binary_threshold: float = 1e-3,
    zoom: int = 1,
    round_out: bool = True,
    logger=None,
) -> int:
    """Compute the FFT Filtering of an opened rasterio input dataset.

    The read and write operations are performed by chunks of whole rows,
    also called strips. This method wraps the `fft_array_filtering`
    implemented in the `core.convolution.fft_filtering` module.

    Parameters
    ----------
    ds_in : rasterio.io.DatasetReader
        Opened input image dataset.

    ds_out : rasterio.io.DatasetWriter
        Opened output dataset.

    band : int
        Band to consider in the input dataset.

    fil : numpy.ndarray
        The filter given as an array in the spatial domain.

    boundary : BoundaryPad
        The edge management rule as a single value (similar for each side) or
        a tuple ``((top, bottom), (left, right))``. The rule is defined by the
        :class:`~gridr.chain.enums.BoundaryPad` enum.

    out_mode : ConvolutionOutputMode
        The output mode for the returned array.

    strip_size : int, optional
        The chunk target number of rows. Defaults to 512.

    binary : bool, optional
        Option to save output as binary (0 or 1). Defaults to False.

    binary_threshold : float, optional
        In case the `binary` option is activated, all values greater or equal
        to `binary_threshold` are set to 1, 0 otherwise. Defaults to 1e-3.

    zoom : int or tuple, optional
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q and given as (P, Q).
        Defaults to 1.

    round_out : bool, optional
        Option to round the written output to the nearest integer.
        Defaults to True.

    logger : logging.Logger, optional
        Python logger object to use for logging. If None, a logger is
        initialized internally. Defaults to None.

    Returns
    -------
    int
        Returns 1 upon successful completion of the filtering process.

    Notes
    -----
    Input chunks do not overlap in order to limit the number of operations.
    The overlap-add method is used to build the whole output image with no
    boundary effects (see `Wikipedia entry for Overlap-add method
    <https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method>`_).

    The two last strips are merged if the last strip does not match the
    defined strip size. Strips are processed sequentially.

    This method directly falls back to the wrapped method depending on
    dataset and filter shapes. More precisely:

    -   The input raster number of rows must be greater than twice the
        strip size.
    -   The input raster number of rows must be greater than twice the
        filter size.

    Examples
    --------
    A binary option can be set to True to activate a binary output. In that case,
    the `binary_threshold` must be set to define the threshold between 0 and 1
    used for the binary conversion. Please also note that the output dataset has
    to be opened with the correct option in order to save it as a true binary
    (1bit per pixel) raster.

    For instance:

    .. code-block:: python

        import rasterio
        import numpy as np
        from gridr.chain.enums import BoundaryPad, ConvolutionOutputMode

        # Assuming ds_in is an opened rasterio dataset
        file_out = "output_binary.tif"
        with rasterio.open(file_out, "w",
                            driver="GTiff",
                            dtype=np.uint8,
                            height=ds_in.height,
                            width=ds_in.width,
                            count=1,
                            nbits=1) as ds_out_binary:
            fft_filtering_oa_strip_chain(
                ds_in=ds_in,
                ds_out=ds_out_binary,
                band=1,
                fil=np.ones((3, 3)), # Example filter
                boundary=BoundaryPad.REFLECT,
                out_mode=ConvolutionOutputMode.SAME,
                binary=True,
                binary_threshold=0.5
            )
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    assert zoom == 1

    logger.debug("precompute output shape")
    # compute output shape from method parameters
    shape_out = fft_array_filter_output_shape(
        arr=ArrayProfile.from_dataset(ds_in),
        fil=fil,
        win=None,
        boundary=boundary,
        out_mode=out_mode,
        zoom=zoom,
        axes=None,
    )
    logger.debug(f"precomputed output shape : {shape_out}")

    fil = fft_odd_filter(fil, axes=None)

    strip_size = check_oa_strip_size(
        arr=ArrayProfile.from_dataset(ds_in), fil=fil, strip_size=strip_size
    )

    # Compute strips definitions
    chunk_boundaries = chunks.get_chunk_boundaries(
        nsize=ds_in.height, chunk_size=strip_size, merge_last=True
    )

    if len(chunk_boundaries) == 1:
        # Full read => fallback to the core method
        fft_array_filter_fallback(
            ds_in,
            ds_out,
            band,
            fil,
            boundary,
            out_mode,
            binary,
            binary_threshold,
            zoom,
            round_out,
        )
    else:
        # The overlap add algorithm is implemented here in order to limit
        # the number of operations.

        # Allocate a dest buffer for the rasterio read operation
        chunk_sizes = np.asarray([c1 - c0 for c0, c1 in chunk_boundaries])
        buffer = np.zeros(
            (max(chunk_sizes), ds_in.width),
            dtype=ds_in.profile["dtype"],
            order="C",
        )

        # Create a buffer to store the strip that overlap between continuous
        # strips. That overlap buffer must store the filter support from all
        # overlap contributions (top and bottom). Those supports are shifted
        # from 1 row => the buffer number of row equals 1 + the filter's support
        oa_nrow_max = min(max(chunk_sizes), fil.shape[0]) + 1

        # NOTE : dtype is set here to float64 in order to match the dtype
        # of fft_array_filter method output array.
        # TODO : float32 may be sufficient and added as a parameter
        oa_buffer = np.zeros((oa_nrow_max, shape_out[1]), dtype=np.float64)

        # Initialize destination start and stop lines.
        dst_start_line = -1
        dst_stop_line = 0
        oa_a = -1
        oa_b = None

        # Loop on chunks => here we have at least 2 chunks
        for chunk_idx, (lower_line, upper_line) in enumerate(chunk_boundaries):
            # Create the current chunk window for read call
            chunk_window = Window.from_slices((lower_line, upper_line), (0, ds_in.width))

            logger.debug(
                f"chunk idx : {chunk_idx}" f"- from line {lower_line} to line {upper_line}"
            )

            # Read the data
            arr = ds_in.read(
                band,
                window=chunk_window,
                out=buffer[0 : upper_line - lower_line, :],
            )

            logger.debug(f"chunk idx : {chunk_idx} " f"- Read array shape : {arr.shape}")

            # Adapt chunk boundary mode in order to respect the overlap add
            # algorithm : we have to take care of overlaps for whose we dont
            # want a boundary management.
            cboundary = np.asarray(tuplify(boundary, ndim=arr.ndim))
            if chunk_idx > 0:
                # all chunks but first - overwrite top boundary mode with
                # no boundary pad in order to compute overlap-add
                # (we dont test chunk_idx < len(chunk_boundaries) -1 because
                # we are ensured that we got at least 2 chunks)
                cboundary[0, 0] = BoundaryPad.NONE
            if chunk_idx < len(chunk_boundaries) - 1:
                # all chunks but last - overwrite bottom boundary mode with
                # no boundary pad in order to compute overlap-add
                cboundary[0, 1] = BoundaryPad.NONE

            # Filter current strip
            # - set win to None in order to take full array
            # - set out_mode to FULL in order to manage the overlap add
            logger.debug(
                f"chunk idx {chunk_idx} " f"- call fft_array_filter with boundary {cboundary}"
            )

            carr_out, cwin_same = fft_array_filter(
                arr=arr,
                fil=fil,
                win=None,  # full array
                boundary=cboundary,
                out_mode=ConvolutionOutputMode.FULL,
                zoom=zoom,
                axes=None,
            )

            logger.debug(
                f"chunk idx {chunk_idx} " f"- current array output shape : {carr_out.shape}"
            )

            col_slice_carr = None
            col_slice_out = None

            # Set the used columns interval depending on the output_mode
            if out_mode == ConvolutionOutputMode.SAME:
                # only take "same" area columns
                col_slice_carr = slice(cwin_same[1, 0], cwin_same[1, 1] + 1)
                col_slice_out = slice(0, shape_out[1])

            elif out_mode == ConvolutionOutputMode.FULL:
                # take all columns
                col_slice_carr = slice(0, carr_out.shape[1])
                col_slice_out = slice(0, carr_out.shape[1])

            else:
                raise NotImplementedError

            dst_start_line = dst_stop_line

            if chunk_idx > 0:
                # Define and get top overlap area
                oa_a = cwin_same[0, 0] + fil.shape[0] // 2
                oa_a_src_slice = slice(0, fil.shape[0])

                # Do the add - take care of the 1 line shift
                oa_buffer[1:, col_slice_out] += carr_out[oa_a_src_slice, col_slice_carr]

                logger.debug(
                    f"chunk idx {chunk_idx}"
                    f" - add oa_buffer from rows {oa_a_src_slice}"
                    f" and column {col_slice_carr}"
                )

                # Write the current overlapping area
                dst_stop_line = dst_start_line + oa_buffer.shape[0]
                oa_dst_window = Window.from_slices((dst_start_line, dst_stop_line), col_slice_out)

                logger.debug(f"chunk idx {chunk_idx}" " - writing top overlapping area")
                logger.debug(
                    f"chunk idx {chunk_idx}"
                    " --- top overlapping area src slice : "
                    f"{oa_a_src_slice}"
                )
                logger.debug(
                    f"chunk idx {chunk_idx}"
                    " --- top overlapping area dest start line : "
                    f"{dst_start_line}"
                )
                logger.debug(
                    f"chunk idx {chunk_idx}"
                    " --- top overlapping area dest stop line (non inc.) : "
                    f"{dst_stop_line}"
                )
                logger.debug(
                    f"chunk idx {chunk_idx}"
                    " --- top overlapping area dest window : "
                    f"{oa_dst_window}"
                )

                if binary:
                    bin_out = (np.abs(oa_buffer[:, col_slice_out]) >= binary_threshold).astype(
                        np.uint8
                    )
                    ds_out.write(bin_out, 1, window=oa_dst_window)
                elif round_out:
                    ds_out.write(
                        np.round(oa_buffer[:, col_slice_out]),
                        1,
                        window=oa_dst_window,
                    )
                else:
                    ds_out.write(oa_buffer[:, col_slice_out], 1, window=oa_dst_window)
                dst_start_line = dst_stop_line

            if chunk_idx < len(chunk_boundaries) - 1:
                # Define and get bottom overlap area
                oa_b = cwin_same[0, 1] - fil.shape[0] // 2
                oa_b_src_slice = slice(oa_b, carr_out.shape[0])
                oa_buffer[0:-1, col_slice_out] = carr_out[oa_b_src_slice, col_slice_carr]
                oa_buffer[-1, col_slice_out] = 0
                logger.debug(
                    f"chunk idx {chunk_idx} "
                    f"- fill_oa_buffer from rows {oa_b_src_slice} "
                    f"and column {col_slice_carr}"
                )

            # Non overlap area

            # Init default
            noa_src_start = oa_a + 1
            noa_src_stop = oa_b

            if chunk_idx == 0:
                if out_mode == ConvolutionOutputMode.SAME:
                    noa_src_start = cwin_same[0, 0]
                elif out_mode == ConvolutionOutputMode.FULL:
                    noa_src_start = 0

            if chunk_idx == len(chunk_boundaries) - 1:
                if out_mode == ConvolutionOutputMode.SAME:
                    noa_src_stop = cwin_same[0, 1] + 1
                elif out_mode == ConvolutionOutputMode.FULL:
                    noa_src_stop = carr_out.shape[0]

            noa_src_slice = slice(noa_src_start, noa_src_stop)
            dst_stop_line = dst_start_line + noa_src_stop - noa_src_start
            noa_dst_window = Window.from_slices((dst_start_line, dst_stop_line), col_slice_out)

            logger.debug(f"chunk idx {chunk_idx} " "- writing non overlapping area")
            logger.debug(
                f"chunk idx {chunk_idx} --- non overlapping area src "
                f"start line : {noa_src_start}"
            )
            logger.debug(
                f"chunk idx {chunk_idx} --- non overlapping area src "
                f"stop line (non inc.) : {noa_src_stop}"
            )
            logger.debug(
                f"chunk idx {chunk_idx} --- non overlapping area dest "
                f"start line : {dst_start_line}"
            )
            logger.debug(
                f"chunk idx {chunk_idx} --- non overlapping area dest "
                f"stop line (non inc.): {dst_stop_line}"
            )
            logger.debug(
                f"chunk idx {chunk_idx} --- non overlapping area dest " f"window {noa_dst_window}"
            )
            # write the non overlap area
            if binary:
                bin_out = (
                    np.abs(carr_out[noa_src_slice, col_slice_carr]) >= binary_threshold
                ).astype(np.uint8)
                ds_out.write(bin_out, 1, window=noa_dst_window)
            elif round_out:
                ds_out.write(
                    np.round(carr_out[noa_src_slice, col_slice_carr]),
                    1,
                    window=noa_dst_window,
                )
            else:
                ds_out.write(
                    carr_out[noa_src_slice, col_slice_carr],
                    1,
                    window=noa_dst_window,
                )
            dst_start_line = dst_stop_line
    return 0


if __name__ == "__main__":
    alogger = logging.getLogger(__name__)
    alogger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    log_rio = rasterio.logging.getLogger()
    log_rio.setLevel(logging.ERROR)

    from artemis_io import artemis_io as aio

    aio.register_all()

    ima_in = (
        "/home/il/kelbera/work_campus/artemis/contrib/worker/"
        + "AstridzImageFilteringWorker_ake/examples/amiens_10_r_6_41_ref_extrait.tif"
    )
    ima_in_ds = rasterio.open(ima_in)

    filter_in = (
        "/home/il/kelbera/work_campus/artemis/contrib/worker/"
        + "AstridzImageFilteringWorker_ake/examples/filtre_dezoom2.f"
    )
    filter_ds = aio.open(filter_in, driver="cnes_orion_filter")
    filter_data = filter_ds.read(1)
    print(ima_in_ds.profile)

    my_shape_out = fft_array_filter_output_shape(
        arr=ArrayProfile.from_dataset(ima_in_ds),
        fil=filter_data,
        win=None,
        boundary=BoundaryPad.REFLECT,
        out_mode=ConvolutionOutputMode.FULL,
        zoom=1,
        axes=None,
    )
    print("output_shape", my_shape_out)

    path_out = "./test_out.tif"
    with rasterio.open(
        path_out,
        "w",
        driver_name="GTiff",
        dtype=np.uint16,
        height=my_shape_out[0],
        width=my_shape_out[1],
        count=1,
    ) as my_ds_out:
        # nbits=1) as ds_out:

        fft_filtering_oa_strip_chain(
            ds_in=ima_in_ds,
            ds_out=my_ds_out,
            band=1,
            fil=filter_data,
            boundary=BoundaryPad.REFLECT,
            out_mode=ConvolutionOutputMode.FULL,
            strip_size=512,
            zoom=1,
            logger=alogger,
        )

    path_out = "./test_out_0.tif"
    with rasterio.open(
        path_out,
        "w",
        driver_name="GTiff",
        dtype=np.uint16,
        height=my_shape_out[0],
        width=my_shape_out[1],
        count=1,
    ) as my_ds_out:
        # nbits=1) as ds_out:

        fft_filtering_oa_strip_chain(
            ds_in=ima_in_ds,
            ds_out=my_ds_out,
            band=1,
            fil=filter_data,
            boundary=BoundaryPad.REFLECT,
            out_mode=ConvolutionOutputMode.FULL,
            strip_size=0,
            zoom=1,
            logger=alogger,
        )
