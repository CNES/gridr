# Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
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
"""FFT filtering chain with overlap-add strip management.

Three frames coexist here. Mixing them up is the main way to get this wrong.

Input frame
    Rows and columns of the source raster. The production window `win` uses it,
    with inclusive bounds, and so does everything read from disk.

Full resolution
    The output a single
    :func:`~gridr.core.convolution.fft_filtering.fft_array_filter` call would
    return for that window with ``zoom=1``. Strip boundaries, kernel margins and
    the overlap-add buffer all live here. Overlap-add sums partial results, so
    the contributions have to be added before any subsampling.

Decimated resolution
    The writes, and only them. Which samples survive is decided from the global
    index of the block being written, since only that index carries the
    decimation phase across strips: how many rows a strip contributes depends on
    where it starts modulo ``Q``.

How overlap-add works here
--------------------------
Strips do not overlap on input. Each one is convolved as if the others did not
exist, so every sample near a cut holds a partial sum.
Overlap-add is what adds those partial sums back together.

Take a 3-row kernel (margin ``m = 1``, so ``k = 3``) and two strips A and B cut
at input row ``c``. Convolving a strip of ``n`` rows produces ``n + k - 1`` rows,
so A spills ``k`` rows past the cut and B is missing A's contribution on its
first ``k`` rows::

      input rows        ...  c-2  c-1  |  c   c+1  ...
                        ---------------|---------------
                            strip A    |    strip B

      global row          c-1      c      c+1     c+2
      A full output      A(n-1)   A(n)   A(n+1)    .    <- tail, k rows
      B full output        .      B(0)    B(1)    B(2)  <- head, k rows

The two spans overlap by ``k - 1`` rows. Stacking them in a buffer of ``k + 1``
rows, offset by one, and summing gives what a single call would have produced::

      oa_buffer row        0       1       2       3
      filled from A      tail0   tail1   tail2     .    (last row zeroed)
      added from B         .     head0   head1   head2
                         -----   -----   -----   -----
      global row          c-1      c      c+1     c+2   <- what gets written

Only the two middle rows are actual sums; the outer ones come from a single
strip. That is why the buffer is ``k + 1`` rows and not ``k``.

Each strip then writes two disjoint blocks: the seam it shares with the previous
strip, taken from the buffer, and its own non-overlapping middle. On an 18-row
raster cut into three 6-row strips, with the same 3-row kernel::

      global output row      0  1  2  3 | 4  5  6  7 |   8  9  | 10 11 12 13 | 14 ... 17
      written by              strip 0   |   buffer   | strip 1 |    buffer   |  strip 2
                               middle   | A + B seam |  middle |  B + C seam |   middle

No row is written twice and none is skipped.

With a production window
------------------------
The window is what gets cut into strips, not the raster. Reads are extended by
the kernel margin at the window edges, using real neighbours wherever they
exist, and not at the internal cuts, whose missing contributions are the ones
overlap-add carries over::

      raster
      +---------------------------------------------------+
      |                                                   |
      |     .......................................       |  <- margin read from
      |     . +---------------------------------+ .       |     real samples
      |     . |         window, strip 0         | .       |
      |     . +- - - - - - - - - - - - - - - - -+ .       |  <- nothing read at
      |     . |         window, strip 1         | .       |     the cut: that is
      |     . +- - - - - - - - - - - - - - - - -+ .       |  <- overlap-add's job
      |     . |         window, strip 2         | .       |
      |     . +---------------------------------+ .       |
      |     .......................................       |  <- margin read from
      |                                                   |     real samples
      +---------------------------------------------------+

A boundary policy applies where the extension leaves the raster, not in the
middle of it, so a window sitting in the interior is filtered identically under
every policy. With ``win=None`` each window edge is a raster edge.

Decimation
----------
The reconstruction above happens at full resolution. Only the writes are
decimated, and the samples kept are chosen from the global row index, not from
a per-strip count.

I/O is restricted to the window extended by the kernel margins, on both axes.
A small window on a large raster therefore costs a small read, and falls back
to a single call rather than being striped into pieces shorter than the kernel.
"""

import logging
from typing import NoReturn

import numpy as np
import rasterio
from numpy.typing import DTypeLike
from rasterio.windows import Window

from gridr.core.convolution.fft_filtering import (
    BoundarySpec,
    ConvolutionMethod,
    DecimationOrigin,
    OutputMode,
    _normalize_boundary_mode,
    align_kernel,
    build_plan,
    decimated_size,
    decimation_offset,
    fft_array_filter,
    kernel_margin,
    normalize_zoom,
    pad_kernel_to_odd,
)
from gridr.core.utils import chunks
from gridr.core.utils.array_utils import ArrayProfile
from gridr.core.utils.array_window import window_normalize
from gridr.core.utils.parameters import tuplify


def extended_extent(
    first: int,
    last: int,
    size: int,
    margin: int,
    extend_before: bool = True,
    extend_after: bool = True,
) -> tuple[int, int]:
    """Range to read so that the kernel margins come from real samples.

    The extension does not depend on the boundary policy. A boundary condition
    describes the edge of the raster, not a seam in the middle of it, so real
    neighbours are read wherever they exist and the policy only governs what is
    synthesised beyond them. Clipping to the array keeps the I/O proportional to
    the window and not to the raster.

    The flags are there for the overlap-add seams: a strip must not see beyond
    itself at an internal cut, whose missing contributions are the ones
    overlap-add carries over.

    Parameters
    ----------
    first, last : int
        Inclusive bounds of the production window along the axis.

    size : int
        Length of the axis in the input array.

    margin : int
        Half-width of the odd-sized kernel along the axis.

    extend_before, extend_after : bool, optional
        Whether each side may look outside the window. Default ``True`` on both
        sides; pass ``False`` at an internal overlap-add seam.

    Returns
    -------
    tuple of two ints
        Inclusive bounds of the range to read.

    Examples
    --------
    >>> extended_extent(10, 20, 100, 3)
    (7, 23)
    >>> extended_extent(1, 20, 100, 3)
    (0, 23)
    >>> extended_extent(10, 20, 100, 3, extend_after=False)
    (7, 20)
    """
    low = first - margin if extend_before else first
    high = last + margin if extend_after else last
    return max(0, low), min(size - 1, high)


def _normalize_boundary_pairs(boundary: BoundarySpec) -> tuple[tuple[str, str], ...]:
    """Expand a boundary specification to one validated ``(before, after)`` pair per axis."""
    return tuple(
        tuple(_normalize_boundary_mode(side) for side in pair)
        for pair in tuplify(boundary, ndim=2, fill=None, strict=True)
    )


def check_oa_strip_size(nrow: int, kernel: np.ndarray, strip_size: int) -> int:
    """Check the strip size against the production window and kernel heights.

    The method returns ``0``, meaning "process in a single chunk", if any of
    the following holds:

    - half the number of produced rows is lesser than the strip size;
    - the number of rows in the kernel is greater than or equal to half the
      number of produced rows;
    - the strip size is lesser than the number of rows in the kernel, in which
      case a strip could not hold the kernel support and the overlap-add
      buffer would be too small to carry every contribution.

    Otherwise it returns the given `strip_size`.

    Parameters
    ----------
    nrow : int
        Number of rows produced, i.e. the height of the production window.

    kernel : numpy.ndarray
        The kernel, already padded to an odd size by
        :func:`~gridr.core.convolution.fft_filtering.pad_kernel_to_odd`. Using
        the raw filter here would under-estimate the margins by one row for an
        even-sized filter.

    strip_size : int
        The chunk target number of rows.

    Returns
    -------
    int
        The original `strip_size`, or ``0``.
    """
    if nrow / 2 < strip_size:
        strip_size = 0
    if kernel.shape[0] >= nrow / 2:
        # There is no use of overlap.
        strip_size = 0
    if 0 < strip_size < kernel.shape[0]:
        # A strip shorter than the kernel cannot carry its own support.
        strip_size = 0
    return strip_size


def decimated_block(
    global_start: int,
    count: int,
    q: int,
    offset: int,
) -> tuple[slice | None, int]:
    """Select the samples of a full-resolution block that survive decimation.

    A block of `count` consecutive samples starting at `global_start` in the
    full-resolution output frame keeps the samples whose global index is
    congruent to `offset` modulo `q`. The answer depends on where the block sits
    in the global frame and not on how many samples were written before it,
    which is how the decimation phase survives from one strip to the next.

    Parameters
    ----------
    global_start : int
        Index, in the full-resolution output frame, of the block's first sample.

    count : int
        Number of consecutive full-resolution samples in the block.

    q : int
        Decimation step, strictly positive.

    offset : int
        Global index of the first sample kept by the decimation, as returned by
        :func:`~gridr.core.convolution.fft_filtering.decimation_offset`. It must
        lie in ``[0, q)``, as that function guarantees.

    Returns
    -------
    tuple of (slice or None, int)
        A slice relative to the block selecting the samples to keep, and
        the destination index of the first of them. ``(None, 0)`` when the
        block keeps nothing.

    Raises
    ------
    ValueError
        If `q` is not strictly positive, or if `offset` is outside ``[0, q)``.

    Examples
    --------
    >>> decimated_block(0, 10, 3, 1)
    (slice(1, 10, 3), 0)
    >>> decimated_block(10, 10, 3, 1)
    (slice(0, 10, 3), 3)
    >>> decimated_block(11, 2, 3, 1)
    (None, 0)
    >>> decimated_block(7, 4, 1, 0)
    (slice(0, 4, 1), 7)
    """
    if q <= 0:
        raise ValueError(f"q must be strictly positive, got {q}")
    if not 0 <= offset < q:
        raise ValueError(f"offset must lie in [0, {q}), got {offset}")
    if count <= 0:
        return None, 0
    # if q == 1:
    #    return slice(0, count), global_start - offset

    first = max(offset, global_start)
    kept = first + (offset - first) % q
    if kept >= global_start + count:
        return None, 0
    return slice(kept - global_start, count, q), (kept - offset) // q


def _write_block(
    ds_out: rasterio.io.DatasetWriter,
    block: np.ndarray,
    window: Window,
    binary: bool,
    binary_threshold: float,
    round_out: bool,
) -> None:
    """Apply the output conversion and write one block."""
    if binary:
        block = (np.abs(block) >= binary_threshold).astype(np.uint8)
    elif round_out:
        block = np.round(block)
    ds_out.write(block, 1, window=window)


def fft_array_filter_fallback(
    ds_in: rasterio.io.DatasetReader,
    ds_out: rasterio.io.DatasetWriter,
    band: int,
    kernel: np.ndarray,
    win: np.ndarray,
    boundary: BoundarySpec,
    out_mode: OutputMode,
    binary: bool = False,
    binary_threshold: float = 1e-3,
    zoom: int | tuple[int, int] = 1,
    decimation: DecimationOrigin = "centered",
    method: ConvolutionMethod = "overlap_add",
    dtype: DTypeLike = None,
    round_out: bool = True,
) -> NoReturn:
    """Wrapper to the `fft_array_filter` core method in case of no strip.

    This function acts as a fallback when strip processing is not required or
    not applicable, directly calling the core `fft_array_filter` method. The
    decimation is delegated to it, since there is no reconstruction to perform.

    Only the production window extended by the kernel margins is read, so a
    small window on a large raster costs a small read.

    Parameters
    ----------
    ds_in : rasterio.io.DatasetReader
        Opened input image dataset.

    ds_out : rasterio.io.DatasetWriter
        Opened output dataset.

    band : int
        Band to consider in the input dataset.

    kernel : numpy.ndarray
        The kernel given as an array in the spatial domain.

    win : numpy.ndarray
        The production window given as ``(2, 2)`` inclusive ``(first, last)``
        bounds, in the input frame.

    boundary : str, None, or sequence of pairs
        The edge management rule, as a single mode string (``"reflect"``,
        ``"symmetric"``, ``"edge"``, ``"wrap"``, ``"constant"``, or ``None`` and
        ``"none"`` for no synthesis) or as ``((top, bottom), (left, right))``.

    out_mode : str
        The output mode for the returned array: ``"same"``, ``"full"`` or
        ``"valid"``.

    binary : bool, optional
        Option to save output as binary (0 or 1). Defaults to False.

    binary_threshold : float, optional
        In case the `binary` option is activated, all values greater or equal
        to `binary_threshold` are set to 1, 0 otherwise. Defaults to 1e-3.

    zoom : int or tuple of two ints, optional
        The rational zoom factor ``P/Q``. Only ``P == 1`` is supported;
        ``Q > 1`` decimates the output. Defaults to 1.

    decimation : DecimationOrigin, optional
        Which sample of each block of ``Q`` is kept, on every axis. The phase is
        counted in the output frame and not in the input one: with
        ``out_mode="same"`` index 0 of the output is the first sample of `win`, so
        ``"leading"`` keeps the window's own first pixel and ``"centered"`` keeps the
        one ``(Q - 1) // 2`` samples further in. With ``out_mode="full"`` index 0
        is the first sample of the convolution support, which sits ahead of the
        window by the kernel margin plus whatever was read or synthesised around
        it, so neither origin lands on the window's first pixel. Defaults to
        ``"centered"``.

    method : ConvolutionMethod, optional
        Convolution backend.

    dtype : data-type, optional
        Working dtype of the convolution. ``None`` lets NumPy promote.

    round_out : bool, optional
        Option to round the written output to the nearest integer.
        Defaults to True.

    Returns
    -------
    NoReturn
        This function performs an operation on `ds_out` and does not return any
        value.
    """
    margins = kernel_margin(kernel, axes=(0, 1))
    row_low, row_high = extended_extent(int(win[0, 0]), int(win[0, 1]), ds_in.height, margins[0])
    col_low, col_high = extended_extent(int(win[1, 0]), int(win[1, 1]), ds_in.width, margins[1])
    read_window = Window.from_slices((row_low, row_high + 1), (col_low, col_high + 1))
    arr = ds_in.read(band, window=read_window)
    local_win = np.asarray(
        [
            (int(win[0, 0]) - row_low, int(win[0, 1]) - row_low),
            (int(win[1, 0]) - col_low, int(win[1, 1]) - col_low),
        ],
        dtype=np.int64,
    )

    arr_out, _ = fft_array_filter(
        arr,
        kernel,
        local_win,
        boundary=boundary,
        out_mode=out_mode,
        zoom=zoom,
        decimation=decimation,
        axes=None,
        dtype=dtype,
        method=method,
    )
    window = Window.from_slices((0, arr_out.shape[0]), (0, arr_out.shape[1]))
    _write_block(ds_out, arr_out, window, binary, binary_threshold, round_out)


def fft_filtering_oa_strip_chain(
    ds_in: rasterio.io.DatasetReader,
    ds_out: rasterio.io.DatasetWriter,
    band: int,
    fil: np.ndarray,
    boundary: BoundarySpec,
    out_mode: OutputMode,
    win: np.ndarray | None = None,
    strip_size: int = 512,
    binary: bool = False,
    binary_threshold: float = 1e-3,
    zoom: int | tuple[int, int] = 1,
    decimation: DecimationOrigin = "centered",
    method: ConvolutionMethod = "overlap_add",
    dtype: DTypeLike = None,
    round_out: bool = True,
    logger=None,
) -> int:
    """Compute the FFT filtering of an opened rasterio input dataset.

    The read and write operations are performed by chunks of whole rows, also
    called strips. This method wraps `fft_array_filter`, implemented in the
    :mod:`gridr.core.convolution.fft_filtering` module, and reconstructs the
    output with the overlap-add method.

    Parameters
    ----------
    ds_in : rasterio.io.DatasetReader
        Opened input image dataset.

    ds_out : rasterio.io.DatasetWriter
        Opened output dataset. Its height and width must match the decimated
        output shape; the method raises otherwise rather than writing a
        silently truncated raster.

    band : int
        Band to consider in the input dataset.

    fil : numpy.ndarray
        The filter given as an array in the spatial domain. An even-sized
        filter is zero-padded to an odd size, as in the core module.

    boundary : str, None, or sequence of pairs
        The edge management rule as a single mode string applying to every side
        (``"reflect"``, ``"symmetric"``, ``"edge"``, ``"wrap"``, ``"constant"``,
        or ``None`` and ``"none"`` for no synthesis) or as
        ``((top, bottom), (left, right))``.

    out_mode : str
        The output mode for the returned array: ``"same"``, ``"full"`` or
        ``"valid"``. ``"valid"`` is not supported by
        this chain.

    win : numpy.ndarray, optional
        The production window as ``(2, 2)`` inclusive ``(first, last)`` bounds,
        in the input frame. Defaults to ``None``, i.e. the whole raster. Only
        the window extended by the kernel margins is read, on both axes.

    strip_size : int, optional
        The chunk target number of rows. Defaults to 512.

    binary : bool, optional
        Option to save output as binary (0 or 1). Defaults to False.

    binary_threshold : float, optional
        In case the `binary` option is activated, all values greater or equal
        to `binary_threshold` are set to 1, 0 otherwise. Defaults to 1e-3.

    zoom : int or tuple of two ints, optional
        The rational zoom factor ``P/Q``. Only ``P == 1`` is supported;
        ``Q > 1`` decimates the output along every axis. Defaults to 1.

    decimation : DecimationOrigin, optional
        Which sample of each block of ``Q`` is kept, on every axis. The phase is
        counted in the output frame and not in the input one: with
        ``out_mode="same"`` index 0 of the output is the first sample of `win`, so
        ``"leading"`` keeps the window's own first pixel and ``"centered"`` keeps the
        one ``(Q - 1) // 2`` samples further in. With ``out_mode="full"`` index 0
        is the first sample of the convolution support, which sits ahead of the
        window by the kernel margin plus whatever was read or synthesised around
        it, so neither origin lands on the window's first pixel. Moving `win`
        therefore moves the sampling grid with it; it does not resample the
        raster on a grid anchored at row 0.

    method : ConvolutionMethod, optional
        Convolution backend, forwarded to every per-strip call.

    dtype : data-type, optional
        Working dtype of the convolution and of the overlap-add buffer.
        ``None`` promotes the input and the kernel with the usual NumPy rules.
        Pinning it to ``float32`` halves the memory of the reconstruction.

    round_out : bool, optional
        Option to round the written output to the nearest integer.
        Defaults to True.

    logger : logging.Logger, optional
        Python logger object to use. If None, a logger is initialized
        internally.

    Returns
    -------
    int
        Returns 0 upon successful completion of the filtering process.

    Raises
    ------
    NotImplementedError
        If `out_mode` is ``"valid"``, or if `boundary` requests ``"wrap"`` on
        the row axis while the window is actually striped. Striping is what
        breaks it: the top margin of the first strip is periodic with the bottom
        of the window, which a strip cannot see. A window processed in a single
        chunk wraps correctly on either axis.
    ValueError
        If the zoom is not a pure decimation, if `win` is malformed or outside
        the raster, or if `ds_out` does not have the decimated output shape.

    Notes
    -----
    Input chunks do not overlap, in order to limit the number of operations.
    The overlap-add method is used to build the whole output image with no
    boundary effects (see `Wikipedia entry for Overlap-add method
    <https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method>`_).

    The reconstruction is performed at full resolution and the decimation is
    applied at write time, from the global index of each block.

    Chunks are cut over the production window, not over the raster height, and
    each read is restricted to the window extended by the kernel margins. A
    window too short to be striped falls back to a single call; see
    :func:`check_oa_strip_size`.

    The two last strips are merged if the last strip does not match the defined
    strip size. Strips are processed sequentially.

    This method falls back to a single monolithic call depending on the dataset
    and kernel shapes; see :func:`check_oa_strip_size`.

    Because the overlap-add reconstruction performs a different sequence of
    floating-point operations than a single call, with different FFT sizes and
    partial sums, the result matches the monolithic output to within
    floating-point tolerance and not bit for bit.

    This method limits the processing to 2D arrays only.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if out_mode == "valid":
        raise NotImplementedError("the overlap-add chain does not support the 'valid' output mode")

    boundary_pairs = _normalize_boundary_pairs(boundary)

    zoom_pq = normalize_zoom(zoom)
    if not zoom_pq.is_supported:
        raise ValueError(
            f"zoom P/Q = {zoom_pq.p}/{zoom_pq.q} is not supported; "
            "only pure decimation (P == 1) is implemented"
        )
    offset = decimation_offset(zoom_pq.q, decimation)

    # Currently limited to 2D data - set axes to None.
    kernel = align_kernel(fil, ndim=2, axes=None)
    kernel = pad_kernel_to_odd(kernel, axes=None)
    margins = kernel_margin(kernel, axes=(0, 1))

    profile_in = ArrayProfile.from_dataset(ds_in)
    shape_in = (ds_in.height, ds_in.width)
    win = window_normalize(win, shape_in)
    first_row, last_row = int(win[0, 0]), int(win[0, 1])
    nrow_win = last_row - first_row + 1

    # Two reference geometries, both computed without touching a pixel.
    #
    # `inner_plan` is the full-resolution frame: the array a single call would
    # return with Q forced to 1. Every strip index is expressed in it.
    inner_plan = build_plan(
        shape_in, kernel, win, boundary=boundary, out_mode=out_mode, zoom=1, axes=None
    )
    # `outer_shape` is what actually reaches the raster, after decimation.
    outer_shape = build_plan(
        shape_in,
        kernel,
        win,
        boundary=boundary,
        out_mode=out_mode,
        zoom=zoom_pq,
        decimation=decimation,
        axes=None,
    ).output_shape

    logger.debug(f"production window             : {win.tolist()}")
    logger.debug(f"full-resolution (inner) shape : {inner_plan.output_shape}")
    logger.debug(f"decimated (outer) shape       : {outer_shape}")

    if (ds_out.height, ds_out.width) != outer_shape:
        raise ValueError(
            f"the output dataset is {(ds_out.height, ds_out.width)} but the filtering "
            f"produces {outer_shape}; open ds_out at the decimated shape"
        )

    work_dtype = np.dtype(dtype) if dtype is not None else np.result_type(profile_in.dtype, kernel)

    strip_size = check_oa_strip_size(nrow=nrow_win, kernel=kernel, strip_size=strip_size)
    chunk_boundaries = chunks.get_chunk_boundaries(
        nsize=nrow_win, chunk_size=strip_size, merge_last=True
    )

    if len(chunk_boundaries) > 1 and "wrap" in boundary_pairs[0]:
        # Striping is what breaks "wrap" on the row axis: the top margin of the
        # first strip is periodic with the bottom of the window, which a strip
        # cannot see. A single chunk reads the whole window at once, so the core
        # wraps it correctly and the restriction does not apply there.
        raise NotImplementedError(
            f"the 'wrap' boundary is not supported on the row axis when the window is "
            f"striped, and it is here in {len(chunk_boundaries)} chunks; pass "
            "strip_size=0 to process it in one call"
        )

    if len(chunk_boundaries) == 1:
        # Single chunk => fallback to the core method, decimation included.
        logger.debug("single chunk : falling back to a monolithic call")
        fft_array_filter_fallback(
            ds_in=ds_in,
            ds_out=ds_out,
            band=band,
            kernel=kernel,
            win=win,
            boundary=boundary,
            out_mode=out_mode,
            binary=binary,
            binary_threshold=binary_threshold,
            zoom=zoom_pq,
            decimation=decimation,
            method=method,
            dtype=work_dtype,
            round_out=round_out,
        )
        return 0

    # ------------------------------------------------------------------ #
    # Overlap-add reconstruction, at full resolution
    # ------------------------------------------------------------------ #
    # Origin of the global full-resolution frame, expressed as the local FULL
    # output row holding the window's first row. In SAME mode the frame starts
    # at that row; in FULL mode it starts `origin` rows earlier.
    base_row = inner_plan.per_axis[0].origin if out_mode == "full" else 0

    # Column geometry is identical for every strip, since a strip spans the
    # full window width: one read extent, one local window, one decimated
    # slice inside the full-resolution output.
    col_extent = extended_extent(int(win[1, 0]), int(win[1, 1]), ds_in.width, margins[1])
    local_col_win = (int(win[1, 0]) - col_extent[0], int(win[1, 1]) - col_extent[0])

    # The first and last strips are read with their row margins, the others are
    # not; size the buffer for the largest case.
    chunk_sizes = [upper - lower for lower, upper in chunk_boundaries]
    buffer = np.zeros(
        (max(chunk_sizes) + 2 * margins[0], col_extent[1] - col_extent[0] + 1),
        dtype=ds_in.profile["dtype"],
        order="C",
    )

    # The overlap buffer stores the kernel support carried from one strip to
    # the next. Top and bottom contributions are shifted by one row, hence the
    # extra row. It is full-resolution in both axes: the additions must happen
    # before any subsampling.
    oa_nrow = kernel.shape[0] + 1
    oa_width = inner_plan.output_shape[1]
    oa_buffer = np.zeros((oa_nrow, oa_width), dtype=work_dtype)

    last_idx = len(chunk_boundaries) - 1

    col_src = None
    col_fullres = None

    for chunk_idx, (lower, upper) in enumerate(chunk_boundaries):
        # Chunk boundaries are relative to the window; shift to the input frame.
        chunk_first = first_row + lower
        chunk_last = first_row + upper - 1

        # Adapt the chunk boundary mode to the overlap-add algorithm: internal
        # seams must not be extended, so that the natural extension of the
        # convolution provides the partial sums to carry over.
        top = boundary_pairs[0][0] if chunk_idx == 0 else "none"
        bottom = boundary_pairs[0][1] if chunk_idx == last_idx else "none"
        cboundary = ((top, bottom), boundary_pairs[1])

        # Read the strip extended by the kernel margins at the window edges,
        # and not at the internal seams, where the missing contributions are the
        # ones overlap-add carries over. Expressing the window inside what was
        # read lets the core recompute the geometry it would have had globally.
        row_extent = extended_extent(
            chunk_first,
            chunk_last,
            ds_in.height,
            margins[0],
            extend_before=chunk_idx == 0,
            extend_after=chunk_idx == last_idx,
        )
        read_window = Window.from_slices(
            (row_extent[0], row_extent[1] + 1), (col_extent[0], col_extent[1] + 1)
        )
        nread = row_extent[1] - row_extent[0] + 1
        arr = ds_in.read(band, window=read_window, out=buffer[0:nread, :])
        local_win = np.asarray(
            [(chunk_first - row_extent[0], chunk_last - row_extent[0]), local_col_win],
            dtype=np.int64,
        )

        logger.debug(
            f"chunk idx {chunk_idx} - window rows {chunk_first}:{chunk_last}, "
            f"read {arr.shape}, local win {local_win.tolist()}, boundary {cboundary}"
        )

        # Q is forced to 1: the reconstruction needs full-resolution partial
        # sums, and the decimation is applied when writing.
        carr_out, cwin_same = fft_array_filter(
            arr,
            kernel,
            local_win,
            boundary=cboundary,
            out_mode="full",
            zoom=1,
            axes=None,
            dtype=work_dtype,
            method=method,
        )
        logger.debug(f"chunk idx {chunk_idx} - current output shape : {carr_out.shape}")

        if col_src is None:
            # Full-resolution column window inside the local FULL output, then
            # the decimated slice inside it. Global column c maps to local
            # column col_fullres.start + c.
            if out_mode == "same":
                col_fullres = slice(int(cwin_same[1, 0]), int(cwin_same[1, 1]) + 1)
            else:
                col_fullres = slice(0, carr_out.shape[1])
            col_src = slice(col_fullres.start + offset, col_fullres.stop, zoom_pq.q)
            n_cols = decimated_size(col_fullres.stop - col_fullres.start, zoom_pq.q, offset)
            if n_cols != outer_shape[1]:  # pragma: no cover - guards the geometry
                raise ValueError(
                    f"decimated column count {n_cols} does not match the output "
                    f"width {outer_shape[1]}"
                )

        # Row index, in the global full-resolution frame, of local FULL row 0.
        # `lower` is already relative to the window, which is where the global
        # frame starts.
        row_phase = lower - int(cwin_same[0, 0]) + base_row

        # -------------------------------------------------------------- #
        # Top overlapping area
        # -------------------------------------------------------------- #
        if chunk_idx > 0:
            # The first `kernel.shape[0]` rows of this strip's FULL output are
            # the head contributions completing the previous strip's tail.
            oa_buffer[1:, :] += carr_out[0 : kernel.shape[0], col_fullres]

            # oa_buffer row 0 sits one row before this strip's local row 0.
            row_slice, dst_row = decimated_block(row_phase - 1, oa_nrow, zoom_pq.q, offset)
            if row_slice is not None:
                block = oa_buffer[row_slice, col_src.start - col_fullres.start :: zoom_pq.q]
                window = Window.from_slices(
                    (dst_row, dst_row + block.shape[0]), (0, block.shape[1])
                )
                logger.debug(
                    f"chunk idx {chunk_idx} - writing overlap area to rows "
                    f"{dst_row}:{dst_row + block.shape[0]}"
                )
                _write_block(ds_out, block, window, binary, binary_threshold, round_out)

        # -------------------------------------------------------------- #
        # Tail contributions carried to the next strip
        # -------------------------------------------------------------- #
        oa_a = int(cwin_same[0, 0]) + margins[0]
        oa_b = int(cwin_same[0, 1]) - margins[0]

        if chunk_idx < last_idx:
            oa_buffer[0:-1, :] = carr_out[oa_b : carr_out.shape[0], col_fullres]
            oa_buffer[-1, :] = 0
            logger.debug(f"chunk idx {chunk_idx} - filled oa_buffer from row {oa_b}")

        # -------------------------------------------------------------- #
        # Non-overlapping area
        # -------------------------------------------------------------- #
        noa_start = oa_a + 1
        noa_stop = oa_b

        if chunk_idx == 0:
            noa_start = int(cwin_same[0, 0]) if out_mode == "same" else 0
        if chunk_idx == last_idx:
            noa_stop = int(cwin_same[0, 1]) + 1 if out_mode == "same" else carr_out.shape[0]

        row_slice, dst_row = decimated_block(
            row_phase + noa_start, noa_stop - noa_start, zoom_pq.q, offset
        )
        if row_slice is not None:
            block = carr_out[
                noa_start + row_slice.start : noa_start + row_slice.stop : row_slice.step,
                col_src,
            ]
            window = Window.from_slices((dst_row, dst_row + block.shape[0]), (0, block.shape[1]))
            logger.debug(
                f"chunk idx {chunk_idx} - writing non-overlap area to rows "
                f"{dst_row}:{dst_row + block.shape[0]}"
            )
            _write_block(ds_out, block, window, binary, binary_threshold, round_out)

    return 0
