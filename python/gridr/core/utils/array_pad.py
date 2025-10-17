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
#
# Portions of this code are derived from NumPy's numpy.pad implementation
# Copyright (c) 2005-2025, NumPy Developers
# All rights reserved.
#
# NumPy's original code is licensed under the BSD-3-Clause License:
# https://github.com/numpy/numpy/blob/main/LICENSE.txt
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the NumPy Developers nor the names of any contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import typing
import warnings
from typing import Final, Iterable, NoReturn, Tuple, Union

import numpy as np

# Constants
NUMPY_VERSION: Final = tuple(map(int, np.__version__.split(".")[:2]))
USE_FALLBACK: Final = NUMPY_VERSION < (2, 0)

if USE_FALLBACK:
    from numpy.lib.arraypad import _as_pairs

    warnings.warn(
        f"NumPy {np.__version__} is older than 2.0.0. "
        "Internal padding functions are not available. "
        "Using fallback implementation which may allocate extra memory. "
        "Consider upgrading: pip install -U 'numpy>=2.0.0'",
        UserWarning,
        stacklevel=1,  # stacklevel=1 car appelé au niveau module
    )
else:
    from numpy.lib._arraypad_impl import (
        _as_pairs,
        _get_edges,
        _set_pad_area,
        _set_reflect_both,
        _set_wrap_both,
        _view_roi,
    )


def _expand_slice_with_padding(src_slice, pad_width_pair):
    """
    Expands a slice by adding padding before and after.

    Converts a source slice into a target slice that includes padding regions.
    This is used to compute the required window in a padded array when the
    original slice references an unpadded array.

    Parameters
    ----------
    src_slice : tuple of slice
        Source slices defining a region in the unpadded array.
        Each slice can have None for start/stop to indicate full extent.

    pad_width_pair : tuple of tuple
        Padding to apply for each dimension as ((before_0, after_0),
        (before_1, after_1), ...).

    Returns
    -------
    tuple of slice
        Expanded slices that include the padding regions.

    Examples
    --------
    >>> src = (slice(10, 20), slice(30, 40))
    >>> pad = ((5, 5), (3, 3))
    >>> _expand_slice_with_padding(src, pad)
    (slice(5, 25), slice(27, 43))

    >>> # Handle full slices (None boundaries)
    >>> src = (slice(None, None), slice(10, 20))
    >>> pad = ((5, 5), (3, 3))
    >>> _expand_slice_with_padding(src, pad)
    (slice(None, None), slice(7, 23))
    """
    target_slice = []

    for aslice, (pad_before, pad_after) in zip(src_slice, pad_width_pair, strict=True):
        # Handle full slice case (both None)
        if aslice.start is None and aslice.stop is None:
            target_slice.append(slice(None, None))
            continue

        # Extract start and stop, handling None cases
        start = aslice.start if aslice.start is not None else 0
        stop = aslice.stop  # Can remain None for open-ended slices

        # Apply padding
        if pad_before > 0:
            new_start = start - pad_before
            # Ensure we don't go negative if start was 0
            new_start = max(0, new_start) if start == 0 else new_start
        else:
            new_start = start

        if pad_after > 0 and stop is not None:
            new_stop = stop + pad_after
        else:
            new_stop = stop

        target_slice.append(slice(new_start, new_stop))

    return tuple(target_slice)


def _pad_simple_inplace(
    array: np.ndarray,
    src_win: Tuple[Tuple[slice]],
    pad_width: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    strict_size: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice]]:
    """
    Mocks the original numpy `_pad_simple` method considering an inplace behaviour.

    Unlike the original method, this implementation does not fill the padded area
    and does not change any original values.

    Parameters
    ----------
    array : np.ndarray
        Array to grow.

    src_win : tuple of slices
        Defines the source window within array from which padding values are derived.
        For a 2D array, use: (slice(row_start, row_end), slice(col_start, col_end))
        For a 1D array, use: (slice(start, end),)
        The window defines the "original data" region.

    pad_width : sequence of tuple[int, int]
        Pad width on both sides for each dimension in `array`.

    strict_size : bool
        If True the input array must exactly match the required shape for padding.

    Returns
    -------
    array : np.ndarray
        Returns the view on the input array corresponding to the source window.
    padded : np.ndarray
        Returns the input array.
    original_area_slice : tuple
        A tuple of slices pointing to the area of the original array.

    Raises
    ------
    ValueError
        If array is too small for requested padding, or if array is larger than
        needed and strict_size is True.
    """

    # Assign padded to array as inplace padding will occure
    padded = array

    # Assign array to the windowed view
    array = array[src_win]

    # Check that array can exactly hold the requested padding
    # First compute required total size
    padded_shape = tuple(
        left + size + right for size, (left, right) in zip(array.shape, pad_width, strict=True)
    )

    # Get original slice
    original_area_slice = tuple(
        slice(left, left + size) for size, (left, right) in zip(array.shape, pad_width, strict=True)
    )

    # Validate array size against padding requirements
    if ~np.all(padded_shape == padded.shape):
        for axis in range(padded.ndim):
            if padded_shape[axis] > padded.shape[axis]:
                raise ValueError(
                    f"Array too small for requested padding on axis {axis}. "
                    f"Required size: {padded_shape[axis]}, "
                    f"pad_before: {pad_width[axis][0]}, pad_after: {pad_width[axis][1]}), "
                    f"but got: {padded.shape[axis]}"
                )
            elif padded_shape[axis] < padded.shape[axis]:
                msg = (
                    f"Array larger than needed on axis {axis}. "
                    f"Required size: {padded_shape[axis]} but got: {padded.shape[axis]}"
                    f"Extra space will be ignored."
                )
                if strict_size:
                    raise ValueError(msg)
                else:
                    # Limit the padded array to the padded region
                    padded_slice = _expand_slice_with_padding(src_win, pad_width)
                    padded = padded[padded_slice]
                    warnings.warn(msg, UserWarning, stacklevel=2)

    return array, padded, original_area_slice


def pad_inplace(
    array: np.ndarray,
    src_win: Tuple[Tuple[slice]],
    pad_width: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    mode="constant",
    strict_size: bool = True,
    **kwargs,
) -> NoReturn:
    """
    Pad an array inplace using a source window as original area.

    Derived from method `numpy.pad` but acting inplace with mode limited to
    `constant`, `edge`, `reflect`, `symmetric` and `wrap`

    Parameters
    ----------
    array : array_like of rank N
        The array to pad inplace. It's shape must be so that it can hold the
        full padded array.

    src_win : tuple of slices
        Defines the source window within array from which padding values are derived.
        For a 2D array, use: (slice(row_start, row_end), slice(col_start, col_end))
        For a 1D array, use: (slice(start, end),)
        The window defines the "original data" region, and the outside of this
        window will be filled according to the padding mode.

    pad_width : {sequence, array_like, int, dict}
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for before = after = pad width
        for all axes.

    mode : str, optional
        One of the following string values or a user supplied function.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.

        'reflect'
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.

        'symmetric'
            Pads with the reflection of the vector mirrored
            along the edge of the array.

        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

    strict_size : bool, optional
        If True the input array must exactly match the required shape for padding.

    constant_values : sequence or scalar, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants
        for each axis.

        ``(before, after)`` or ``((before, after),)`` yields same before
        and after constants for each axis.

        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default is 0.

    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.

    Returns
    -------
    None
        This function modifies the array in-place and returns None.

    Notes
    -----
    For an array with rank greater than 1, some of the padding of later
    axes is calculated from padding of previous axes.  This is easiest to
    think about with a rank 2 array where the corners of the padded array
    are calculated by using padded values from the first axis.

    The padding function, if used, should modify a rank 1 array in-place. It
    has the following signature::

        padding_func(vector, iaxis_pad_width, iaxis, kwargs)

    where

    vector : ndarray
        A rank 1 array already padded with zeros.  Padded values are
        vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:].
    iaxis_pad_width : tuple
        A 2-tuple of ints, iaxis_pad_width[0] represents the number of
        values padded at the beginning of vector where
        iaxis_pad_width[1] represents the number of values padded at
        the end of vector.
    iaxis : int
        The axis currently being calculated.
    kwargs : dict
        Any keyword arguments the function requires.

    Examples
    --------
    >>> import numpy as np
    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
    >>> a
    array([4, 4, 1, ..., 6, 6, 6])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'edge')
    >>> a
    array([1, 1, 1, ..., 5, 5, 5])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'reflect')
    >>> a
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')
    >>> a
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'symmetric')
    >>> a
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')
    >>> a
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> a = [1, 2, 3, 4, 5]
    >>> np.pad(a, (2, 3), 'wrap')
    >>> a
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])
    """
    array = np.asarray(array)
    if isinstance(pad_width, dict):
        seq = [(0, 0)] * array.ndim
        for axis, width in pad_width.items():
            match width:
                case int(both):
                    seq[axis] = both, both
                case tuple((int(before), int(after))):
                    seq[axis] = before, after
                case _ as invalid:
                    typing.assert_never(invalid)
        pad_width = seq
    pad_width = np.asarray(pad_width)

    if not pad_width.dtype.kind == "i":
        raise TypeError("`pad_width` must be of integral type.")

    # Broadcast to shape (array.ndim, 2)
    pad_width = _as_pairs(pad_width, array.ndim, as_index=True)

    # Make sure that no unsupported keywords were passed for the current mode
    allowed_kwargs = {
        "edge": [],
        "wrap": [],
        "constant": ["constant_values"],
        "reflect": ["reflect_type"],
        "symmetric": ["reflect_type"],
    }
    try:
        unsupported_kwargs = set(kwargs) - set(allowed_kwargs[mode])
    except KeyError:
        raise ValueError(f"mode '{mode!r}' is not supported") from None
    if unsupported_kwargs:
        raise ValueError(
            "unsupported keyword arguments for mode " f"'{mode!r}': {unsupported_kwargs}"
        )

    # Create array with final shape and original values
    # (padded area is undefined)
    array, padded, original_area_slice = _pad_simple_inplace(array, src_win, pad_width, strict_size)

    # And prepare iteration over all dimensions
    # (zipping may be more readable than using enumerate)
    axes = range(padded.ndim)

    if mode == "constant":
        values = kwargs.get("constant_values", 0)
        values = _as_pairs(values, padded.ndim)
        for axis, width_pair, value_pair in zip(axes, pad_width, values, strict=True):
            roi = _view_roi(padded, original_area_slice, axis)
            _set_pad_area(roi, axis, width_pair, value_pair)

    elif mode == "edge":
        for axis, width_pair in zip(axes, pad_width, strict=True):
            roi = _view_roi(padded, original_area_slice, axis)
            edge_pair = _get_edges(roi, axis, width_pair)
            _set_pad_area(roi, axis, width_pair, edge_pair)

    elif mode in {"reflect", "symmetric"}:
        method = kwargs.get("reflect_type", "even")
        include_edge = mode == "symmetric"
        for axis, (left_index, right_index) in zip(axes, pad_width, strict=True):
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                edge_pair = _get_edges(padded, axis, (left_index, right_index))
                _set_pad_area(padded, axis, (left_index, right_index), edge_pair)
                continue

            roi = _view_roi(padded, original_area_slice, axis)
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_reflect_both(
                    roi, axis, (left_index, right_index), method, array.shape[axis], include_edge
                )

    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width, strict=True):
            roi = _view_roi(padded, original_area_slice, axis)
            original_period = padded.shape[axis] - right_index - left_index
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_wrap_both(
                    roi, axis, (left_index, right_index), original_period
                )


def pad_inplace_fallback(
    array: np.ndarray,
    src_win: Tuple[Tuple[slice]],
    pad_width: Union[int, Tuple[int, int], Iterable[Tuple[int, int]]],
    mode="constant",
    strict_size: bool = True,
    **kwargs,
) -> NoReturn:
    """
    A fallback method in case the import of numpy internal library functions
    fail (ie. numpy version < 2, or not yet known refactoring)

    See `pad_inplace` for arguments definitions.

    This fallback directly calls numpy.pad. With doing so it does allocate
    a non necessary buffer.
    """
    array = np.asarray(array)
    if isinstance(pad_width, dict):
        seq = [(0, 0)] * array.ndim
        for axis, width in pad_width.items():
            match width:
                case int(both):
                    seq[axis] = both, both
                case tuple((int(before), int(after))):
                    seq[axis] = before, after
                case _ as invalid:
                    typing.assert_never(invalid)
        pad_width = seq
    pad_width = np.asarray(pad_width)

    # Broadcast to shape (array.ndim, 2)
    pad_width_pair = _as_pairs(pad_width, array.ndim, as_index=True)

    # Call this method to get the strict_size related behaviour
    _, _, _ = _pad_simple_inplace(array, src_win, pad_width_pair, strict_size)

    tmp = np.pad(array[src_win], pad_width=pad_width, mode=mode, **kwargs)
    if ~np.all(tmp.shape == array.shape):
        # fill only the padded area
        padded_slice = _expand_slice_with_padding(src_win, pad_width_pair)
        array[padded_slice] = tmp[:]
    else:
        array[:] = tmp[:]


# Select pad_inplace function
pad_inplace = pad_inplace_fallback if USE_FALLBACK else pad_inplace
