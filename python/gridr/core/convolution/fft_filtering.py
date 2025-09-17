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
FFT Filtering core module
"""
from enum import IntEnum
from typing import Iterable, Tuple, Union

import numpy as np
from scipy import signal

from gridr.core.utils.array_window import window_check, window_extend, window_overflow
from gridr.core.utils.parameters import tuplify


class BoundaryPad(IntEnum):
    """Boundary pad mode enumeration."""

    NONE = 1
    REFLECT = 2


class ConvolutionOutputMode(IntEnum):
    """Convolution area output mode enumeration."""

    SAME = 1
    FULL = 2
    VALID = 3


def get_filter_margin(
    fil: np.ndarray,
    zoom: int = 1,
    axes=None,
) -> Tuple[int]:
    """Compute the required margin for filter in order to avoid edge effect.

    In case of zoom = 1 it corresponds to the half size of the filter.

    Parameters
    ----------
    fil : np.ndarray
        The input filter as ndarray.

    zoom : int, optional
        Zoom, by default 1.

    axes : {None, int, tuple of int}, optional
        The axes that will be used for margin computation, by default None.

    Returns
    -------
    Tuple[int]
        The margins array along all dimensions.
    """
    assert zoom == 1
    if axes is None:
        axes = range(fil.ndim)
    margins = [0 if i not in axes else fil.shape[i] // 2 for i in axes]
    return margins


def pad_array(
    arr: np.ndarray,
    win: np.ndarray,
    pad: Tuple[int, int, int, int],
    boundary: Union[BoundaryPad, Tuple[Tuple[BoundaryPad, BoundaryPad]]],
    axes=None,
) -> np.ndarray:
    """Pad an array with respect to the rules set for edge management.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    win : np.ndarray
        The production window given as a list of tuple containing the
        first and last index for each dimension. E.g., for a 2D array:
        ``((first_row, last_row), (first_col, last_col))``.

    pad : Tuple[int, int, int, int]
        The size of padding for each side as a 4-element tuple
        (top, bottom, right, left).

    boundary : Union[BoundaryPad, Tuple[Tuple[BoundaryPad, BoundaryPad]]]
        The edge management rule as a single value (similar for each side)
        or a tuple ((top, bottom), (left, right)). The rule is defined
        by the `BoundaryPad` enum.

    axes : {None, int, tuple of int}, optional
        The axes on which to perform the padding, by default None.

    Returns
    -------
    np.ndarray
        The padded array.
    """
    if axes is None:
        axes = range(arr.ndim)

    out = arr
    boundary_set = list(
        {b for b in np.asarray(boundary).flat if b not in [BoundaryPad.NONE, None, np.nan]}
    )
    if len(boundary_set) == 0:
        pass
    elif len(boundary_set) == 1:
        mode = None
        if boundary_set[0] == BoundaryPad.REFLECT:
            mode = "reflect"
        else:
            raise Exception(f"Not valid padding mode {boundary_set[0]}")

        indices = tuple(
            (
                slice(None, None) if i not in axes else slice(win[i][0], win[i][1] + 1)
                for i in range(arr.ndim)
            )
        )
        out = np.pad(arr[indices], pad, mode=mode)
    else:
        raise Exception("Only one not NONE BoundaryPad mode is implemented")
    return out


def fft_odd_filter(
    fil: np.ndarray,
    axes=None,
) -> np.ndarray:
    """Check that the filter has an odd length along specified axes.

    If it is not the case it is right/bottom padded with zero on the
    corresponding axe(s).

    Parameters
    ----------
    fil : np.ndarray
        The filter as a numpy ndarray.

    axes : {None, int, tuple of int}, optional
        The axes that will be used for convolution computation,
        by default None.

    Returns
    -------
    np.ndarray
        The odd filter as numpy ndarray.
    """
    if axes is None:
        axes = range(fil.ndim)

    # If filter has an even size, we first pad with a 0 on the right et lower
    # edge
    pad_fil = [0 if i not in axes else 1 - fil.shape[i] % 2 for i in range(fil.ndim)]
    if np.any(pad_fil):
        pad_arg = tuple(((0, pad_fil[i]) for i in range(fil.ndim)))
        fil = np.pad(fil, pad_arg, mode="constant", constant_values=0)
    return fil


def fft_array_filter_check_data(
    arr: np.ndarray,
    fil: np.ndarray,
    win: Union[np.ndarray or None],
    zoom: Union[int, Tuple[int, int]] = 1,
    axes=None,
) -> Tuple[np.ndarray, np.ndarray, Iterable, Tuple]:
    """Performs checks on input data to ensure expected types and profiles.

    This function handles:

        - Converting axes to explicit definitions if `None` is given.
        - Converting the window to an explicit definition if `None` is given,
          and ensuring it's an `ndarray` type.
        - Making sure the filter has an odd size along each dimension.
        - Computing convolution margins.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    fil : np.ndarray
        The filter given as an array in the spatial domain.

    win : np.ndarray or None
        The production window given as a list of tuples containing the
        first and last index for each dimension. For example, for a 2D array:
        ``((first_row, last_row), (first_col, last_col))``.

    zoom : int or Tuple[int, int], optional
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q (e.g., (P, Q)),
        by default 1.

    axes : {None, int, tuple of int}, optional
        The axes on which to perform the convolution, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Iterable, Tuple]
        A tuple containing the filter, the window, the axes, and the margins.
    """
    if axes is None:
        axes = range(arr.ndim)

    # If filter has an even size, we first pad with a 0 on the right et lower edge
    fil = fft_odd_filter(fil, axes)

    # Compute the margin needed to avoid edge effect
    conv_margins = get_filter_margin(fil=fil, zoom=zoom, axes=axes)

    # Set window to full array if not given
    if win is None:
        # Define a correct 2d window matching the array dimensions
        win = [(None, None) if i not in axes else (0, arr.shape[i] - 1) for i in range(arr.ndim)]
    win = np.asarray(win)

    # Check process_window with input arr
    if not window_check(arr, win, axes):
        raise Exception("Target window error : not contained in input data")

    return fil, win, axes, conv_margins


def fft_array_filter_output_shape(
    arr: np.ndarray,
    fil: np.ndarray,
    win: Union[np.ndarray or None],
    boundary: Union[BoundaryPad, Tuple[Tuple[BoundaryPad, BoundaryPad]]] = BoundaryPad.NONE,
    out_mode: ConvolutionOutputMode = ConvolutionOutputMode.SAME,
    zoom: Union[int, Tuple[int, int]] = 1,
    axes=None,
) -> np.ndarray:
    """Compute `fft_array_filter` expected output shape along all axes.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    fil : np.ndarray
        The filter given as an array in the spatial domain.

    win : np.ndarray or None
        The production window given as a list of tuples containing the
        first and last index for each dimension. For example, for a 2D array:
        ``((first_row, last_row), (first_col, last_col))``.

    boundary : Union[BoundaryPad, Tuple[Tuple[BoundaryPad, BoundaryPad]]], optional
        The edge management rule as a single value (similar for each side)
        or a tuple ((top, bottom), (left, right)). The rule is defined
        by the `BoundaryPad` enum, by default `BoundaryPad.NONE`.

    out_mode : ConvolutionOutputMode, optional
        The output mode for the returned array.
        Default to `ConvolutionOutputMode.SAME`.

    zoom : int or Tuple[int, int], optional
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q (e.g., (P, Q)),
        by default 1.

    axes : {None, int, tuple of int}, optional
        The axes on which to perform the convolution, by default None.

    Returns
    -------
    np.ndarray
        An array containing the output shape.

    Notes
    -----
    Currently, this function only supports a `zoom` factor of 1. An assertion
    will fail if a different zoom value is provided, as other zoom factors are
    not yet implemented.
    """
    # zoom different from 1 not yet implemented
    assert zoom == 1
    out = np.nan

    # check data and compute convolution margins
    fil, win, axes, conv_margins = fft_array_filter_check_data(arr, fil, win, zoom, axes)
    win_margins = win

    # Get the boundary management
    boundary = np.asarray(tuplify(boundary, ndim=arr.ndim))

    if np.any(boundary != BoundaryPad.NONE):
        # We want to manage at least one edge with either outer data
        # or padding.
        # Define the margins array
        margins = np.repeat(conv_margins, 2).reshape((len(conv_margins), 2))

        # Margins are computed regardless the boundary mode on each edge.
        # Here we make it compliant with the boundary definition.
        # If BoundaryPad.NONE => set the corresponding margin to 0
        margins = np.where(boundary != BoundaryPad.NONE, margins, 0)

        # Apply the margin to the production window
        win_margins = window_extend(win, margins, reverse=False)

    if out_mode == ConvolutionOutputMode.FULL:
        # It returns the full data with eventually applied margins
        out = [
            (
                arr.shape[i]
                if i not in axes
                else win_margins[i][1] - win_margins[i][0] + 1 + 2 * conv_margins[i]
            )
            for i in range(arr.ndim)
        ]
    elif out_mode == ConvolutionOutputMode.SAME:
        # It returns the data corresponding to the input window
        # Please note that this mode takes into account the optional
        # padding that may be performed
        out = [
            arr.shape[i] if i not in axes else win[i][1] - win[i][0] + 1 for i in range(arr.ndim)
        ]

    else:
        raise NotImplementedError

    return np.asarray(out)


def fft_array_filter(
    arr: np.ndarray,
    fil: np.ndarray,
    win: Union[np.ndarray or None],
    boundary: Union[BoundaryPad, Tuple[Tuple[BoundaryPad, BoundaryPad]]] = BoundaryPad.NONE,
    out_mode: ConvolutionOutputMode = ConvolutionOutputMode.SAME,
    zoom: Union[int, Tuple[int, int]] = 1,
    axes=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FFT convolve between an array and a filter.

    This method wraps the `scipy.signal.oaconvolve` method by adding some
    functionalities:

    - The filter is assumed to have an odd size; if it is not the case, it's
      padded on the right and bottom edges with zeros.
    - The user can specify the production window to limit the data given to the
      convolution method. Note that what's actually given to the FFT convolution
      depends on the `boundary` argument.
    - An edge management option is available to precisely define how boundaries
      are handled on each side. In detail:

        - `BoundaryPad.NONE`: No padding is applied on the edge of the
            production window.
        - `BoundaryPad.REFLECT`: Padding is applied. The padding length is
            calculated from the filter size. If data is available in the full
            array, it's considered. If data is not available or only partially
            available, a "mirror" pad is applied. In this case, the array given
            to the FFT convolution method is extended; note that the convolution
            method still applies zero padding internally.

        Note that the padding rule may differ for each side.

    - The output window can differ depending on the `out_mode`:

        - In mode "SAME": The output window matches the input production
          window.
        - In mode "FULL": The output directly corresponds to the "full" mode of
          the internal convolution method, thus embedding both the margins (from
          the filter) and the extent from the `BoundaryPad` mode. In this case,
          the second element of the output can be used to get the position of
          the production window origin.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    fil : np.ndarray
        The filter given as an array in the spatial domain.

    win : np.ndarray or None
        The production window given as a list of tuples containing the
        first and last index for each dimension. For example, for a 2D array:
        ``((first_row, last_row), (first_col, last_col))``.

    boundary : Union[BoundaryPad, Tuple[Tuple[BoundaryPad, BoundaryPad]]], optional
        The edge management rule as a single value (similar for each side)
        or a tuple ((top, bottom), (left, right)). The rule is defined by
        the `BoundaryPad` enum, by default `BoundaryPad.NONE`.

    out_mode : ConvolutionOutputMode, optional
        The output mode for the returned array, by default `ConvolutionOutputMode.SAME`.

    zoom : int or Tuple[int, int], optional
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q (e.g., (P, Q)),
        by default 1.

    axes : {None, int, tuple of int}, optional
        The axes on which to perform the convolution. WARNING: Not yet used,
        by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:

        -   The filtered array whose size depends on the convolution mode.
        -   The output coordinates of the production window considering a
            "full" mode output.

    Notes
    -----
    Currently, this function only supports a `zoom` factor of 1. An assertion
    will fail if a different zoom value is provided, as other zoom
    factors are not yet implemented.
    """
    # zoom different from 1 not yet implemented
    assert zoom == 1

    # check data and compute convolution margins
    fil, win, axes, conv_margins = fft_array_filter_check_data(arr, fil, win, zoom, axes)

    convol_fct = signal.oaconvolve
    conv_arr = None

    # Compute the shift to apply to the full output in order to get
    # the same window as input.
    # This default computation corresponds to the case where
    # BoundaryPad is set to NONE for all edges
    shift_same = np.asarray([0 if i not in axes else fil.shape[i] // 2 for i in range(fil.ndim)])

    # Get the boundary management
    boundary = np.asarray(tuplify(boundary, ndim=arr.ndim))

    if np.all(boundary == BoundaryPad.NONE):
        # final window corresponds to the input window
        indices = tuple(
            (
                slice(None, None) if i not in axes else slice(int(win[i][0]), int(win[i][1] + 1))
                for i in range(arr.ndim)
            )
        )
        conv_arr = arr[indices]

    elif np.any(boundary != BoundaryPad.NONE):
        # We want to manage at least one edge with either outer data
        # or padding.
        # Define the margins array
        margins = np.repeat(conv_margins, 2).reshape((len(conv_margins), 2))

        # Margins are computed regardless the boundary mode on each edge.
        # Here we make it compliant with the boundary definition.
        # If BoundaryPad.NONE => set the corresponding margin to 0
        margins = np.where(boundary != BoundaryPad.NONE, margins, 0)

        # For output : in order to get the same window we have to take
        # account of used margins to shift the window
        shift_same += margins[:, 0]

        # Apply the margin to the production window
        win_margins = window_extend(win, margins, reverse=False)

        # Next compute the padding
        # Here 0 means that no padding is required
        pad = window_overflow(arr, win_margins, axes)

        if np.all(pad == 0):
            # Nothing more to do, just take the window with margins
            indices = tuple(
                (
                    (
                        slice(None, None)
                        if i not in axes
                        else slice(win_margins[i][0], win_margins[i][1] + 1)
                    )
                    for i in range(arr.ndim)
                )
            )
            conv_arr = arr[indices]
        else:
            # Perform the padding - it directly gives the conv array
            win_pad = window_extend(win_margins, pad, reverse=True)
            conv_arr = pad_array(arr=arr, win=win_pad, pad=pad, boundary=boundary, axes=axes)

    # Perform the convolution with mode = 'full' in order to master the
    # returned window
    out = convol_fct(conv_arr, fil, mode="full", axes=axes)

    if out_mode == ConvolutionOutputMode.FULL:
        # It returns the full data with eventually applied margins
        # That directly correspond to the output
        pass
    elif out_mode == ConvolutionOutputMode.SAME:
        # It returns the data corresponding to the input window
        # Please note that this mode takes into account the optional
        # padding that may be performed
        indices = tuple(
            (
                (
                    slice(None, None)
                    if i not in axes
                    else slice(shift_same[i], shift_same[i] + win[i][1] - win[i][0] + 1)
                )
                for i in range(arr.ndim)
            )
        )
        out = out[indices]

    else:
        raise NotImplementedError

    win_same = np.asarray([shift_same, shift_same + win[:, 1] - win[:, 0]]).T

    return out, win_same
