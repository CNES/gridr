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
import math
from enum import IntEnum
from typing import Iterable, Tuple, Union

import numpy as np
from scipy import signal

from gridr.core.utils.array_window import (
    window_check,
    window_extend,
    window_overflow,
    compose_slice,
)
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


def normalize_zoom_arg(zoom: int | tuple[int, int]) -> tuple[int, int]:
    """
    Normalize the zoom argument to a simplified tuple of integers.

    This function converts the zoom argument to a tuple of integers (P, Q) and 
    simplifies the fraction by dividing both values by their greatest common 
    divisor (GCD).

    Parameters
    ----------
    zoom : int or tuple[int, int]
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q and given as (P, Q).

    Returns
    -------
    tuple[int, int]
        A tuple containing the normalized and simplified P and Q values.

    Raises
    ------
    TypeError
        If the input type is not int or tuple[int, int].
    ValueError
        If the tuple does not contain exactly two integers.
    """
    if isinstance(zoom, int):
        P, Q = zoom, 1
    elif isinstance(zoom, tuple):
        if len(zoom) != 2:
            raise ValueError("Tuple must contain exactly two integers")
        P, Q = zoom
        if not all(isinstance(x, int) for x in (P, Q)):
            raise ValueError("Both elements in the tuple must be integers")
            
        # Simplify the fraction by dividing by GCD
        gcd = math.gcd(P, Q)
        if gcd != 0:  # To avoid division by zero
            P = P // gcd
            Q = Q // gcd
        
    else:
        raise TypeError(
            "Unsupported type for the `zoom` argument. Expected int or tuple[int, int]"
        )
    
    if P <= 0:
        raise ValueError(f"Zoom factor P (={P}) cannot be zero or less")
        
    if Q <= 0:
        raise ValueError(f"Zoom factor Q (={Q}) cannot be zero or less")
    
    return P, Q


def zoom_is_supported(
    zoom: tuple[int, int]
) -> bool:
    """Returns True if the zoom is supported by the current implementation.
    
    Parameters
    ----------
    zoom : tuple[int, int]
        The zoom factor as a tuple of two integers representing the rational 
        P/Q and given as (P, Q).
    """
    return zoom[0] == 1 and zoom[1] >= 1


def decimated_size(N: int, Q: int, offset: int = 0) -> int:
    """
    Computes length of a decimated iterable.

    Parameters
    ----------
    N: int
        Length of the original list (>= 0)
    
    Q: int
        Decimation factor (strictly positive integer)
    
    offset: int, optional
        Starting offset (>= 0). Default 0.
        
    Returns
    -------
    int:
        Number of elements after decimation
    
    Raises
    ------
    ValueError
        If a parameter is out of domain (negative, or Q <= 0)
    """
    if N < 0:
        raise ValueError(f"N must be >= 0, got {N}")
    if Q <= 0:
        raise ValueError(f"Q must be > 0, got {Q}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")

    if offset >= N:
        return 0

    return (N - offset - 1) // Q + 1


def normalize_axes(axes: int | tuple | None, ndim: int) -> tuple:
    """
    Normalize axes input (int, tuple, or None) to a tuple of positive axis
    indices.
    
    Parameters:
    -----------
    axes: int, tuple of ints, or None
        Target axes
    
    ndim: int
        Total number of dimensions of the input array related to axes.
    
    Returns:
    --------
    tuple of ints
        Normalized non-negative axis indices.
    """
    if axes is None:
        return tuple(range(ndim))
        
    axes_tuple = (axes, ) if isinstance(axes, int) else tuple(axes)
    
    # Convert negative indices (e.g., -1 -> ndim - 1) and check boundaries
    normalized = []
    for a in axes_tuple:
        if abs(a) >= ndim:
            raise ValueError(
                f"Axis {a} is out of bounds for ndim={ndim}"
            )
        normalized.append(a % ndim)
    
    return tuple(normalized)


def align_filter_dim(
    fil: np.ndarray,
    ndim: int,
    axes=None,
) -> np.ndarray:
    """
    Aligns a filter `fil` to match `ndim` dimensions.
    
    Parameters
    ----------
    fil : np.ndarray
        The input filter as ndarray.
    
    ndim: int
        Total number of dimensions of the input array related to axes.
    
    axes: int, tuple of ints, or None
        Target axes
        
    Returns
    -------
    np.ndarray
        Reshaped filter guaranteed to have `ndim` dimensions.
    """
    fil = np.asarray(fil)
    target_axes = normalize_axes(axes, ndim)

    # Case 1: Filter rank matches the number of specified axes   
    if fil.ndim == len(target_axes):
        new_shape = [1] * ndim
        for axis, size in zip(target_axes, fil.shape):
            new_shape[axis] = size
        return fil.reshape(new_shape)
    
    # Case 2: Filter alreay has ndim dimensions
    if fil.ndim == ndim:        
        for i in range(ndim):
            if i not in target_axes and fil.shape[i] != 1:
                raise ValueError(
                    f"Axis {i} is not in target axes {target_axes}, "
                    f"so its size in filter must be 1 (got {fil.shape[i]})."
                )
        return fil
    
    raise ValueError(
        f"Filter dimension mismatch: filter has {fil.ndim}D, but "
        f"{len(target_axes)} target axis/axes were specified for a {ndim}D "
        "array."
    )


def get_filter_margin(
    fil: np.ndarray,
    zoom: int | tuple[int, int],
    ndim: int,
    axes=None,
) -> Tuple[int]:
    """Compute the required margin for filter in order to avoid edge effect.

    In case of zoom = 1 it corresponds to the half size of the filter.

    Parameters
    ----------
    fil : np.ndarray
        The input filter as ndarray.

    zoom : int or tuple[int, int]
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q and given as (P, Q).
    
    ndim: int
        Total number of dimensions of the input array related to axes.

    axes : {None, int, tuple of int}, optional
        The axes that will be used for margin computation, by default None.

    Returns
    -------
    Tuple[int]
        The margins array along all dimensions of the input filter.
    """
    # Verify that zoom P=1 and Q=1 (compatibility with this implementation)
    zoom_pq = normalize_zoom_arg(zoom)
    if not zoom_is_supported(zoom_pq):
        raise ValueError(
            f"Zoom P/Q = ({zoom_pq[0]}/{zoom_pq[1]}) value is not yet supported"
        )
    
    axes = normalize_axes(axes, ndim)
    margins = [0 if i not in axes else fil.shape[i] // 2 for i in range(ndim)]
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


def fft_array_filter_check_args(
    out_mode: ConvolutionOutputMode = ConvolutionOutputMode.SAME,
    zoom: int | tuple[int, int] = 1,
):
    """Validate the configuration of convolution parameters.

    This function checks that the combination of output mode and zoom factor
    is supported by the convolution implementation.
    
    Parameters
    ----------
    out_mode : ConvolutionOutputMode, optional
        The output mode for the returned array.
        Default to `ConvolutionOutputMode.SAME`.

    zoom : int or Tuple[int, int], optional
        The zoom factor. It can either be a single integer or a tuple of
        two integers representing the rational P/Q (e.g., (P, Q)),
        by default 1.
    
    Raises
    ------
    ValueError
        If the combination of output mode and zoom factor is not supported.
    TypeError
        If the zoom argument has an invalid type.
    """
    # Normalize and validate the zoom argument
    try:
        P, Q = normalize_zoom_arg(zoom)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Invalid zoom argument: {str(e)}") from e

    if not zoom_is_supported((P, Q)):
        raise ValueError(
            f"Zoom P/Q = ({P}/{Q}) value is not yet supported"
        )

    # Check for supported combinations of out_mode and zoom
    if Q > 1 and out_mode != ConvolutionOutputMode.SAME:
        raise ValueError(
            f"Zoom factor with Q={Q} is not supported with output mode {out_mode}. "
            f"Only Q=1 is supported with output modes other than SAME. "
            f"Consider using ConvolutionOutputMode.SAME or a zoom factor with Q=1."
        )


def fft_array_filter_check_data(
    arr: np.ndarray,
    fil: np.ndarray,
    win: Union[np.ndarray or None],
    zoom: int | tuple[int, int] = 1,
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
    axes = normalize_axes(axes, arr.ndim)

    # Normalize filter dimension
    fil = align_filter_dim(fil, arr.ndim, axes)
    
    # If filter has an even size, we first pad with a 0 on the right et lower edge
    fil = fft_odd_filter(fil, axes)

    # Compute the margin needed to avoid edge effect
    conv_margins = get_filter_margin(fil=fil, zoom=zoom, ndim=arr.ndim, axes=axes)

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
    zoom: int | tuple[int, int] = 1,
    centered_decimation: bool = True,
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
    
    centered_decimation : bool, optional
        If True, applies centering to the decimation when Q>1. If False,
        performs simple decimation without centering. Default is True.
        
        Please note the centering is computed through an offset depending of
        the parity of Q : Q // 2 if Q is even, (Q - 1) / 2 if Q is odd.
        

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
    (P, Q) = normalize_zoom_arg(zoom)
    
    # Check the combination of parameters `out_mode` and `zoom_pq`
    # If zoom Q factor is greater than 1, a decimation will be performed thus impacting
    # the output shape.
    fft_array_filter_check_args(out_mode, (P, Q)) 
    
    out = np.nan

    # check data and compute convolution margins
    fil, win, axes, conv_margins = fft_array_filter_check_data(
        arr, fil, win, (P, Q), axes
    )
    win_margins = win

    # Get the boundary management - ensure the number of pairs aligns with
    # `ndim` by left-padding with fill=BoundaryPad.NONE which is neutral
    # for the process that will follow.
    boundary = np.asarray(
        tuplify(boundary, ndim=arr.ndim, fill=BoundaryPad.NONE, strict=False)
    )

    if np.any(boundary != BoundaryPad.NONE):
        
        # Note : Zoom Q factor > 1 not supported in this mode.
        
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
            arr.shape[i] if i not in axes else win[i][1] - win[i][0] + 1
            for i in range(arr.ndim)
        ]
        
        # Zoom Q > 1 => a downsampling will be performed
        if Q > 1:
            decimation_offset = 0
            if centered_decimation:
                if Q % 2:
                    decimation_offset = Q // 2
                else:
                    decimation_offset = (Q - 1) // 2
            
            # Update output size
            out = [
                out[i] if i not in axes
                else decimated_size(out[i], Q, decimation_offset)
                for i in range(arr.ndim)
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
    centered_decimation: bool = True,
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
    
    - For zoom factors (P, Q) with Q>1, the method performs decimation after convolution.
      The centering of the decimation can be controlled with the `centered_decimation`
      parameter.

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
    
    centered_decimation : bool, optional
        If True, applies centering to the decimation when Q>1. If False,
        performs simple decimation without centering. Default is True.
        
        Please note the centering is computed through an offset depending of
        the parity of Q : Q // 2 if Q is even, (Q - 1) / 2 if Q is odd.
        
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
    (P, Q) = normalize_zoom_arg(zoom)
    
    # Check the combination of parameters `out_mode` and `zoom_pq`
    # If zoom Q factor is greater than 1, a decimation will be performed thus
    # impacting the output shape.
    fft_array_filter_check_args(out_mode, (P, Q))
    
    # check data and compute convolution margins
    fil, win, axes, conv_margins = fft_array_filter_check_data(
        arr, fil, win, (P, Q), axes
    )

    convol_fct = signal.oaconvolve
    conv_arr = None

    # Compute the shift to apply to the full output in order to get the same
    # window as input.
    # This default computation corresponds to the case where BoundaryPad is set
    # to NONE for all edges
    shift_same = np.asarray(
        [0 if i not in axes else fil.shape[i] // 2 for i in range(fil.ndim)]
    )

    # Get the boundary management - ensure the number of pairs aligns with
    # `ndim` by left-padding with fill=BoundaryPad.NONE which is neutral
    # for the process that will follow.
    boundary = np.asarray(
        tuplify(boundary, ndim=arr.ndim, fill=BoundaryPad.NONE, strict=False)
    )

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
        try:
            margins = np.where(boundary != BoundaryPad.NONE, margins, 0)
        except ValueError as err:
            raise ValueError(
                f"shift_same : {shift_same}\n",
                f"boundary: {boundary}\n",
                f"margins : {margins}\n",
                f"margins[:, 0]: {margins[:,0]}"
                f"conv_margins : {conv_margins}\n",
                f"fil : {fil}\n",
                f"fil.shape : {fil.shape}\n",
            ) from err

        # For output : in order to get the same window we have to take
        # account of used margins to shift the window
        try:
            shift_same += margins[:, 0]
        except ValueError as err:
            raise ValueError(
                f"shift_same : {shift_same}\n"
                f"margins : {margins}\n",
                f"margins[:, 0]: {margins[:,0]}"
                f"conv_margins : {conv_margins}\n",
                f"fil : {fil}\n",
                f"fil.shape : {fil.shape}\n",
            ) from err

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
        
        if Q > 1:
            decimation_offset = 0
            if centered_decimation:
                if Q % 2:
                    decimation_offset = Q // 2
                else:
                    decimation_offset = (Q - 1) // 2

            # Compose the indices with the decimation
            indices = tuple(
                (
                    (
                        slice(None, None)
                        if i not in axes
                        else compose_slice(indices[i], slice(decimation_offset, None, Q), out.shape[i])
                    )
                    for i in range(arr.ndim)
                )
            )

        out = out[indices]
    
    else:
        raise NotImplementedError

    win_same = np.asarray([shift_same, shift_same + win[:, 1] - win[:, 0]]).T

    return out, win_same
