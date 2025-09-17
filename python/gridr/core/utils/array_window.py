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
Module for operations on array's window : check, extend, overflow

---

Window Convention (``win``)

Throughout this module, parameters representing a **window** (often named `win`)
or a subset selection of data adhere to a specific convention. These parameters
are consistently represented as a **2D NumPy array**, where each row corresponds
to a dimension of the underlying data object. Each row contains a pair of
indices: `(min_idx, max_idx)`. It's crucial to note that **both `min_idx` and
`max_idx` are inclusive**.

For example:
::

    # For a 2D array, selecting rows 10 to 20 and columns 5 to 15:
    win_param = np.array([[10, 20], [5, 15]])


**Note on "Chunk" Convention vs. "Window" Convention**:

While "window" parameters use an **inclusive** `min_idx` and `max_idx` for both
bounds, the "chunk" convention (as adopted in GridR for certain functions)
follows Python's standard slicing, where the `start_index` is inclusive but the
`stop_index` is **exclusive**. Always refer to the specific function's docstring
for its expected input/output convention to avoid off-by-one errors.

"""
from typing import Optional, Tuple, Union

import numpy as np
from rasterio.windows import Window

# Inside to outside signs for each edge
WINDOW_EDGE_OUTER_SIGNS = np.array((-1, 1))
# Outside to inside signs for each edge
WINDOW_EDGE_INNER_SIGNS = np.array((1, -1))


def window_expand_ndim(win: np.ndarray, insert: np.ndarray, pos: int = 0) -> np.ndarray:
    """Expand a window by inserting a new dimension at the beginning or end.

    This function takes an existing window (typically a 2D NumPy array where
    each row represents a dimension's `(start, end)` bounds) and inserts a new
    dimension's bounds at a specified position.

    Parameters
    ----------
    win : numpy.ndarray
        The input window. This array will not be modified by the function.
        It's expected to be a 2D array-like where each row is a
        `(min_idx, max_idx)` pair.

    insert : numpy.ndarray or tuple[int, int]
        The element to insert as the new dimension's bounds. This should be
        a 1D array-like or a tuple of two integers `(min_idx, max_idx)`.

    pos : int, default 0
        The position of insertion.

          - `0` : Inserts the new dimension at the beginning (index 0).
          - `-1` : Inserts the new dimension at the end.

    Returns
    -------
    numpy.ndarray
        The expanded window with the new dimension inserted at the specified
        position.

    Raises
    ------
    ValueError
        If the argument `pos` is neither `0` nor `-1`.
    """
    if pos not in (0, -1):
        raise ValueError("The argument 'pos' must be either 0 or -1")
    insert = np.copy(np.asarray(insert))
    win = np.copy(win)

    if pos == 0:
        win = np.vstack((insert, win[:]))
    else:
        win = np.vstack((win[:], insert))
    return win


def window_shift(
    win: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Shift an existing window by a scalar bias defined for each dimension.

    This function adjusts the boundaries of an N-dimensional window by adding
    a corresponding shift value to both the start and end index of each
    dimension.

    For example:
    ::

        [(a,b), (c,d)] shifted by [u, v] becomes [(a+u, b+u), (c+v, d+v)]

    Parameters
    ----------
    win : numpy.ndarray
        The input window. This should be a 2D NumPy array where each row
        represents a dimension, and the two columns represent the inclusive
        first and last index for that dimension, e.g., `((first_row, last_row),
        (first_col, last_col))`.
        The number of rows in `win` must correspond to the number of dimensions
        being considered. If certain axes are not being shifted, their
        corresponding rows in `win` should still be present.

    shift : numpy.ndarray
        A 1D NumPy array containing the scalar shift value for each dimension.
        The length of this array must match the number of dimensions (rows) in
        `win`.

    Returns
    -------
    numpy.ndarray
        The new window with shifted boundaries. The original `win` array is not
        modified.

    Raises
    ------
    AssertionError
        If `shift` is not a 1-dimensional array.

    AssertionError
        If the number of elements in `shift` does not match the number of
        dimensions (rows) in `win`.

    """
    assert shift.ndim == 1
    assert shift.shape[0] == win.shape[-2]
    return np.swapaxes(np.swapaxes(win, -2, -1) + shift, -2, -1)


def window_from_chunk(
    chunk: np.ndarray,
    origin: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns a window from a chunk definition.

    This function converts a 'chunk' definition, which uses Python slicing
    conventions (inclusive start, exclusive stop), into a 'window' definition.
    The 'window' convention, as used throughout this module, uses **inclusive**
    start and end indices for both bounds.

    The conversion involves adjusting the 'stop' index of each dimension in the
    chunk by subtracting one to make it inclusive for the window. An optional
    `origin` array can also be applied as a bias to these indices.

    Parameters
    ----------
    chunk : numpy.ndarray
        An N-dimensional chunk definition. This is typically a 2D NumPy array
        where each row represents a dimension, and the two columns are
        `(start_index, stop_index)` following Python's slicing (inclusive start,
        exclusive stop).

    origin : numpy.ndarray, optional
        A 1-dimensional NumPy array containing a bias to apply to the indices
        of each dimension. The i-th element in the `origin` array is applied
        to the i-th axis. Defaults to ``None``, meaning no bias is applied.

    Returns
    -------
    numpy.ndarray
        The converted chunk in the module's window convention
        (inclusive start, inclusive end for each dimension).

    """
    win = np.asarray(chunk)
    # change convention
    win[..., 1] -= 1
    if origin is not None:
        win = window_shift(win, origin)
    return win


def window_indices(
    win: np.ndarray,
    reset_origin: bool = False,
    axes: Optional[Union[int, Tuple[int, ...], np.ndarray]] = None,
) -> Tuple[slice]:
    """Get slicing indices for an array from a window definition.

    This function converts a window definition (using the module's inclusive
    convention) into a tuple of `slice` objects. These slices can then be
    directly used to obtain a view of a NumPy array.

    Parameters
    ----------
    win : numpy.ndarray
        The input window. This is a 2D NumPy array where each row represents a
        dimension, and the two columns are the **inclusive first and inclusive
        last index** for that dimension, e.g., `((first_row, last_row),
        (first_col, last_col))`.
        The number of rows in `win` must match the number of dimensions of the
        array you intend to slice. If certain axes are not meant to be
        constrained by the window, their corresponding rows in `win` should
        still be present, or `axes` should be used to specify which dimensions
        to consider.

    reset_origin : bool, default False
        If `True`, each window interval will be adjusted by subtracting its
        `min_idx`. This results in slices whose starting elements are `0`,
        effectively making the view relative to the start of the window.

    axes : int or tuple of int or numpy.ndarray, optional
        Specifies the axes (dimensions) of the array on which to apply the
        window constraints. If `None`, the window is applied to all dimensions
        defined by `win.shape[0]`. Defaults to `None`.

    Returns
    -------
    tuple of slice
        A tuple of `slice` objects, where each slice corresponds to a dimension
        of the array. These slices are ready for direct use in NumPy array
        indexing.
    """
    win = np.asarray(win)

    if axes is None:
        axes = range(win.shape[0])
        axes = np.atleast_1d(axes)  # pylint: disable=R0204

    if reset_origin:
        win = win - np.vstack(win[:, 0])

    indices = tuple(
        (
            slice(None, None) if i not in axes else slice(int(win[i][0]), int(win[i][1] + 1))
            for i in range(win.shape[0])
        )
    )
    return indices


def window_from_indices(
    indices: Tuple[slice],
    original_shape: Tuple[int, ...],
    axes: Optional[Union[int, Tuple[int, ...], np.ndarray]] = None,
) -> np.ndarray:
    """Reconstructs a window (``win`` convention) from slicing indices.

    This function converts a tuple of `slice` objects (which follow Python's
    inclusive start, exclusive stop convention) back into the module's `win`
    convention (inclusive start, inclusive end for both bounds).
    It requires the `original_shape` to correctly determine the `stop`
    index for dimensions where the slice is `None` or implicitly covers the
    entire dimension.

    Parameters
    ----------
    indices : tuple of slice
        A tuple of `slice` objects. Each slice follows Python's standard `start`
        (inclusive) and `stop` (exclusive) convention

    original_shape : tuple of int
        The full N-dimensional shape of the array to which these `indices`
        would be applied. This is necessary to correctly determine the `max_idx`
        for slices where `slice.stop` is `None`.

    axes : int or tuple of int or numpy.ndarray, optional
        The axes (dimensions) that were considered when creating the `indices`.
        If `None`, it is assumed that `indices` contains slices for all
        dimensions and that all axes were considered. Defaults to `None`.

    Returns
    -------
    numpy.ndarray
        The reconstructed window, formatted as a 2D NumPy array, where each row
        represents a dimension and contains `(min_idx, max_idx)` with both indices
        being **inclusive**, adhering to the module's "window" convention.
    """
    ndim = len(indices)
    win = np.zeros((ndim, 2), dtype=int)

    if axes is None:
        axes = range(ndim)
    axes = np.atleast_1d(axes)

    for i in range(ndim):
        if i in axes:
            s = indices[i]
            # Invert slice(start, stop) to get (start, stop-1)
            # The stop value in slice is exclusive, so we subtract 1 to get the
            # last index
            start = s.start if s.start is not None else 0
            # Default to last element if None
            stop = s.stop - 1 if s.stop is not None else original_shape[i] - 1

            win[i, 0] = start
            win[i, 1] = stop
        else:
            # For axes not included in 'axes' (which result in slice(None, None)),
            # the window effectively covers the entire dimension of the original
            # array.
            # So, we set the start to 0 and the end to original_shape[i] - 1.
            win[i, 0] = 0
            win[i, 1] = original_shape[i] - 1

    return win


def window_apply(
    arr: np.ndarray,
    win: np.ndarray,
    axes: Optional[Union[int, Tuple[int, ...], np.ndarray]] = None,
    check: bool = True,
) -> np.ndarray:
    """Applies a window to an array and returns the windowed view.

    This function provides a convenient way to extract a sub-array (view) from
    a NumPy array using the module's established "window" convention. It
    converts the window definition into NumPy-compatible slices and applies them.

    You can disable the consistency check between the array and the window if it
    has already been performed or is not desired. Be aware that if `check` is
    disabled, the function **does not verify if the window lies entirely within
    the array's boundaries**. In such cases, NumPy will not raise an
    `IndexError`; instead, it will silently limit the window to the available
    data, which can lead to unexpected or "awkward" behavior if the window
    extends beyond the array.

    Parameters
    ----------
    arr : numpy.ndarray
        The input N-dimensional array to which the window will be applied

    win : numpy.ndarray
        The window to apply. This is a 2D NumPy array where each row represents
        a dimension and contains `(min_idx, max_idx)`, with **both indices being
        inclusive**, following the module's "window" convention. Its number of
        rows must match the number of dimensions in `arr`, or the number of
        axes specified.

    axes : int or tuple of int or numpy.ndarray, optional
        Specifies the axes (dimensions) of the array on which to apply the
        window constraints. If `None`, the window is applied to all dimensions
        corresponding to the rows in `win`. Defaults to `None`.

    check : bool, default True
        If `True`, performs input consistency checks, including verifying that
        the `win` dimensions match `arr`'s dimensions (or specified `axes`) and
        that the window's bounds lie within the array's boundaries. Set to
        `False` to skip these checks.

    Returns
    -------
    numpy.ndarray
        A view of the input array, constrained by the applied window.

    Raises
    ------
    ValueError
        If `check` is `True` and the `window_check` function fails, indicating
        an inconsistency between the array and the window, or if the window
        lies outside the array's bounds.

    """
    win = np.asarray(win)
    ret = arr

    if check:
        if not window_check(arr, win, axes):
            raise ValueError("window check fails : check window/array " "consistency")
    indices = window_indices(win, reset_origin=False, axes=axes)
    ret = arr[indices]
    return ret


def window_check(
    arr: np.ndarray, win: np.ndarray, axes: Optional[Union[int, Tuple[int, ...], np.ndarray]] = None
) -> bool:
    """Checks if a window lies entirely within an array's shape.

    This method validates if the given window's boundaries are consistent with
    the array's dimensions and if the indices within the window are ordered
    correctly (start index less than or equal to end index).

    The function applies checks based on the module's established "window"
    convention, where both `min_idx` and `max_idx` are inclusive.

    Parameters
    ----------
    arr : numpy.ndarray
        The input N-dimensional array whose shape will be checked against the
        window.

    win : numpy.ndarray
        The window to test. This is a 2D NumPy array where each row represents
        a dimension and contains `(min_idx, max_idx)`. **Both `min_idx` and
        `max_idx` are inclusive**, following the module's "window" convention.
        The number of rows in `win` must correspond to the number of dimensions
        in `arr` if `axes` is `None`.

    axes : int or tuple of int or numpy.ndarray, optional
        The axes (dimensions) of the array on which the check is performed.
        If `None`, the check is performed on all dimensions of `arr`
        corresponding to the rows of `win`. Defaults to `None`.

    Returns
    -------
    bool
        Returns `True` if the window lies entirely within the array's shape
        and its indices are correctly ordered. Returns `False` otherwise (e.g.,
        if arrays are empty).

    Raises
    ------
    ValueError
        If `arr` or `win` are scalar inputs (0-dimensional arrays).

    ValueError
        If the array's number of dimensions (`arr.ndim`) does not equal the
        window's first dimension length (`win.shape[0]`).

    IndexError
        If at least one of the window's dimension ranges has an invalid order
        (i.e., `max_idx < min_idx`).

    """
    win = np.asarray(win)
    ret = True

    if arr.ndim == 0 or win.ndim == 0:  # scalar inputs
        raise ValueError("at least one input array is a scalar")
    elif arr.ndim != win.shape[0]:
        raise ValueError(
            "array's number of dimension should be equal to the " "window's first dimension length"
        )
    elif arr.size == 0 or win.size == 0:  # empty arrays
        ret = False

    if ret:
        if axes is None:
            axes = range(arr.ndim)
        axes = np.atleast_1d(axes)  # pylint: disable=R0204

        # check that first index is greater or equal the last index
        order_test = [np.nan if i not in axes else win[i][1] - win[i][0] >= 0 for i in axes]
        # please note here that nan number are considered as True in np.all
        if ~np.all(order_test):
            raise IndexError("At least one window's dimension range has invalid " "order")

        # the order is ok ; now check that the window lies in the array
        within_test = [
            np.nan if i not in axes else win[i][0] >= 0 and win[i][1] < arr.shape[i]
            for i in range(arr.ndim)
        ]
        ret = np.all(within_test)  # pylint: disable=R0204
    return ret


def window_extend(win: np.ndarray, extent: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Extends or shrinks a window by a specified extent.

    This function adjusts the boundaries of an N-dimensional window by adding
    or subtracting an `extent` value to its `min_idx` and `max_idx` for each
    dimension. The direction of the extension (inward or outward) is controlled
    by the `reverse` parameter and internal sign constants.

    Parameters
    ----------
    win : numpy.ndarray
        The window to extend or shrink. This is a 2D NumPy array where each row
        represents a dimension and contains `(min_idx, max_idx)`.
        **Both `min_idx` and `max_idx` are inclusive**, adhering to the module's
        "window" convention.

    extent : numpy.ndarray
        The integer extents to apply to the window's boundaries. This should be
        a 2D NumPy array where each row corresponds to a dimension of `win`, and
        contains two elements `(start_extent, end_extent)`.
        For example, for a 2D window, it would be
        `((up_extent, bottom_extent), (left_extent, right_extent))`.

    reverse : bool, default False
        Controls the direction of the extension.

          - If `False` (default), the extent is applied from
            **inside to outside**, effectively expanding the window (using
            `WINDOW_EDGE_OUTER_SIGNS`).
          - If `True`, the extent is applied from **outside to inside**,
            effectively shrinking the window (using `WINDOW_EDGE_INNER_SIGNS`).

    Returns
    -------
    numpy.ndarray
        The adjusted window with its boundaries extended or shrunk. The original
        `win` array is not modified.

    """
    win = np.asarray(win)
    extent = np.asarray(extent)

    signs = WINDOW_EDGE_OUTER_SIGNS
    if reverse:
        signs = WINDOW_EDGE_INNER_SIGNS
    return win + signs * extent


def window_overflow(
    arr: np.ndarray, win: np.ndarray, axes: Optional[Union[int, Tuple[int, ...], np.ndarray]] = None
) -> np.ndarray:
    """Computes the overflow of a window relative to an array's shape.

    This function calculates the extent to which each side of a window
    (`min_idx` and `max_idx`) extends beyond the corresponding array dimension's
    boundaries (0 to `shape[i]-1`).

    Overflow values are set to 0 for dimensions (axes) that are not explicitly
    selected via the `axes` argument.

    Parameters
    ----------
    arr : numpy.ndarray
        The input N-dimensional array. Its shape defines the boundaries against
        which the `win` parameter is checked.

    win : numpy.ndarray
        The window to check for overflow. This is a 2D NumPy array where each
        row represents a dimension and contains `(min_idx, max_idx)`. **Both
        `min_idx` and `max_idx` are inclusive**, following the module's "window"
        convention. The number of rows in `win` should typically match the
        number of dimensions in `arr`.

    axes : int or tuple of int or numpy.ndarray, optional
        The axes (dimensions) of `arr` on which to compute the overflow.
        If `None`, the overflow is computed for all dimensions of `arr` (up
        to `arr.ndim`). Overflow values for dimensions not included in `axes`
        will be `0`. Defaults to `None`.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the overflow for each dimension. Each row
        corresponds to a dimension of `arr`, and contains
        `(left_overflow, right_overflow)`.
        `left_overflow` is `abs(min(0, win[i][0]))`, and `right_overflow` is
        `max(0, win[i][1] - arr.shape[i] + 1)`.

    """
    win = np.asarray(win)

    if axes is None:
        axes = range(arr.ndim)
    axes = np.atleast_1d(axes)  # pylint: disable=R0204

    overflow = [
        [0, 0] if i not in axes else [abs(min(0, win[i][0])), max(0, win[i][1] - arr.shape[i] + 1)]
        for i in range(arr.ndim)
    ]

    return np.asarray(overflow)


def window_shape(
    win: np.ndarray, axes: Optional[Union[int, Tuple[int, ...], np.ndarray]] = None
) -> Tuple[int, ...]:
    """Computes the shape of a given window.

    This function calculates the length of each dimension defined by a window,
    following the module's "window" convention where both `min_idx` and `max_idx`
    are inclusive.

    For dimensions not included in `axes`, the corresponding shape element in
    the returned tuple will be `None`, indicating that this dimension's length
    is not computed or relevant for the window's explicit boundaries.

    Parameters
    ----------
    win : numpy.ndarray
        The window for which to compute the shape. This is a 2D NumPy array
        where each row represents a dimension and contains `(min_idx, max_idx)`.
        **Both `min_idx` and `max_idx` are inclusive**, adhering to the module's
        "window" convention. The number of rows in `win` indicates the total
        number of dimensions covered by the window definition.

    axes : int or tuple of int or numpy.ndarray, optional
        The axes (dimensions) for which to compute the length.
        If `None`, the length is computed for all dimensions defined by `win`.
        For dimensions not specified in `axes`, the corresponding element in the
        returned shape tuple will be `None`. Defaults to `None`.

    Returns
    -------
    tuple of int or None
        A tuple representing the shape of the window. Each element is an integer
        representing the length of the dimension, or `None` if that dimension
        was not included in `axes`.

    """
    win = np.asarray(win)

    if axes is None:
        axes = range(win.shape[0])
    axes = np.atleast_1d(axes)  # pylint: disable=R0204

    shape = tuple(
        [None if i not in axes else win[i, 1] - win[i, 0] + 1 for i in range(win.shape[0])]
    )
    return shape


def as_rio_window(win: np.ndarray) -> Window:
    """Converts a window definition to a `rasterio.windows.Window` object.

    This function translates the module's "window" convention (inclusive start
    and end indices) into `rasterio.windows.Window` object, which internally
    uses a slice-like convention (inclusive start, exclusive stop).

    Parameters
    ----------
    win : numpy.ndarray
        The window to convert. This is a 2D NumPy array where each row
        represents a dimension and contains `(min_idx, max_idx)`.
        **Both `min_idx` and `max_idx` are inclusive**, following the module's
        "window" convention.
        The number of rows in `win` must correspond to the number of dimensions
        expected by the `rasterio.windows.Window.from_slices` method (e.g.,
        typically 2 for (row, col) windows).

    Returns
    -------
    rasterio.windows.Window
        The corresponding `rasterio.windows.Window` object.
    """
    win = np.asarray(win)
    args = [(incl_idx_0, incl_idx_1 + 1) for incl_idx_0, incl_idx_1 in win]
    return Window.from_slices(*args)


def from_rio_window(rio_win: Window) -> np.ndarray:
    """Converts a `rasterio.windows.Window` object to a GridR window.

    This function translates a `rasterio.windows.Window` object, which
    internally uses a slice-like convention (inclusive start, exclusive stop),
    into this module's "window" convention (inclusive start and inclusive end
    indices).

    Parameters
    ----------
    rio_win : rasterio.windows.Window
        The `rasterio.windows.Window` object to convert. This object typically
        defines a 2D window using `row_off`, `col_off`, `height`, and `width`
        attributes.

    Returns
    -------
    numpy.ndarray
        The corresponding window in the module's convention. This is a 2D NumPy
        array where the first row defines the row bounds
        `(min_row_idx, max_row_idx)` and the second row defines the column
        bounds `(min_col_idx, max_col_idx)`.
        **Both `min_idx` and `max_idx` are inclusive**.
    """
    win = np.array(
        [
            [rio_win.row_off, rio_win.row_off + rio_win.height - 1],
            [rio_win.col_off, rio_win.col_off + rio_win.width - 1],
        ]
    )
    return win
