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
Array utils module
"""
# pylint: disable=C0413
import sys
from typing import Any, Literal, NoReturn, Optional, Tuple, Union

import numpy as np
import rasterio
from gridr.cdylib import (
    PyArrayWindow2,
    py_array1_replace_f32_i8,
    py_array1_replace_f32_u8,
    py_array1_replace_f64_i8,
    py_array1_replace_f64_u8,
    py_array1_replace_i8,
    py_array1_replace_u8,
)

PY311 = sys.version_info >= (3, 11)

if PY311:
    from typing import Self  # noqa: E402, F401
else:
    from typing_extensions import Self  # noqa: E402, F401
# pylint: enable=C0413


def array_replace(
    array: np.ndarray,
    val_cond: Union[int, float],
    val_true: Union[int, float],
    val_false: Union[int, float],
    array_cond: Optional[np.ndarray] = None,
    array_cond_val: Optional[Union[int, float]] = None,
    win: Optional[np.ndarray] = None,
) -> NoReturn:
    """Replaces elements within an array in-place based on specified conditions.

    This method is a Python wrapper around the Rust function
    `py_array1_replace_*`, designed for efficient in-place modification of NumPy
    arrays. It allows conditional replacement based on either the `array` itself
    or an optional `array_cond` (condition array).

    Parameters
    ----------
    array : numpy.ndarray
        The array into which elements are replaced in-place.
        Must be C-contiguous and have a `dtype` of `int8`, `uint8`, `float32`,
        or `float64`.

    val_cond : int or float
        The primary condition value. Elements in `array` (or `array_cond` if
        provided) equal to `val_cond` will be affected.

    val_true : int or float
        The value to replace elements that satisfy the condition (i.e., input
        value equals `val_cond`).

    val_false : int or float
        The value to replace elements that do *not* satisfy the condition (i.e.,
        input value does not equal `val_cond`).

    array_cond : numpy.ndarray, optional
        An optional 1D or 2D array on which to apply the condition. If provided,
        the replacement in `array` is based on the values in `array_cond`. Must
        have a `dtype` of `int8` or `uint8`. Defaults to `None`, in which case
        the `array` itself is used for the condition.

    array_cond_val : int or float, optional
        The condition value to use if `array_cond` is defined. This value is
        compared against elements in `array_cond`. This parameter is required
        if `array_cond` is provided. Defaults to `None`.

    win : numpy.ndarray, optional
        A window `win` to restrict the operation to a specific region of the
        `array`.
        This is a 2D NumPy array where each row represents a dimension and
        contains `(min_idx, max_idx)`. **Both `min_idx` and `max_idx` are
        inclusive**, adhering to the GridR's "window" convention. The window's
        dimensions must match those of the `array`. Defaults to `None`, meaning
        the operation applies to the entire array.

    Returns
    -------
    NoReturn
        This function modifies the `array` in-place and does not return any
        value.

    Raises
    ------
    AssertionError
        If `array`'s `dtype` is not one of `int8`, `uint8`, `float32`, or
        `float64`.

    AssertionError
        If `array` is not C-contiguous (`array.flags.c_contiguous` is `False`).

    AssertionError
        If `array_cond` is provided and its `dtype` is not `int8` or `uint8`.

    AssertionError
        If `array_cond` is provided but `array_cond_val` is `None`.

    AssertionError
        If `array_cond_val` is provided but `array_cond` is `None`.

    """
    assert array.dtype in (np.int8, np.uint8, np.float32, np.float64)
    assert array.flags.c_contiguous is True
    if array_cond is not None:
        assert array_cond.dtype in (np.int8, np.uint8)
        assert array_cond_val is not None
    if array_cond_val is not None:
        assert array_cond is not None

    py_window = None
    if win is not None:
        py_window = PyArrayWindow2(
            start_row=win[0][0], end_row=win[0][1], start_col=win[1][0], end_col=win[1][1]
        )

    nrow, ncol = array.shape
    array = array.reshape(-1)
    if array_cond is not None:
        array_cond = array_cond.reshape(-1)

    py_array_replace_func = {
        (np.dtype("int8"), np.dtype("int8")): py_array1_replace_i8,
        (np.dtype("float32"), np.dtype("int8")): py_array1_replace_f32_i8,
        (np.dtype("float64"), np.dtype("int8")): py_array1_replace_f64_i8,
        (np.dtype("uint8"), np.dtype("uint8")): py_array1_replace_u8,
        (np.dtype("float32"), np.dtype("uint8")): py_array1_replace_f32_u8,
        (np.dtype("float64"), np.dtype("uint8")): py_array1_replace_f64_u8,
    }
    if array_cond is not None:
        py_array_replace_func[(array.dtype, array_cond.dtype)](
            array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, py_window
        )
    else:
        py_array_replace_func[(array.dtype, array.dtype)](
            array, nrow, ncol, val_cond, val_true, val_false, None, None, py_window
        )


def is_clip_required(in_dtype: np.dtype, out_dtype: np.dtype) -> bool:
    """
    Determines if clipping is required when converting between data types.

    Args:
        in_dtype: The source data type
        out_dtype: The destination data type

    Returns:
        bool: True if clipping is required, False otherwise

    Raises:
        ValueError: If the conversion between types is not managed
    """
    in_dtype_info = np.iinfo(in_dtype) if in_dtype.kind in "iu" else np.finfo(in_dtype)
    out_dtype_info = np.iinfo(out_dtype) if out_dtype.kind in "iu" else np.finfo(out_dtype)
    in_min, in_max = np.float64(in_dtype_info.min), np.float64(in_dtype_info.max)
    out_min, out_max = np.float64(out_dtype_info.min), np.float64(out_dtype_info.max)

    match (in_dtype.kind, out_dtype.kind):
        # Cases where we need to check if source values fit in destination range
        case ("f", "f") | ("i", "f") | ("i", "i") | ("f", "i"):
            clip_required = in_min < out_min or in_max > out_max

        # Cases where clipping is always required
        case ("f", "u") | ("i", "u"):
            clip_required = True

        # Unsigned to unsigned conversion
        case ("u", "u") | ("u", "i") | ("u", "f"):
            clip_required = in_max > out_max

        case _:
            raise ValueError(f"Conversion from {in_dtype} to {out_dtype} is not managed")

    return clip_required


def is_clip_to_dtype_limits_safe(in_dtype: np.dtype, out_dtype: np.dtype) -> bool:
    """
    Determines whether clipping from an input type to the type limits of a target data type is safe.

    This function checks if converting from a data type to a target data type and clipping it to the
    target type's limits will preserve all values without overflow.

    Parameters
    ----------
    in_dtype : np.dtype
        The input type to be checked for safe clipping.
    out_dtype : np.dtype
        The target data type to which the array would be converted.

    Returns
    -------
    bool
        True if clipping to the target type limits is safe (no overflow will occur and the target
        limit can be expressed in the input data type with precision), False otherwise.

    Notes
    -----
    This function is necessary when performing type conversions between different numerical
    data types, especially when converting between floating-point and integer types or
    between different floating-point precisions. The main concern is to prevent overflow
    when clipping values to the target type's limits.

    The function performs the following checks:
    
        1. Only floating-point input types are considered for this check (integer inputs
           are assumed to be safe by default).
       
        2. Checks if clipping is actually required between the input and output types.
    
        3. Attempts to convert the maximum value of the output type to the input type and convert it
           back to the output type.
       
        4. If this conversion results in an OverflowError or if the converted value
           doesn't match the expected maximum value of the output type, returns False.

    The function is particularly important when processing numerical data where preserving
    the integrity of values is critical, such as in scientific computing, financial
    applications, or any domain where numerical precision matters.

    The test is only performed on the max as the min of an integer type is a power of 2 and can
    safely expressed when clipping is required.
    """
    is_safe = True
    # The test has only to be done considering float inputs
    if in_dtype.kind == "f" and is_clip_required(in_dtype, out_dtype):
        type_info = None
        if out_dtype.kind == "f":
            type_info = np.finfo(out_dtype)
        else:
            type_info = np.iinfo(out_dtype)
        try:
            is_safe = out_dtype.type(in_dtype.type(type_info.max)) == type_info.max
        except OverflowError:
            is_safe = False
    return is_safe


def array_convert(
    array_in: np.ndarray,
    array_out: np.ndarray,
    clip: Union[Literal["auto"], Tuple[Any, Any]] = "auto",
    safe: bool = True,
    rounding_method: Optional[Literal["round", "ceil", "floor"]] = "round",
) -> None:
    """
    Convert an input array to a target dtype with optional clipping and rounding.

    The function performs type conversion from the input array to the output array,
    with optional clipping and rounding operations based on the specified parameters.

    Parameters
    ----------
    array_in : numpy.ndarray
        The input array containing the data to convert. The dtype of this array
        determines the source data type.

    array_out : numpy.ndarray
        The output array that will contain the converted data. The dtype of this
        array determines the target data type.

    clip : Union[Literal['auto'], Tuple[Any, Any]]]
        Controls value clipping behavior:

        -   'auto': Automatic clipping based on the target dtype's range
        -   Tuple: Specific numeric range for clipping (e.g., (min_value, max_value)).
                 The tuple values should correspond to the target dtype's range

    safe : bool
        If True, check are performed to garanty that the automatic clipping can be performed without
        overflow caused by floating point precision loss.

    rounding_method : Optional[Literal['round', 'ceil', 'floor']], optional
        Specifies the rounding method to use when converting from float to integer:

        -   'round': Standard rounding (default)
        -   'ceil': Round up to nearest integer
        -   'floor': Round down to nearest integer
        -   None: No rounding is performed

    Returns
    -------
    None
        The function modifies both the input array and the output array inplace
        and returns nothing.

    Notes
    -----

    -   The function checks if clipping is required based on the input and output
        dtypes and the specified clipping range.
    -   When 'auto' clipping is specified, the function automatically clips to the
        range of the output dtype.
    -   The function currently raises an Exception if clipping from input type to output type is not
        safe.
    -   The function raises ValueError if the specified clipping range is invalid
        for the output dtype.
    -   No safety check is performed when the clipping range is not automatic

    """
    in_dtype = array_in.dtype
    out_dtype = array_out.dtype

    # Get output dtype min and max codable values.
    out_dtype_info = np.iinfo(out_dtype) if out_dtype.kind in "iu" else np.finfo(out_dtype)
    out_dtype_range = (out_dtype_info.min, out_dtype_info.max)

    # Determines from types if clip is required
    clip_required = is_clip_required(in_dtype, out_dtype)

    # Define clip range
    clip_range = None
    if clip == "auto" and clip_required:
        clip_range = (out_dtype_range[0], out_dtype_range[1])

        if safe and not is_clip_to_dtype_limits_safe(in_dtype, out_dtype):
            raise Exception(
                f"Clipping to output dtype limits is not safe from {in_dtype} to {out_dtype}"
            )

    elif clip != "auto":
        clip_range = clip
        if min(clip_range) < out_dtype_info.min or max(clip_range) > out_dtype_info.max:
            raise ValueError(f"Clipping range {clip_range} is not allowed on type {out_dtype}")

    match rounding_method:
        case "round":
            np.round(array_in, decimals=0, out=array_in)

        case "ceil":
            np.ceil(array_in, out=array_in)

        case "floor":
            np.floor(array_in, out=array_in)

        case _:
            pass

    if clip_range is not None:
        clip_range_in = (in_dtype.type(clip_range[0]), in_dtype.type(clip_range[1]))
        np.clip(array_in, a_min=clip_range_in[0], a_max=clip_range_in[1], out=array_in)
        overflow = (array_in < clip_range[0]) | (array_in > clip_range[1])
        if np.any(overflow):
            raise Exception(f"Some value(s) are out of {out_dtype} range after clipping")
        # raise Exception(array_in)
        array_out[:] = array_in

    else:
        array_out[:] = array_in


class ArrayProfile(object):
    """
    A class to define array attributes for mocking or descriptive purposes.

    This class is designed to hold essential attributes of a NumPy array, such
    as its shape, number of dimensions, data type, and total size, allowing
    access to these members similar to a `numpy.ndarray` object without
    instantiating a full array.
    """

    def __init__(self, shape: Tuple[int, ...], ndim: int, dtype: np.dtype):
        """
        Initializes an `ArrayProfile` object.

        Parameters
        ----------
        shape : tuple of int
            The shape of the array, e.g., `(rows, cols, bands)`.

        ndim : int
            The number of dimensions of the array.

        dtype : numpy.dtype
            The data type of the array's elements, e.g., `np.int16`,
            `np.float32`.

        Returns
        -------
        None
            This constructor initializes the object's attributes.
        """
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(self.shape)

    @classmethod
    def from_dataset(cls, ds: rasterio.io.DatasetReader) -> Self:
        """
        Creates an `ArrayProfile` object from a `rasterio.io.DatasetReader`.

        This class method extracts relevant array attributes (shape, number of
        dimensions, and data type) from an opened Rasterio dataset. It adjusts
        the `ndim` and `shape` for single-band datasets to represent them as 2D
        arrays, consistent with typical image processing.

        Parameters
        ----------
        ds : rasterio.io.DatasetReader
            A Rasterio dataset reader object, typically obtained via `
            rasterio.open()`.

        Returns
        -------
        ArrayProfile
            An instantiated `ArrayProfile` object populated with attributes
            derived from the provided Rasterio dataset.
        """
        shape = (ds.count, ds.height, ds.width)
        ndim = 3
        if ds.count == 1:
            shape = (ds.height, ds.width)
            ndim = 2
        return cls(shape=shape, ndim=ndim, dtype=np.dtype(ds.profile["dtype"]))
