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
GridR interpolators interface
"""
# pylint: disable=C0413
from typing import Any, TypeAlias, Union

from gridr.cdylib import (
    BSpline3Interpolator,
    BSpline5Interpolator,
    BSpline7Interpolator,
    BSpline9Interpolator,
    BSpline11Interpolator,
    LinearInterpolator,
    NearestInterpolator,
    OptimizedBicubicInterpolator,
    PyInterpolatorType,
)

# Dictionary mapping short names to interpolator types
INTERPOLATOR_TYPES = {
    "nearest": PyInterpolatorType.Nearest,
    "linear": PyInterpolatorType.Linear,
    "cubic": PyInterpolatorType.OptimizedBicubic,
    "bspline3": PyInterpolatorType.BSpline3,
    "bspline5": PyInterpolatorType.BSpline5,
    "bspline7": PyInterpolatorType.BSpline7,
    "bspline9": PyInterpolatorType.BSpline9,
    "bspline11": PyInterpolatorType.BSpline11,
}

# Dictionary mapping interpolator types to their classes
INTERPOLATOR_TYPE_CLASSES = {
    PyInterpolatorType.Nearest: NearestInterpolator,
    PyInterpolatorType.Linear: LinearInterpolator,
    PyInterpolatorType.OptimizedBicubic: OptimizedBicubicInterpolator,
    PyInterpolatorType.BSpline3: BSpline3Interpolator,
    PyInterpolatorType.BSpline5: BSpline5Interpolator,
    PyInterpolatorType.BSpline7: BSpline7Interpolator,
    PyInterpolatorType.BSpline9: BSpline9Interpolator,
    PyInterpolatorType.BSpline11: BSpline11Interpolator,
}

# Type alias for interpolator classes
Interpolator: TypeAlias = Union[
    NearestInterpolator,
    LinearInterpolator,
    OptimizedBicubicInterpolator,
    BSpline3Interpolator,
    BSpline5Interpolator,
    BSpline7Interpolator,
    BSpline9Interpolator,
    BSpline11Interpolator,
]

BSplineInterpolator: TypeAlias = Union[
    BSpline3Interpolator,
    BSpline5Interpolator,
    BSpline7Interpolator,
    BSpline9Interpolator,
    BSpline11Interpolator,
]

# Type alias for all accepted interpolator identifier
InterpolatorIdentifier: TypeAlias = Union[str, PyInterpolatorType, Interpolator]


def get_interpolator(
    interp: InterpolatorIdentifier,
    **kwargs: Any,
) -> Interpolator:
    """
    Get an instance of an interpolator either from its short name, from its enum type, or from an
    existing object.

    If `interp` is either a `str` or a `PyInterpolatorType`, the corresponding interpolator object
    is instantiated with the provided keyword arguments. If an existing interpolator object is
    passed as an argument, the method returns it as is without any modification.

    Parameters
    ----------
    interp : Union[str, PyInterpolatorType, InterpolatorClasses]
        The interpolator identifier. It can be:

        - A string representing the interpolator name (e.g., "nearest", "linear", "cubic", etc.).
        - A `PyInterpolatorType` enum value.
        - An instance of an interpolator class.

    **kwargs : Any
        Additional keyword arguments to pass to the interpolator constructor if a new instance is
        created.

    Returns
    -------
    InterpolatorClasses
        An instance of the specified interpolator.

    Raises
    ------
    Exception
        If the interpolator name or type is not recognized.
    Exception
        If the type of the `interp` parameter is not supported.

    Examples
    --------
    >>> # Create a nearest interpolator
    >>> nearest_interp = get_interpolator("nearest")
    >>> # Use an existing interpolator
    >>> existing_interp = NearestInterpolator()
    >>> returned_interp = get_interpolator(existing_interp)
    >>> returned_interp is existing_interp  # True
    """
    ret = interp
    if isinstance(interp, str):
        try:
            ret = INTERPOLATOR_TYPE_CLASSES[INTERPOLATOR_TYPES[interp]](**kwargs)
        except KeyError as err:
            raise Exception(f"Unknown interpolator {interp!r}") from err
    elif isinstance(interp, PyInterpolatorType):
        try:
            ret = INTERPOLATOR_TYPE_CLASSES[interp](**kwargs)
        except KeyError as err:
            raise Exception(f"Unknown interpolator {interp!r}") from err
    elif isinstance(interp, tuple(INTERPOLATOR_TYPE_CLASSES.values())):
        pass
    else:
        raise Exception("Bad type for the interp parameter")
    return ret


def is_bspline(
    interp: InterpolatorIdentifier,
) -> bool:
    """
    Check if an interpolator is a B-Spline.

    Parameters
    ----------
    interp : Union[str, PyInterpolatorType, InterpolatorClasses]
        The interpolator identifier. It can be:

        - A string representing the interpolator name (e.g., "nearest", "linear", "cubic", etc.).
        - A `PyInterpolatorType` enum value.
        - An interpolator class
        - An instance of an interpolator class.

    Returns
    -------
    bool
        `True`  if the interpolator is a B-Spline, `False` otherwise
    """
    check_type = interp in (
        "bspline3",
        PyInterpolatorType.BSpline3,
        BSpline3Interpolator,
        "bspline5",
        PyInterpolatorType.BSpline5,
        BSpline5Interpolator,
        "bspline7",
        PyInterpolatorType.BSpline7,
        BSpline7Interpolator,
        "bspline9",
        PyInterpolatorType.BSpline9,
        BSpline9Interpolator,
        "bspline11",
        PyInterpolatorType.BSpline11,
        BSpline11Interpolator,
    )
    check_object = (
        isinstance(interp, BSpline3Interpolator)
        or isinstance(interp, BSpline5Interpolator)
        or isinstance(interp, BSpline7Interpolator)
        or isinstance(interp, BSpline9Interpolator)
        or isinstance(interp, BSpline11Interpolator)
    )
    return check_type or check_object
