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
Cardinal B-spline prefiltering for B-spline interpolation.

This module wraps the rust functions implementing the prefiltering operations
required for cardinal B-spline interpolation, following the theoretical
framework described in Briand & Monasse (2018) :cite:`briand2018theory`
"""
# pylint: disable=C0413
from typing import NoReturn, Optional

import numpy as np

from gridr.cdylib import (
    py_array1_bspline_prefiltering_f64,
    py_compute_2d_domain_extension,
    py_compute_2d_truncation_index,
)
from gridr.core.interp.interpolator import BSplineInterpolator, is_bspline


def compute_2d_truncation_index(n: int, precision: int = 6) -> np.ndarray:
    r"""Compute the bspline prefiltering truncation index for max order n

    The truncation index is computed as an array of math:`N(i, \epsilon)`
    for :math:`1 ≤ i ≤ \tilde{n}`.
    The first element of the returned array corresponds to :math:`N(1, \epsilon`.
    The array is of fixed size and contains zeros on non computed indexes.

    This method wraps the Rust function `py_compute_2d_truncation_index`.

    Parameters
    ----------
    n: int
        The bspline order

    precision : int
        The precision parameter for the truncation index calculation as a number
        of decimal

    Returns
    -------
    np.ndarray
        A numpy array with dtype `np.uintp` containing the truncation indexes.
    """
    fprec = 10.0 ** (-precision)
    return py_compute_2d_truncation_index(n, fprec)


def compute_2d_domain_extension(n: int, precision: int = 6) -> np.ndarray:
    r"""Computes the extended domain lengths for B-spline pre-filtering using
    Approach 1 (Extended Domain) Eq. (54).

    This method wraps the Rust function `py_compute_2d_domain_extension`.

    This function implements the first approach for computing pre-filtering
    coefficients with precision, as described in the paper. It calculates the
    extended domain lengths :math:`L_j^{(n, \epsilon)}` for the B-spline
    pre-filtering process.

    The addition of the number of poles and the total sum of the domain lengths
    extension gives the margin required for the full bspline interpolation
    process in order to achieve a precision of `precision` decimals.

    The extended domain length is computed recursively using the formula:

    .. math::
        L_{\tilde{n}}^{(n, \epsilon)} = \tilde{n}
        L_{j}^{(n, \epsilon)} = L_{j+1}^{(n, \epsilon)} + N^{(j+1, \epsilon)}, j = \tilde{n}-1 \to 0

    Where:

      - :math:`n` is the spline order
      - :math:`\epsilon` is the precision parameter
      - :math:`\tilde{n} = \lfloor \frac{n}{2} \rfloor`
      - :math:`N^{(i, \epsilon)}` is the truncation index computed by `compute_2d_truncation_index`

    Parameters
    ----------
    n: int
        The bspline order

    precision : int
        The precision parameter for the truncation index calculation as a number
        of decimal

    Returns
    -------
    np.ndarray
        A numpy array with dtype `np.uintp` containing the domain extension
        lengths indexes.
    """
    fprec = 10.0 ** (-precision)
    return py_compute_2d_domain_extension(n, fprec)


def compute_bspline_total_margin(
    n: str,
    precision: int = 6,
) -> int:
    """Compute the BSpline required margin to perfom both pre-filtering and
    interpolation.

    The total margin is obtained through the first element of the domain
    extension array.

    Parameters
    ----------
    n: int
        The bspline order

    precision : int
        The precision parameter for the truncation index calculation as a number
        of decimal

    Returns
    -------
    int
        The margin required for the full BSpline interpolation process, ie. the
        pre-filtering and the interpolation itself.
    """
    return compute_2d_domain_extension(n, precision)[0]


def array_bspline_prefiltering(
    array_in: np.ndarray,
    array_in_mask: Optional[np.ndarray] = None,
    interp: Optional[BSplineInterpolator] = None,
    n: Optional[int] = None,
    trunc_idx: Optional[np.ndarray] = None,
    precision: Optional[int] = 6,
    mask_influence_threshold: Optional[float] = 0.001,
) -> NoReturn:
    """In-place B-spline prefiltering for cardinal B-spline interpolation.

    This method applies causal and anti-causal recursive prefiltering to prepare
    data for B-spline interpolation, following Algorithm 4 from Briand & Monasse
    (2018). It delegates computation to an optimized Rust implementation for
    performance.

    The prefiltering approximates theoretically infinite recursive filters using
    finite sums, with precision controlled by the truncation index (derived from
    `precision` parameter).

    This method provides two mutually exclusive modes:

      - **Via interpolator object**: Provide `interp` (must be initialized)
      - **Direct specification**: Provide `n` with either `trunc_idx` or
        `precision` (they are mutually exclusive)


    When a mask is provided, invalid pixels are propagated based on the
    exponential decay of recursive filters. The influence at distance `d` decays
    as :math:`|z_k|^d`, where :math:`z_k` is the dominant B-spline pole. The
    propagation radius is computed as:

    .. math::
        d \\geq \\frac{\\ln(s)}{\\ln(|z_k|)}

    where `s` is the `mask_influence_threshold`. This uses Manhattan distance,
    creating a diamond-shaped influence zone. Border regions corresponding to
    domain extension are automatically invalidated.

    **Important**: The threshold `s` is relative to invalid pixel values. For
    outliers (e.g., value 10000 in a 0-255 image), even small thresholds
    (s=0.001) may require large dilation radii. Choose `s` to ensure absolute
    contamination remains acceptable.

    Parameters
    ----------
    array_in : np.ndarray
        Input array to be prefiltered in-place. Must be C-contiguous with shape
        (nrow, ncol) or (nvar, nrow, ncol). Boundary extension must be applied
        beforehand.

    array_in_mask : np.ndarray, optional
        Validity mask (uint8: 1=valid, 0=invalid) with shape (nrow, ncol). If
        provided, invalid regions are dilated by the computed influence radius.
        Modified in-place.

    interp : BSplineInterpolator, optional
        Initialized B-spline interpolator object. Mutually exclusive with `n`.

    n : int, optional
        B-spline order (must be odd: 3, 5, 7, 9, or 11). Mutually exclusive with
        `interp`.

    trunc_idx : np.ndarray, optional
        Precomputed truncation indices :math:`N(i, ε)`. If not provided,
        computed from `precision`.
        This parameter is not used if `interp` is defined.

    precision : int, default=6
        Number of decimal places for precision parameter
        :math:`ε = 10^{-precision}`.
        Used to compute truncation indices when `trunc_idx` is not provided.
        Smaller values require larger margins but higher accuracy.
        This parameter is not used if `interp` is defined.

    mask_influence_threshold : float, default=0.001
        Residual influence threshold `s` (0 < s < 1) for invalid pixel
        propagation. Value of 1 means no propagation.
        This parameter is not used if `interp` is defined.


    Raises
    ------
    ValueError
        If both `interp` and `n` are provided, if neither is provided, or if
        `interp` is not a B-spline interpolator.

    AssertionError
        If input arrays are not C-contiguous, if mask dtype is not uint8, or if
        array shapes are incompatible.

    Exception
        If `precision` or `mask_influence_threshold` is missing when using direct
        mode.

    Returns
    -------
    None

    Notes
    -----
    - All arrays must be C-contiguous
    - When using `interp`, it must be initialized via `interp.initialize()`
    - The input array is modified in-place
    - This is a Python wrapper around the optimized Rust function
      `array1_bspline_prefiltering_ext_gene`
    - When using mask, a temporary buffer is internally allocated in the Rust function
      in order to perform mask dilatation.

    References
    ----------
    :cite:`briand2018theory` Briand, T., & Monasse, P. (2018). Theory and Practice of Image B-Spline
    Interpolation. *Image Processing On Line*, 8, 99-141.

    Examples
    --------
    >>> import numpy as np
    >>> from gridr import array_bspline_prefiltering
    >>>
    >>> # Direct mode with order specification
    >>> data = np.random.rand(100, 100)
    >>> array_bspline_prefiltering(data, n=5, precision=6)
    >>>
    >>> # With mask
    >>> mask = np.ones((100, 100), dtype=np.uint8)
    >>> mask[40:60, 40:60] = 0  # Invalid region
    >>> array_bspline_prefiltering(data, array_in_mask=mask, n=5,
    ...                            mask_influence_threshold=0.001)

    """
    # Manage interpolator definition mode
    if interp is not None and n is not None:
        raise ValueError(
            "The parameters `interp` and `n` are mutually exclusives. "
            "Please only provide one of them"
        )
    elif interp is None and n is None:
        raise ValueError(
            "Both parameters `interp` and `n` are undefined. "
            "Please provide one (and only one) of them"
        )
        
    assert array_in.flags.c_contiguous is True

    array_in_shape = array_in.shape
    if len(array_in_shape) == 2:
        array_in_shape = (1,) + array_in_shape
    array_in = array_in.reshape(-1)

    # Manage optional input mask
    # array_in_mask_dtype = np.dtype("uint8")
    if array_in_mask is not None:
        assert array_in_mask.flags.c_contiguous is True
        # check shape
        assert array_in_mask.dtype == np.dtype("uint8")
        assert array_in_mask.shape[0] == array_in_shape[1]
        assert array_in_mask.shape[1] == array_in_shape[2]
        # reshape
        array_in_mask = array_in_mask.reshape(-1)

    # Functional branch
    if n is not None:
        if trunc_idx is None and precision is None:
            raise Exception(
                "You must provide either a value for the `precision` parameter"
                " or for the `trunc_idx` parameter"
            )
        if trunc_idx is not None and precision is not None:
            raise Exception(
                "The `precision` parameter and the `trunc_idx` parameter are"
                " mutually exclusive"
            )
            
        # Manage optional trunc_idx and precision
        if trunc_idx is not None:
            assert trunc_idx.flags.c_contiguous is True
            assert trunc_idx.dtype.kind in "iu"
            if trunc_idx.dtype != np.dtype("uintp"):
                trunc_idx = trunc_idx.astype(np.dtype("uintp"))

            # Set a valid value for unused precision
            if precision is None:
                precision = 0.0
        else:
            precision = 10 ** (-precision)

        if array_in_mask is not None and mask_influence_threshold is None:
            raise Exception("Missing a value for the `mask_influence_threshold` parameter")

        # Call the rust method
        py_array1_bspline_prefiltering_f64(
            n=n,
            epsilon=precision,
            array_in=array_in,
            array_in_shape=array_in_shape,
            array_in_mask=array_in_mask,
            trunc_idx=trunc_idx,
            mask_influence_threshold=mask_influence_threshold,
        )

    # Object branch
    else:
        if not is_bspline(interp):
            raise ValueError(f"The interpolator {interp} is not a B-Spline interpolator")
            
        interp.array1_bspline_prefiltering_ext_f64(
            array_in=array_in,
            array_in_shape=array_in_shape,
            array_in_mask=array_in_mask,
        )
