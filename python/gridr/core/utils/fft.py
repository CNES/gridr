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
"""Fast Fourier Transform related functions.

This module provides functions for computing inverse Fast Fourier Transforms.

Functions
---------
ifft : Compute the inverse Fast Fourier Transform of an array with optional
       frequency domain shifting
"""
import numpy as np


def ifft(array: np.ndarray, shift: bool = True, shift_after: bool = True) -> np.ndarray:
    """Compute the inverse Fast Fourier Transform of the input array.

    This function computes the inverse FFT of a 1D or 2D array and provides
    optional frequency domain shifting before and after the transform.

    Parameters
    ----------
    array : numpy.ndarray
        Input array (1D or 2D) representing frequency domain data with
        zero-frequency component at the center of the spectrum
    shift : bool, default True
        If True, shifts the input array so that the zero-frequency component
        is at the beginning of the spectrum before computing the inverse FFT
    shift_after : bool, default True
        If True, shifts the result so that the zero spatial frequency index
        is at the center of the output array

    Returns
    -------
    numpy.ndarray
        The inverse FFT result with appropriate frequency domain shifting
        applied

    Notes
    -----
    The function assumes the input array has its zero-frequency component at
    the center of the spectrum. This is typical for FFT results where the DC
    component is positioned at the middle of the array.

    Examples
    --------
    >>> import numpy as np
    >>> freq_data = np.random.rand(100)
    >>> spatial_data = ifft(freq_data, shift=True, shift_after=True)
    """
    if shift:
        array = np.fft.ifftshift(array)

    if array.ndim == 2:
        ifft_result = np.fft.ifft2(array)
    elif array.ndim == 1:
        ifft_result = np.fft.ifft(array)
    else:
        raise ValueError(
            f"Unsupported array dimension: {array.ndim}. " f"Only 1D and 2D arrays are supported."
        )

    if shift_after:
        ifft_result = np.fft.fftshift(ifft_result)

    return ifft_result
