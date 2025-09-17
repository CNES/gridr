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
"""Two-dimensional filter processing utilities.

Classes
-------
Filter2d : Two-dimensional filter class for frequency domain processing
"""
import numpy as np

from gridr.core.utils import fft


class Filter2d(object):
    """Two-dimensional filter class for frequency domain processing.

    This class represents a 2D filter with methods for computing spatial kernels
    and checking filter properties. The filter operates in the frequency domain
    and can be converted to spatial domain representations.

    Attributes
    ----------
    fil : numpy.ndarray
        Frequency domain representation of the filter
    freq_x : numpy.ndarray
        X-frequency coordinates
    freq_y : numpy.ndarray
        Y-frequency coordinates

    Examples
    --------
    >>> filter_obj = Filter2d()
    >>> kernel = filter_obj.get_spatial_kernel(real=True)
    >>> is_dirac = filter_obj.is_dirac()
    """

    def __init__(self, **kwargs):
        """Initialise the Filter2d object.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for filter configuration
        """
        self.fil = None
        self.freq_x = None
        self.freq_y = None

    def get_spatial_kernel(self, real: bool = True) -> np.ndarray:
        """Compute the spatial convolution kernel from frequency domain
        representation.

        This method applies the inverse FFT to convert the frequency domain
        filter to the spatial domain, with proper frequency shifting to position
        the zero-frequency component at the center of the kernel.

        Parameters
        ----------
        real : bool, default True
            If True, returns only the real part of the kernel. If False, returns
            the complex kernel including imaginary components.

        Returns
        -------
        numpy.ndarray
            The spatial convolution kernel with zero-frequency component at the
            center of the array
        """
        # Apply inverse FFT with frequency shifting
        kernel = fft.ifft(self.fil, shift=True, shift_after=True)

        if real:
            kernel = kernel.real

        return kernel

    def is_dirac(self) -> bool:
        """Check if the current filter represents a Dirac delta function.

        A filter is considered a Dirac if its spatial kernel satisfies:
        - Sum of all coefficients equals 1
        - Maximum coefficient value equals 1
        - Minimum coefficient value equals 0

        Returns
        -------
        bool
            True if the filter is a Dirac delta function, False otherwise

        Notes
        -----
        This method uses the real part of the spatial kernel for evaluation.
        The Dirac property is commonly used to verify that a filter preserves
        the original signal when applied.
        """
        kernel = self.get_spatial_kernel(real=True)
        check_sum = np.sum(kernel) == 1
        check_max = np.max(kernel) == 1
        check_min = np.min(kernel) == 0
        return check_sum and check_max and check_min
