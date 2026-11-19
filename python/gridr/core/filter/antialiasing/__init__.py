#
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
"""
Antialiasing filters.

An antialiasing filter removes, before a resampling that reduces the
resolution, the source frequencies that the destination geometry cannot
represent. Each method of construction lives in a submodule of this package
and is described by a specification class implementing
:class:`AntialiasingFilter`: from the transition matrix ``W`` of the geometry, its
method ``design`` returns the filter as a :class:`KernelFilter`, or None when
no filter is needed, with diagnostics specific to the method.

Available methods:

- :mod:`gridr.core.filter.antialiasing.reciprocal_cell`: filter matched to a
  cell of the reciprocal lattice of the destination geometry
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class KernelFilter:
    """
    Antialiasing filter given by an odd-sized 2D kernel.

    It is the output of every method of construction, and wraps a user
    kernel too.

    Attributes
    ----------
    kernel : np.ndarray
        2D kernel with odd dimensions, centered, applied by convolution.
    """

    kernel: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate the kernel.

        Raises
        ------
        ValueError
            If the kernel is not a finite 2D array with odd dimensions.
        """
        k = np.asarray(self.kernel)
        if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
            raise ValueError("The kernel must be 2D with odd dimensions")
        if not np.all(np.isfinite(k)):
            raise ValueError("The kernel must be finite")

    def margins(self) -> tuple[int, int, int, int]:
        """
        Margins read by the kernel around an array.

        Returns
        -------
        tuple[int, int, int, int]
            ``(top, bottom, left, right)`` half-sizes, in pixels.
        """
        hr, hc = self.kernel.shape[0] // 2, self.kernel.shape[1] // 2
        return hr, hr, hc, hc


@runtime_checkable
class AntialiasingFilter(Protocol):
    """
    Specification of an antialiasing filter, whatever its method.

    A method builds the filter from the transition matrix ``W = [w1 w2]`` of
    the geometry, where ``w1`` and ``w2`` are the source displacements for one
    destination column and one destination row.
    """

    def design(self, transition_matrix: np.ndarray) -> tuple[KernelFilter | None, Any]:
        """
        Build the filter for a transition matrix.

        Parameters
        ----------
        transition_matrix : np.ndarray
            Transition matrix ``W = [w1 w2]`` of the geometry.

        Returns
        -------
        tuple[KernelFilter or None, Any]
            The filter, None when no filter is needed, and the diagnostics of
            the method.
        """
        ...
