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
from typing import Tuple

import numpy as np

from gridr.core.filter.filter import Filter2d
from gridr.core.filter.lattice import Lattice2d


class CellModel(object):
    pass


class ReciprocalCellModel(CellModel):
    """A reciprocal cell model with tunable frequency cutoff characteristics.

    This class extends the base CellModel.

    Attributes
    ----------
    cell_geometry : Lattice2d
        The basis of the vectorial space used for the cell definition.
        Please note it is different from the working geometry, ie. the lattice
        effectively defining a regular resampling operation in the physical
        space.

    cutoff_tanh_slope : float
        The slope parameter for the hyperbolic tangent function applied to the
        frequency cutoff, controlling the sharpness of the transition

    cutoff_shift : float
        The shift parameter for the frequency cutoff, allowing adjustment of
        the cutoff position in the frequency domain

    """

    def __init__(
        self,
        cell_geometry: Lattice2d,
        cutoff_tanh_slope: float = 10.0,
        cutoff_shift: float = 0.0,
    ):
        """Initialize the ReciprocalCellModel

        Parameters
        ----------
        cell_geometry : Lattice2d
            The basis of the vectorial space used for the cell definition.
            Please note it is different from the working geometry, ie. the
            lattice effectively defining a regular resampling operation in the
            physical space.

        cutoff_tanh_slope : float, optional
            The slope parameter for the hyperbolic tangent function applied to
            the frequency cutoff (default is 10.0)

        cutoff_shift : float, optional
            The shift parameter for the frequency cutoff (default is 0.0)
        """
        self.cell_geometry = cell_geometry
        self.cutoff_tanh_slope = cutoff_tanh_slope
        self.cutoff_shift = cutoff_shift

    def compute(self, freq1d_x: np.array, freq1d_y: np.array, working_geometry: Lattice2d):
        """Compute the filter

        Parameters
        ----------
        freq1d_x : numpy.ndarray
            One-dimensional X-frequency coordinates

        freq1d_y : numpy.ndarray
            One-dimensional Y-frequency coordinates

        working_geometry: Lattice2d
            The lattice defining the regular resampling operation in either the
            physical space or the reciprocal space.

        Returns
        -------
        np.ndarray
            The computed filter for the meshgrid of the input frequencies.
        """
        # Computation on the cell geometry lattice
        self.cell_geometry.check_singularity()

        # The voronoi paving cell computation has to be performed in the dual
        # space.
        v_geometry = self.cell_geometry
        if v_geometry.is_spatial_domain():
            v_geometry = Lattice2d.get_dual_domain_lattice(v_geometry)

        v1, v2, v3 = Lattice2d.voronoi_paving_cell(lattice2d=v_geometry)

        # Defining the working geometry
        if working_geometry.is_spatial_domain():
            working_geometry = Lattice2d.get_dual_domain_geometry(working_geometry)

        u1 = working_geometry.v1
        u2 = working_geometry.v2

        # Construct the mesh of the spatial frequencies aligned with the
        # source axis.
        freq_x_grid, freq_y_grid = np.meshgrid(freq1d_x, freq1d_y)

        # The working geometry is used to make the change of coordinates
        # to the filter geometry
        #
        #  /       \     /            \     /    \
        # |  fx_fil |   |  u1_x  u2_x  |   |  fx  |
        # |         | = |              | . |      |
        # |  fy_fil |   |  u1_y  u2_y  |   |  fy  |
        #  \       /     \            /     \    /
        #
        fil_freq_x = freq_x_grid * u1[0] + freq_y_grid * u2[0]
        fil_freq_y = freq_x_grid * u1[1] + freq_y_grid * u2[1]

        # Compute the filter coefficients
        freq1 = fil_freq_x * v1[0] + fil_freq_y * v1[1]
        freq2 = fil_freq_x * v2[0] + fil_freq_y * v2[1]
        freq3 = None
        if v3 is not None:
            freq3 = fil_freq_x * v3[0] + fil_freq_y * v3[1]

        # Init the filter to ones everywhere
        fil = np.ones(freq1.shape, dtype=np.float64)

        if self.cutoff_tanh_slope is None:
            # tensorial product for all frequencies
            fil[np.where(np.abs(freq1) > (0.5 + self.cutoff_shift))] = 0.0
            fil[np.where(np.abs(freq2) > (0.5 + self.cutoff_shift))] = 0.0
            if v3 is not None:
                fil[np.where(np.abs(freq3) > (0.5 + self.cutoff_shift))] = 0.0
        else:
            # tanh
            def freq_tanh(x, r, slope):
                d = 2 * slope
                t = r + 0.5
                p = 0.5 * (np.tanh(d * (x + t)) - np.tanh(d * (x - t)))
                return p

            fil = freq_tanh(freq1, self.cutoff_shift, self.cutoff_tanh_slope)
            fil *= freq_tanh(freq2, self.cutoff_shift, self.cutoff_tanh_slope)
            if v3 is not None:
                fil *= freq_tanh(freq3, self.cutoff_shift, self.cutoff_tanh_slope)

            # Normalize the filter so that it equals 1 at hot spot
            hot_spot_x = fil.shape[1] // 2
            hot_spot_y = fil.shape[0] // 2
            if fil[hot_spot_y, hot_spot_x] != 0:
                fil /= fil[hot_spot_y, hot_spot_x]

        return fil


class FrequentialInterpolator2d(Filter2d):
    """
    Frequential interpolator 2d
    """

    def __init__(self, working_geometry: Lattice2d, model: CellModel):
        """
        Init a FrequentialInterpolator2d
        :param working_geometry: the working geometry
        :param model: the cell model definition
        """
        super().__init__()
        if not isinstance(model, ReciprocalCellModel):
            raise NotImplementedError
        self.working_geometry = working_geometry
        self.model = model

    def compute(
        self, nrow: int, ncol: int, oversampling_row: float, oversampling_col: float
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Compute the filter on nrow x ncol samples.

        Convention of the filter :
        - the hot spot is located at [nrow // 2, ncol // 2]
        - the first index corresponds to the row indices
        - the second index corresponds to the col indices
        - the filter is expressed in the frequency domain

        :param nrow: number of rows for the computed filter
        :param ncol: number of columns for the computed filter
        :param oversampling_row: oversampling factor for the rows frequencies (y frequencies)
        :param oversampling_col: oversampling factor for the columns frequencies (x frequencies)
        :return: the computed filter.
        """
        # Check working geometry
        if np.linalg.det(self.working_geometry.as_matrix()) == 0:
            raise Exception("Working geometry has a nul determinant")

        if self.working_geometry.is_spatial_domain():
            u_geometry = Lattice2d.get_dual_domain_lattice(self.working_geometry)
        else:
            u_geometry = self.working_geometry

        # Compute frequencies - the zero-frequency component is shifted at the center
        # of the spectrum.
        self.freq_x = np.fft.fftshift(np.fft.fftfreq(ncol)) * oversampling_col
        self.freq_y = np.fft.fftshift(np.fft.fftfreq(nrow)) * oversampling_row

        self.fil = self.model.compute(
            freq1d_x=self.freq_x, freq1d_y=self.freq_y, working_geometry=u_geometry
        )
