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
Grid antialiasing module
"""
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from gridr.core.filter.frequential_interp_2d import FrequentialInterpolator2d, ReciprocalCellModel
from gridr.core.filter.lattice import Domain, Lattice2d
from gridr.core.grid.grid_utils import array_compute_resampling_grid_geometries


class ComputeAntialiasingFilterStatus(Enum):
    """Enum class to represent `compute_antialiasing_filter_from_grid` status"""

    NA = 0
    NOT_REQUIRED = 1
    REQUIRED_FOR_ZOOM = 2
    REQUIRED = 3


def compute_antialiasing_filter_from_grid(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    grid_resolution: Tuple[int, int],
    filter_nrow: int,
    filter_ncol: int,
    filter_cutoff_tanh_slope: Optional[float] = 10.0,
    filter_cutoff_shift: float = 0.0,
    win: Optional[np.ndarray] = None,
    grid_mask: Optional[np.ndarray] = None,
    grid_mask_valid_value: Optional[int] = 1,
    grid_nodata: Optional[float] = None,
) -> Tuple[ComputeAntialiasingFilterStatus, np.ndarray]:
    """
    Create an antialiasing filter adapted to the grid geometric transformation.
    The filter will be well adapted to regular grid.

    Parameters
    ----------
    grid_row : np.ndarray
        A 2D array representing the row coordinates of the target grid, with
        the same shape as `grid_col`. The coordinates target row positions in
        an input array.

    grid_col : np.ndarray
        A 2D array representing the column coordinates of the target grid,
        with the same shape as `grid_row`. The coordinates target column
        positions in an input array.

    grid_resolution : Tuple[int, int]
        A tuple specifying the oversampling factor for the grid for rows and
        columns. The resolution value of 1 represents full resolution, and
        higher values indicate lower resolution grids.

    filter_nrow : int
         The output filter number of rows

    filter_ncol : int
         The output filter number of columns

    filter_cutoff_tanh_slope : Optional[float], default 10.
        The slope parameter for the hyperbolic tangent function applied to the
        frequency cutoff, controlling the sharpness of the transition

    filter_cutoff_shift : float, default = 0.
        The shift parameter for the frequency cutoff, allowing adjustment of
        the cutoff position in the frequency domain

    win : Optional[np.ndarray], default None
        A window (or sub-region) of the full resolution grid to limit the
        resampling to a specific target region. The window is defined as a list
        of tuples containing the first and last indices for each dimension.
        If `None`, the entire grid is processed.

    grid_mask : Optional[np.ndarray], default None
        An optional integer mask array for the grid. Grid cells corresponding to
        `grid_mask_valid_value` are considered **valid**; all other values
        indicate **invalid** cells and will result in `nodata_out` in the output
        array. If not provided, the entire grid is considered valid. The grid
        mask must have the same shape as `grid_row` and `grid_col`.

    grid_mask_valid_value : Optional[int], default 1
        The value in `grid_mask` that designates a **valid** grid cell.
        All values in `grid_mask` that differ from this will be treated as
        **invalid**. This parameter is required if `grid_mask` is provided.

    grid_nodata : Optional[float], default None
        The value in `grid_row` and `grid_col` to consider as **invalid**
        cells. Please note this option is exclusive with `grid_mask`. The
        exclusivity is managed within the bound core method.

    Returns
    -------
    Tuple[ComputeAntialiasingFilterStatus, np.ndarray]
        A tuple containing :

        -   The computing status
        -   The spatial kernel corresponding to the filter as a bi-dimensional
            numpy array

    """
    # Compute the PyGridGeometriesMetricsF64 object for the input grid and
    # related parameters
    grid_metrics = array_compute_resampling_grid_geometries(
        grid_row=grid_row,
        grid_col=grid_col,
        grid_resolution=grid_resolution,
        win=win,
        grid_mask=grid_mask,
        grid_mask_valid_value=grid_mask_valid_value,
        grid_nodata=grid_nodata,
    )

    if grid_metrics is None:
        raise Exception("The computation of the grid metrics failed.")

    # Get the transition matrix from the grid_metrics
    W = np.asarray(grid_metrics.transition_matrix.w()).reshape((2, 2))

    # Compute the matrix of the two vectors v1 and v2 of the sampling lattice
    # in the source array expressed in the target geometry
    V = None
    try:
        V = np.linalg.inv(W)
    except np.linalg.LinAlgError as e:
        raise Exception("The grid transition matrix W is singular") from e

    # Compute the reciprocal lattice of V
    # U is equal to (transpose(V))^-1 = transpose(V^-1) => U = transpose(W)
    U = np.transpose(W)
    # Compute the norm
    norm_U = np.linalg.norm(U, axis=0)
    norm_U1 = norm_U[0] ** 2
    norm_U2 = norm_U[1] ** 2

    case = ComputeAntialiasingFilterStatus.NA

    working_geometry = Lattice2d(v1=V[:, 0], v2=V[:, 1], domain=Domain.SPATIAL_DOMAIN)

    cell_geometry = Lattice2d(
        v1=np.array([1.0, 0.0]), v2=np.array([0.0, 1.0]), domain=Domain.SPATIAL_DOMAIN
    )

    if norm_U1 <= 1 and norm_U2 <= 1:
        # The grid performs a zoom in both directions
        # Lets compute the adapted filter and check if antialisaing is needed.
        filter_interp = FrequentialInterpolator2d(
            working_geometry=working_geometry,
            model=ReciprocalCellModel(
                cell_geometry=cell_geometry, cutoff_tanh_slope=None, cutoff_shift=0.0
            ),
        )
        filter_interp.compute(nrow=128, ncol=128, oversampling_row=1, oversampling_col=1)

        # Lets check if the filter corresponds to a dirac
        if filter_interp.is_dirac():
            # Antialiasing is not needed
            case = ComputeAntialiasingFilterStatus.NOT_REQUIRED
        else:
            # Antialiasing is needed
            case = ComputeAntialiasingFilterStatus.REQUIRED_FOR_ZOOM
    else:
        # The grid performs a zoom-out in at least one direction.
        # Antialiasing is needed
        case = ComputeAntialiasingFilterStatus.REQUIRED

    # Init the antialiasing kernel
    aa_kernel = None

    if case in (
        ComputeAntialiasingFilterStatus.REQUIRED,
        ComputeAntialiasingFilterStatus.REQUIRED_FOR_ZOOM,
    ):
        # Init the filter
        filter_interp = FrequentialInterpolator2d(
            working_geometry=working_geometry,
            model=ReciprocalCellModel(
                cell_geometry=cell_geometry,
                cutoff_tanh_slope=filter_cutoff_tanh_slope,
                cutoff_shift=filter_cutoff_shift,
            ),
        )
        # Compute the coefficients
        filter_interp.compute(
            nrow=filter_nrow, ncol=filter_ncol, oversampling_row=1, oversampling_col=1
        )
        # Get the real spatial kernel
        aa_kernel = filter_interp.get_spatial_kernel(real=True)

    return case, aa_kernel
