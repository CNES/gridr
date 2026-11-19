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
Tests for the gridr.core.grid.grid_antialiasing module.

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_antialiasing.py
"""  # noqa: E501

import numpy as np
import pytest

from gridr.core.filter.antialiasing import KernelFilter
from gridr.core.filter.antialiasing.reciprocal_cell import (
    ReciprocalCellFilter,
    tanh_gate,
)
from gridr.core.grid import grid_antialiasing as ga
from gridr.core.grid.grid_commons import grid_resolution_window_safe
from gridr.core.grid.grid_utils import array_compute_resampling_grid_geometries


def rot(deg: float) -> np.ndarray:
    """Rotation matrix of ``deg`` degrees."""
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def affine_grid(
    w: np.ndarray,
    shape: tuple[int, int] = (30, 30),
    resolution: tuple[int, int] = (1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Affine backward grid ``(grid_row, grid_col)`` of transition matrix ``w``."""
    r, c = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    jp, ip = c * resolution[1], r * resolution[0]
    return w[1, 0] * jp + w[1, 1] * ip + 500.0, w[0, 0] * jp + w[0, 1] * ip + 700.0


def curved_grid(n: int = 48, a: float = 4e-3) -> tuple[np.ndarray, np.ndarray]:
    """Conformal backward grid ``z -> 2 z + a z^2`` on ``n x n`` nodes."""
    r, c = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 2.0 * (c + 1j * r) + a * (c + 1j * r) ** 2
    return z.imag + 100.0, z.real + 100.0


def longest_distance_oracle(
    grid_row: np.ndarray,
    grid_col: np.ndarray,
    resolution: tuple[int, int],
    valid: np.ndarray,
) -> np.ndarray:
    """Independent NumPy computation of the longest-distance transition matrix."""

    def span(axis: int) -> np.ndarray:
        """Source displacement per destination pixel over the longest valid row
        (axis 1) or column (axis 0)."""
        best, vec = 0, None
        for k in range(valid.shape[1 - axis]):
            line = valid[k, :] if axis == 1 else valid[:, k]
            idx = np.flatnonzero(line)
            if idx.size and idx[-1] - idx[0] > best:
                best = idx[-1] - idx[0]
                pa = (k, idx[0]) if axis == 1 else (idx[0], k)
                pb = (k, idx[-1]) if axis == 1 else (idx[-1], k)
                step = resolution[1] if axis == 1 else resolution[0]
                vec = np.array(
                    [grid_col[pb] - grid_col[pa], grid_row[pb] - grid_row[pa]]
                ) / (best * step)
        return vec

    return np.c_[span(1), span(0)]


class TestTransitionMatrix:
    """Estimation of the transition matrix."""

    @pytest.mark.parametrize("resolution", [(1, 1), (4, 8), (8, 4)])
    @pytest.mark.parametrize("mode", ["longest-distance", "least-square-fit"])
    def test_exact_on_affine_grids(self, resolution, mode):
        """Both modes are exact on affine grids, with or without mask or nodata."""
        w = rot(30) @ np.diag([3.0, 1.5]) @ np.array([[1.0, 0.4], [0.0, 1.0]])
        grid_row, grid_col = affine_grid(w, resolution=resolution)
        mask = np.ones(grid_row.shape, dtype=np.uint8)
        mask[5:20, 3:15] = 0
        mask[0, :] = 0
        for kwargs in ({}, {"grid_mask": mask}):
            est = ga.estimate_transition_matrix(
                grid_row, grid_col, resolution, mode=mode, **kwargs
            )
            assert np.abs(est - w).max() < 1e-10
        nodata_row = np.where(mask == 1, grid_row, -9999.0)
        est = ga.estimate_transition_matrix(
            nodata_row, grid_col, resolution, grid_nodata=-9999.0, mode=mode
        )
        assert np.abs(est - w).max() < 1e-10

    def test_affine_residual(self):
        """The least-squares residual vanishes on an affine grid only."""
        w = 2.0 * rot(20)
        _, p0, residual = ga.fit_affine_least_squares(*affine_grid(w))
        assert residual < 1e-9
        assert np.allclose(p0, [700.0, 500.0])
        assert ga.fit_affine_least_squares(*curved_grid())[2] > 1.0

    def test_longest_distance_matches_oracle(self):
        """The longest-distance mode matches an independent computation."""
        rng = np.random.default_rng(5)
        grid_row, grid_col = curved_grid()
        for _ in range(10):
            valid = rng.random(grid_row.shape) > 0.3
            res = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
            est = ga.estimate_transition_matrix(
                grid_row,
                grid_col,
                res,
                grid_mask=(3 * valid).astype(np.uint8),
                grid_mask_valid_value=3,
                mode="longest-distance",
            )
            ref = longest_distance_oracle(grid_row, grid_col, res, valid)
            assert np.abs(est - ref).max() < 1e-10

    def test_least_squares_on_a_curved_grid(self):
        """On a curved grid, the least-squares fit gives the mean Jacobian."""
        grid_row, grid_col = curved_grid()
        dydr, dydc = np.gradient(grid_row)
        dxdr, dxdc = np.gradient(grid_col)
        # exact here because the grid is quadratic: the mean of the central
        # differences then equals the least-squares slope
        w_mean = np.array([[dxdc.mean(), dxdr.mean()], [dydc.mean(), dydr.mean()]])
        w = ga.estimate_transition_matrix(grid_row, grid_col, mode="least-square-fit")
        np.testing.assert_allclose(w, w_mean, atol=1e-9)

    def test_errors(self):
        """An unknown mode or too few valid nodes raise ValueError."""
        grid_row, grid_col = affine_grid(np.eye(2))
        with pytest.raises(ValueError):
            ga.estimate_transition_matrix(grid_row, grid_col, mode="unknown")
        mask = np.zeros(grid_row.shape, dtype=np.uint8)
        mask[0, :2] = 1
        with pytest.raises(ValueError):
            ga.fit_affine_least_squares(grid_row, grid_col, grid_mask=mask)

    @pytest.mark.parametrize("mode", ["longest-distance", "least-square-fit"])
    def test_single_valid_row(self, mode):
        """A single valid row does not define W: both modes say so."""
        grid_row, grid_col = affine_grid(2.0 * np.eye(2))
        mask = np.zeros(grid_row.shape, dtype=np.uint8)
        mask[5, :] = 1
        match = "collinear" if mode == "least-square-fit" else "single valid row"
        with pytest.raises(ValueError, match=match):
            ga.estimate_transition_matrix(grid_row, grid_col, grid_mask=mask, mode=mode)

    @pytest.mark.parametrize("mode", ["longest-distance", "least-square-fit"])
    def test_validity_arguments(self, mode):
        """grid_mask and grid_nodata are exclusive; a mask needs its valid value."""
        grid_row, grid_col = affine_grid(2.0 * np.eye(2))
        mask = np.ones(grid_row.shape, dtype=np.uint8)
        with pytest.raises(ValueError):
            ga.estimate_transition_matrix(
                grid_row, grid_col, grid_mask=mask, grid_nodata=-9999.0, mode=mode
            )
        with pytest.raises(ValueError):
            ga.estimate_transition_matrix(
                grid_row,
                grid_col,
                grid_mask=mask,
                grid_mask_valid_value=None,
                mode=mode,
            )

    @pytest.mark.parametrize("resolution", [(1, 1), (4, 4), (3, 5)])
    def test_window_is_full_resolution(self, resolution):
        """win is a full resolution window, reduced to the nodes that cover it."""
        grid_row, grid_col = curved_grid()
        win = np.array([[5, 20], [7, 30]]) * np.array(resolution)[:, None] + 1
        node_win, _ = grid_resolution_window_safe(resolution, win, grid_row.shape)
        rows = slice(node_win[0, 0], node_win[0, 1] + 1)
        cols = slice(node_win[1, 0], node_win[1, 1] + 1)
        # least squares: same fit as on the nodes of the window alone
        w_win, _, res_win = ga.fit_affine_least_squares(
            grid_row, grid_col, resolution, win=win
        )
        w_sub, _, res_sub = ga.fit_affine_least_squares(
            grid_row[rows, cols].copy(), grid_col[rows, cols].copy(), resolution
        )
        assert np.abs(w_win - w_sub).max() < 1e-9 and abs(res_win - res_sub) < 1e-9
        # longest distance: the grid geometry analysis on the window of nodes
        metrics = array_compute_resampling_grid_geometries(
            grid_row=grid_row,
            grid_col=grid_col,
            grid_resolution=resolution,
            win=node_win,
        )
        expected = np.asarray(metrics.transition_matrix.w()).reshape(2, 2)
        est = ga.estimate_transition_matrix(
            grid_row, grid_col, resolution, win=win, mode="longest-distance"
        )
        assert np.array_equal(est, expected)


class TestResolution:
    """Resolution of a filter specification on a grid."""

    def test_passthrough(self):
        """None and a kernel filter are returned unchanged, without estimation."""
        k = KernelFilter(np.ones((3, 5)) / 15)
        grid = affine_grid(np.eye(2))
        assert ga.resolve_antialiasing_filter(k, *grid) == (k, None)
        assert ga.resolve_antialiasing_filter(None, *grid) == (None, None)

    def test_rejects_unknown_specifications(self):
        """A specification without a design method raises TypeError."""
        with pytest.raises(TypeError):
            ga.resolve_antialiasing_filter(np.ones((3, 3)), *affine_grid(np.eye(2)))

    def test_zoom_resolves_to_none(self):
        """A zoom needs no filter; the diagnostics are still returned."""
        kf, diag = ga.resolve_antialiasing_filter(
            ReciprocalCellFilter(), *affine_grid(0.5 * np.eye(2))
        )
        assert kf is None and diag.filter.filter_required is False

    @pytest.mark.parametrize("mode", ["longest-distance", "least-square-fit"])
    def test_resolve_calls_design(self, mode):
        """The resolution estimates W, then calls the design of the specification."""
        w = 2.0 * rot(30)
        grid_row, grid_col = affine_grid(w, resolution=(4, 4))
        spec = ReciprocalCellFilter(aliasing_tolerance=0.1)
        kf, diag = ga.resolve_antialiasing_filter(
            spec, grid_row, grid_col, (4, 4), transition_matrix_estimation=mode
        )
        assert np.allclose(diag.transition_matrix, w)
        assert diag.transition_matrix_estimation == mode
        assert diag.affine_residual < 1e-9
        expected, filter_diag = spec.design(diag.transition_matrix)
        assert np.array_equal(kf.kernel, expected.kernel)
        assert diag.filter.kernel_size == filter_diag.kernel_size
        assert diag.filter.response_error == filter_diag.response_error

    @pytest.mark.parametrize("mode", ["longest-distance", "least-square-fit"])
    def test_node_selection_is_passed_through(self, mode):
        """win, grid_mask and grid_nodata reach the estimation and the residual."""
        grid_row, grid_col = curved_grid()
        mask = np.ones(grid_row.shape, dtype=np.uint8)
        mask[:, 30:] = 0
        win = np.array([[4, 60], [8, 70]])
        nodes = {"win": win, "grid_mask": mask}
        _, diag = ga.resolve_antialiasing_filter(
            ReciprocalCellFilter(),
            grid_row,
            grid_col,
            (2, 2),
            transition_matrix_estimation=mode,
            **nodes,
        )
        expected = ga.estimate_transition_matrix(
            grid_row, grid_col, (2, 2), mode=mode, **nodes
        )
        residual = ga.fit_affine_least_squares(grid_row, grid_col, (2, 2), **nodes)[2]
        assert np.array_equal(diag.transition_matrix, expected)
        assert diag.affine_residual == residual
        assert not np.array_equal(
            expected,
            ga.estimate_transition_matrix(grid_row, grid_col, (2, 2), mode=mode),
        )

# TODO: remove in 0.9.0 with the deprecated methods
class TestDeprecatedWrapper:
    """compute_antialiasing_filter_from_grid."""

    def test_wrapper(self):
        """The wrapper warns and returns the kernel of the new resolution."""
        grid_row, grid_col = affine_grid(2.0 * rot(30))
        with pytest.warns(DeprecationWarning):
            status, kernel = ga.compute_antialiasing_filter_from_grid(
                grid_row, grid_col, (1, 1), 21, 21, 10.0, -0.1
            )
        assert status == ga.ComputeAntialiasingFilterStatus.REQUIRED
        tolerance = float(tanh_gate(np.array(0.5), 10.0, 0.4))
        expected, _ = ga.resolve_antialiasing_filter(
            ReciprocalCellFilter(aliasing_tolerance=tolerance, kernel_size=21),
            grid_row,
            grid_col,
        )
        assert np.allclose(kernel, expected.kernel)
        with pytest.warns(DeprecationWarning):
            status, kernel = ga.compute_antialiasing_filter_from_grid(
                *affine_grid(0.5 * np.eye(2)), (1, 1), 21, 21
            )
        assert (
            status == ga.ComputeAntialiasingFilterStatus.NOT_REQUIRED and kernel is None
        )