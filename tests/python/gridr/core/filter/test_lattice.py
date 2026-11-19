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
Tests for the gridr.core.filter.lattice module.

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/filter/test_lattice.py
"""  # noqa: E501

import numpy as np
import pytest

from gridr.core.filter.lattice import (
    Domain,
    Lattice2d,
    gauss_reduce,
    relevant_vectors,
    voronoi_cell_facets,
    voronoi_cell_vertices,
)


def rot(deg: float) -> np.ndarray:
    """Rotation matrix of ``deg`` degrees."""
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# rotated and sheared basis, oblique enough to give a hexagon
SHEAR = rot(20) @ np.array([[1.0, 0.5], [0.0, 1.0]])


def random_bases(seed: int, count: int) -> list[np.ndarray]:
    """Random lattice bases, unreduced, one vector per column."""
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(count):
        reduced = rot(rng.uniform(0, 180)) @ np.diag(rng.uniform(0.3, 3.0, 2))
        reduced = reduced @ np.array([[1.0, rng.uniform(-0.5, 0.5)], [0.0, 1.0]])
        # unimodular matrix to make the returned base non-reduced.
        # same lattice but skewed basis
        unimodular = np.array([[1, rng.integers(-3, 4)], [0, 1]]) @ np.array(
            [[1, 0], [rng.integers(-3, 4), 1]]
        )
        bases.append(reduced @ unimodular)
    return bases


def brute_force_cell(
    basis: np.ndarray, metric: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Mask of the Voronoi cell of the origin, by nearest-point search."""
    # Go to euclidian metric
    chol = np.linalg.cholesky(metric)
    u1, u2 = gauss_reduce(*(chol.T @ basis).T)
    # Sample the lattice points
    points = np.array([a * u1 + b * u2 for a in range(-4, 5) for b in range(-4, 5)])
    # Coordinate transform of the evaluated points x, y
    h = np.stack([x, y], -1) @ chol
    # Compute the squared distance  of each evaluated point to each point of the
    # lattice
    d = (h[..., None, 0] - points[:, 0]) ** 2 + (h[..., None, 1] - points[:, 1]) ** 2
    # the origin of the lattice is at len(points) // 2 by construction
    # return a mask that is True when the nearest point is the origin, ie
    # the point is in the cell.
    return d.argmin(-1) == len(points) // 2


class TestGaussReduce:
    """Lagrange-Gauss reduction."""

    def test_reduced_basis(self):
        """The result is ordered, reduced and spans the same lattice."""
        for basis in random_bases(1, 100):
            u1, u2 = gauss_reduce(basis[:, 0], basis[:, 1])
            # assert the base is ordered
            assert u1 @ u1 <= u2 @ u2 + 1e-9
            # assert the base is reduced
            assert abs(u1 @ u2) <= 0.5 * (u1 @ u1) + 1e-9
            # coordinates of u1, u2 in the input basis
            c = np.linalg.solve(basis, np.stack([u1, u2], -1))
            # integer coordinates: u1, u2 are lattice points
            assert np.allclose(c, np.round(c), atol=1e-9)
            # det = +-1: c^-1 is integer too, so both bases span the same lattice
            assert np.isclose(abs(np.linalg.det(c)), 1.0)

    def test_nearest_integer(self):
        """A projection ratio of 0.9 is rounded to 1, not truncated to 0."""
        u1, u2 = gauss_reduce(np.array([1.0, 0.0]), np.array([0.9, 1.0]))
        assert np.allclose(u1, [1.0, 0.0])
        assert np.allclose(u2, [-0.1, 1.0])


class TestVoronoiCell:
    """Voronoi cell of a lattice in a metric."""

    @pytest.mark.parametrize("kind", ["euclidean", "anisotropic"])
    def test_matches_brute_force(self, kind):
        """The cell matches a brute-force nearest-point search."""
        f = np.linspace(-1.5, 1.5, 241)
        x, y = np.meshgrid(f, f)
        for k, basis in enumerate(random_bases(2, 20)):
            metric = np.eye(2)
            if kind == "anisotropic":
                d = rot(17.0 * k) @ np.diag([1.0, 2.5])
                metric = d @ d.T
            facets = voronoi_cell_facets(basis, metric)
            cell = np.all([np.abs(x * a[0] + y * a[1]) <= 0.5 for a in facets], axis=0)
            # only pixels on the boundary may be classified differently
            assert np.mean(cell != brute_force_cell(basis, metric, x, y)) < 2e-3

    def test_lattice_metric_gives_parallelogram(self):
        """
        In the metric D D^T, with D = B^-T, the cell is the parallelogram
        spanned by B.
        """
        for basis in random_bases(3, 50):
            dual = np.linalg.inv(basis).T
            facets = voronoi_cell_facets(basis, dual @ dual.T)
            assert len(facets) == 2
            got = sorted(np.abs(np.array(facets) @ basis).round(9).tolist())
            assert np.allclose(got, [[0.0, 1.0], [1.0, 0.0]])

    def test_hexagon_area(self):
        """An oblique lattice gives a hexagon of the area of the lattice cell."""
        basis = 2.0 * SHEAR
        facets = voronoi_cell_facets(basis)
        assert len(facets) == 3
        p = voronoi_cell_vertices(facets)
        area = 0.5 * abs(p[:, 0] @ np.roll(p[:, 1], 1) - p[:, 1] @ np.roll(p[:, 0], 1))
        assert len(p) == 6
        assert np.isclose(area, abs(np.linalg.det(basis)))

    def test_rectangle(self):
        """An orthogonal basis gives a rectangle."""
        facets = voronoi_cell_facets(np.diag([1.0, 2.0]))
        assert len(facets) == 2
        assert len(voronoi_cell_vertices(facets)) == 4

    def test_paving_cell_square(self):
        """The identity lattice gives the unit square, without third facet."""
        lattice = Lattice2d(
            np.array([1.0, 0.0]), np.array([0.0, 1.0]), Domain.SPATIAL_DOMAIN
        )
        a1, a2, a3 = Lattice2d.voronoi_paving_cell(lattice)
        assert a3 is None
        assert np.allclose(sorted(np.abs([a1, a2]).tolist()), [[0, 1], [1, 0]])

    def test_relevant_vectors_obtuse(self):
        """The relevant vectors come from an obtuse superbase."""
        for basis in random_bases(4, 50):
            vectors = relevant_vectors(basis[:, 0], basis[:, 1])
            assert vectors[0] @ vectors[1] <= 1e-12
            if len(vectors) == 3:
                assert np.allclose(vectors[2], vectors[0] + vectors[1])

    def test_public_methods(self):
        """Lattice2d methods delegate to the module functions and keep their formats."""
        basis = 2.0 * SHEAR
        lattice = Lattice2d(basis[:, 0], basis[:, 1], Domain.FREQUENCY_DOMAIN)
        reduced = Lattice2d.gaussian_lattice_reduction(lattice)
        u1, u2 = gauss_reduce(lattice.v1, lattice.v2)
        assert np.allclose(reduced.v1, u1) and np.allclose(reduced.v2, u2)
        assert reduced.domain == Domain.FREQUENCY_DOMAIN
        # a hexagon: three facets, a3 is set
        a1, a2, a3 = Lattice2d.voronoi_paving_cell(lattice)
        for got, ref in zip((a1, a2, a3), voronoi_cell_facets(basis), strict=True):
            assert np.allclose(got, ref)