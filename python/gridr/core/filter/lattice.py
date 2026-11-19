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
Two-dimensional lattices and their Voronoi cells.
"""

from enum import Enum

import numpy as np


class Domain(Enum):
    """
    Defines the domain : spatial or frequential
    """

    FREQUENCY_DOMAIN = 0
    SPATIAL_DOMAIN = 1


class Lattice2d:
    r"""
    Defines a 2D lattices by 2 vectors :math:`\mathbf{v}_1`, :math:`\mathbf{v}_2`
    and a definition domain
    """

    def __init__(self, v1: np.array, v2: np.array, domain: Domain):
        """
        Init

        Parameters
        ----------
        v1: np.array
            First geometry generator vector

        v2: np.array
            Second geometry generator vector

        domain: Domain
            The definition domain
        """
        self.v1 = v1
        self.v2 = v2
        self.domain = domain

    def as_matrix(self) -> np.ndarray:
        r"""
        Converts the current lattice as a matrix containing the geometry
        vectors in its columns

        .. math::

           M = \begin{bmatrix}
           v_{1x} & v_{2x} \\
           v_{1y} & v_{2y}
           \end{bmatrix}

        Returns
        -------
        np.ndarray
            The matrix representation of the lattice.
        """
        return np.array((self.v1, self.v2)).transpose()

    def is_spatial_domain(self) -> bool:
        """Returns True if the lattice is defined in the spatial domain"""
        return self.domain == Domain.SPATIAL_DOMAIN

    def is_frequency_domain(self) -> bool:
        """Returns True if the lattice is defined in the frequency domain"""
        return self.domain == Domain.FREQUENCY_DOMAIN

    def check_singularity(self) -> bool:
        """
        Check if determinant is nul and raises an Exception if it is the case

        Returns
        -------
        bool
            True if determinant is not nul, otherwise the method raises an
            exception.
        """
        if np.linalg.det(self.as_matrix()) == 0:
            raise Exception("Geometry 2D matrix as a nul determinant")
        return True

    @classmethod
    def get_dual_domain(cls, lattice2d: "Lattice2d") -> Domain:
        """
        Returns the type of the dual domain.

        Parameters
        ----------
        lattice2d: Lattice2d
            The input 2D lattice for which we want to compute the paving cell.

        Returns
        -------
        Domain
            Domain.FREQUENCY_DOMAIN if the lattice is defined in the spatial
            domain, Domain.SPATIAL_DOMAIN otherwise
        """
        if lattice2d.is_frequency_domain():
            return Domain.SPATIAL_DOMAIN
        else:
            return Domain.FREQUENCY_DOMAIN

    @classmethod
    def get_dual_domain_lattice(cls, lattice2d: "Lattice2d") -> "Lattice2d":
        r"""
        Computes and returns the lattice in the dual domain.
        The dual lattice matrix U is obtained by taking the transpose of the
        inverse of the input lattice matrix V:

        .. math::

           V = \begin{bmatrix}
           v_{1x} & v_{2x} \\
           v_{1y} & v_{2y}
           \end{bmatrix}

           U = \frac{1}{Det(V)} \begin{bmatrix}
           v_{2y} & -v_{1y} \\
           -v_{2x} & v_{1x}
           \end{bmatrix}

        Parameters
        ----------
        lattice2d: Lattice2d
            The input 2D lattice for which we want to compute the paving cell.

        Returns
        -------
        Lattice2d
            The dual lattice
        """
        lattice2d.check_singularity()
        v_matrix = lattice2d.as_matrix()
        u_matrix = np.linalg.inv(v_matrix).transpose()
        out = Lattice2d(
            v1=np.array(u_matrix[:, 0]).reshape(-1),
            v2=np.array(u_matrix[:, 1]).reshape(-1),
            domain=cls.get_dual_domain(lattice2d),
        )
        return out

    @classmethod
    def voronoi_paving_cell(
        cls,
        lattice2d: "Lattice2d",
        rtol: float = 1e-9,
        reduction_itermax: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Compute the facet vectors of the Voronoi cell of a lattice.

        The Voronoi cell of the origin is the set of points closer to the
        origin than to any other lattice point, in the Euclidean metric. Its
        translates by the lattice vectors pave the plane. In two dimensions the
        cell has two or three pairs of opposite edges, defined by the
        perpendicular bisectors of the segments joining the origin to its
        relevant vectors ``v``. Each pair is described by the facet vector
        ``a = v / |v|^2``: the cell is ``{x : |x . a| <= 1/2}`` over the facets.

        The relevant vectors are obtained from an obtuse superbase of the
        reduced basis (see :func:`relevant_vectors`).

        Parameters
        ----------
        lattice2d : Lattice2d
            The input 2D lattice.

        rtol : float, optional
            Relative tolerance on the scalar product of the reduced basis
            vectors below which the basis is considered orthogonal and the
            cell is a rectangle instead of a hexagon.

        reduction_itermax : int, optional
            Maximum number of iterations of the lattice reduction.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray | None]
            The facet vectors ``(a1, a2, a3)``. ``a3`` is None when the cell is
            a rectangle.
        """
        lattice2d.check_singularity()
        facets = voronoi_cell_facets(
            lattice2d.as_matrix(), rtol=rtol, reduction_itermax=reduction_itermax
        )
        a3 = facets[2] if len(facets) == 3 else None
        return facets[0], facets[1], a3

    @classmethod
    def gaussian_lattice_reduction(
        cls,
        lattice2d: "Lattice2d",
        itermax: int = 100,
    ) -> "Lattice2d":
        r"""A Gaussian Lattice Reduction algorithm implementation.

        This method reduces a 2D lattice basis to find the smallest vectors. (see
        :func:`gauss_reduce`).

        Parameters
        ----------
        lattice2d : Lattice2d
            The input 2D lattice for which we want to compute a reduced basis.

        itermax : int, optional
            Maximum number of iterations to perform, by default 100.

        Returns
        -------
        Lattice2d
            A new Lattice2d object whose vectors are the smallest one.
        """
        u1, u2 = gauss_reduce(lattice2d.v1, lattice2d.v2, itermax)
        return cls(v1=u1, v2=u2, domain=lattice2d.domain)


def gauss_reduce(
    b1: np.ndarray, b2: np.ndarray, itermax: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Lagrange-Gauss reduction of a 2D lattice basis.

    The algorithm orders the basis so that ``|u1| <= |u2|``, replaces ``u2``
    by ``u2 - m u1`` with ``m`` the integer nearest to ``u1 . u2 / |u1|^2``,
    and repeats until ``m = 0``. The basis is ordered once more at the end.

    Parameters
    ----------
    b1, b2 : np.ndarray
        Basis vectors.

    itermax : int, optional
        Safety bound on the number of iterations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reduced basis ``(u1, u2)`` with ``|u1| <= |u2|`` and
        ``|u1 . u2| <= |u1|^2 / 2``.
    """
    u1 = np.asarray(b1, dtype=np.float64).copy()
    u2 = np.asarray(b2, dtype=np.float64).copy()
    for _ in range(itermax):
        if u2 @ u2 < u1 @ u1:
            u1, u2 = u2, u1
        m = int(np.rint((u1 @ u2) / (u1 @ u1)))
        if m == 0:
            break
        u2 = u2 - m * u1
    if u2 @ u2 < u1 @ u1:
        u1, u2 = u2, u1
    return u1, u2


def relevant_vectors(
    b1: np.ndarray, b2: np.ndarray, rtol: float = 1e-9, itermax: int = 100
) -> list[np.ndarray]:
    """
    Voronoi-relevant vectors of a 2D lattice, from an obtuse superbase.

    After reduction, the sign of ``u2`` is chosen so that ``u1 . u2 <= 0``;
    ``u1``, ``u2`` and ``-(u1 + u2)`` then form an obtuse superbase
    :cite:`kurlin2024mathematics`. The relevant vectors are ``u1``, ``u2``
    and, unless the reduced basis is orthogonal, ``u1 + u2``.
    The Voronoi cell of the origin is ``{x : |x . v| <= |v|^2 / 2}`` over
    these vectors: a hexagon, or a rectangle in the orthogonal case.

    Parameters
    ----------
    b1, b2 : np.ndarray
        Lattice basis.

    rtol : float, optional
        Relative tolerance on ``u1 . u2`` below which the reduced basis is
        considered orthogonal.

    itermax : int, optional
        Maximum number of iterations of the reduction.

    Returns
    -------
    list[np.ndarray]
        Two or three relevant vectors.
    """
    u1, u2 = gauss_reduce(b1, b2, itermax)
    if u1 @ u2 > 0:
        u2 = -u2
    vectors = [u1, u2]
    if abs(u1 @ u2) > rtol * np.sqrt((u1 @ u1) * (u2 @ u2)):
        vectors.append(u1 + u2)
    return vectors


def voronoi_cell_facets(
    basis: np.ndarray,
    metric: np.ndarray | None = None,
    rtol: float = 1e-9,
    reduction_itermax: int = 100,
) -> list[np.ndarray]:
    """
    Facet vectors of the Voronoi cell of a lattice in a given metric.

    The metric ``G`` is symmetric positive definite; distances are measured
    as ``|x|_G^2 = x^T G x``. With the Cholesky factorisation ``G = L L^T``,
    the change of variables ``y = L^T x`` makes the metric Euclidean. The
    relevant vectors ``v`` of the mapped lattice ``L^T B Z^2`` give the facet
    vectors ``a = L v / |v|^2``, and the cell is ``{x : |x . a| <= 1/2}``.

    Parameters
    ----------
    basis : np.ndarray
        Lattice basis ``B``, one vector per column, shape ``(2, 2)``.

    metric : np.ndarray or None, optional
        Metric ``G``, shape ``(2, 2)``. None means the Euclidean metric.

    rtol : float, optional
        Relative tolerance of the orthogonality test (see
        :func:`relevant_vectors`).

    reduction_itermax : int, optional
        Maximum number of iterations of the reduction.

    Returns
    -------
    list[np.ndarray]
        Two or three facet vectors.
    """
    basis = np.asarray(basis, dtype=np.float64)
    if metric is None:
        chol = np.eye(2)
    else:
        chol = np.linalg.cholesky(np.asarray(metric, dtype=np.float64))
    # y = L^T x makes the metric Euclidean
    mapped = chol.T @ basis
    vectors = relevant_vectors(mapped[:, 0], mapped[:, 1], rtol, reduction_itermax)
    # back to the metric G
    return [chol @ v / (v @ v) for v in vectors]


def voronoi_cell_vertices(facets: list[np.ndarray]) -> np.ndarray:
    """
    Ordered vertices of the convex cell ``{x : |x . a| <= 1/2}``.

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors.

    Returns
    -------
    np.ndarray
        Vertices, shape ``(k, 2)``, ordered counter-clockwise.
    """
    # Each facet gives 2 boundary lines, x . a = 1/2 and x . (-a) = 1/2.
    lines = [sign * np.asarray(a, dtype=np.float64) for a in facets for sign in (1, -1)]
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            system = np.array([lines[i], lines[j]])
            # Parallel lines (opposite edges) have no intersection
            if abs(np.linalg.det(system)) < 1e-12:
                continue
            p = np.linalg.solve(system, [0.5, 0.5])
            # keep the intersection only if it lies in the cell
            if all(abs(p @ a) <= 0.5 + 1e-9 for a in facets):
                points.append(p)
    # rounding can make duplicates around zero
    points = np.unique(np.round(np.array(points), 12), axis=0)
    # Sorting : the cell is convex by definition and centrally symmetric : the
    # mean is inside the cell => angular sort around it.
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]
