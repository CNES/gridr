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
Lattice
"""
from enum import Enum

import numpy as np


class Domain(Enum):
    """
    Defines the domain : spatial or frequential
    """

    FREQUENCY_DOMAIN = 0
    SPATIAL_DOMAIN = 1


class Lattice2d(object):
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
        cosine_treshold: float = 0.01,
        reduction_itermax: int = 100,
    ):
        """
        Compute the normalized normal vectors of the edges of a minimal Voronoi
        paving cell.

        Given a lattice in two-dimensional space, the minimal Voronoi paving
        cell is defined as follows:

        -   For every point within the cell, the closest point in the lattice is
            the cell's centroid
        -   The collection of all such cells, each centered on a different lattice
            point, completely covers the plane without overlaps

        This implementation calculates the unit-length normal vectors for each
        edge of the Voronoi cell. These vectors:

        -   Are perpendicular to their corresponding edges
        -   Have unit length (normalized)

        The six nearest neighbor point in the lattice to any given cell centroid
        are determined by all possible integer linear combinations of the basis
        vectors where the coefficients are restricted to the set {-1, 0, 1}.

        Parameters
        ----------
        lattice2d: Lattice2d
            The input 2D lattice for which we want to compute the paving cell.

        cosine_treshold: float, optional
            The angular threshold parameter that determines the orthogonality
            condition between the two basis vectors of the reduced lattice. When
            the cosine of the angle between these vectors falls below this
            threshold, the algorithm considers them sufficiently orthogonal to
            produce a quadrilateral Voronoi cell rather than a hexagonal one.

        reduction_itermax : int, optional
            Maximum number of iterations for the lattice reduction
            preprocessing.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, NoReturn]]
            A tuple containing the three basis vectors (w1, w2, w3) for the
            Voronoi cell. w3 will be None if the cell is quadrilateral (cosine
            below threshold).
        """
        lattice2d.check_singularity()

        # Step 1: Lattice reduction to obtain a reduced basis
        w_geometry = cls.gaussian_lattice_reduction(lattice2d, reduction_itermax)

        # Step 2: Extract the reduced basis vectors
        # - v1 is orthogonal to the perpendicular bisector edge between the
        # centroid and neighbors at ±v1 (with v2=0)
        # - v2 is orthogonal to the perpendicular bisector edge between the
        # centroid and neighbors at ±v2 (with v1=0)
        w1 = w_geometry.v1  # First basis vector
        w2 = w_geometry.v2  # Second basis vector

        # Step 3: Compute the cosine of the angle between w1 and w2
        # This determines whether we'll have a hexagonal or quadrilateral cell
        cos_w1_w2 = np.dot(w1, w2) / (np.dot(w1, w1) * np.dot(w2, w2)) ** 0.5

        # Step 4: Determine cell geometry based on angle cosine
        if np.abs(cos_w1_w2) > cosine_treshold:
            # Hexagonal cell case - compute the third basis vector
            # Candidates are w1 + w2 and w1 - w2
            w3_a = w1 + w2
            w3_b = w1 - w2

            # Select the candidate with the smallest norm
            if np.dot(w3_b, w3_b) < np.dot(w3_a, w3_a):
                w3 = w3_b
            else:
                w3 = w3_a

            # Normalize and adjust sign for consistent orientation
            w3 = w3 / np.dot(w3, w3)
            w3[1] = -w3[1]  # Flip y-component for standard orientation
        else:
            w3 = None  # Quadrilateral cell case

        # Step 5: Normalize and standardize orientation for all basis vectors
        w1 = w1 / np.dot(w1, w1)
        w1[1] = -w1[1]
        w2 = w2 / np.dot(w2, w2)
        w2[1] = -w2[1]

        return w1, w2, w3

    @classmethod
    def gaussian_lattice_reduction(
        cls,
        lattice2d: "Lattice2d",
        itermax: int = 100,
    ) -> "Lattice2d":
        r"""A Gaussian Lattice Reduction algorithm implementation.

        This method reduces a 2D lattice basis to find the smallest vectors.

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

        Notes
        -----
        The algorithm works as follows:

        1.  If :math:`\|\mathbf{v}_2\| < \|\mathbf{v}_1\|`, swap :math:`\mathbf{v}_1` and
            :math:`\mathbf{v}_2`.

        2.  Compute the projection scalar :math:`m` :
            :math:`m = \lfloor \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\mathbf{v}_1 \cdot \mathbf{v}_1} \rfloor` # noqa: E501, B950

        3.  If :math:`m = 0`, return the original basis vectors :math:`\mathbf{v}_1` and
            :math:`\mathbf{v}_2`.

        4.  Replace :math:`\mathbf{v}_2` with :math:`\mathbf{v}_2 - m \mathbf{v}_1`

        5.  Repeat the process until convergence or until the vectors are orthogonal or `itermax`
            is reached

        """
        u1 = lattice2d.v1.copy()
        u2 = lattice2d.v2.copy()
        counter = 0

        while counter < itermax:
            # Calculate norms
            norm_u1_sq = np.dot(u1, u1)
            norm_u2_sq = np.dot(u2, u2)

            # Swap if necessary
            if norm_u2_sq < norm_u1_sq:
                # swap u1 and u2
                u1, u2 = u2, u1
                norm_u1_sq, norm_u2_sq = norm_u2_sq, norm_u1_sq

            # Calculate projection
            m = int(np.dot(u1, u2) / norm_u1_sq)

            if m == 0:
                break

            # Update u2
            u2 = u2 - m * u1
            counter += 1

        return cls(v1=u1, v2=u2, domain=lattice2d.domain)
