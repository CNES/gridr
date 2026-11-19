.. _aa-transition-matrix-page:

======================================
Transition matrix of an irregular grid
======================================

The antialiasing methods work on the lattice of the transition matrix :math:`W`, the local Jacobian of the backward grid
(:ref:`aa-sec-lattice`). When the transformation is affine, as for a decimation, a rotation or a shear, :math:`W` is the
same everywhere and any two rows and columns of the grid give it exactly. On an irregular backward grid, the local
Jacobian varies across the image and some nodes may be invalid. A method that designs a single filter for the whole
image, as :doc:`source prefiltering <source_prefiltering>`, then needs one transition matrix that stands for the whole
grid. This page describes how GridR estimates it, and how far the grid is from an affine one.

.. _aa-sec-estimation:

Estimation modes
================

The transition matrix is estimated once, on the valid nodes of the grid. GridR offers two estimation modes. In both,
node :math:`(r, c)` of a grid stored with resolution :math:`(\rho_r, \rho_c)` corresponds to the destination pixel
:math:`(i', j') = (r\rho_r, c\rho_c)`, and masked nodes or nodes equal to ``grid_nodata`` are left out.

``longest-distance`` (default). In every row of the grid, the first and the last valid nodes are found. The row where
they are farthest apart, in number of nodes, gives :math:`\mathbf w_1`: their coordinate difference divided by their
index distance times :math:`\rho_c`. The column of longest valid span gives :math:`\mathbf w_2` in the same way, with
:math:`\rho_r`.

``least-square-fit``. The affine model :math:`(x, y)^\top = \mathbf p_0 + W\,(j', i')^\top` (:math:`\mathbf p_0` the
origin of the direct lattice) is fitted by least squares on all valid nodes, once for :math:`x` and once for :math:`y`.

Both modes are exact on affine grids. They differ on curved grids (:ref:`Figure 27 <aa-fig-estimation>`).
``least-square-fit`` returns the mean of the local Jacobian. ``longest-distance`` only sees the first row and column,
where this grid is least stretched, and gives a larger cell: the filter is weaker than the mean Jacobian asks. It is
the default because it is the matrix that the grid geometry analysis of the resampling already computes.

Whatever the mode, the largest distance between a valid node and the least-squares affine model is reported as
``affine_residual``, in source pixels.

.. _aa-fig-estimation:

.. figure:: /images/antialiasing/fig_estimation.png
   :width: 555px
   :align: center

   Figure 27. Estimating the transition matrix. (a) A curved grid
   (:math:`z \mapsto 2z + 0.004\,z^2`, with :math:`z = c + \mathrm{i}\,r` for the node of row :math:`r` and column
   :math:`c`, one node in three shown); in red,
   the first row and the first column, which have the longest valid span
   and are used by ``longest-distance``. (b) Cells obtained from the two
   estimates, against the cell of the mean Jacobian (grey).
