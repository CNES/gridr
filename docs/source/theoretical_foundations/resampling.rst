.. resampling:

==========
Resampling
==========

Definition
==========

General definition
******************
Let :

- :math:`I_s` be the **source discrete image**, defined on the regular sampling grid

.. math::
    \mathscr{G}_s = \{ (x,y) = (j,i) \in \mathbb{Z}^2 \}
 
- :math:`F_s : \Omega_s \to \mathbb{R}` be the **continuous image** associated with  :math:`I_s`, defined on the continuous footprint

.. math::

    \Omega_{F_s} = \left[ - \frac{1}{2}, H_s + \frac{1}{2} \right] \times \left[ - \frac{1}{2}, W_s + \frac{1}{2} \right]

- :math:`I_d` be the **destination discrete image**, defined on its own sampling grid

.. math::
    \mathscr{G}_d = \{ (x',y') = (j',i') \in \mathbb{Z}^2 \}

**Resampling** is the process of determining, for each destination sampling point :math:`(x', y') \in \mathscr{G}_d`, a corresponding location :math:`(u,v)` in the source continous domain :math:`\Omega_s` and assigning a value based on the continuous image :math:`F_s`:

.. math::
    I_d[i',j'] \approx F_s(u,v) 

Formally, resampling requires a rule that associates to every :math:`(x',y') \in \mathscr{G}_d` a point :math:`(u,v) \in \Omega_s`. This rule may come from:

- a simple geometric transform (zoom, rotation, shear),
- a known analytic mapping,
- or a general remapping grid

Because the continuous image :math:`F` is not explicitly known, it must be reconstructed from the discrete source image using an interpolation model (nearest, linear, cubic, cardinal B-Splines, physical kernels, etc.).

Thus, resampling consists conceptually of two steps:

1) A geometric mapping
2) A radiometric interpolation 


Forward and Backward Mappings
*****************************
There are two fundamental ways to define the correspondance between source and destination sampling grids.

Forward Mapping (Source :math:`\to` Destination)
------------------------------------------------
A **forward** (or *push-forward*) transformation specifies how each source sampling point moves in the continuous destination plane:

.. math::

    g: \mathscr{G}_s \to \Omega_d,\;  (x', y') = g_x(x,y), g_y(x,y)

Conceptually:

- each **soure pixel center** :math:`(x,y) \in \mathscr{G}_s` is mapped to a continuous location :math:`(x', y')`,
- these mapped points must then be projected back onto the destination sampling grid :math:`\mathscr{G}_d` to produce the discrete image.

Backward Mapping (Destination :math:`\to` Source)
-------------------------------------------------
A **backward** (or *pull-back*) transformation directly expresses, for each destination sampling point, where to sample the source:

.. math::

    f: \mathscr{G}_d \to \Omega_s,\;  (u, v) = f_x(x',y'), f_y(x',y')

Thus each destination pixel center :math:`(x',y')` retrieves its value by sampling the continuous source image at (u, v).

Classical Geometric Transformations
***********************************
Both forward and backward mappings can represent common geometric transformations:

- Zoom / Scaling
    Changes pixel spacing; resampling evaluates :math:`F_s` on a uniformly scaled grid
- Dezoom / Downsampling
    Similar to scaling but coarser; often requires low-pass filtering to limit aliasing artefacts
- Rotation
    The new sampling grid is a rotated version of the original one.
- General Remapping Functions
    Arbitrary mappings :math:`(x',y') \longmapsto  (x,y)`, e.g., warping fields, distortion corrections, etc.
- User-provided Sampling Grids
    Arbitrary sets of continuous coordinates, possibly subsampled.
- translation,


User-provided Sampling Grids
============================
`GridR` implements a geometric transformation by evaluating images on arbitrary sets of continuous coordinates.

This process is referred to as *grid resampling* in `GridR`.

For this resampling pipeline, the sampling grid is always defined in the **backward** (destination :math:`\to` source) sense: for each pixel center :math:`(j,i)` of the destination discrete image, the grid provides the continuous coordinates :math:`(x,y)` in the source image from which the value must be interpolated.

This is consistent with the pixel-center convention adopted throughout `GridR`, where the center of pixel :math:`(i,j)` is located at continuous coordinates :math:`(j,i)`.

Formally, given a destination pixel center :math:`(j,i)`, the backward grid provides

.. math::

    (x,y) = \Phi(i,j)

where :math:`\Phi : \mathbb{Z}^2 \to \mathbb{R}^2` is the backward geometric transformation

Subsampled Grids
****************
To reduce disk/memory footprint and/or to limit the upstream computation required to build the transformation, `GridR` allows the backward sampling grid :math:`\Phi` to be stored at a lower spatial resolution than the destination image.

In this case, full-resolution coordinates :math:`(x,y)` for every destination pixel center are reconstructed using **bilinear interpolation** on the subsampled grid.
