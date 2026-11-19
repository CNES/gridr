.. _aa-reciprocal-lattice-page:

===================================
Aliasing and the reciprocal lattice
===================================

.. _aa-sec-aliasing:

Aliasing on a zone plate
========================

Grid resampling evaluates the continuous source image :math:`F_s` at the points :math:`\Phi(i', j')`, where
:math:`\Phi` is the backward grid and :math:`(i', j')` the center of a destination pixel (row :math:`i'`, column :math:`j'`).

The following pages use an apodized Fresnel zone plate to show the effects of aliasing on the destination image.

.. note::

   **The zone plate.** The test pattern is a Fresnel zone plate on :math:`512 \times 512` pixels, with a soft edge. With
   :math:`\mathbf p = (x, y)^\top` a point of the image, :math:`x` along columns and :math:`y` along rows,
   and :math:`\mathbf c` the center of the image,

   .. math::

      u(\mathbf p) = \tfrac12 + \tfrac12\, w\big(\lVert \mathbf p - \mathbf c\rVert\big)\cos\big(\pi k\,\lVert \mathbf p - \mathbf c\rVert^2\big), \qquad k = 0.45/256 ,

   where :math:`k` sets the rate at which the rings tighten, and :math:`w` is an apodization function of the distance
   :math:`r` to the center, which fades the rings over the last 16 pixels of the disk in order to avoid a sharp
   discontinuity:

   .. math::

      w(r) =
      \begin{cases}
         1 & \text{if } r \le 240, \\
         \tfrac12 + \tfrac12\cos\big(\pi\,(r - 240)/16\big) & \text{if } 240 < r < 256, \\
         0 & \text{if } r \ge 256.
      \end{cases}

   Near a point :math:`\mathbf p`, the pattern is a plane
   sinusoid whose frequency vector is :math:`\boldsymbol\nu = k\,(\mathbf p - \mathbf c)`: the local
   frequency at :math:`\mathbf p` (:ref:`Figure 1 <aa-fig-zp_source>`).
   A position in the image thus codes a frequency: the distance to the center gives its modulus, the direction gives its
   orientation. This is why the axes of the zone plate images are graduated in cycles per pixel, and why regions of the
   frequency plane can be drawn on them.

   The rings reach 0.45 cycle per pixel at the edge of the disk, just below the Nyquist frequency of the source.
   Outside the disk the pattern is a uniform grey: in the corners the local frequency would reach
   :math:`0.45\sqrt2 \approx 0.64` cycle per pixel and the source would itself alias. Without the window, the sharp
   edge of the disk would add a visible pattern of its own to the resampled images, easy to mistake for aliasing.
   The spectrum (:ref:`Figure 2 <aa-fig-zp_spectrum>`) holds every frequency up to :math:`240\,k \approx 0.42` cycle
   per pixel with the same level on average, and falls to zero between 0.42 and 0.45, where the window fades the
   rings. Its dark rings and moiré come from the Discrete Fourier Transform and the pattern itself, not from any
   resampling.

.. _aa-fig-zp_source:

.. figure:: /images/antialiasing/fig_zp_source.png
   :width: 560px
   :align: center

   Figure 1. The source zone plate, shown pixel for pixel, axes graduated in local frequency
   :math:`\boldsymbol\nu = k(\mathbf p - \mathbf c)` (cycles per pixel). :math:`\mathbf c` is the center of the pattern and
   :math:`\mathbf p` a point. Outside the disk of radius 256 pixels, the pattern is a uniform grey; the rings fade into it
   over the last 16 pixels of radius.

.. _aa-fig-zp_spectrum:

.. figure:: /images/antialiasing/fig_zp_spectrum.png
   :width: 606px
   :align: center

   Figure 2. Spectrum of the zone plate, in dB below its peak. Its mean level is constant up to about 0.42 cycle per pixel and falls to
   zero at 0.45, where the rings fade into the grey; its structure (dark rings, moiré) comes from the pattern and the DFT.

.. warning::

   The zone plate and its spectrum are shown pixel for pixel, at their true resolution. When the page is zoomed, or when
   the operating system scales the display, the browser resamples the images and may
   show false patterns. This is aliasing too, but it comes from the display, not from the resampling studied here.
   To check a figure, view it at 100 % zoom, or open the image on its own.

   The resampled images are shown without interpolation, each pixel as a block of 2 × 2 screen pixels.


:ref:`Figure 3 <aa-fig-problem>` resamples the zone plate on a grid that combines a decimation by 3, a rotation by
20° and a shear of 0.5, with the quintic B-spline interpolator and no antialiasing. This grid is the running example of
these pages (:ref:`next section <aa-sec-lattice>` gives its matrix). The output (a) reproduces the central rings, but
false rings appear all around them, and two complete false zone plates show in opposite corners: rings too fine for
the destination have folded back into coarser ones.
The reference (b) is the image the destination should show: the rings it can represent, and a uniform grey beyond. It
is computed from the source image, whose spectrum is cut to the reciprocal cell (:ref:`aa-sec-reciprocal`), here the
default parallelogram, and then sampled on the grid without interpolation error. The error (c) is the
difference between the output and the reference.

.. _aa-fig-problem:

.. figure:: /images/antialiasing/fig_problem.png
   :width: 744px
   :align: center

   Figure 3. The zone plate resampled on the grid of the running example (decimation by 3, rotation by 20°, shear 0.5),
   with the quintic B-spline interpolator and no antialiasing. (a) Output; the two grey corners map outside the disk of
   the zone plate. (b) Reference, the image the destination should show. (c) Difference between the output and the
   reference, with enhanced contrast (x3), and its rms. The three images have 116 × 116 pixels.

.. _aa-sec-lattice:

The direct lattice
==================

Near a point, the backward grid is affine:

.. math::


   \Phi(i', j') \approx \mathbf p_0 + W \begin{pmatrix} j' \\ i' \end{pmatrix}, \qquad W = [\mathbf w_1\ \mathbf w_2].

Here :math:`\mathbf p_0` is a point of the source plane, and :math:`\mathbf w_1 = \partial\Phi/\partial j'`
and :math:`\mathbf w_2 = \partial\Phi/\partial i'` are the source displacements for one destination column and one
destination row. We take the origin of the source plane at :math:`\mathbf p_0`. The destination pixel centers then
form the direct lattice :math:`\Lambda = W\mathbb{Z}^2` in the source plane (:ref:`Figure 4 <aa-fig-lattice>`).
The cell spanned by :math:`\mathbf w_1` and :math:`\mathbf w_2`
has area :math:`\lvert \det W\rvert`, the mean number of source pixels per destination pixel.

:math:`W` is the transition matrix of the grid, the local Jacobian of :math:`\Phi`. For the running example:

.. math::

   W = 3\begin{pmatrix}\cos 20^\circ & -\sin 20^\circ\\ \sin 20^\circ & \cos 20^\circ\end{pmatrix}\begin{pmatrix}1 & 0.5\\ 0 & 1\end{pmatrix}.

Here :math:`\lvert\det W\rvert = 9`: each destination pixel covers nine source pixels on average. The shear makes
:math:`\mathbf w_1` and :math:`\mathbf w_2` unequal in length (3 and 3.35 source pixels) and not orthogonal (63°).

.. _aa-fig-lattice:

.. figure:: /images/antialiasing/fig_lattice.png
   :width: 643px
   :align: center

   Figure 4. (a) Destination grid :math:`\mathscr{G}_d`. (b) Its image
   :math:`\Phi(\mathscr{G}_d)` on the source grid for the running example:
   the direct lattice :math:`\Lambda`, spanned by :math:`\mathbf w_1` and
   :math:`\mathbf w_2`.



.. _aa-sec-reciprocal:

The reciprocal lattice
======================

The reciprocal lattice of :math:`\Lambda` is

.. math::


   \Lambda^* = \{ \boldsymbol\nu : \boldsymbol\nu\cdot{\boldsymbol\lambda} \in \mathbb{Z} \text{ for all } {\boldsymbol\lambda} \in W\mathbb{Z}^2 \} = W^{-\top}\mathbb{Z}^2 ,

where :math:`\boldsymbol\nu = (\nu_x, \nu_y)^\top` is a source frequency, in cycles per source pixel, and :math:`{\boldsymbol\lambda}` a vector of
the direct lattice.

Its basis :math:`B = W^{-\top} = [\mathbf b_1\ \mathbf b_2]` satisfies
:math:`\mathbf b_i\cdot \mathbf w_j = \delta_{ij}`: :math:`\mathbf b_1` is orthogonal to :math:`\mathbf w_2` and
:math:`\mathbf b_2` to :math:`\mathbf w_1`. Its cell has area :math:`1/\lvert\det W\rvert`.

A coarser destination grid gives a denser reciprocal lattice. In destination units, :math:`\boldsymbol\mu = W^\top \boldsymbol\nu`
(cycles per destination pixel), the reciprocal lattice is :math:`\mathbb{Z}^2`.

Sampling the continuous source image :math:`F_s` on :math:`\Lambda` amounts to multiplying it by a comb of Dirac
impulses placed at the points of :math:`\Lambda`: :math:`F_\Lambda = F_s\cdot\sum_{{\boldsymbol\lambda}\in\Lambda}\delta_{\boldsymbol\lambda}`,
where :math:`\delta_{\boldsymbol\lambda}` is the Dirac impulse at :math:`{\boldsymbol\lambda}`. The Fourier transform of
a comb on :math:`\Lambda` is a comb on the reciprocal lattice :math:`\Lambda^*`, scaled by :math:`1/\lvert\det W\rvert`.
The product in space becomes a convolution in frequency: with :math:`\hat F_s` the Fourier transform of :math:`F_s`,
each impulse of the second comb places a copy of :math:`\hat F_s`, and the spectrum of the sampled image is periodised:

.. math::

   \hat F_\Lambda(\boldsymbol\nu) = \frac{1}{\lvert\det W\rvert}\sum_{{\boldsymbol\lambda}^*\in\Lambda^*}\hat F_s(\boldsymbol\nu - {\boldsymbol\lambda}^*).

The sampled spectrum is the sum of copies of :math:`\hat F_s` shifted by the points of :math:`\Lambda^*`. Another
choice of origin would only change the phase of each copy, not its position or modulus.
Where two copies overlap, distinct frequencies of :math:`F_s` add up and cannot be separated: this is aliasing
(:ref:`Figure 5 <aa-fig-periodisation>` a).
A frequency :math:`\boldsymbol\nu` and its translates :math:`\boldsymbol\nu + {\boldsymbol\lambda}^*` form a class.
After sampling, they give the same destination frequency.

A cell of the reciprocal lattice, or fundamental domain, is a set :math:`S` whose translates :math:`S + {\boldsymbol\lambda}^*` tile
the plane: they are disjoint, up to their boundaries, and cover it. A cell contains exactly one frequency of each class
and has area :math:`1/\lvert\det W\rvert`. A lattice has many cells (:ref:`Figure 5 <aa-fig-periodisation>` b and c).

.. _aa-fig-periodisation:

.. figure:: /images/antialiasing/fig_periodisation.png
   :width: 785px
   :align: center

   Figure 5. (a) A spread spectrum (red disk) copied at every point of
   :math:`\Lambda^*`: the copies overlap. (b) A spectrum limited to a cell:
   the copies tile the plane. (c) Another cell of the same lattice gives
   another tiling. Dashed square: Nyquist square of the source.

.. admonition:: Antialiasing condition
   :class: important

   Sampling on the lattice :math:`\Lambda` adds to the spectrum of the sampled signal all its translates by the vectors
   of :math:`\Lambda^*`. No aliasing occurs if and only if, before sampling, the spectrum lies inside a cell of the
   reciprocal lattice: the copies then do not overlap, and each class keeps a single frequency. Whatever the method,
   avoiding aliasing therefore means keeping at most one frequency of each class before sampling.
   On a locally regular grid, the condition holds locally, with the lattice of the local Jacobian.

:ref:`Figure 6 <aa-fig-aliasing_source>` to :ref:`Figure 8 <aa-fig-aliasing_cell>` apply the condition to the
resampling of :ref:`Figure 3 <aa-fig-problem>`. The reciprocal lattice is spanned by :math:`\mathbf b_1` and :math:`\mathbf b_2`.
The cell used here is the parallelogram :math:`\{\boldsymbol\nu : \lVert W^\top \boldsymbol\nu\rVert_\infty \le \tfrac12\}`:
in destination units, it is the Nyquist square of the destination,
:math:`\lvert\mu_1\rvert \le \tfrac12` and :math:`\lvert\mu_2\rvert \le \tfrac12`. It is the default cell of GridR; the
next page discusses the choice of a cell.

Since a position of the zone plate codes a frequency, the cell and the points of :math:`\Lambda^*` can be drawn on the
zone plate itself (:ref:`Figure 6 <aa-fig-aliasing_source>`). The rings inside the parallelogram are those the destination
can hold; all the others fold at sampling. The dashed outline is the part of the source that the output of
:ref:`Figure 3 <aa-fig-problem>` covers: it contains the cell, and two points of :math:`\Lambda^*`,
:math:`\pm(\mathbf b_1 + \mathbf b_2)`. The spectrum of the zone plate fills the disk of radius 0.45 and overflows the cell
widely (:ref:`Figure 7 <aa-fig-aliasing_spectrum>`).

.. _aa-fig-aliasing_source:

.. figure:: /images/antialiasing/fig_aliasing_source.png
   :width: 560px
   :align: center

   Figure 6. The source zone plate, shown pixel for pixel, axes graduated in local frequency (cycles per pixel), with
   the cell :math:`\{\boldsymbol\nu : \lVert W^\top \boldsymbol\nu\rVert_\infty \le \tfrac12\}` of the running example
   (red), the points of its reciprocal lattice (blue), and the part of the source covered by the output of
   :ref:`Figure 3 <aa-fig-problem>` (dashed green).

.. _aa-fig-aliasing_spectrum:

.. figure:: /images/antialiasing/fig_aliasing_spectrum.png
   :width: 606px
   :align: center

   Figure 7. Spectrum of the zone plate (dB), points of the reciprocal
   lattice of the running example (blue) and the cell
   :math:`\{\boldsymbol\nu : \lVert W^\top \boldsymbol\nu\rVert_\infty \le \tfrac12\}`
   (white): the spectrum overflows the cell.

In the same way, each output pixel can be labelled by the local frequency of its source point. The red parallelogram
of :ref:`Figure 8 <aa-fig-aliasing_cell>` bounds the output pixels whose local frequency lies inside the cell, about 30 %
of the output: it is the red parallelogram of :ref:`Figure 6 <aa-fig-aliasing_source>`, carried to the output. The false
rings all lie outside it: they are the frequencies outside the cell, folded at sampling. The centers of the two false
zone plates are the output pixels whose local frequency is :math:`\pm(\mathbf b_1 + \mathbf b_2)`: there, the frequency
folds to zero.

.. _aa-fig-aliasing_cell:

.. figure:: /images/antialiasing/fig_aliasing_cell.png
   :width: 496px
   :align: center

   Figure 8. (a) Output of Figure 3; red: the output pixels whose source
   point has its local frequency on the boundary of the cell. (b) Its
   error, contrast multiplied by 3: the false rings lie outside the red
   parallelogram.
