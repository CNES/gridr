.. _aa-source-prefiltering:

==================================================
Source prefiltering: the reciprocal cell filter
==================================================

.. admonition:: Scope
   :class: important

   This method avoids aliasing with a low-pass filter applied to the source before interpolation. From the transition
   matrix :math:`W` of the grid, it takes as reciprocal cell the Voronoi cell of the reciprocal lattice in a stated
   metric, turns the cell into a smooth template with a bounded leak on its boundary, and realises the template with an
   odd-sized kernel whose response is checked against the template. The kernel is the same over the whole image. This page takes :math:`W` as given; its
   estimation on an irregular grid is described in :doc:`transition_matrix_estimation`.

The method applies the antialiasing condition of the :ref:`previous page <aa-sec-reciprocal>` to the lattice of
:math:`W`. Its principle follows the presentation of image sampling in the CNES reference book
:cite:`cnes2012satellite`. This page details the design, measures the effect of the filter on the zone plate and
gives the interface. Two experimental options have a page of their own (:doc:`experimental_options`).

.. _aa-sec-cell:

Fundamental domains and the choice of a cell
============================================

The antialiasing condition asks for a cell of the reciprocal lattice, a fundamental domain, but does not single out one.
Two kinds of domain are valid for any lattice:

-  parallelograms :math:`B'[-\tfrac12, \tfrac12]^2` built on a basis :math:`B'` of :math:`\Lambda^*`; any basis will do,
   and the longer its vectors, the thinner the parallelogram;
-  Voronoi cells: the points closer to the origin than to any other lattice point, for a given metric.

:ref:`Figure 9 <aa-fig-tilings>` shows three domains for the running example, and
:ref:`Figure 10 <aa-fig-tilings_source>` draws them on the zone plate, where a position codes a frequency. The three have the same area,
:math:`1/\lvert\det W\rvert`. They differ by their shape, hence by the frequencies they keep.

.. _aa-fig-tilings:

.. figure:: /images/antialiasing/fig_tilings.png
   :width: 795px
   :align: center

   Figure 9. Three fundamental domains of the same reciprocal lattice (blue points).
   The central cell (colour) and its translates (grey); dotted circle:
   largest inscribed disk; dashed square: source Nyquist square.

.. _aa-fig-tilings_source:

.. figure:: /images/antialiasing/fig_tilings_source.png
   :width: 560px
   :align: center

   Figure 10. The three domains of Figure 9 drawn on the source zone plate, shown pixel
   for pixel, axes in local frequency. The rings inside each outline are the frequencies
   that the domain keeps; nothing is filtered here.

After sampling, the frequencies of a class merge into one, so a fundamental domain keeps one frequency per class and
the low-pass filter applied to the source must remove the others.
The spectrum of natural scenes decreases with the frequency: the domain that loses
the least keeps, in every class, the frequency closest to the origin. This is the Voronoi cell of :math:`\Lambda^*`,
the choice of the CNES reference book :cite:`cnes2012satellite`.
Distance computation needs a metric, a symmetric positive definite matrix :math:`G` that measures a frequency by
:math:`\lVert \boldsymbol\nu\rVert_G^2 = \boldsymbol\nu^\top G\,\boldsymbol\nu`, and the cell depends on it
(:ref:`Figure 11 <aa-fig-metric>`).

.. admonition:: Design choice: the metric is a parameter
   :class: tip

   GridR takes the Voronoi cell and leaves the metric to the user, as ``isotropic_frame``: the space in which the scene
   is isotropic is a property of the data, not of the grid. The same transition matrix can come from a square-pixel
   sensor resampled to a sheared geometry, or from an oblique sensor resampled to a square-pixel map, and the two call
   for different metrics.

The cases implemented in GridR are summarised below.

+------------------------------------------------+----------------------------------------------------------------------------------+
| ``isotropic_frame``                            | Cell                                                                             |
+================================================+==================================================================================+
| ``"destination"``, :math:`G = WW^\top`         | parallelogram :math:`\{\lVert W^\top \boldsymbol\nu\rVert_\infty \le \tfrac12\}` |
+------------------------------------------------+----------------------------------------------------------------------------------+
| ``"source"``, :math:`G = I`                    | hexagon, or rectangle when :math:`\Lambda^*` has an orthogonal basis             |
+------------------------------------------------+----------------------------------------------------------------------------------+
| matrix :math:`G`, e.g. :math:`D^{-1}D^{-\top}` | hexagon or rectangle                                                             |
| with :math:`D` the local Jacobian from source  |                                                                                  |
| pixels to a target coordinate system (e.g.     |                                                                                  |
| ground coordinates)                            |                                                                                  |
+------------------------------------------------+----------------------------------------------------------------------------------+

The default is ``"destination"``: it always gives a parallelogram aligned with the destination grid.
:ref:`Figure 11 <aa-fig-metric>` shows the cells of the running example in three metrics. The choice matters only on
sheared grids: the destination and source cells coincide when :math:`\mathbf w_1` and :math:`\mathbf w_2` are
orthogonal.

.. _aa-fig-metric:

.. figure:: /images/antialiasing/fig_metric.png
   :width: 795px
   :align: center

   Figure 11. Voronoi cells of the same reciprocal lattice in three metrics. Dotted
   ellipse: largest disk of the metric inside the cell, the frequencies kept in every
   direction of that metric.

The zone plate is isotropic in source pixels, so the hexagon of the source metric suits it better than the default
parallelogram. Both cells have the same area, but the hexagon keeps the rings about as far in every direction, while
the parallelogram reaches further in its corners and less far elsewhere. The rest of the page uses the default
parallelogram, the cell obtained when no metric is given, and shows the hexagon next to it where the comparison is
useful.

.. _aa-sec-voronoi:

Computing the Voronoi cell
==========================

The Voronoi cell is a polygon whose edges, or facets, are the perpendicular bisectors of the segments joining the
origin to a few lattice points, its relevant vectors. In two dimensions there are two or three pairs of them, found from
a reduced basis of the lattice :cite:`kurlin2024mathematics`.

The cell is computed in a space where the metric is the usual Euclidean norm. Any matrix :math:`L` with
:math:`G = LL^\top` gives such a space: with :math:`\tilde{\boldsymbol\nu} = L^\top \boldsymbol\nu`,

.. math::

   \lVert \boldsymbol\nu\rVert_G^2 = \boldsymbol\nu^\top L L^\top \boldsymbol\nu = \lVert L^\top \boldsymbol\nu\rVert^2
   = \lVert \tilde{\boldsymbol\nu}\rVert^2.

GridR takes for :math:`L` the Cholesky factor of :math:`G`, which exists for any symmetric positive definite matrix.
A lattice point :math:`W^{-\top}\mathbf k` becomes :math:`L^\top W^{-\top}\mathbf k`, so the reciprocal lattice becomes
the lattice of basis :math:`L^\top W^{-\top}`. Distances are preserved, and the Voronoi cell of :math:`\Lambda^*` for
:math:`G` is the image by :math:`L^{-\top}` of the Euclidean Voronoi cell of this lattice.
The computation then takes three steps:

1. the Lagrange-Gauss algorithm reduces the basis to the two shortest independent vectors :math:`\mathbf e_1` and
   :math:`\mathbf e_2`, with :math:`\lVert \mathbf e_1\rVert \le \lVert \mathbf e_2\rVert` and
   :math:`\lvert \mathbf e_1\cdot \mathbf e_2\rvert \le \lVert \mathbf e_1\rVert^2/2`;
2. the sign of :math:`\mathbf e_2` is chosen so that :math:`\mathbf e_1\cdot \mathbf e_2 \le 0`; then :math:`\mathbf e_1`,
   :math:`\mathbf e_2` and :math:`\mathbf e_0 = -(\mathbf e_1 + \mathbf e_2)` form an obtuse superbase: no two of them
   make an acute angle;
3. the cell is bounded by the perpendicular bisectors of :math:`\pm \mathbf e_1`, :math:`\pm \mathbf e_2` and
   :math:`\pm \mathbf e_0`, the relevant vectors. It is a hexagon, or a rectangle when :math:`\mathbf e_1\cdot \mathbf e_2 = 0`,
   since :math:`\pm \mathbf e_0` then add no side.

Back in source frequencies, the bisector of a relevant vector :math:`\mathbf v` becomes the facet
:math:`\mathbf a = L\,\mathbf v/\lVert \mathbf v\rVert^2`, and the cell is the set of :math:`\boldsymbol\nu` such that
:math:`\lvert \boldsymbol\nu\cdot \mathbf a\rvert \le \tfrac12` for every facet.


.. _aa-fig-reduction:

.. figure:: /images/antialiasing/fig_reduction.png
   :width: 789px
   :align: center

   Figure 12. (a) A basis of the reciprocal lattice of the running example (grey) and
   the reduced obtuse basis (red). (b) Relevant vectors and the hexagonal
   Voronoi cell in the source metric. (c) Shortest hexagon edge against the
   angle between :math:`\mathbf e_1` and :math:`\mathbf e_2`: the hexagon
   turns continuously into a rectangle at 90°.

The passage from the hexagon to the rectangle is continuous: as the angle between :math:`\mathbf e_1` and
:math:`\mathbf e_2` tends to 90°, two opposite edges shrink to zero (:ref:`Figure 12 <aa-fig-reduction>` c).

In the destination metric, :math:`G = WW^\top`, the mapped lattice is a rotated square lattice and the facets are the
columns of :math:`W`. The cell is the parallelogram :math:`\{\boldsymbol\nu : \lVert W^\top \boldsymbol\nu\rVert_\infty \le \tfrac12\}`
of the :ref:`previous page <aa-sec-reciprocal>`, whatever the rotation, anisotropy or shear of the grid.

In the destination metric, :math:`G = WW^\top`, the Cholesky factor is :math:`L = WQ` with :math:`Q` orthogonal. The
mapped lattice has basis :math:`L^\top W^{-\top} = Q^\top`: it is a rotated square lattice, and the facets are the
columns of :math:`W`. The cell is the parallelogram :math:`\{\boldsymbol\nu : \lVert W^\top \boldsymbol\nu\rVert_\infty \le \tfrac12\}`
of the :ref:`previous page <aa-sec-reciprocal>`, whatever the rotation, anisotropy or shear of the grid.

.. _aa-sec-decision:

Deciding whether to filter
==========================

The source is itself sampled on :math:`\mathbb{Z}^2`: its spectrum is :math:`\mathbb{Z}^2`-periodic, and one period is
the Nyquist square :math:`[-\tfrac12, \tfrac12)^2`. If the square lies inside the cell, no source frequency can fold and
the prefilter is skipped. The cell is convex, so it is enough to test the four corners :math:`(\pm\tfrac12, \pm\tfrac12)`:
each must satisfy :math:`\lvert \boldsymbol\nu\cdot \mathbf a\rvert \le \tfrac12` for every facet :math:`\mathbf a` of
the cell (:ref:`aa-sec-voronoi`).

.. _aa-fig-decision:

.. figure:: /images/antialiasing/fig_decision.png
   :width: 771px
   :align: center

   Figure 13. Decision to filter. Dashed square: source Nyquist square; green: cell. Corners inside the cell in green,
   outside in red.

The test is exact: the square lies inside the cell if and only if its four corners do. A zoom, the identity and a
zoom along one axis need no filter. A rotation at unit scale needs one: the square and the rotated cell have the same
area, and the corners of the square stick out (:ref:`Figure 13 <aa-fig-decision>`).

The decision is all or nothing: a grid scaled by slightly more than 1 along one axis is filtered along both axes,
while the identity is not filtered at all. The progressive gates of :doc:`experimental_options` make this transition
gradual.

.. _aa-sec-template:

Frequency template
==================

.. _aa-gate-and-tolerance:

Gate and tolerance
------------------

The indicator of the cell cannot be realised by a finite kernel. GridR replaces each pair of facets by a smooth gate on
the normalised coordinate :math:`\gamma = \boldsymbol\nu\cdot \mathbf a`:

.. math::

   \varphi(\gamma) = \frac{\tanh\big(d(\gamma + t)\big) - \tanh\big(d(\gamma - t)\big)}{2\tanh(d\,t)}, \qquad d = 2\cdot\text{slope},

where :math:`\tanh` is the hyperbolic tangent,

.. math::

   \tanh x = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}.

It goes from :math:`-1` to 1 and crosses 0 at :math:`x = 0`: it is a smooth step, and the factor :math:`d` in front of
its argument sets how fast it climbs.

In the gate, each tanh is such a step, one rising at :math:`\gamma = -t`, the other
falling at :math:`\gamma = +t`, and the gate is a plateau at 1 between them. :math:`t` is its half-width, the gate is
close to 1/2 at :math:`\gamma = \pm t`, and the slope sets the steepness of its edges. Changing :math:`t` moves the edges
without changing their shape; changing the slope makes them steeper or softer around the same points
(:ref:`Figure 14 <aa-fig-gate_shape>`).

.. _aa-fig-gate_shape:

.. figure:: /images/antialiasing/fig_gate_shape.png
   :width: 769px
   :align: center

   Figure 14. The gate :math:`\varphi` over the facet coordinate, with the ideal gate (black) and the cell boundary
   :math:`\gamma = \pm\tfrac12` (dotted). (a) Slope 10, three half-widths. (b) Half-width 0.4, four slopes.

The frequency template :math:`H` is the response the filter aims at. By default, it is the product of the gates of all the
facets of the cell:

.. math::

   H(\boldsymbol\nu) = \prod_{\mathbf a \in \text{cell}} \varphi(\boldsymbol\nu\cdot\mathbf a).

:math:`H` is close to 1 inside the cell, falls across its boundary and tends to 0 outside: it is a smooth version of
the indicator of the cell. Two :doc:`experimental options <experimental_options>` change the list of gates.

The half-width :math:`t` is not a free parameter: it is set by the tolerance on the cell boundary, :math:`\gamma = \tfrac12`,

.. math::

   \varphi(\tfrac12) = \varepsilon,

where :math:`\varepsilon` is the aliasing tolerance. The equation has a closed form,

.. math::

   \cosh(2\,\text{slope}\;t) = \sinh(\text{slope})\,\sqrt{\frac{\varepsilon}{1 - \varepsilon}} ,

which has a solution when the right-hand side is at least 1. GridR rejects a slope too small for the tolerance.

At slope 10, :math:`t` is 0.500, 0.445 and 0.385 for :math:`\varepsilon` equal to 0.5, 0.1 and 0.01.

The tolerance bounds the folded amplitude. After sampling, a component at :math:`\gamma = \tfrac12 + \delta` folds onto
:math:`\lvert\gamma\rvert = \tfrac12 - \delta` with the amplitude :math:`\varphi(\tfrac12 + \delta)`, which is at most
:math:`\varepsilon` and decays exponentially with :math:`\delta` (:ref:`Figure 15 <aa-fig-gate>` b). The kernel adds at
most its response error :math:`\eta` (:ref:`aa-sec-synthesis`).

.. _aa-fig-gate:

.. figure:: /images/antialiasing/fig_gate.png
   :width: 781px
   :align: center

   Figure 15. (a) Gate profiles at slope 10 for three tolerances. (b) Amplitude of the
   alias as a function of the facet coordinate where it lands.

:ref:`Figure 16 <aa-fig-template_plate>` applies the default template of the running example to the zone plate, for
the two cells. The rings survive inside each cell and fade across its boundary.

.. _aa-fig-template_plate:

.. figure:: /images/antialiasing/fig_template_plate.png
   :width: 768px
   :align: center

   Figure 16. The central part of the source zone plate filtered by the default template
   of the running example (:math:`\varepsilon = 0.1`, slope 10), shown pixel for pixel,
   axes in local frequency. (a) Parallelogram of the destination metric. (b) Hexagon of
   the source metric. Dashed: the cell.

.. _aa-sec-kernel:

Kernel synthesis
================

.. _aa-sec-synthesis:

From the template to the kernel
-------------------------------

The kernel is an :math:`n \times n` array :math:`h`, with :math:`n` odd. Its response,
:math:`H_n(\boldsymbol\nu) = \sum_{\mathbf q} h(\mathbf q)\, e^{-2i\pi\,\boldsymbol\nu\cdot\mathbf q}`, should follow the
template :math:`H` defined above. GridR computes the ideal kernel on a fine grid and keeps its center
(:ref:`Figure 17 <aa-fig-synthesis_steps>`):

1. :math:`H` is sampled over the Nyquist square, at the frequencies :math:`\boldsymbol\nu = \mathbf m / N_f` for integer
   vectors :math:`\mathbf m`, with :math:`N_f` odd and about four times the size estimate.
2. An inverse DFT of these samples gives a fine kernel of :math:`N_f \times N_f` pixels. The template is smooth, so the
   ideal kernel decays quickly and does not wrap around on such a large grid. The fine kernel is then the ideal kernel
   sampled on the pixels, except for the part of the template outside the Nyquist square, which the samples do not
   see. This part is negligible, except near unit scale on an oblique grid (:ref:`aa-sec-periodicity`). The fine
   kernel is real and even, like :math:`H`.
3. The central :math:`n \times n` part of the fine kernel is extracted and normalized to a unit sum so that a uniform image is left
   unchanged. This is the kernel :math:`h`.
4. Its final response :math:`H_n` is computed by a zero-padded DFT and compared with :math:`H`. The largest difference
   over :math:`\lVert\boldsymbol\nu\rVert_\infty \le 0.45` is the response error. The band along the border of the
   Nyquist square is left out: the periodic template can have a corner or a jump there. The response is therefore not
   controlled in that band. The accuracy :math:`\eta`, set by ``kernel_accuracy``, is the bound asked on the response
   error.

:ref:`Figure 17 <aa-fig-synthesis_steps>` uses the parallelogram. The steps are the same for the hexagon, which has
shorter facet vectors and gives a :math:`53\times53` kernel instead of :math:`61\times61`
(:ref:`Figure 18 <aa-fig-synthesis>`).

The size :math:`n` is the smallest odd size whose response error is below :math:`\eta`. GridR starts from the
:ref:`size estimate <aa-sec-size>` below and replaces :math:`n` by :math:`2n+1` until the accuracy is met. It then
bisects between the last size that failed, or 1 when the estimate passed, and the first one that passed. The bisection
ends on an accurate size :math:`n` whose neighbour :math:`n-2` failed. The error does not always decrease with the
size: it can rise and fall again by a little. GridR therefore also tries :math:`n-4` and :math:`n-6`, and keeps the
smallest accurate size. GridR truncates the kernel when it would exceed a size bound
(:ref:`aa-sec-sizebound`).

.. admonition:: Design choice: odd-sized kernels cropped from a fine kernel
   :class: tip

   With an odd size, the kernel has a center pixel. It is then even, its response is real and the filtered image is not
   shifted. Before normalisation, the crop of the ideal kernel is the kernel of that support closest to the template in
   mean square over the Nyquist square (using Parseval's idendity). Taking the inverse DFT of :math:`n \times n` samples
   of the template would be simpler, but the kernel would only match the template at those :math:`n^2` frequencies and
   would be less accurate in between (:ref:`Figure 18 <aa-fig-synthesis>`).

.. _aa-fig-synthesis_steps:

.. figure:: /images/antialiasing/fig_synthesis_steps.png
   :width: 698px
   :align: center

   Figure 17. The synthesis steps for the running example, every sample shown. 1: template sampled on the fine grid.
   2: inverse DFT, the fine kernel on :math:`255\times255` pixels (:math:`\log_{10}\lvert h\rvert`), with the selected
   crop; it has decayed well before the edges, so it does not wrap around. 3: the :math:`61\times61` kernel, on a
   symmetric log scale so that its lobes show next to the center. 4: deviation between its true response and the
   template, below 0.01 everywhere; white: the cell.

.. _aa-fig-synthesis:

.. figure:: /images/antialiasing/fig_synthesis.png
   :width: 409px
   :align: center

   Figure 18. Response error against the kernel size for the running example: crop of the fine kernel and inverse DFT
   of :math:`n\times n` samples for the parallelogram, crop for the hexagon of the source metric (orange,
   :math:`53\times53`). Dotted line: accuracy 0.01.

.. _aa-sec-size:

Size estimate
-------------

The search starts from an estimate of the size.
The size of a kernel is set by the sharpness of its template: a sharp transition in frequency needs a long kernel in
space, a soft one a short kernel. For the template of gates, the estimate is

.. math::

   n \approx \frac{4}{\pi^2}\,\ln\frac{1}{\eta}\;\text{slope}\;\max\lVert \mathbf a\rVert ,

the maximum being taken over the facets of the template. Its derivation will be given in a technical note. It was
checked empirically on random grids, from zoom to decimation by 4, with rotation, anisotropy and shear. It is an
estimate, not a bound, and it only seeds the search: the accuracy is always measured on the kernel itself.
It holds when the template is negligible on the border of the Nyquist square, which excludes oblique grids near unit
scale (:ref:`aa-sec-periodicity`).

The size is proportional to the slope and to the longest facet vector, and depends only logarithmically on the
accuracy.

.. _aa-sec-sizebound:

Size bound
----------

By default the size bound :math:`n_{max}` is twice the size estimate, rounded up to an odd size. A user can also set
an explicit bound.

.. note::

   The factor 2 is empirical. It was set on the same random grids as the size estimate, and it covers the accurate
   sizes of these grids except on oblique grids near unit scale (:ref:`aa-sec-periodicity`). The details will be
   given in a technical note.

When the accurate size exceeds the bound, GridR truncates the accurate kernel to the bound, leaves the template as it
is, and reports that the accuracy is not met.

.. admonition:: Design choice: the template is never changed implicitly
   :class: tip

   The template is what the user asked for. Under a size bound, GridR gives the best kernel of that support and says
   how far it is from the template. Trading sharpness for a smaller kernel is up to the user, who can lower the slope.

.. _aa-sec-prefilter-effect:

Effect on the zone plate
========================

:ref:`Figure 19 <aa-fig-moire>` repeats the resampling of :ref:`Figure 3 <aa-fig-problem>` with the prefilter. It uses
the parallelogram at :math:`\varepsilon = 0.5` and at the default 0.1, and the hexagon at the default tolerance.

The second row of :ref:`Figure 19 <aa-fig-moire>` isolates the aliased part of each output, the frequencies outside
the cell that went through the prefilter and folded at sampling. It is the difference between two near-exact
samplings: one of the prefiltered source, one of its part inside the cell used by the prefilter. With
:math:`\varepsilon = 0.5`, half of the amplitude passes on the
cell boundary and a residue is still visible. With the default :math:`\varepsilon = 0.1`, it is no longer visible. A
stricter tolerance lowers it further, roughly in proportion to :math:`\varepsilon`.

With the hexagon and the default tolerance, the aliased part is about the same as with the parallelogram, for a
smaller kernel. The two cells differ in what they keep rather than in
what they let fold. They have the same area and the spectrum of the zone plate is flat, so they keep the same amount of
signal, but not in the same place. The hexagon keeps the rings up to the same local frequency in every direction of
the source, which the grid turns into an ellipse of rings on the output (:ref:`Figure 19 <aa-fig-moire>` c). The
parallelogram keeps them further along its long diagonal, out to the corners of the output, and cuts them earlier
across it (b).

.. _aa-fig-moire:

.. figure:: /images/antialiasing/fig_moire.png
   :width: 744px
   :align: center

   Figure 19. The resampling of Figure 3 with the prefilter (quintic B-spline interpolation). (a, b, c) Outputs with the
   parallelogram at :math:`\varepsilon = 0.5` and at the default :math:`\varepsilon = 0.1`, and with the hexagon at
   :math:`\varepsilon = 0.1`. (d, e, f) Aliased part of each output, the exact sampling of the prefiltered source minus
   the exact sampling of its part inside the cell of the prefilter, contrast multiplied by 10, with its rms.

.. _aa-fig-moire_spectra:

.. figure:: /images/antialiasing/fig_moire_spectra.png
   :width: 606px
   :align: center

   Figure 20. Spectra (dB) of the three prefiltered sources of Figure 19,
   bin for bin, with the cell of each prefilter (white). The spectrum of a real image is centrally symmetric:
   the upper half holds all of it.

:ref:`Figure 20 <aa-fig-moire_spectra>` shows the spectra of the prefiltered sources with the cell: whatever lies outside the cell folds at sampling.

Interface
=========

The method is implemented by the class ``ReciprocalCellFilter`` of the module
``gridr.core.filter.antialiasing.reciprocal_cell``. It only holds the parameters below. Its method ``design`` applies
the steps of this page to a transition matrix: it computes the cell, decides whether to filter, and synthesizes the
kernel. On a grid, ``resolve_antialiasing_filter`` estimates the transition matrix (:doc:`transition_matrix_estimation`)
and calls ``design``. The :doc:`API reference </api_python/core/filter/antialiasing/reciprocal_cell>` documents the
returned kernel, the diagnostics and the warnings.

.. the :doc:`tutorial </_notebooks/generated/antialiasing_001_filter_construction>` shows the filter on a grid and the effect of each parameter.


+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| Field of ``ReciprocalCellFilter``   | Default           | Meaning                                                                                                      |
+=====================================+===================+==============================================================================================================+
| ``cutoff_tanh_slope``               | 10.0              | steepness of the transition                                                                                  |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``aliasing_tolerance``              | 0.1               | :math:`\varepsilon`, template value on the cell boundary, in :math:`(0, 0.5]`                                |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``isotropic_frame``                 | ``"destination"`` | metric of the Voronoi cell: ``"destination"``, ``"source"`` or a :math:`2\times2` matrix :math:`G`           |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``kernel_accuracy``                 | 0.01              | target sup error between the kernel response and the template                                                |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``max_kernel_size``                 | ``None``          | odd bound on the kernel size; ``None`` sets it to twice the size estimate                                    |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``kernel_size``                     | ``None``          | forced odd size; ``None`` selects the smallest accurate size                                                 |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``experimental_nyquist_gates``      | ``False``         | experimental: gates on the Nyquist square where the template jumps (:ref:`aa-sec-periodicity`)               |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
| ``experimental_progressive_gates``  | ``False``         | experimental: tolerance of each facet ramped by how far it cuts the Nyquist square (:ref:`aa-sec-active`)    |
+-------------------------------------+-------------------+--------------------------------------------------------------------------------------------------------------+
