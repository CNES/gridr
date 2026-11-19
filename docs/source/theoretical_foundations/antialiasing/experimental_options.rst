.. _aa-experimental-options:

==================================================
Experimental options of the reciprocal cell filter
==================================================

By default, the template of the :doc:`reciprocal cell filter <source_prefiltering>` is the product of the gates of all
the facets of the cell. The filter has two experimental options that change this template. The first adds gates on the
border of the Nyquist square. It is meant for oblique grids close to unit scale, where no kernel of reasonable size
follows the default template. The second weakens the gates of the facets that can fold little or nothing. It is meant
for grids that keep or zoom an axis, or are close to unit scale. Both options have only been assessed on synthetic
grids.

.. _aa-sec-periodicity:

Nyquist gates near unit scale
=============================

The template seen by a kernel
-----------------------------

A discrete kernel applied on the source grid has the response :math:`H_n` of the
:ref:`kernel synthesis <aa-sec-synthesis>`. On a pixel grid, the frequencies :math:`\boldsymbol\nu` and :math:`\boldsymbol\nu + \mathbf m`, for an integer vector
:math:`\mathbf m`, give the same samples, so the response of a kernel is periodic, of period 1 along each axis. It is
enough to know it on the Nyquist square :math:`[-\tfrac12, \tfrac12)^2`, whose opposite edges are the same
frequencies.

The template is defined on the whole plane, but a kernel can only follow its part inside the Nyquist square, repeated
with period 1. When the cell lies well inside the square, the template is close to zero along the border and the
repetition is continuous. Near unit scale on an oblique grid, the cell crosses the border of the square
(:ref:`Figure 21 <aa-fig-periodic>` a). The template then takes different values on two opposite edges, so the
repeated template jumps and the periodicity is not satisfied.
The response of a kernel is smooth and cannot follow a jump. It oscillates around it, as in the Gibbs phenomenon,
and the synthesis stops at the size bound without reaching the requested accuracy. Outside the cell the response goes
well above the tolerance (:ref:`Figure 21 <aa-fig-periodic>` b).

.. _aa-fig-periodic:

.. figure:: /images/antialiasing/fig_periodic.png
   :width: 762px
   :align: center

   Figure 21. A 30° rotation at unit scale, default tolerance :math:`\varepsilon = 0.1` and default size bound. (a) The cell (green outline)
   and the Nyquist square (dashed); in red, the parts of the cell outside the square; dash-dotted, the line of the cut.
   (b) Cut of the default template along :math:`\nu_x` at :math:`\nu_y = 0.122`, a line where the jump is large, on the plane (dotted) and repeated
   with period 1 as a discrete kernel sees it (blue), with the true response of the kernel synthesised by GridR (red);
   shaded: the Nyquist square. The repetition jumps at :math:`\nu_x = \pm 0.5`, and the kernel oscillates around the
   jump at its size bound.

With the parallelogram cell this happens sooner than with the hexagon. The parallelogram reaches further
from the origin, along its long diagonal, than the hexagon of the same grid (:ref:`Figure 22 <aa-fig-cells-nyquist>`).

.. _aa-fig-cells-nyquist:

.. figure:: /images/antialiasing/fig_cells_nyquist.png
   :width: 568px
   :align: center

   Figure 22. The running example with a decimation factor of 1.25 instead of 3. (a) The parallelogram overflows the
   Nyquist square at two corners (red). (b) The hexagon of the same grid stays inside.

Gates on the Nyquist square
---------------------------

An option of the filter adds a gate on the edges of the Nyquist square, on each axis where the template jumps. With
gates on both axes, the template :math:`H` becomes

.. math::

   H(\boldsymbol\nu) = \prod_{\mathbf a \in \text{cell}} \varphi(\boldsymbol\nu\cdot\mathbf a)\;\times\;\varphi(\nu_x)\;\varphi(\nu_y),

a smooth version of the indicator of the intersection of the cell and the Nyquist square. A sharp cut on the border
would only move the jump, so the added edges require a smooth transition, like the edges of the cell. The added gates
have the same slope and the same half-width as the gates of the cell, and equal the tolerance :math:`\varepsilon` on
the border of the square. The jump left is at most :math:`\varepsilon`
times the jump without them (:ref:`Figure 23 <aa-fig-periodic-nyquist>`). A kernel of about the same size then follows
the template and reaches the accuracy (:ref:`Figure 24 <aa-fig-nyquist>`).

.. _aa-fig-periodic-nyquist:

.. figure:: /images/antialiasing/fig_periodic_nyquist.png
   :width: 506px
   :align: center

   Figure 23. The cut of :ref:`Figure 21 <aa-fig-periodic>`, with the Nyquist gates. The jump at
   :math:`\nu_x = \pm 0.5` is ten times smaller and a :math:`21\times21` kernel follows the template.

.. _aa-fig-nyquist:

.. figure:: /images/antialiasing/fig_nyquist.png
   :width: 547px
   :align: center

   Figure 24. The same rotation, default tolerance. (a) Default template: it does not vanish on the border of the
   square. (b) Template with the Nyquist gates. (c, d) True responses of the :math:`21\times21` kernels cropped from (a)
   and (b), with their largest error on :math:`\lVert\boldsymbol\nu\rVert_\infty \le 0.45`.

What the gates restore is the continuity of the values across the border. GridR therefore adds a gate on an axis only
when the values of the template on the two opposite edges differ by more than a small threshold. For a rotation
by 45°, the cell is symmetric about both axes, the values match on opposite edges and no gate is added.

The gates have a cost. Near the border of the Nyquist square, they attenuate frequencies that lie inside the cell. The
kernel with the gates therefore keeps less of the in-cell energy than the default kernel. In exchange, it stays below the
tolerance outside the cell, where the default kernel goes far above it.

.. _aa-sec-active:

Progressive gates
=================

The :ref:`decision to filter <aa-sec-decision>` tests the facets of the cell one by one. The band
:math:`\lvert\boldsymbol\nu\cdot\mathbf a\rvert \le \tfrac12` of a facet reaches furthest over the Nyquist square at a
corner :math:`(\pm\tfrac12, \pm\tfrac12)`, where :math:`\lvert\boldsymbol\nu\cdot\mathbf a\rvert` equals
:math:`(\lvert a_x\rvert + \lvert a_y\rvert)/2`. The band therefore cuts the square only when
:math:`\lVert\mathbf a\rVert_1 = \lvert a_x\rvert + \lvert a_y\rvert > 1`; otherwise, nothing can fold across that facet.
When no facet cuts the square, there is no filter. As soon as one facet cuts it, the default template gates every
facet with :math:`\varepsilon`, including the facets whose band contains the whole square. Their gates blur
frequencies that cannot fold.

Take rows decimated by 2 and columns kept, :math:`W = \operatorname{diag}(1, 2)`. The cell is the rectangle
:math:`\lvert \nu_x\rvert \le \tfrac12,\ \lvert \nu_y\rvert \le \tfrac14`. The facet :math:`(0, 2)` of the decimated
axis cuts the square, so the cell is filtered. The facet :math:`(1, 0)` of the kept axis has
:math:`\lVert\mathbf a\rVert_1 = 1`: its band is the whole square, and the vertical edges of the cell lie on its border.
The default template still puts a gate on them. This gate falls to :math:`\varepsilon` at :math:`\nu_x = \pm\tfrac12`
and attenuates the highest frequencies of the kept axis (:ref:`Figure 25 <aa-fig-blur>` a).

An option of the filter uses the test of each facet for its own gate. Each facet gets its own tolerance :math:`\tau`,
the value of its gate on the cell boundary:

.. math::

   \tau(\mathbf a) =
   \begin{cases}
      1 & \text{if } \lVert\mathbf a\rVert_1 \le 1, \\
      1 - (1 - \varepsilon)\,\dfrac{\lVert\mathbf a\rVert_1 - 1}{\mu} & \text{if } 1 < \lVert\mathbf a\rVert_1 < 1 + \mu, \\
      \varepsilon & \text{if } \lVert\mathbf a\rVert_1 \ge 1 + \mu,
   \end{cases}

A facet with :math:`\tau = 1` gets no gate. Its band contains the Nyquist square, so dropping it does not change the
part of the cell inside the square. In the example, the facet of the kept axis has :math:`\lVert\mathbf a\rVert_1 = 1`:
the template stays at 1 up to the border along :math:`\nu_x` (:ref:`Figure 25 <aa-fig-blur>` b).

The ramp makes the template a continuous function of the facets of the cell, and therefore of :math:`W` as long as
the number of facets stays the same. With the destination metric, the cell is always a parallelogram whose facets are
the columns of :math:`W`. With another metric, the cell turns from a hexagon into a rectangle when
:math:`\mathbf e_1\cdot\mathbf e_2` vanishes, and loses two facets.

Scale the kept axis by :math:`s`. Its facet is :math:`(s, 0)` and :math:`\lVert\mathbf a\rVert_1 = s`. With a hard
threshold at 1, a scale just above 1 would bring back the whole gate, although only the frequencies between
:math:`1/(2s)` and :math:`\tfrac12` fold, and this band vanishes when :math:`s` goes to 1. With the ramp, the gate
comes back progressively and reaches the default gate at :math:`s = 1 + \mu` (:ref:`Figure 25 <aa-fig-blur>` c).

The width of the ramp depends on the facet. With :math:`d = 2\cdot\text{slope}` and
:math:`a_{\min} = \min(\lvert a_x\rvert, \lvert a_y\rvert)`,

.. math::

   \mu(\mathbf a) = \min\Big(\frac{c}{d} + a_{\min},\ \frac{c + \sqrt{c^2 + 2m}}{d}\Big), \qquad
   c = \ln\frac{1}{\varepsilon(1 - \varepsilon)} - 1, \qquad m = -\operatorname{Li}_2(-K) - \ln(1 + K), \qquad
   K = \frac{1 - \varepsilon}{\varepsilon} ,

where :math:`\operatorname{Li}_2` is the dilogarithm. At the default tolerance and slope, :math:`\mu` is 0.070 for a
facet aligned with an axis and grows to 0.185 for an oblique facet. Past :math:`1 + \mu`, on a flat spectrum, the full
gate does at least as well as no gate, with blur and folding counted equally. The formula comes from a model of one
facet on a flat spectrum, valid for a slope of about 5 or more. Its derivation will be given in a technical note in
preparation; the formula was checked numerically against the model it comes from.

Inside the ramp, the tolerance is looser than :math:`\varepsilon`. The frequencies that can then pass with more than
:math:`\varepsilon` lie in a band narrower than :math:`\mu/2` below the source Nyquist frequency, where natural images
carry little energy. Past the ramp, the template is the default one and the tolerance holds.

With the same decimation on both axes, or a rotation far from unit scale, every facet is past the ramp and the option
changes nothing. :ref:`Figure 26 <aa-fig-ramp-region>` shows where the ramp acts for a grid
:math:`W = s\,R(\theta)`, at the default tolerance and slope. It covers two families of grids. The first is a slight
decimation: a scale from 1 to 1.07 along an axis. The second is a moderate zoom combined with a marked rotation: at 45
degrees, scales from 0.71 to 0.84, where the corners of the rotated cell barely cross the Nyquist square. At unit
scale, it covers rotations up to about 12 degrees. Only the frequencies that fold get the looser tolerance: a band at
most 0.033 cycles per pixel wide below the source Nyquist frequency for an axis facet, and up to 0.11 along the normal
of an oblique facet, in the corners of the square. The diagnostics of the filter give the gate value applied to each
facet, ``template_tolerances``: it tells, for a given grid, whether the tolerance was loosened and by how much.

Below the threshold, the option only removes blur that prevents no folding. Past the ramp, it changes nothing.
Inside the ramp, it lets a little more folding through to avoid blur. The default favours the removal of folding,
which is why the option is off by default.

The option remains experimental. The width of the ramp comes from a model of one facet on a flat spectrum, and the
option has only been assessed on synthetic grids. On oblique grids the template left may jump on the border of the
Nyquist square, which calls for the Nyquist gates.

.. _aa-fig-blur:

.. figure:: /images/antialiasing/fig_blur.png
   :width: 793px
   :align: center

   Figure 25. Rows decimated by 2, default tolerance and slope. (a) Default template, columns kept. (b) Template with
   the progressive gates, columns kept. Dashed: the edges of the cell; the vertical ones lie on the border of the
   Nyquist square. (c) Template with the progressive gates along the kept axis, for scales :math:`s` of that axis
   across the ramp, and the default template at :math:`s = 1` (dashed).

.. _aa-fig-ramp-region:

.. figure:: /images/antialiasing/fig_ramp_region.png
   :width: 521px
   :align: center

   Figure 26. Grids :math:`W = s\,R(\theta)` with the progressive gates, default tolerance and slope. Colour: the
   loosest gate on the cell boundary among the facets that cut the Nyquist square. Black: every such facet has the
   default tolerance, as without the option. Grey: no filter. In between, at least one facet is inside the ramp.
   Dashed: unit scale.
