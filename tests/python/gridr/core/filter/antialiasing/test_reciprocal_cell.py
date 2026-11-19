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
Tests for the gridr.core.filter.antialiasing package and its reciprocal_cell module.

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/filter/antialiasing/test_reciprocal_cell.py
"""  # noqa: E501

import warnings

import numpy as np
import pytest

from gridr.core.filter.antialiasing import AntialiasingFilter, KernelFilter
from gridr.core.filter.antialiasing import reciprocal_cell as rc
from gridr.core.filter.lattice import voronoi_cell_facets

def rot(deg: float) -> np.ndarray:
    """Rotation matrix of ``deg`` degrees."""
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# rotated and sheared grid; 3 * SHEAR is the running example of the doc
SHEAR = rot(20) @ np.array([[1.0, 0.5], [0.0, 1.0]])


class TestCell:
    """Cell of the reciprocal lattice and decision to filter."""

    def test_destination_is_parallelogram(self):
        """In the destination metric the facets are w1 and w2."""
        w = 2.0 * SHEAR
        facets = rc.cell_facets(w, "destination")
        assert len(facets) == 2
        got = sorted(np.abs(np.array(facets) @ np.linalg.inv(w).T).round(9).tolist())
        assert np.allclose(got, [[0.0, 1.0], [1.0, 0.0]])

    def test_source_is_hexagon_when_oblique(self):
        """In the source metric, a sheared grid gives a hexagon."""
        assert len(rc.cell_facets(2.0 * SHEAR, "source")) == 3
        assert len(rc.cell_facets(2.0 * rot(20), "source")) == 2

    def test_explicit_metric(self):
        """An explicit metric is passed as is to the Voronoi construction."""
        w = 2.0 * SHEAR
        d = rot(40) @ np.diag([1.0, 2.5])
        g = d @ d.T
        expected = voronoi_cell_facets(rc.reciprocal_basis(w), g)
        for a, b in zip(rc.cell_facets(w, g), expected, strict=True):
            assert np.allclose(a, b)

    @pytest.mark.parametrize(
        "w, expected",
        [
            (0.5 * np.eye(2), False),
            (np.eye(2), False),
            (np.diag([0.7, 1.0]), False),
            (np.array([[0.72, 0.72], [0.18, -0.18]]), False),
            (2.0 * np.eye(2), True),
            (np.diag([1.0, 2.0]), True),
            (rot(30), True),
        ],
    )
    def test_filter_required(self, w, expected):
        """A filter is needed if a Nyquist corner leaves the cell."""
        assert rc.filter_required(rc.cell_facets(w)) is expected


class TestTemplate:
    """Gate and frequency template."""

    @pytest.mark.parametrize("slope", [5.0, 10.0, 20.0])
    @pytest.mark.parametrize("tolerance", [0.5, 0.1, 0.05, 0.01])
    def test_gate_meets_tolerance(self, slope, tolerance):
        """The gate is 1 at the origin and the tolerance on the boundary."""
        t = rc.cutoff_half_width(slope, tolerance)
        assert abs(rc.tanh_gate(np.array(0.0), slope, t) - 1.0) < 1e-12
        assert abs(rc.tanh_gate(np.array(0.5), slope, t) - tolerance) < 1e-9

    def test_half_width_at_half_tolerance(self):
        """A tolerance of 0.5 gives a half-width of 1/2."""
        assert abs(rc.cutoff_half_width(10.0, 0.5) - 0.5) < 1e-6

    @pytest.mark.parametrize(
        "slope, tolerance",
        [(10.0, 0.0), (10.0, 1.0), (10.0, -0.1), (10.0, 1.5), (1.0, 0.2)],
    )
    def test_half_width_rejects(self, slope, tolerance):
        """A tolerance out of (0, 1), or a slope too small for it, is rejected."""
        with pytest.raises(ValueError):
            rc.cutoff_half_width(slope, tolerance)

    def test_slope_too_small(self):
        """A gate too soft to fall to the tolerance on the boundary is rejected."""
        with pytest.raises(ValueError, match="too small"):
            rc.cutoff_half_width(1.0, 0.2)

    def test_template_product(self):
        """The template gates every facet and stays below the tolerance outside."""
        w = 2.0 * rot(30)
        facets = rc.cell_facets(w)
        t = rc.cutoff_half_width(10.0, 0.1)
        f = np.linspace(-0.5, 0.5, 101)
        fx, fy = np.meshgrid(f, f)
        expected = rc.tanh_gate(fx * w[0, 0] + fy * w[1, 0], 10.0, t) * rc.tanh_gate(
            fx * w[0, 1] + fy * w[1, 1], 10.0, t
        )
        h = rc.template(facets, fx, fy, 10.0, t)
        assert np.allclose(h, expected)
        outside = ~np.all([np.abs(fx * a[0] + fy * a[1]) <= 0.5 for a in facets], 0)
        assert h[outside].max() <= 0.1 + 1e-9


class TestSynthesis:
    """Kernel synthesis."""

    @pytest.mark.parametrize(
        "w, frame",
        [
            (2.0 * rot(30), "destination"),
            (2.0 * SHEAR, "source"),
            (np.diag([1.0, 3.0]), "destination"),
        ],
    )
    def test_kernel_properties(self, w, frame):
        """The kernel is odd, symmetric, of unit sum and accurate."""
        facets = rc.cell_facets(w, frame)
        kernel, info = rc.synthesize_kernel(facets, 10.0, 0.1, 0.01, 129)
        n = kernel.shape[0]
        assert n % 2 == 1 and kernel.shape == (n, n)
        assert np.allclose(kernel, kernel[::-1, ::-1])
        assert abs(kernel.sum() - 1.0) < 1e-12
        assert info["accuracy_met"] and info["response_error"] <= 0.01
        assert info["accurate_size"] == n

    def test_cell_boundary_amplitude(self):
        """On and just outside the cell boundary, the response stays within tolerance + accuracy."""
        facets = rc.cell_facets(2.0 * rot(30))
        kernel, _ = rc.synthesize_kernel(facets, 10.0, 0.1, 0.01)
        f, response = rc.kernel_response(kernel)
        fx, fy = np.meshgrid(f, f)
        # g > 0.5 in the cell
        # g = 0.5 along the cell border
        # g > 1/2 outside the cell
        g = np.max(np.abs(np.stack([fx, fy], -1) @ np.array(facets).T), axis=-1)
        # one DFT step outside the boundary, where the template is at most the tolerance
        band = (g >= 0.5) & (g < 0.5 + 2e-3)
        assert np.abs(response[band]).max() <= 0.1 + 0.01

    def test_size_bound_truncates(self):
        """Under an active bound, the kernel is the crop of the accurate kernel."""
        facets = rc.cell_facets(2.0 * rot(30))
        accurate, info = rc.synthesize_kernel(facets, 10.0, 0.1, 0.01)
        bounded, binfo = rc.synthesize_kernel(facets, 10.0, 0.1, 0.01, max_size=21)
        assert accurate.shape[0] > 21 and bounded.shape == (21, 21)
        assert not binfo["accuracy_met"]
        assert binfo["accurate_size"] == info["accurate_size"]
        c, h = accurate.shape[0] // 2, 21 // 2
        crop = accurate[c - h : c + h + 1, c - h : c + h + 1]
        assert np.allclose(bounded, crop / crop.sum())

    def test_forced_size(self):
        """A forced size is used as is; even sizes are rejected."""
        facets = rc.cell_facets(2.0 * rot(30))
        kernel, info = rc.synthesize_kernel(facets, 10.0, 0.1, size=15)
        assert kernel.shape == (15, 15) and info["accurate_size"] is None
        with pytest.raises(ValueError):
            rc.synthesize_kernel(facets, 10.0, 0.1, size=16)
        with pytest.raises(ValueError):
            rc.synthesize_kernel(facets, 10.0, 0.1, max_size=64)

    def test_forced_size_beyond_the_bound(self):
        """A forced size larger than the automatic bound fits in the fine grid."""
        # first run with automatic bound
        facets = rc.cell_facets(2.0 * np.eye(2))
        _, info = rc.synthesize_kernel(facets, 10.0, 0.1)
        # force twice the automatic bound
        n = 2 * info["size_bound"] + 1
        kernel, info = rc.synthesize_kernel(facets, 10.0, 0.1, size=n)
        assert kernel.shape == (n, n)
        assert info["n_fine"] >= 2 * n + 1

    def test_error_domain(self):
        """A corner of the template on the Nyquist border does not saturate the size.

        With rows decimated by 2 and a tolerance of 0.5, the gate of the facet
        a = (1, 0) is 0.5 on both edges f_x = +-1/2: the template, repeated with
        period 1, has a corner there that no small kernel follows. The accuracy is
        measured on ||f||_inf <= 0.45 only: it is met there, not on the whole square.
        """
        facets = rc.cell_facets(np.diag([1.0, 2.0]))
        kernel, info = rc.synthesize_kernel(facets, 10.0, 0.5, 0.01, 65)
        # measured on ||f||_inf <= 0.45: met, the size is lesser than 65
        assert info["accuracy_met"] and kernel.shape[0] < 65
        # measured on the whole square, corner included: missed
        f, response = rc.kernel_response(kernel)
        fx, fy = np.meshgrid(f, f)
        full = np.abs(response - rc.template(facets, fx, fy, 10.0, info["t"])).max()
        assert full > 0.01

    def test_fine_grid_does_not_wrap(self):
        """
        The fine grid is large enough for the kernel not to wrap: a much larger one
        gives the same kernel.
        """
        facets = rc.cell_facets(3.0 * SHEAR)
        kernel, info = rc.synthesize_kernel(facets, 10.0, 0.1, 0.01)
        large = rc.crop_kernel(
            rc.fine_kernel(facets, 10.0, info["t"], 8 * info["n_fine"] + 1), len(kernel)
        )
        assert np.abs(kernel - large).max() < 1e-9


class TestSpecification:
    """Specification, kernel filter and design."""

    def test_spec_validation(self):
        """Out-of-domain parameters are rejected."""
        for kwargs in (
            {"aliasing_tolerance": 0.0},
            {"aliasing_tolerance": 0.6},
            {"cutoff_tanh_slope": 1.0, "aliasing_tolerance": 0.001},
            {"kernel_size": 32},
            {"max_kernel_size": 64},
            {"kernel_accuracy": 0.0},
            {"isotropic_frame": "ground"},
            {"isotropic_frame": np.array([[1.0, 2.0], [0.0, 1.0]])},
        ):
            with pytest.raises(ValueError):
                rc.ReciprocalCellFilter(**kwargs)

    def test_kernel_filter(self):
        """A kernel filter checks its kernel and gives its margins."""
        k = KernelFilter(np.ones((3, 5)) / 15)
        assert k.margins() == (1, 1, 2, 2)
        with pytest.raises(ValueError):
            KernelFilter(np.ones((4, 3)))
        with pytest.raises(ValueError):
            KernelFilter(np.full((3, 3), np.nan))

    def test_protocol(self):
        """The specification implements the AntialiasingFilter protocol."""
        assert isinstance(rc.ReciprocalCellFilter(), AntialiasingFilter)
        assert not isinstance(KernelFilter(np.ones((3, 3)) / 9), AntialiasingFilter)

    def test_zoom_designs_none(self):
        """A zoom needs no filter."""
        kf, diag = rc.ReciprocalCellFilter().design(0.5 * np.eye(2))
        assert kf is None and diag.filter_required is False
        assert diag.kernel_size is None and diag.template_tolerances is None

    def test_design(self):
        """A decimation is designed into a kernel with its diagnostics."""
        w = 2.0 * rot(30)
        kf, diag = rc.ReciprocalCellFilter(aliasing_tolerance=0.1).design(w)
        assert diag.filter_required
        assert diag.kernel_size == kf.kernel.shape[0]
        assert diag.accuracy_met and diag.response_error <= 0.01
        assert len(diag.template_facets) == 2
        assert np.allclose(diag.template_tolerances, [0.1, 0.1])
        kernel, _ = rc.synthesize_kernel(rc.cell_facets(w), 10.0, 0.1)
        assert np.array_equal(kf.kernel, kernel)

    def test_both_options(self):
        """Both experimental options together give a valid design."""
        spec = rc.ReciprocalCellFilter(
            experimental_nyquist_gates=True, experimental_progressive_gates=True
        )
        # worst case scenario : unit scale with a 30 deg rotation.
        kf, diag = spec.design(rot(30))
        assert len(diag.template_facets) == 4
        assert diag.accuracy_met and diag.kernel_size == kf.kernel.shape[0]


class TestNyquistGates:
    """Experimental Nyquist gates of the template."""

    def test_off_by_default(self):
        """By default the template gates the facets of the cell only."""
        rng = np.random.default_rng(5)
        t = rc.cutoff_half_width(10.0, 0.01)
        f = np.linspace(-0.5, 0.5, 101)
        fx, fy = np.meshgrid(f, f)
        for _ in range(20):
            w = rot(rng.uniform(0, 90)) @ np.diag(rng.uniform(0.6, 2.5, 2))
            facets = rc.cell_facets(w)
            assert len(rc.template_gates(facets, 10.0, t)[0]) == len(facets)
            assert np.allclose(
                rc.template(facets, fx, fy, 10.0, t),
                rc.cell_response(facets, fx, fy, 10.0, t),
            )

    def test_periodic_for_random_grids(self):
        """With the gates added where needed, every template is periodic."""
        rng = np.random.default_rng(8)
        t = rc.cutoff_half_width(10.0, 0.01)
        options = rc.TemplateOptions(nyquist_gates=True)
        for _ in range(50):
            w = rot(rng.uniform(0, 90)) @ np.diag(rng.uniform(0.6, 2.0, 2))
            gates, ts = rc.template_gates(rc.cell_facets(w), 10.0, t, options)
            for axis in (0, 1):
                assert rc.border_mismatch(gates, axis, 10.0, ts) <= 0.01 + 1e-9

    @pytest.mark.parametrize("tolerance", [0.01, 0.5])
    @pytest.mark.parametrize("deg", [10, 30])
    def test_nyquist_gates_reduce_jump(self, deg, tolerance):
        """The Nyquist gates divide the jump across the border by at least 1/tolerance."""
        facets = rc.cell_facets(rot(deg))
        t = rc.cutoff_half_width(10.0, tolerance)
        options = rc.TemplateOptions(nyquist_gates=True)
        gates, _ = rc.template_gates(facets, 10.0, t, options)
        assert len(gates) == 4
        for axis in (0, 1):
            before = rc.border_mismatch(facets, axis, 10.0, t)
            after = rc.border_mismatch(gates, axis, 10.0, t)
            assert after <= tolerance * before + 1e-9

    def test_no_gate_on_a_slope_change(self):
        """At 45 degrees the template takes the same values on opposite edges."""
        facets = rc.cell_facets(rot(45))
        options = rc.TemplateOptions(nyquist_gates=True)
        for tolerance in (0.01, 0.1, 0.5):
            t = rc.cutoff_half_width(10.0, tolerance)
            assert len(rc.template_gates(facets, 10.0, t, options)[0]) == 2

    def test_design(self):
        """The option reaches the synthesis and the diagnostics."""
        with pytest.warns(RuntimeWarning, match="jumps"):
            _, plain = rc.ReciprocalCellFilter().design(rot(30))
        kf, diag = rc.ReciprocalCellFilter(experimental_nyquist_gates=True).design(
            rot(30)
        )
        assert len(plain.template_facets) == 2 and len(diag.template_facets) == 4
        assert diag.accuracy_met and diag.kernel_size == kf.kernel.shape[0]
        assert diag.kernel_size < plain.kernel_size


class TestProgressiveGates:
    """Experimental gates whose tolerance depends continuously on the facet."""

    OPTIONS = rc.TemplateOptions(progressive_gates=True)

    def test_off_by_default(self):
        """By default every facet of the cell is gated with the tolerance."""
        facets = rc.cell_facets(np.diag([1.0, 2.0]))
        gates, ts = rc.template_gates(facets, 10.0, 0.4)
        assert len(gates) == 2 and ts == [0.4, 0.4]

    def test_ramp_width(self):
        """The axis width c / d, and the symmetries of the Nyquist square."""
        # c = ln(1 / (tol (1 - tol))) - 1 and d = 2 slope
        mu0 = (np.log(1 / (0.1 * 0.9)) - 1) / 20.0
        assert abs(rc.ramp_width(np.array([1.0, 0.0]), 10.0, 0.1) - mu0) < 1e-12
        # sign changes and axis swap leave the Nyquist square invariant
        ref = rc.ramp_width(np.array([0.9, 0.13]), 10.0, 0.1)
        for a in ((-0.9, 0.13), (0.13, -0.9), (-0.13, -0.9)):
            assert rc.ramp_width(np.array(a), 10.0, 0.1) == ref

    def test_ramp_width_moment(self):
        """The first moment of the blur in the oblique width, against a quadrature."""
        x = np.linspace(-60.0, 0.0, 60001)
        w = 9.0 * np.exp(x)
        m = np.trapezoid(-x * (w / (1 + w)) ** 2, x)
        c = np.log(1 / (0.1 * 0.9)) - 1
        oblique = (c + np.sqrt(c * c + 2 * m)) / 20.0
        assert abs(rc.ramp_width(np.array([0.6, 0.6]), 10.0, 0.1) - oblique) < 1e-7

    def test_facet_tolerance(self):
        """1 up to |a|_1 = 1, then a linear ramp down to the tolerance."""
        ramp = rc.ramp_width(np.array([1.0, 0.0]), 10.0, 0.1)
        assert rc.facet_tolerance(np.array([0.8, 0.0]), 10.0, 0.1) == 1.0
        assert rc.facet_tolerance(np.array([1.0, 0.0]), 10.0, 0.1) == 1.0
        mid = rc.facet_tolerance(np.array([1.0 + ramp / 2, 0.0]), 10.0, 0.1)
        assert abs(mid - 0.55) < 1e-12
        assert rc.facet_tolerance(np.array([1.0 + 1.001 * ramp, 0.0]), 10.0, 0.1) == 0.1
        assert rc.facet_tolerance(np.array([0.6, 0.6]), 10.0, 0.1) == 0.1
        # a quarter of the ramp, at other slopes and tolerances
        for slope, tol in ((5.0, 0.01), (20.0, 0.3)):
            mu = rc.ramp_width(np.array([1.0, 0.0]), slope, tol)
            tau = rc.facet_tolerance(np.array([1.0 + mu / 4, 0.0]), slope, tol)
            assert abs(tau - (1 - (1 - tol) / 4)) < 1e-12
        # sign changes and axis swap give the same tolerance
        ref = rc.facet_tolerance(np.array([1.02, -0.05]), 10.0, 0.1)
        for a in ((-1.02, 0.05), (0.05, 1.02), (-0.05, -1.02)):
            assert rc.facet_tolerance(np.array(a), 10.0, 0.1) == ref

    @pytest.mark.parametrize("tolerance", [0.6, 0.9, 0.99])
    def test_half_width_beyond_the_filter_tolerance(self, tolerance):
        """The ramp needs gate values on the boundary up to 1."""
        t = rc.cutoff_half_width(10.0, tolerance)
        assert abs(rc.tanh_gate(np.array(0.5), 10.0, t) - tolerance) < 1e-9

    def test_no_blur_along_a_kept_axis(self):
        """Columns kept or zoomed, rows decimated: no attenuation along fx."""
        t = rc.cutoff_half_width(10.0, 0.1)
        fx = np.linspace(-0.5, 0.5, 201)
        for sx in (1.0, 0.9):
            facets = rc.cell_facets(np.diag([sx, 2.0]))
            gates, _ = rc.template_gates(facets, 10.0, t, self.OPTIONS)
            assert len(gates) == 1 and np.allclose(np.abs(gates[0]), [0.0, 2.0])
            h = rc.template(facets, fx, 0.0 * fx, 10.0, t, self.OPTIONS)
            assert np.abs(h - 1.0).max() < 1e-6
        # without the option, the gate of the kept axis attenuates near Nyquist
        facets = rc.cell_facets(np.diag([1.0, 2.0]))
        assert rc.template(facets, np.array(0.45), np.array(0.0), 10.0, t) < 0.5

    def test_continuous_in_the_grid(self):
        """The template changes little for a small change of the kept axis."""
        t = rc.cutoff_half_width(10.0, 0.1)
        fx = np.linspace(-0.5, 0.5, 401)
        scales = np.linspace(0.98, 1.2, 221)
        cuts = [
            rc.template(
                rc.cell_facets(np.diag([sx, 2.0])), fx, 0 * fx, 10.0, t, self.OPTIONS
            )
            for sx in scales
        ]
        assert np.abs(np.diff(np.array(cuts), axis=0)).max() < 0.05
        # past the ramp, the default template
        facets = rc.cell_facets(np.diag([1.2, 2.0]))
        assert np.allclose(cuts[-1], rc.template(facets, fx, 0 * fx, 10.0, t))

    def test_domain_is_kept(self):
        """Dropping the facets without gate leaves the cell unchanged in the Nyquist
        square, and a filter is required exactly when a gate remains."""
        rng = np.random.default_rng(7)
        t = rc.cutoff_half_width(10.0, 0.1)
        f = np.linspace(-0.5, 0.5, 201)
        xy = np.stack(np.meshgrid(f, f), -1)
        for _ in range(50):
            shear = np.array([[1.0, rng.uniform(-0.6, 0.6)], [0.0, 1.0]])
            w = rot(rng.uniform(0, 90)) @ np.diag(rng.uniform(0.4, 3.0, 2)) @ shear
            frame = "source" if rng.random() < 0.5 else "destination"
            facets = rc.cell_facets(w, frame)
            gates, _ = rc.template_gates(facets, 10.0, t, self.OPTIONS)
            full = np.all(np.abs(xy @ np.array(facets).T) <= 0.5, -1)
            part = np.ones_like(full)
            if gates:
                part = np.all(np.abs(xy @ np.array(gates).T) <= 0.5, -1)
            assert np.array_equal(full, part)
            assert rc.filter_required(facets) == bool(gates)

    def test_tolerance_of_each_facet(self):
        """Outside the cell, the template stays below the tolerance of the facet."""
        rng = np.random.default_rng(9)
        t = rc.cutoff_half_width(10.0, 0.1)
        f = np.linspace(-0.5, 0.5, 201)
        fx, fy = np.meshgrid(f, f)
        for _ in range(60):
            w = rot(rng.uniform(0, 90)) @ np.diag(rng.uniform(0.8, 1.6, 2))
            facets = rc.cell_facets(w)
            h = rc.template(facets, fx, fy, 10.0, t, self.OPTIONS)
            bound = np.ones_like(fx)
            for a in facets:
                out = np.abs(fx * a[0] + fy * a[1]) > 0.5
                bound[out] = np.minimum(bound[out], rc.facet_tolerance(a, 10.0, 0.1))
            assert (h <= bound + 1e-9).all()

    def test_design(self):
        """The option reaches the synthesis and the diagnostics."""
        spec = rc.ReciprocalCellFilter(experimental_progressive_gates=True)
        kf, diag = spec.design(np.diag([1.0, 2.0]))
        assert len(diag.facets) == 2 and len(diag.template_facets) == 1
        assert diag.accuracy_met and kf is not None
        assert np.allclose(diag.template_tolerances, [0.1])
        _, diag = spec.design(np.diag([1.05, 2.0]))
        assert len(diag.template_facets) == 2 and diag.accuracy_met
        for a, tau in zip(diag.template_facets, diag.template_tolerances, strict=True):
            assert abs(tau - rc.facet_tolerance(a, 10.0, 0.1)) < 1e-9
        # the facet of the columns lies in the ramp
        assert 0.1 < max(diag.template_tolerances) < 1.0


class TestSize:
    """Size bound, size search and warning."""

    @pytest.mark.parametrize("w", [3.0 * SHEAR, 8.0 * np.eye(2), np.diag([3.0, 2.0])])
    def test_default_bound(self, w):
        """By default the accurate kernel fits under the bound, without warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, diag = rc.ReciprocalCellFilter().design(w)
        assert diag.accuracy_met and diag.kernel_size == diag.accurate_size
        assert diag.kernel_size < diag.size_bound and diag.border_jump < 1e-3

    def test_explicit_bound(self):
        """Under an explicit bound, the accurate size beyond it is still found."""
        w = 8.0 * np.eye(2)
        _, free = rc.ReciprocalCellFilter().design(w)
        with pytest.warns(RuntimeWarning, match="raise max_kernel_size"):
            _, bounded = rc.ReciprocalCellFilter(max_kernel_size=65).design(w)
        assert bounded.kernel_size == 65 and bounded.size_bound == 65
        assert not bounded.accuracy_met
        assert bounded.accurate_size == free.kernel_size

    @pytest.mark.parametrize(
        "rule, start, n_max, expected",
        [
            (lambda n: n >= 29, 41, 81, 29),
            (lambda n: False, 41, 81, None),
            # the bisection ends on 37 with 35 failing; 33 is found at lo - 2
            (lambda n: n >= 37 or n == 33, 41, 81, 33),
            # the bisection ends on 39 with 37 failing; 33 is found at lo - 4
            (lambda n: n >= 39 or n == 33, 45, 91, 33),
        ],
    )
    def test_search(self, rule, start, n_max, expected):
        """Smallest accurate size, also when the error rises and falls again."""
        tried = []

        def accurate(n):
            tried.append(n)
            return rule(n)

        assert rc.smallest_accurate_size(accurate, start, n_max) == expected
        # each size costs a kernel synthesis: none is tried twice
        assert len(tried) == len(set(tried))

    def test_warning_on_a_border_jump(self):
        """A 30 degree rotation at unit scale misses the accuracy and says why."""
        with pytest.warns(RuntimeWarning, match="experimental_nyquist_gates"):
            _, diag = rc.ReciprocalCellFilter().design(rot(30))
        assert not diag.accuracy_met and diag.border_jump > 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spec = rc.ReciprocalCellFilter(experimental_nyquist_gates=True)
            _, diag = spec.design(rot(30))
        assert diag.accuracy_met and diag.border_jump < 0.1

    def test_warning_without_a_bound(self):
        """With the Nyquist gates on, the warning suggests setting the bound."""
        spec = rc.ReciprocalCellFilter(
            kernel_accuracy=0.001, experimental_nyquist_gates=True
        )
        with pytest.warns(RuntimeWarning, match="set max_kernel_size"):
            _, diag = spec.design(rot(30))
        assert not diag.accuracy_met

    def test_no_warning_when_the_size_is_forced(self):
        """A size forced by the caller is reported, not warned about."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, diag = rc.ReciprocalCellFilter(kernel_size=21).design(rot(30))
        assert not diag.accuracy_met