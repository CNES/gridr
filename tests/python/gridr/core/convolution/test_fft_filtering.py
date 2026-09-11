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
"""Tests for :mod:`gridr.core.convolution.fft_filtering`.

Run:
PYTHONPATH=$PWD/python:$PYTHONPATH pytest tests/python/gridr/core/convolution/test_fft_filtering.py

Three levels of confidence:

``TestPlan*``
    Pure integer geometry against hand-written expectations. No array is
    allocated, so a failure points at the arithmetic.

``TestAgainstReference``
    The whole pipeline against :mod:`scipy.ndimage`, an independent
    implementation. A systematic sign, offset or axis-order error shows up
    here and nowhere else.

``TestProperties``
    Algebraic invariants that hold whatever the kernel: linearity, translation
    by a shifted Dirac, separability, tile against whole raster.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import ndimage, signal

from gridr.core.convolution.fft_filtering import (
    BOUNDARY_MODES,
    CONVOLUTION_METHODS,
    DECIMATION_ORIGINS,
    OUTPUT_MODES,
    Zoom,
    _make_convolution_input,
    align_kernel,
    build_plan,
    decimated_size,
    decimation_offset,
    fft_array_filter,
    fft_array_filter_output_shape,
    kernel_margin,
    normalize_axes,
    normalize_zoom,
    pad_kernel_to_odd,
)

# --------------------------------------------------------------------------- #
# Fixtures and shared data
# --------------------------------------------------------------------------- #
#: Non-square and non-symmetric on purpose: a square symmetric kernel makes a
#: transposed axis mapping and a flipped convolution indistinguishable from a
#: correct result.
ASYMMETRIC_KERNEL = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
    ]
)
DIRAC_KERNEL = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
GAUSSIAN_KERNEL = (1.0 / 16.0) * np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])

#: Boundary policies mapped onto the equivalent :mod:`scipy.ndimage` mode.
PAD_TO_NDIMAGE = {
    "reflect": "mirror",
    "symmetric": "reflect",
    "edge": "nearest",
    "wrap": "grid-wrap",
    "constant": "constant",
}


def _boundary_matrix() -> list:
    """Every scalar policy plus the full ``2 ** 4`` grid of per-side policies."""
    scalars = ["none", "reflect"]
    sides = ["none", "reflect"]
    pairs = [
        ((top, bottom), (left, right))
        for top, bottom, left, right in itertools.product(sides, repeat=4)
    ]
    return scalars + pairs


def _boundary_id(boundary) -> str:
    if isinstance(boundary, str):
        return boundary
    return "-".join(side[0] for pair in boundary for side in pair)


BOUNDARY_MATRIX = _boundary_matrix()
BOUNDARY_IDS = [_boundary_id(b) for b in BOUNDARY_MATRIX]


@pytest.fixture(name="raster")
def fixture_raster() -> np.ndarray:
    """A deterministic pseudo-random raster."""
    return np.random.default_rng(20250903).standard_normal((50, 60))


@pytest.fixture(name="stack")
def fixture_stack() -> np.ndarray:
    """A 3-D stack, to exercise the pass-through of non-target axes."""
    return np.random.default_rng(7).standard_normal((3, 40, 45))


def assert_array(actual, expected, *, dtype=None, rtol=1e-10, atol=0.0, err_msg=""):
    """Assert shape, dtype and values, with no dependency on the NumPy version."""
    actual = np.asanyarray(actual)
    expected = np.asanyarray(expected)
    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}. {err_msg}"
    if dtype is not None:
        assert actual.dtype == np.dtype(dtype), f"dtype {actual.dtype} != {dtype}. {err_msg}"
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #
class TestNormalizeZoom:
    """Zoom parsing and simplification."""

    @pytest.mark.parametrize(
        ("zoom", "expected"),
        [
            (1, Zoom(1, 1)),
            (2, Zoom(2, 1)),
            (np.int64(2), Zoom(2, 1)),
            ((1, 1), Zoom(1, 1)),
            ((1, 2), Zoom(1, 2)),
            ((3, 5), Zoom(3, 5)),
            ((2, 6), Zoom(1, 3)),
            ([2, 6], Zoom(1, 3)),
            (np.array([2, 6]), Zoom(1, 3)),
        ],
    )
    def test_accepts_and_simplifies(self, zoom, expected):
        assert normalize_zoom(zoom) == expected

    @pytest.mark.parametrize("zoom", [0, -1, (1, 0), (1, -1), (0, 1), (-2, -4)])
    def test_non_positive_is_value_error(self, zoom):
        with pytest.raises(ValueError):
            normalize_zoom(zoom)

    @pytest.mark.parametrize("zoom", [1.0, True, "2", (1.5, 2), (True, 1), None])
    def test_non_integral_is_type_error(self, zoom):
        """A wrong type raises ``TypeError``, a wrong value ``ValueError``.

        Booleans are integers to Python, so ``zoom=True`` would otherwise be
        read as ``1``.
        """
        with pytest.raises(TypeError):
            normalize_zoom(zoom)

    @pytest.mark.parametrize("zoom", [(1,), (1, 2, 3)])
    def test_wrong_arity_is_value_error(self, zoom):
        with pytest.raises(ValueError):
            normalize_zoom(zoom)

    @pytest.mark.parametrize(
        ("zoom", "supported"),
        [((1, 1), True), ((1, 2), True), ((1, 50), True), ((2, 1), False), ((3, 5), False)],
    )
    def test_is_supported(self, zoom, supported):
        assert normalize_zoom(zoom).is_supported is supported


class TestBoundaryModes:
    """The vocabulary itself, independent of any filtering."""

    @pytest.mark.parametrize("policy", [m for m in BOUNDARY_MODES if m != "none"])
    def test_every_policy_is_a_numpy_pad_mode(self, policy):
        assert np.pad(np.arange(5.0), (2, 2), mode=policy).shape == (9,)

    def test_none_is_the_only_one_without_a_numpy_equivalent(self):
        with pytest.raises(ValueError, match="mode"):
            np.pad(np.arange(5.0), (2, 2), mode="none")

    @pytest.mark.parametrize("spelling", [None, "none"])
    def test_none_and_None_are_the_same_policy(self, raster, spelling):
        assert_array(
            fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary=spelling).data,
            fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary="none").data,
            rtol=0,
            atol=0,
        )

    def test_an_unknown_mode_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="unknown boundary mode"):
            fft_array_filter(np.zeros((5, 5)), np.ones((3, 3)), boundary="mirror")

    @pytest.mark.parametrize(
        ("option", "value"),
        [("out_mode", "SAME"), ("method", "overlap"), ("decimation", "middle")],
    )
    def test_every_vocabulary_rejects_a_near_miss(self, option, value):
        with pytest.raises(ValueError, match=f"unknown {option}"):
            build_plan((10, 10), DIRAC_KERNEL, **{option: value})


class TestNormalizeAxes:
    """Axis specification handling."""

    @pytest.mark.parametrize(
        ("axes", "ndim", "expected"),
        [
            (None, 3, (0, 1, 2)),
            (0, 2, (0,)),
            (-1, 3, (2,)),
            (-3, 3, (0,)),
            ((1, 2), 3, (1, 2)),
            ((1, -1), 4, (1, 3)),
            ([2, 0], 3, (0, 2)),  # axes is a set: the result is sorted
            (np.array([1]), 2, (1,)),
        ],
    )
    def test_normalizes(self, axes, ndim, expected):
        assert normalize_axes(axes, ndim) == expected

    @pytest.mark.parametrize(("axes", "ndim"), [(2, 2), (-3, 2), ((0, 3), 3), ((0, 0), 2)])
    def test_out_of_bounds_or_repeated(self, axes, ndim):
        with pytest.raises(ValueError):
            normalize_axes(axes, ndim)

    def test_negative_ndim_is_rejected(self):
        with pytest.raises(ValueError, match="ndim must be >= 0"):
            normalize_axes(0, -1)

    @pytest.mark.parametrize("ragged", [[[1, 2], [3]], [[0], [1, 2, 3]]])
    def test_ragged_input_is_a_type_error(self, ragged):
        """A ragged sequence makes ``numpy.ndim`` raise; it must not leak out."""
        with pytest.raises(TypeError):
            normalize_axes(ragged, 3)
        with pytest.raises(TypeError):
            normalize_zoom(ragged)


class TestDecimation:
    """Decimation offset and resulting length."""

    @pytest.mark.parametrize("q", range(1, 12))
    def test_centered_offset_is_block_centre(self, q):
        """The centre of a block of ``q`` samples, lower one when ``q`` is even."""
        assert decimation_offset(q, "centered") == (q - 1) // 2
        assert decimation_offset(q, "leading") == 0

    @pytest.mark.parametrize(("q", "expected"), [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (8, 3)])
    def test_centered_offset_reference_values(self, q, expected):
        assert decimation_offset(q) == expected

    def test_offset_rejects_non_positive(self):
        with pytest.raises(ValueError):
            decimation_offset(0)

    @pytest.mark.parametrize("size", range(0, 25))
    @pytest.mark.parametrize("q", range(1, 8))
    def test_size_matches_slicing(self, size, q):
        sequence = list(range(size))
        for offset in range(0, size + 3):
            assert decimated_size(size, q, offset) == len(sequence[offset::q])

    @pytest.mark.parametrize(
        ("size", "q", "offset"), [(-1, 1, 0), (5, 0, 0), (5, -1, 0), (5, 1, -1)]
    )
    def test_invalid_domain(self, size, q, offset):
        with pytest.raises(ValueError):
            decimated_size(size, q, offset)


class TestKernelPreparation:
    """Kernel alignment, odd-size padding and margins."""

    @pytest.mark.parametrize(
        ("kernel_shape", "ndim", "axes", "expected"),
        [
            ((3, 3), 2, (0, 1), (3, 3)),
            ((3, 5), 2, (0, 1), (3, 5)),
            ((3, 5), 3, (1, 2), (1, 3, 5)),
            ((3, 5), 3, (2, 1), (1, 3, 5)),  # same as (1, 2): axes is a set
            ((5,), 2, (1,), (1, 5)),
            ((5,), 3, (0,), (5, 1, 1)),
        ],
    )
    def test_align_shapes(self, kernel_shape, ndim, axes, expected):
        kernel = np.arange(float(np.prod(kernel_shape))).reshape(kernel_shape)
        assert align_kernel(kernel, ndim, axes).shape == expected

    def test_axis_order_does_not_change_the_mapping(self):
        kernel = np.arange(15.0).reshape(3, 5)
        increasing = align_kernel(kernel, 3, normalize_axes((1, 2), 3))
        reversed_spelling = align_kernel(kernel, 3, normalize_axes((2, 1), 3))
        assert_array(increasing, reversed_spelling)
        assert_array(increasing.reshape(3, 5), kernel)

    def test_axis_order_does_not_change_the_result(self, raster):
        """The same holds end to end, consistently with scipy.signal."""
        stack = raster[np.newaxis, ...]
        forward = fft_array_filter(stack, ASYMMETRIC_KERNEL, axes=(1, 2)).data
        backward = fft_array_filter(stack, ASYMMETRIC_KERNEL, axes=(2, 1)).data
        assert_array(forward, backward, rtol=0, atol=0)

    def test_align_full_rank_is_left_alone(self):
        kernel = np.ones((1, 3, 5))
        assert align_kernel(kernel, 3, (1, 2)) is kernel

    @pytest.mark.parametrize(
        ("kernel_shape", "ndim", "axes"),
        [
            ((3, 3), 2, (0,)),  # rank matches neither reading
            ((3, 3, 3), 4, (1, 2)),
            ((2, 3, 3), 3, (1, 2)),  # not singleton outside the target axes
        ],
    )
    def test_align_rejects_mismatch(self, kernel_shape, ndim, axes):
        with pytest.raises(ValueError):
            align_kernel(np.zeros(kernel_shape), ndim, axes)

    @pytest.mark.parametrize(
        ("shape", "axes", "expected"),
        [
            ((5, 5), (0, 1), (5, 5)),
            ((4, 4), (0, 1), (5, 5)),
            ((4, 4), (0,), (5, 4)),
            ((4, 7), (1,), (4, 7)),
        ],
    )
    def test_pad_to_odd(self, shape, axes, expected):
        kernel = np.ones(shape)
        padded = pad_kernel_to_odd(kernel, axes)
        assert padded.shape == expected
        assert_array(padded[tuple(slice(0, size) for size in shape)], kernel)
        assert padded.sum() == pytest.approx(kernel.sum()), "zero padding must not add energy"

    @pytest.mark.parametrize("helper", [pad_kernel_to_odd, kernel_margin])
    def test_helpers_normalize_axes_themselves(self, helper):
        kernel = np.ones((3, 5))
        assert helper(kernel) is not None
        assert np.array_equal(np.asarray(helper(kernel, None)), np.asarray(helper(kernel, (0, 1))))

    @pytest.mark.parametrize(
        ("shape", "axes", "expected"),
        [
            ((3, 3), (0, 1), (1, 1)),
            ((3, 5), (0, 1), (1, 2)),
            ((3, 5), (1,), (0, 2)),
            ((1, 7, 9), (1, 2), (0, 3, 4)),
        ],
    )
    def test_margin(self, shape, axes, expected):
        assert kernel_margin(np.zeros(shape), axes) == expected


# --------------------------------------------------------------------------- #
# Plan geometry — no data involved
# --------------------------------------------------------------------------- #
class TestPlanGeometry:
    """The plan is a pure function of shapes; assert it as such."""

    def test_same_mode_without_margin(self):
        plan = build_plan((50, 60), DIRAC_KERNEL)
        assert plan.output_shape == (50, 60)
        assert plan.source == (slice(0, 50), slice(0, 60))
        assert not plan.needs_padding
        assert plan.window.tolist() == [[1, 50], [1, 60]]  # window applied to the full output

    def test_interior_window_reads_real_neighbours(self):
        """A window away from the edges needs no synthetic samples at all."""
        plan = build_plan((50, 60), DIRAC_KERNEL, ((10, 20), (30, 40)), boundary="reflect")
        assert plan.source == (slice(9, 22), slice(29, 42))
        assert not plan.needs_padding
        assert plan.output_shape == (11, 11)

    @pytest.mark.parametrize("boundary", list(BOUNDARY_MODES))
    def test_no_policy_ever_applies_away_from_the_array_edge(self, boundary):
        """A boundary condition describes the edge of the array, not a seam.

        When the margins all exist in the array, every policy including
        ``"none"`` reads them and synthesises nothing, so the plans match.
        """
        plan = build_plan((50, 60), ASYMMETRIC_KERNEL, ((10, 20), (30, 40)), boundary=boundary)
        assert not plan.needs_padding
        assert plan.per_axis[0].source == slice(9, 22)
        assert plan.per_axis[1].source == slice(28, 43)
        assert plan.output_shape == (11, 11)

    @pytest.mark.parametrize("boundary", list(BOUNDARY_MODES))
    def test_interior_window_gives_the_same_samples_whatever_the_policy(self, raster, boundary):
        """The same statement, checked on the values rather than on the plan."""
        reference = fft_array_filter(
            raster, ASYMMETRIC_KERNEL, ((10, 20), (30, 40)), boundary="symmetric"
        ).data
        produced = fft_array_filter(
            raster, ASYMMETRIC_KERNEL, ((10, 20), (30, 40)), boundary=boundary
        ).data
        assert_array(produced, reference, rtol=0, atol=0)

    def test_none_still_synthesises_nothing_at_the_array_edge(self):
        """``NONE`` reads what exists and adds nothing where nothing exists."""
        plan = build_plan((50, 60), ASYMMETRIC_KERNEL, boundary="none")
        assert not plan.needs_padding
        assert plan.per_axis[0].source == slice(0, 50)
        assert plan.per_axis[0].origin == ASYMMETRIC_KERNEL.shape[0] // 2

    def test_edge_window_is_padded_on_the_touching_side_only(self):
        plan = build_plan((50, 60), DIRAC_KERNEL, ((0, 20), (30, 40)), boundary="reflect")
        assert plan.per_axis[0].pad_width == (1, 0)
        assert plan.per_axis[1].pad_width == (0, 0)

    @pytest.mark.parametrize(
        ("out_mode", "expected"),
        [
            ("same", (50, 60)),
            ("full", (52, 62)),
            ("valid", (48, 58)),
        ],
    )
    def test_output_modes(self, out_mode, expected):
        assert build_plan((50, 60), DIRAC_KERNEL, out_mode=out_mode).output_shape == expected

    def test_non_target_axes_are_passed_through(self):
        plan = build_plan((3, 40, 45), DIRAC_KERNEL, axes=(1, 2))
        assert plan.output_shape == (3, 40, 45)
        assert plan.per_axis[0].pad_width == (0, 0)

    def test_default_window_with_axes_subset(self):
        plan = build_plan((3, 40, 45), DIRAC_KERNEL, None, axes=(1, 2))
        assert plan.window.dtype == np.int64
        assert plan.window.tolist() == [[0, 2], [1, 40], [1, 45]]

    @pytest.mark.parametrize("q", [2, 3, 5, 50])
    @pytest.mark.parametrize("origin", list(DECIMATION_ORIGINS))
    def test_decimated_shape_matches_manual_slicing(self, q, origin):
        undecimated = build_plan((50, 60), DIRAC_KERNEL, ((10, 20), (30, 42))).output_shape
        decimated = build_plan(
            (50, 60), DIRAC_KERNEL, ((10, 20), (30, 42)), zoom=(1, q), decimation=origin
        ).output_shape
        offset = decimation_offset(q, origin)
        expected = tuple(len(range(size)[offset::q]) for size in undecimated)
        assert decimated == expected

    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"zoom": (2, 1)}, ValueError),  # upsampling not implemented
            ({"zoom": (1, 0)}, ValueError),
            ({"zoom": 1.5}, TypeError),
            ({"axes": (0, 0)}, ValueError),
            ({"axes": 5}, ValueError),
            ({"win": ((0, 100), (0, 10))}, ValueError),  # window outside the array
            ({"win": ((0, 10),)}, ValueError),  # wrong window rank
            ({"win": ((None, None), (0, 10))}, TypeError),  # None is not an index
            ({"method": "FFT"}, ValueError),
            ({"decimation": True}, TypeError),
        ],
    )
    def test_invalid_options(self, kwargs, exc):
        with pytest.raises(exc):
            build_plan((50, 60), DIRAC_KERNEL, **kwargs)

    @pytest.mark.parametrize(
        "boundary",
        [
            (("wrap", "edge"), ("none", "none")),
            (("reflect", "none"), ("constant", "none")),
        ],
        ids=["same-axis", "across-axes"],
    )
    def test_mixed_policies_are_refused(self, boundary):
        with pytest.raises(ValueError, match="single padding policy"):
            build_plan((50, 60), DIRAC_KERNEL, boundary=boundary)

    def test_one_policy_on_a_subset_of_sides_is_fine(self):
        boundary = (
            ("reflect", "none"),
            ("none", "reflect"),
        )
        plan = build_plan((50, 60), DIRAC_KERNEL, boundary=boundary)
        assert plan.pad_mode == "reflect"
        assert plan.pad_width == ((1, 0), (0, 1))
        assert plan.needs_padding

    def test_pad_mode_is_none_when_nothing_is_synthesised(self):
        plan = build_plan((50, 60), DIRAC_KERNEL, ((10, 20), (30, 40)), boundary="reflect")
        assert not plan.needs_padding
        assert plan.pad_mode == "none"
        assert plan.conv_shape == (13, 13)
        assert plan.src_win == (slice(0, 13), slice(0, 13))

    @pytest.mark.parametrize("boundary", BOUNDARY_MATRIX, ids=BOUNDARY_IDS)
    @pytest.mark.parametrize("win", [None, ((0, 20), (30, 42)), ((10, 20), (30, 42))])
    def test_pad_mode_and_needs_padding_agree(self, boundary, win):
        """The invariant that makes ``pad_mode`` readable without cross-checking."""
        plan = build_plan((50, 60), ASYMMETRIC_KERNEL, win, boundary=boundary)
        assert (plan.pad_mode == "none") == (not plan.needs_padding)

    @pytest.mark.parametrize(
        ("boundary", "exc", "match"),
        [
            (
                (("none", "none"),) * 3,
                ValueError,
                "3 pairs for a 2-d array",
            ),
            (
                (("none", "none", "none"), (0, 0)),
                ValueError,
                "before, after",
            ),
            (((1, 2), (3, 4)), TypeError, "must be a string"),
            (42, TypeError, "mode string, None, or a sequence"),
        ],
        ids=["too-many-pairs", "not-a-pair", "not-a-policy", "not-a-sequence"],
    )
    def test_malformed_boundary(self, boundary, exc, match):
        with pytest.raises(exc, match=match):
            build_plan((50, 60), DIRAC_KERNEL, boundary=boundary)

    def test_wrap_needs_the_window_to_span_the_axis(self):
        with pytest.raises(ValueError, match="'wrap' boundary needs the window to span axis 1"):
            build_plan((50, 60), ASYMMETRIC_KERNEL, ((0, 49), (0, 40)), boundary="wrap")

    def test_wrap_is_fine_when_the_window_spans_the_axis(self):
        plan = build_plan((50, 60), ASYMMETRIC_KERNEL, ((10, 20), (0, 59)), boundary="wrap")
        assert plan.pad_mode == "wrap"
        assert plan.per_axis[1].pad_width == (2, 2)

    def test_wrap_on_an_interior_window_never_pads_so_never_raises(self):
        """No margin is synthesised, so the policy simply does not apply."""
        plan = build_plan((50, 60), ASYMMETRIC_KERNEL, ((10, 20), (30, 40)), boundary="wrap")
        assert not plan.needs_padding
        assert plan.pad_mode == "none"

    def test_valid_mode_needs_room(self):
        with pytest.raises(ValueError, match=r"the 'valid' output is empty"):
            build_plan((3, 3), np.ones((5, 5)), out_mode="valid")
            
    @pytest.mark.parametrize("boundary", list(BOUNDARY_MODES))
    def test_valid_shape_does_not_depend_on_the_boundary(self, boundary):
        plan = build_plan((50, 60), ASYMMETRIC_KERNEL, boundary=boundary, out_mode="valid")
        margins = kernel_margin(ASYMMETRIC_KERNEL)
        assert plan.output_shape == (50 - 2 * margins[0], 60 - 2 * margins[1])


class TestPredictedShape:
    """``fft_array_filter_output_shape`` must agree with the produced array."""

    @pytest.mark.parametrize("boundary", ["none", "symmetric"])
    @pytest.mark.parametrize("out_mode", list(OUTPUT_MODES))
    @pytest.mark.parametrize("zoom", [1, (1, 3), (1, 50)])
    @pytest.mark.parametrize("origin", list(DECIMATION_ORIGINS))
    def test_prediction_matches_production(self, raster, boundary, out_mode, zoom, origin):
        kwargs = {
            "win": ((10, 20), (30, 42)),
            "boundary": boundary,
            "out_mode": out_mode,
            "zoom": zoom,
            "decimation": origin,
        }
        predicted = fft_array_filter_output_shape(raster.shape, ASYMMETRIC_KERNEL, **kwargs)
        produced = fft_array_filter(raster, ASYMMETRIC_KERNEL, **kwargs).data
        assert predicted == produced.shape

    def test_accepts_a_bare_shape(self):
        assert fft_array_filter_output_shape((50, 60), DIRAC_KERNEL, zoom=(1, 5)) == (10, 12)


# --------------------------------------------------------------------------- #
# Comparison with an independent implementation
# --------------------------------------------------------------------------- #
class TestAgainstReference:
    """Compare the `fft_array_filter` with :mod:`scipy.ndimage`."""

    @pytest.mark.parametrize("boundary", list(PAD_TO_NDIMAGE))
    def test_whole_raster_matches_ndimage(self, raster, boundary):
        expected = ndimage.convolve(raster, ASYMMETRIC_KERNEL, mode=PAD_TO_NDIMAGE[boundary])
        produced = fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary=boundary).data
        assert_array(produced, expected, rtol=0, atol=1e-9, err_msg=f"boundary={boundary}")

    def test_kernel_is_convolved_not_correlated(self, raster):
        """Guards the orientation: a symmetric kernel cannot detect a flip."""
        convolved = fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary="constant").data
        correlated = ndimage.correlate(raster, ASYMMETRIC_KERNEL, mode="constant")
        assert not np.allclose(convolved, correlated), "the two must differ for this kernel"
        assert_array(
            convolved,
            ndimage.correlate(raster, ASYMMETRIC_KERNEL[::-1, ::-1], mode="constant"),
            rtol=0,
            atol=1e-9,
        )

    def test_stack_is_filtered_plane_by_plane(self, stack):
        produced = fft_array_filter(
            stack, ASYMMETRIC_KERNEL, boundary="symmetric", axes=(1, 2)
        ).data
        for index, plane in enumerate(stack):
            expected = ndimage.convolve(plane, ASYMMETRIC_KERNEL, mode="reflect")
            assert_array(produced[index], expected, rtol=0, atol=1e-9, err_msg=f"plane {index}")

    @pytest.mark.parametrize("method", list(CONVOLUTION_METHODS))
    def test_backends_agree(self, raster, method):
        reference = fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary="symmetric").data
        produced = fft_array_filter(
            raster, ASYMMETRIC_KERNEL, boundary="symmetric", method=method
        ).data
        assert_array(produced, reference, rtol=0, atol=1e-9)
    
    @pytest.mark.parametrize("boundary", list(BOUNDARY_MODES))
    def test_valid_matches_scipy(self, raster, boundary):
        expected = signal.convolve(raster, ASYMMETRIC_KERNEL, mode="valid")
        produced = fft_array_filter(
            raster, ASYMMETRIC_KERNEL, boundary=boundary, out_mode="valid"
        ).data
        assert_array(produced, expected, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Algebraic properties
# --------------------------------------------------------------------------- #
class TestProperties:
    """Invariants that hold for every kernel and every window."""

    @pytest.mark.parametrize("boundary", BOUNDARY_MATRIX, ids=BOUNDARY_IDS)
    def test_dirac_is_the_identity_on_the_window(self, raster, boundary):
        produced = fft_array_filter(
            raster, DIRAC_KERNEL, ((10, 20), (30, 42)), boundary=boundary
        ).data
        assert_array(produced, raster[10:21, 30:43], rtol=0, atol=1e-12)

    @pytest.mark.parametrize(("row", "col"), [(0, 1), (2, 1), (1, 0), (1, 2)])
    def test_shifted_dirac_translates(self, raster, row, col):
        """A Dirac off-centre translates by one sample"""
        kernel = np.zeros((3, 3))
        kernel[row, col] = 1.0
        produced = fft_array_filter(raster, kernel, ((10, 20), (30, 42)), boundary="symmetric").data
        shift_row, shift_col = 1 - row, 1 - col
        expected = raster[10 + shift_row : 21 + shift_row, 30 + shift_col : 43 + shift_col]
        assert_array(produced, expected, rtol=0, atol=1e-12)

    def test_linearity(self, raster):
        rng = np.random.default_rng(3)
        other = rng.standard_normal(raster.shape)
        alpha, beta = 2.5, -0.75

        def filtered(array):
            return fft_array_filter(array, ASYMMETRIC_KERNEL, boundary="constant").data

        assert_array(
            filtered(alpha * raster + beta * other),
            alpha * filtered(raster) + beta * filtered(other),
            rtol=0,
            atol=1e-9,
        )

    def test_separable_kernel_equals_two_passes(self, raster):
        rows = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        cols = np.array([1.0, 2.0, 1.0]) / 4.0
        separable = np.outer(rows, cols)

        one_pass = fft_array_filter(raster, separable, boundary="constant").data
        first = fft_array_filter(raster, rows[:, None], boundary="constant").data
        two_passes = fft_array_filter(first, cols[None, :], boundary="constant").data
        assert_array(one_pass, two_passes, rtol=0, atol=1e-9)

    @pytest.mark.parametrize(
        "window", [((0, 24), (0, 29)), ((25, 49), (30, 59)), ((10, 20), (5, 8))]
    )
    def test_tile_matches_whole_raster(self, raster, window):
        """Filtering a tile with real margins gives what filtering the full
        raster gives at the same place."""
        whole = fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary="symmetric").data
        tile = fft_array_filter(raster, ASYMMETRIC_KERNEL, window, boundary="symmetric").data
        (first_row, last_row), (first_col, last_col) = window
        assert_array(
            tile,
            whole[first_row : last_row + 1, first_col : last_col + 1],
            rtol=0,
            atol=1e-9,
        )

    @pytest.mark.parametrize("q", [2, 3, 5])
    @pytest.mark.parametrize("origin", list(DECIMATION_ORIGINS))
    def test_decimation_equals_filter_then_subsample(self, raster, q, origin):
        kwargs = {"win": ((10, 20), (30, 42)), "boundary": "symmetric"}
        decimated = fft_array_filter(
            raster, ASYMMETRIC_KERNEL, zoom=(1, q), decimation=origin, **kwargs
        ).data
        full_rate = fft_array_filter(raster, ASYMMETRIC_KERNEL, **kwargs).data
        offset = decimation_offset(q, origin)
        assert_array(decimated, full_rate[offset::q, offset::q], rtol=0, atol=1e-12)

    @pytest.mark.parametrize("boundary", list(BOUNDARY_MODES))
    @pytest.mark.parametrize("out_mode", list(OUTPUT_MODES))
    @pytest.mark.parametrize("win", [None, ((10, 40), (10, 50))], ids=["whole", "interior"])
    def test_validity_window_stability(self, raster, boundary, out_mode, win):
        """Checks the validity window is the same whatever the boundary and out_mode"""
        reference = fft_array_filter(
            raster, ASYMMETRIC_KERNEL, win, boundary=None, out_mode="valid", dtype=np.float64
        ).data
        produced = fft_array_filter(
            raster,
            ASYMMETRIC_KERNEL,
            win,
            boundary=boundary,
            out_mode=out_mode,
            dtype=np.float64,
        ).data
        
        plan = build_plan(
            raster.shape, ASYMMETRIC_KERNEL, win, boundary=boundary, out_mode=out_mode
        )
        margins = kernel_margin(ASYMMETRIC_KERNEL)
        inside = []
        for axis, geometry in enumerate(plan.per_axis):
            start = geometry.origin + margins[axis] - geometry.output.start
            inside.append(slice(start, start + reference.shape[axis]))
        assert_array(produced[tuple(inside)], reference, rtol=0, atol=1e-9)

# --------------------------------------------------------------------------- #
# Contracts of the public entry point
# --------------------------------------------------------------------------- #
class TestFilterContract:
    """Everything a caller can rely on that is not a pixel value."""

    def test_result_unpacks_as_a_pair(self, raster):
        result = fft_array_filter(raster, DIRAC_KERNEL)
        data, window = result
        assert data is result.data
        assert window is result.window

    def test_window_locates_the_production_area_in_full_output(self, raster):
        window_spec = ((10, 20), (30, 42))
        full = fft_array_filter(raster, ASYMMETRIC_KERNEL, window_spec, out_mode="full")
        same = fft_array_filter(raster, ASYMMETRIC_KERNEL, window_spec, out_mode="same")
        (first_row, last_row), (first_col, last_col) = full.window
        assert_array(
            full.data[first_row : last_row + 1, first_col : last_col + 1],
            same.data,
            rtol=0,
            atol=1e-12,
        )

    def test_default_dtype_is_the_numpy_promotion(self, raster):
        result = fft_array_filter(raster.astype(np.float32), ASYMMETRIC_KERNEL)
        assert result.data.dtype == np.float64, "a float64 kernel promotes a float32 raster"

    def test_dtype_can_be_pinned(self, raster):
        result = fft_array_filter(raster.astype(np.float32), ASYMMETRIC_KERNEL, dtype=np.float32)
        assert result.data.dtype == np.float32

    def test_padding_matches_numpy_pad(self, raster):
        """The in-place fill must be indistinguishable from ``numpy.pad``.

        Guards against going back to one :func:`numpy.pad` call per side, which
        allocates per side and wraps around the already-extended array.
        """
        plan = build_plan(raster.shape, ASYMMETRIC_KERNEL, boundary="wrap")
        produced = _make_convolution_input(raster, plan)
        expected = np.pad(raster[plan.source], plan.pad_width, mode="wrap")
        assert_array(produced, expected, rtol=0, atol=0)

    def test_input_is_never_modified(self, raster):
        original = raster.copy()
        fft_array_filter(raster, ASYMMETRIC_KERNEL, boundary="symmetric")
        assert_array(raster, original, rtol=0, atol=0)

    def test_even_kernel_is_padded_to_odd(self, raster):
        even = np.ones((4, 4)) / 16.0
        padded = np.zeros((5, 5))
        padded[:4, :4] = even
        assert_array(
            fft_array_filter(raster, even, boundary="constant").data,
            fft_array_filter(raster, padded, boundary="constant").data,
            rtol=0,
            atol=1e-12,
        )

    def test_a_plan_can_be_reused_across_tiles(self, raster):
        """Tiled processing builds the geometry once and applies it many times."""
        plan = build_plan(raster.shape, ASYMMETRIC_KERNEL, boundary="symmetric")
        first = fft_array_filter(raster, ASYMMETRIC_KERNEL, plan=plan).data
        second = fft_array_filter(raster * 2.0, ASYMMETRIC_KERNEL, plan=plan).data
        assert_array(second, 2.0 * first, rtol=0, atol=1e-9)

    def test_plan_rank_is_checked(self, raster, stack):
        """Plan with incorrect rank is not allowed."""
        plan = build_plan(raster.shape, DIRAC_KERNEL)
        with pytest.raises(ValueError, match="plan was built"):
            fft_array_filter(stack, DIRAC_KERNEL, plan=plan)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Exhaustive coverage of the small integer domains (Q, size, offset, kernel size)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", range(1, 18))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 7, 16, 33, 200])
def test_decimated_size_over_every_offset(size, q):
    for offset in range(size + 3):
        assert decimated_size(size, q, offset) == len(range(size)[offset::q])


@pytest.mark.parametrize("half_cols", range(0, 5))
@pytest.mark.parametrize("half_rows", range(0, 5))
@pytest.mark.parametrize(("rows", "cols"), [(1, 1), (1, 40), (40, 1), (17, 23), (40, 40)])
def test_same_mode_always_returns_the_window(rows, cols, half_rows, half_cols):
    """``same`` returns the production window whatever the kernel size."""
    kernel = np.ones((2 * half_rows + 1, 2 * half_cols + 1))
    shape = fft_array_filter_output_shape((rows, cols), kernel, out_mode="same", boundary="symmetric")
    assert shape == (rows, cols)
