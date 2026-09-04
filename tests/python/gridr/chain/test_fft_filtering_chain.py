# coding: utf8
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
"""Tests for :mod:`gridr.chain.fft_filtering_chain`.

The reference is the monolithic :func:`fft_array_filter` call: the chain exists
to bound memory and I/O, not to compute something different. Every geometric
test therefore ends in a comparison against that call.

Equality is to within floating-point tolerance, not bit for bit: overlap-add
performs a different sequence of operations — other FFT sizes, and partial sums
— so the two results differ at the last bits. The tolerance below is relative
to the data magnitude and sits far under any meaningful signal.

``TestDecimatedBlock`` carries most of the value per second of runtime: it
checks, on pure integers, the one invariant the whole chain rests on — that
slicing a partitioned full-resolution range block by block yields exactly the
globally decimated sequence.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from gridr.chain.fft_filtering_chain import (
    check_oa_strip_size,
    decimated_block,
    extended_extent,
    fft_filtering_oa_strip_chain,
    normalize_win,
)
from gridr.core.convolution.fft_filtering import (
    BoundaryPad,
    ConvolutionMethod,
    ConvolutionOutputMode,
    DecimationOrigin,
    decimation_offset,
    fft_array_filter,
    fft_array_filter_output_shape,
)
from gridr.core.utils import chunks

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
#: Non-square and non-symmetric: a square symmetric kernel hides a transposed
#: axis mapping, a flipped convolution and an off-by-one in the strip margins.
ASYMMETRIC_KERNEL = np.arange(1.0, 16.0).reshape(5, 3) / 120.0
#: Even-sized, to exercise the zero-padding to an odd size. The chain must use
#: the padded kernel for its margins, not the raw filter.
EVEN_KERNEL = np.ones((4, 4)) / 16.0
WIDE_KERNEL = np.ones((7, 7)) / 49.0

#: A plausible UTM transform. Georeferencing the fixtures avoids
#: ``NotGeoreferencedWarning`` noise without filtering warnings away, which
#: would also hide real ones. ``Affine.identity`` is not enough: GDAL warns
#: about it too.
TRANSFORM = Affine.translation(300000.0, 4800000.0) @ Affine.scale(10.0, -10.0)

#: Production windows exercised as an axis of the equivalence tests, on the
#: 300x137 fixture with ``strip_size=64``. Three families matter and they test
#: different things:
#:
#: - **raster edges**, where a boundary policy actually has to synthesise;
#: - **interior**, where it must not, whatever the policy;
#: - **chunk boundaries**, where the window height lands exactly on a strip cut,
#:   one row over, one row under, or too short to stripe at all.
#:
#: Most heights are 192 rows, i.e. three strips, so that a window has a first,
#: a middle and a last chunk rather than degenerating into two.
WINDOW_CASES = {
    "none": None,
    "explicit-full": ((0, 299), (0, 136)),
    "interior": ((50, 249), (30, 119)),
    "top-edge": ((0, 191), (30, 119)),
    "bottom-edge": ((108, 299), (30, 119)),
    "left-edge": ((50, 241), (0, 99)),
    "right-edge": ((50, 241), (37, 136)),
    "corner-top-left": ((0, 191), (0, 99)),
    "corner-bottom-right": ((108, 299), (37, 136)),
    "chunk-aligned": ((40, 167), (20, 119)),
    "chunk-merged": ((40, 168), (20, 119)),
    "three-chunks": ((40, 231), (20, 119)),
    "just-under-two-chunks": ((40, 166), (20, 119)),
}
WINDOW_IDS = list(WINDOW_CASES)
WINDOW_VALUES = [None if w is None else np.asarray(w) for w in WINDOW_CASES.values()]

#: A cheap subset for the tests that already carry a wide option matrix.
SAMPLE_WINDOW_IDS = ["none", "interior", "corner-bottom-right", "chunk-merged"]
SAMPLE_WINDOW_VALUES = [
    None if WINDOW_CASES[name] is None else np.asarray(WINDOW_CASES[name])
    for name in SAMPLE_WINDOW_IDS
]

#: Local policies only. ``WRAP`` is excluded by construction — see
#: ``TestChainContract.test_row_axis_wrap_is_refused``.
LOCAL_POLICIES = [
    BoundaryPad.NONE,
    BoundaryPad.REFLECT,
    BoundaryPad.SYMMETRIC,
    BoundaryPad.EDGE,
    BoundaryPad.ZERO,
]


@pytest.fixture(name="raster")
def fixture_raster(tmp_path):
    """Write a deterministic pseudo-random raster and hand back a factory.

    Random rather than a ramp: a linear ramp is reproduced almost exactly by
    any normalised smoothing kernel, which would mask an offset error in the
    strip reconstruction — precisely what these tests are for.
    """

    def _make(nrow: int = 300, ncol: int = 137, scale: float = 100.0) -> tuple:
        data = np.random.default_rng(20260903).standard_normal((nrow, ncol))
        data = (data * scale).astype(np.float32)
        path = tmp_path / f"in_{nrow}x{ncol}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=nrow,
            width=ncol,
            count=1,
            dtype="float32",
            transform=TRANSFORM,
            crs="EPSG:32631",
        ) as dataset:
            dataset.write(data, 1)
        return path, data

    return _make


def run_chain(path_in, out_path, out_shape, out_dtype="float64", **kwargs) -> np.ndarray:
    """Run the chain into a fresh dataset of `out_shape` and read it back."""
    with (
        rasterio.open(path_in) as ds_in,
        rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype=out_dtype,
            transform=TRANSFORM,
            crs="EPSG:32631",
        ) as ds_out,
    ):
        fft_filtering_oa_strip_chain(ds_in=ds_in, ds_out=ds_out, band=1, **kwargs)
    with rasterio.open(out_path) as dataset:
        return dataset.read(1)


def run_and_compare(path_in, data, out_path, kernel, win=None, *, strip_size=64, **options):
    """Run the chain on `win` and compare with the same call made in one go.

    Every equivalence test funnels through here, so the chain and the core are
    always given exactly the same window and options.
    """
    shape = fft_array_filter_output_shape(data.shape, kernel, win, **options)
    produced = run_chain(
        path_in,
        out_path,
        shape,
        fil=kernel,
        win=win,
        strip_size=strip_size,
        dtype=np.float64,
        round_out=False,
        **options,
    )
    assert_matches_monolithic(produced, data, kernel, win=win, **options)
    return produced


def assert_matches_monolithic(produced, data, kernel, win=None, **kwargs) -> None:
    """Compare against the single-call reference, scaled to the data magnitude."""
    expected, _ = fft_array_filter(data, kernel, win, dtype=np.float64, **kwargs)
    assert produced.shape == expected.shape, f"{produced.shape} != {expected.shape}"
    scale = float(np.max(np.abs(expected))) or 1.0
    deviation = float(np.max(np.abs(produced - expected)))
    assert deviation <= 1e-9 * scale, f"max|d| = {deviation:.3e} for a magnitude of {scale:.3e}"


# --------------------------------------------------------------------------- #
# The invariant the chain rests on
# --------------------------------------------------------------------------- #
class TestDecimatedBlock:
    """Pure integer arithmetic: no raster, no convolution, no I/O."""

    @pytest.mark.parametrize(
        ("global_start", "count", "q", "offset", "expected"),
        [
            (0, 10, 1, 0, (slice(0, 10, 1), 0)),
            (7, 10, 1, 0, (slice(0, 10, 1), 7)),
            (0, 10, 3, 1, (slice(1, 10, 3), 0)),
            (10, 10, 3, 1, (slice(0, 10, 3), 3)),
            (11, 2, 3, 1, (None, 0)),
            (0, 1, 5, 2, (None, 0)),
            (0, 3, 5, 2, (slice(2, 3, 5), 0)),
        ],
    )
    def test_reference_values(self, global_start, count, q, offset, expected):
        assert decimated_block(global_start, count, q, offset) == expected

    @pytest.mark.parametrize("count", [0, -1])
    def test_empty_block_keeps_nothing(self, count):
        assert decimated_block(5, count, 3, 1) == (None, 0)

    def test_non_positive_step_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            decimated_block(0, 10, 0, 0)

    @pytest.mark.parametrize("offset", [-1, 3, 7])
    def test_offset_outside_one_period_is_rejected(self, offset):
        """``decimation_offset`` always returns a value in ``[0, q)``."""
        with pytest.raises(ValueError, match=r"offset must lie in \[0, 3\)"):
            decimated_block(0, 10, 3, offset)

    @pytest.mark.parametrize("global_start", [0, 5, 100])
    def test_q_one_keeps_everything_at_its_global_index(self, global_start):
        """No special case for ``q == 1``: the general formula already covers it."""
        assert decimated_block(global_start, 10, 1, 0) == (slice(0, 10, 1), global_start)

    @pytest.mark.parametrize("q", range(1, 8))
    @pytest.mark.parametrize("origin", list(DecimationOrigin))
    @pytest.mark.parametrize("cuts", [(0, 7, 19, 40), (0, 1, 2, 40), (0, 13, 26, 40), (0, 40)])
    def test_partition_reconstructs_the_global_decimation(self, q, origin, cuts):
        """Blocks written one after another must tile the decimated output.

        This is the property a per-strip counter breaks: the number of samples
        a block contributes depends on where it starts modulo ``q``, so the
        destination index has to come from the global frame.
        """
        total = cuts[-1]
        offset = decimation_offset(q, origin)
        expected = list(range(total))[offset::q]

        rebuilt = [None] * len(expected)
        for start, stop in itertools.pairwise(cuts):
            block = list(range(start, stop))
            local, destination = decimated_block(start, stop - start, q, offset)
            if local is None:
                continue
            kept = block[local]
            for index, value in enumerate(kept):
                position = destination + index
                assert rebuilt[position] is None, f"position {position} written twice"
                rebuilt[position] = value

        assert rebuilt == expected


class TestCheckOaStripSize:
    """When striping is worth it, and when it cannot work at all."""

    @pytest.mark.parametrize(
        ("nrow", "kernel_rows", "strip_size", "expected"),
        [
            (1000, 3, 128, 128),  # nominal
            (1000, 3, 600, 0),  # strip larger than half the raster
            (1000, 501, 128, 0),  # kernel larger than half the raster
            (1000, 200, 128, 0),  # strip shorter than the kernel
            (1000, 128, 128, 128),  # strip exactly the kernel height
            (1000, 3, 0, 0),  # already monolithic
        ],
    )
    def test_decision(self, nrow, kernel_rows, strip_size, expected):
        kernel = np.ones((kernel_rows, 3))
        assert check_oa_strip_size(nrow=nrow, kernel=kernel, strip_size=strip_size) == expected

    def test_a_strip_shorter_than_the_kernel_cannot_carry_its_support(self):
        """Without this rule the overlap buffer is silently under-sized.

        A 1001-row kernel on a 3000-row raster does not trip the "half the
        raster" rule, so 512-row strips were kept and the buffer allocated
        513 rows instead of 1002.
        """
        assert check_oa_strip_size(nrow=3000, kernel=np.ones((1001, 3)), strip_size=512) == 0


# --------------------------------------------------------------------------- #
# Equivalence with the monolithic call
# --------------------------------------------------------------------------- #
class TestChainMatchesMonolithic:
    """The chain bounds memory and I/O; it must not change the numbers.

    ``win`` is an axis of these tests rather than a separate family, because a
    window is not a variant of the problem: it changes where the strips are
    cut, which margins are read and where the boundary policies apply. Testing
    it apart would leave every option matrix below unverified for windows.
    """

    @pytest.mark.parametrize("win", WINDOW_VALUES, ids=WINDOW_IDS)
    @pytest.mark.parametrize("out_mode", [ConvolutionOutputMode.SAME, ConvolutionOutputMode.FULL])
    @pytest.mark.parametrize("q", [1, 3])
    def test_every_window_position(self, raster, tmp_path, win, out_mode, q):
        """Raster edges, interior, and windows landing on a chunk cut."""
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            ASYMMETRIC_KERNEL,
            win,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=out_mode,
            zoom=(1, q),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    @pytest.mark.parametrize("boundary", LOCAL_POLICIES, ids=lambda p: p.name)
    @pytest.mark.parametrize("out_mode", [ConvolutionOutputMode.SAME, ConvolutionOutputMode.FULL])
    @pytest.mark.parametrize("q", [1, 2, 5])
    def test_boundary_output_mode_and_decimation(
        self, raster, tmp_path, win, boundary, out_mode, q
    ):
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            ASYMMETRIC_KERNEL,
            win,
            boundary=boundary,
            out_mode=out_mode,
            zoom=(1, q),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    @pytest.mark.parametrize("origin", list(DecimationOrigin), ids=lambda o: o.name)
    @pytest.mark.parametrize("q", [2, 3, 4, 7, 13])
    def test_decimation_phase_survives_every_strip_size(self, raster, tmp_path, win, origin, q):
        """Strip heights that are not multiples of ``Q`` are the interesting case."""
        path_in, data = raster(nrow=301, ncol=138)
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            ASYMMETRIC_KERNEL,
            win,
            strip_size=97,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, q),
            decimation=origin,
        )

    @pytest.mark.parametrize("win", WINDOW_VALUES, ids=WINDOW_IDS)
    @pytest.mark.parametrize("strip_size", [64, 71, 97, 128, 149])
    def test_strip_size_does_not_change_the_result(self, raster, tmp_path, win, strip_size):
        """Where the cuts fall must be invisible, wherever the window starts."""
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            WIDE_KERNEL,
            win,
            strip_size=strip_size,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, 3),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    @pytest.mark.parametrize("q", [1, 3])
    def test_even_sized_kernel(self, raster, tmp_path, win, q):
        """The chain must derive its margins from the odd-padded kernel.

        Using the raw filter shifts every strip boundary by one row.
        """
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            EVEN_KERNEL,
            win,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, q),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    def test_per_side_policies(self, raster, tmp_path, win):
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            ASYMMETRIC_KERNEL,
            win,
            boundary=(
                (BoundaryPad.SYMMETRIC, BoundaryPad.NONE),
                (BoundaryPad.NONE, BoundaryPad.SYMMETRIC),
            ),
            out_mode=ConvolutionOutputMode.FULL,
            zoom=(1, 2),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    def test_column_axis_wrap_is_supported(self, raster, tmp_path, win):
        """Every strip spans the full window width, so a column seam is local."""
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            ASYMMETRIC_KERNEL,
            win,
            boundary=((BoundaryPad.NONE, BoundaryPad.NONE), (BoundaryPad.WRAP, BoundaryPad.WRAP)),
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, 3),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    @pytest.mark.parametrize("method", list(ConvolutionMethod), ids=lambda m: m.name)
    def test_every_backend_agrees(self, raster, tmp_path, win, method):
        path_in, data = raster()
        shape = fft_array_filter_output_shape(
            data.shape,
            ASYMMETRIC_KERNEL,
            win,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, 2),
        )
        produced = run_chain(
            path_in,
            tmp_path / "out.tif",
            shape,
            fil=ASYMMETRIC_KERNEL,
            win=win,
            strip_size=64,
            method=method,
            dtype=np.float64,
            round_out=False,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, 2),
        )
        assert_matches_monolithic(
            produced,
            data,
            ASYMMETRIC_KERNEL,
            win=win,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, 2),
        )

    @pytest.mark.parametrize("win", SAMPLE_WINDOW_VALUES, ids=SAMPLE_WINDOW_IDS)
    def test_single_chunk_fallback_matches_too(self, raster, tmp_path, win):
        """A strip shorter than the kernel forces the monolithic path."""
        path_in, data = raster()
        run_and_compare(
            path_in,
            data,
            tmp_path / "out.tif",
            WIDE_KERNEL,
            win,
            strip_size=4,
            boundary=BoundaryPad.SYMMETRIC,
            out_mode=ConvolutionOutputMode.SAME,
            zoom=(1, 3),
        )


# --------------------------------------------------------------------------- #
# Production window
# --------------------------------------------------------------------------- #
class TestNormalizeWin:
    """Window parsing, before any I/O happens."""

    def test_none_is_the_whole_array(self):
        assert normalize_win(None, (50, 60)).tolist() == [[0, 49], [0, 59]]

    def test_bounds_are_inclusive_and_integer(self):
        window = normalize_win(((10, 20), (30, 40)), (50, 60))
        assert window.tolist() == [[10, 20], [30, 40]]
        assert np.issubdtype(window.dtype, np.integer)

    @pytest.mark.parametrize(
        ("win", "match"),
        [
            (((10, 20),), "shape"),
            (((10, 20), (30, 40), (0, 1)), "shape"),
            (((10.0, 20.0), (30.0, 40.0)), "shape"),
            (((20, 10), (30, 40)), "empty"),
            (((-1, 20), (30, 40)), "not contained"),
            (((10, 50), (30, 40)), "not contained"),
            (((10, 20), (30, 60)), "not contained"),
        ],
        ids=["too-few", "too-many", "float", "empty", "negative", "past-rows", "past-cols"],
    )
    def test_malformed(self, win, match):
        with pytest.raises(ValueError, match=match):
            normalize_win(win, (50, 60))


class TestExtendedExtent:
    """Which samples have to be read around the window."""

    @pytest.mark.parametrize(
        ("first", "last", "margin", "expected"),
        [
            (10, 20, 3, (7, 23)),
            (1, 20, 3, (0, 23)),
            (10, 98, 3, (7, 99)),
            (0, 99, 3, (0, 99)),
            (10, 20, 0, (10, 20)),
        ],
        ids=["interior", "clipped-low", "clipped-high", "whole-axis", "no-margin"],
    )
    def test_extent_is_independent_of_any_policy(self, first, last, margin, expected):
        """Real neighbours are read whenever they exist.

        A boundary condition describes the edge of the raster; it must never
        decide whether data in the middle of it is read.
        """
        assert extended_extent(first, last, 100, margin) == expected

    @pytest.mark.parametrize(
        ("extend_before", "extend_after", "expected"),
        [
            (True, True, (7, 23)),
            (False, True, (10, 23)),
            (True, False, (7, 20)),
            (False, False, (10, 20)),
        ],
        ids=["both", "seam-above", "seam-below", "both-seams"],
    )
    def test_seam_flags_suppress_the_extension(self, extend_before, extend_after, expected):
        """The flags exist for overlap-add cuts, not for boundary policies.

        A strip must not see beyond itself at an internal seam: those are the
        contributions overlap-add carries over.
        """
        assert (
            extended_extent(10, 20, 100, 3, extend_before=extend_before, extend_after=extend_after)
            == expected
        )


class TestWindowChunking:
    """The window cases must actually produce the strip structures they claim.

    A case named "chunk-aligned" that silently falls back to a single call
    tests nothing while looking like it tests something. These expectations
    pin the structure so that a change to the chunking rules is caught here,
    where it is readable, instead of quietly emptying the matrix above.
    """

    @pytest.mark.parametrize(
        ("name", "height", "strip_size", "chunk_sizes"),
        [
            ("none", 300, 64, [64, 64, 64, 108]),
            ("interior", 200, 64, [64, 64, 72]),
            ("top-edge", 192, 64, [64, 64, 64]),
            ("chunk-aligned", 128, 64, [64, 64]),
            ("chunk-merged", 129, 64, [64, 65]),
            ("just-under-two-chunks", 127, 0, [127]),
        ],
    )
    def test_window_height_drives_the_strip_layout(self, name, height, strip_size, chunk_sizes):
        window = WINDOW_CASES[name]
        if window is not None:
            assert window[0][1] - window[0][0] + 1 == height, "the case no longer has that height"

        effective = check_oa_strip_size(nrow=height, kernel=ASYMMETRIC_KERNEL, strip_size=64)
        assert effective == strip_size
        boundaries = chunks.get_chunk_boundaries(
            nsize=height, chunk_size=effective, merge_last=True
        )
        assert [int(upper) - int(lower) for lower, upper in boundaries] == chunk_sizes

    def test_the_matrix_covers_more_than_one_chunk_layout(self):
        """Guards against every window degenerating to the same shape."""
        layouts = set()
        for window in WINDOW_CASES.values():
            height = 300 if window is None else window[0][1] - window[0][0] + 1
            effective = check_oa_strip_size(nrow=height, kernel=ASYMMETRIC_KERNEL, strip_size=64)
            boundaries = chunks.get_chunk_boundaries(
                nsize=height, chunk_size=effective, merge_last=True
            )
            layouts.add(len(boundaries))
        assert layouts >= {1, 2, 3, 4}, f"only {sorted(layouts)} chunk counts exercised"


class TestProductionWindow:
    """A window must give the same numbers as the same window filtered alone."""

    @pytest.mark.parametrize("boundary", LOCAL_POLICIES, ids=lambda p: p.name)
    def test_window_margins_come_from_real_neighbours(self, raster, tmp_path, boundary):
        """An interior window is padded on no side, whatever the policy.

        The margins exist in the raster, so the policy never applies and every
        policy must give the same numbers.
        """
        path_in, data = raster()
        window = np.asarray(((50, 249), (30, 129)))
        options = {"boundary": boundary, "out_mode": ConvolutionOutputMode.SAME, "zoom": (1, 2)}
        shape = fft_array_filter_output_shape(data.shape, ASYMMETRIC_KERNEL, window, **options)
        produced = run_chain(
            path_in,
            tmp_path / f"out_{boundary.name}.tif",
            shape,
            fil=ASYMMETRIC_KERNEL,
            win=window,
            strip_size=64,
            dtype=np.float64,
            round_out=False,
            **options,
        )
        assert_matches_monolithic(produced, data, ASYMMETRIC_KERNEL, win=window, **options)

    @pytest.mark.parametrize(
        "win",
        [((100, 111), (100, 108)), ((0, 5), (0, 5)), ((290, 299), (130, 136))],
        ids=["tiny-interior", "tiny-corner", "tiny-far-corner"],
    )
    def test_a_window_too_short_to_stripe_falls_back(self, raster, tmp_path, win):
        """The fallback must handle the cases striping cannot, not crash on them."""
        path_in, data = raster()
        window = np.asarray(win)
        options = {
            "boundary": BoundaryPad.SYMMETRIC,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, 2),
        }
        shape = fft_array_filter_output_shape(data.shape, WIDE_KERNEL, window, **options)
        produced = run_chain(
            path_in,
            tmp_path / "out_tiny.tif",
            shape,
            fil=WIDE_KERNEL,
            win=window,
            strip_size=512,
            dtype=np.float64,
            round_out=False,
            **options,
        )
        assert_matches_monolithic(produced, data, WIDE_KERNEL, win=window, **options)

    def test_a_window_narrower_than_the_kernel(self, raster, tmp_path):
        path_in, data = raster()
        window = np.asarray(((10, 289), (120, 122)))
        options = {
            "boundary": BoundaryPad.ZERO,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, 1),
        }
        shape = fft_array_filter_output_shape(data.shape, WIDE_KERNEL, window, **options)
        produced = run_chain(
            path_in,
            tmp_path / "out_narrow.tif",
            shape,
            fil=WIDE_KERNEL,
            win=window,
            strip_size=64,
            dtype=np.float64,
            round_out=False,
            **options,
        )
        assert_matches_monolithic(produced, data, WIDE_KERNEL, win=window, **options)

    def test_reads_are_proportional_to_the_window(self, raster, tmp_path):
        """The point of the whole thing: a small window costs a small read.

        Reading whole strips of the raster would be correct and unusable on a
        large image; this counts the pixels actually pulled from the dataset.
        """
        path_in, data = raster(nrow=400, ncol=400)
        window = np.asarray(((150, 249), (200, 279)))
        options = {
            "boundary": BoundaryPad.SYMMETRIC,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, 2),
        }
        shape = fft_array_filter_output_shape(data.shape, WIDE_KERNEL, window, **options)

        pixels_read = 0

        with (
            rasterio.open(path_in) as ds_in,
            rasterio.open(
                tmp_path / "out_io.tif",
                "w",
                driver="GTiff",
                height=shape[0],
                width=shape[1],
                count=1,
                dtype="float64",
                transform=TRANSFORM,
                crs="EPSG:32631",
            ) as ds_out,
        ):
            original_read = ds_in.read

            def counting_read(*args, **kwargs):
                nonlocal pixels_read
                block = original_read(*args, **kwargs)
                pixels_read += int(np.prod(block.shape))
                return block

            ds_in.read = counting_read
            fft_filtering_oa_strip_chain(
                ds_in=ds_in,
                ds_out=ds_out,
                band=1,
                fil=WIDE_KERNEL,
                win=window,
                strip_size=512,
                dtype=np.float64,
                round_out=False,
                **options,
            )

        margin = WIDE_KERNEL.shape[0] // 2
        expected = (100 + 2 * margin) * (80 + 2 * margin)
        assert pixels_read == expected, f"read {pixels_read} pixels, window needs {expected}"
        assert pixels_read < 0.1 * data.size


# --------------------------------------------------------------------------- #
# Contracts of the chain itself
# --------------------------------------------------------------------------- #
class TestChainContract:
    """What a caller can rely on beyond the pixel values."""

    @pytest.mark.parametrize(
        "boundary",
        [
            BoundaryPad.WRAP,
            ((BoundaryPad.WRAP, BoundaryPad.NONE), (BoundaryPad.NONE, BoundaryPad.NONE)),
            ((BoundaryPad.NONE, BoundaryPad.WRAP), (BoundaryPad.NONE, BoundaryPad.NONE)),
        ],
        ids=["scalar", "top-only", "bottom-only"],
    )
    def test_row_axis_wrap_is_refused(self, raster, tmp_path, boundary):
        """The only non-local policy, and the only one a strip cannot honour.

        The top margin of the first strip comes from the bottom of the raster.
        Padding each strip independently wraps it around the strip instead,
        which is wrong by a wide margin rather than by a rounding error.
        """
        path_in, data = raster(nrow=200, ncol=80)
        with pytest.raises(NotImplementedError, match="WRAP is not supported on the row axis"):
            run_chain(
                path_in,
                tmp_path / "out.tif",
                data.shape,
                fil=WIDE_KERNEL,
                boundary=boundary,
                out_mode=ConvolutionOutputMode.SAME,
                strip_size=64,
            )

    def test_valid_output_mode_is_refused(self, raster, tmp_path):
        path_in, data = raster(nrow=200, ncol=80)
        with pytest.raises(NotImplementedError, match="VALID"):
            run_chain(
                path_in,
                tmp_path / "out.tif",
                data.shape,
                fil=WIDE_KERNEL,
                boundary=BoundaryPad.SYMMETRIC,
                out_mode=ConvolutionOutputMode.VALID,
                strip_size=64,
            )

    def test_upsampling_is_refused(self, raster, tmp_path):
        path_in, data = raster(nrow=200, ncol=80)
        with pytest.raises(ValueError, match="only pure decimation"):
            run_chain(
                path_in,
                tmp_path / "out.tif",
                data.shape,
                fil=WIDE_KERNEL,
                boundary=BoundaryPad.SYMMETRIC,
                out_mode=ConvolutionOutputMode.SAME,
                zoom=(2, 1),
                strip_size=64,
            )

    @pytest.mark.parametrize("shape", [(200, 80), (67, 80), (200, 27)])
    def test_output_dataset_shape_is_checked(self, raster, tmp_path, shape):
        """A silently truncated raster is the failure mode this prevents."""
        path_in, _ = raster(nrow=200, ncol=80)
        with pytest.raises(ValueError, match="open ds_out at the decimated shape"):
            run_chain(
                path_in,
                tmp_path / "out.tif",
                shape,
                fil=WIDE_KERNEL,
                boundary=BoundaryPad.SYMMETRIC,
                out_mode=ConvolutionOutputMode.SAME,
                zoom=(1, 3),
                strip_size=64,
            )

    @pytest.mark.parametrize("q", [1, 3])
    def test_round_out_matches_the_rounded_reference(self, raster, tmp_path, q):
        path_in, data = raster(nrow=240, ncol=80)
        options = {
            "boundary": BoundaryPad.EDGE,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, q),
        }
        shape = fft_array_filter_output_shape(data.shape, WIDE_KERNEL, None, **options)
        produced = run_chain(
            path_in,
            tmp_path / f"out_round_{q}.tif",
            shape,
            fil=WIDE_KERNEL,
            strip_size=60,
            dtype=np.float64,
            round_out=True,
            **options,
        )
        expected, _ = fft_array_filter(data, WIDE_KERNEL, None, dtype=np.float64, **options)
        np.testing.assert_array_equal(produced, np.round(expected))

    def test_binary_output(self, raster, tmp_path):
        """Thresholding amplifies any reconstruction error near the threshold.

        With a magnitude well above the threshold the decision is unambiguous,
        which makes this an exact-equality test.
        """
        path_in, data = raster(nrow=240, ncol=80, scale=100.0)
        options = {
            "boundary": BoundaryPad.EDGE,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, 3),
        }
        shape = fft_array_filter_output_shape(data.shape, WIDE_KERNEL, None, **options)
        produced = run_chain(
            path_in,
            tmp_path / "out_binary.tif",
            shape,
            out_dtype="uint8",
            fil=WIDE_KERNEL,
            strip_size=60,
            dtype=np.float64,
            binary=True,
            binary_threshold=1.0,
            **options,
        )
        expected, _ = fft_array_filter(data, WIDE_KERNEL, None, dtype=np.float64, **options)
        np.testing.assert_array_equal(produced, (np.abs(expected) >= 1.0).astype(np.uint8))

    def test_working_dtype_can_be_pinned(self, raster, tmp_path):
        """``float32`` halves the reconstruction memory; it must stay accurate."""
        path_in, data = raster(nrow=300, ncol=90, scale=1.0)
        options = {
            "boundary": BoundaryPad.ZERO,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, 2),
        }
        shape = fft_array_filter_output_shape(data.shape, ASYMMETRIC_KERNEL, None, **options)
        produced = run_chain(
            path_in,
            tmp_path / "out_f32.tif",
            shape,
            fil=ASYMMETRIC_KERNEL,
            strip_size=64,
            dtype=np.float32,
            round_out=False,
            **options,
        )
        expected, _ = fft_array_filter(data, ASYMMETRIC_KERNEL, None, dtype=np.float32, **options)
        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-5)

    def test_input_raster_is_not_modified(self, raster, tmp_path):
        path_in, data = raster(nrow=200, ncol=80)
        options = {
            "boundary": BoundaryPad.SYMMETRIC,
            "out_mode": ConvolutionOutputMode.SAME,
            "zoom": (1, 2),
        }
        shape = fft_array_filter_output_shape(data.shape, ASYMMETRIC_KERNEL, None, **options)
        run_chain(
            path_in,
            tmp_path / "out.tif",
            shape,
            fil=ASYMMETRIC_KERNEL,
            strip_size=64,
            dtype=np.float64,
            **options,
        )
        with rasterio.open(path_in) as dataset:
            np.testing.assert_array_equal(dataset.read(1), data)
