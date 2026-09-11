# Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.core.utils.array_window module

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/utils/test_array_window.py
"""
import numpy as np
import pytest
import random
import rasterio

from gridr.core.utils.array_window import (
    as_rio_window,
    complementary_window_indices,
    window_apply,
    window_check,
    window_extend,
    window_indices,
    window_overflow,
    window_shape,
    compose_slice,
    window_normalize,
)

ARRAY_00 = np.arange(4 * 7).reshape(4, 7)
WIN_00_01, AXES_00_01, WIN_ARRAY_00_01, NOCHECK_EXPECT_00_01 = (
    [(0, 3), (0, 6)],
    None,
    ARRAY_00[0:4, 0:7],
    ARRAY_00[0:4, 0:7],
)
WIN_00_02, AXES_00_02, WIN_ARRAY_00_02, NOCHECK_EXPECT_00_02 = (
    [(0, 4), (0, 6)],
    None,
    ValueError,
    ARRAY_00[0:5, 0:7],
)
WIN_00_03, AXES_00_03, WIN_ARRAY_00_03, NOCHECK_EXPECT_00_03 = (
    [(0, 3), (0, 7)],
    None,
    ValueError,
    ARRAY_00[0:4, 0:8],
)
WIN_00_04, AXES_00_04, WIN_ARRAY_00_04, NOCHECK_EXPECT_00_04 = (
    [(1, 2), (0, 6)],
    None,
    ARRAY_00[1:3, 0:7],
    ARRAY_00[1:3, 0:7],
)
WIN_00_05, AXES_00_05, WIN_ARRAY_00_05, NOCHECK_EXPECT_00_05 = (
    [(0, 3), (3, 4)],
    None,
    ARRAY_00[0:4, 3:5],
    ARRAY_00[0:4, 3:5],
)
WIN_00_06, AXES_00_06, WIN_ARRAY_00_06, NOCHECK_EXPECT_00_06 = (
    [(1, 2), (3, 3)],
    None,
    ARRAY_00[1:3, 3:4],
    ARRAY_00[1:3, 3:4],
)
WIN_00_07, AXES_00_07, WIN_ARRAY_00_07, NOCHECK_EXPECT_00_07 = (
    [(1, 2), (3, 3)],
    (0,),
    ARRAY_00[1:3, 0:7],
    ARRAY_00[1:3, 0:7],
)

# --- Helper ---


def assert_complement_covers(arr_shape, win, axes=None):
    """Assert that window + complement covers the full array without overlap."""
    inside = window_indices(win, axes=axes)
    comp = complementary_window_indices(win, arr_shape, axes=axes)

    mask = np.zeros(arr_shape, dtype=int)
    mask[inside] += 1
    for s in comp:
        mask[s] += 1

    np.testing.assert_array_equal(mask, np.ones(arr_shape, dtype=int))


class TestArrayWindow:
    """Class for test"""

    @pytest.mark.parametrize(
        "data, expected",
        [
            (([(2, 3), (3, 6)], False, None), (slice(2, 4), slice(3, 7))),
            (([(2, 3), (3, 6)], True, None), (slice(0, 2), slice(0, 4))),
            (([(2, 3), (3, 6)], False, [0]), (slice(2, 4), slice(None, None))),
            (([(2, 3), (3, 6)], False, [1]), (slice(None, None), slice(3, 7))),
        ],
    )
    def test_window_indices(self, data, expected):
        """Test window_indices method"""
        win, reset_origin, axes = data
        try:
            indices = window_indices(win, reset_origin, axes)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except Exception:
                pass

            assert indices == expected

    # --- complementary_window_indices ---

    @pytest.mark.parametrize(
        "shape, win, axes, expected_regions",
        [
            # --- 2D ---
            # centre : 4 regions
            ((5, 6), [[1, 3], [2, 4]], None, 4),
            # top-left corner
            ((4, 5), [[0, 1], [0, 2]], None, 2),
            # botom-right corner
            ((4, 5), [[2, 3], [3, 4]], None, 2),
            # full width
            ((5, 4), [[1, 3], [0, 3]], None, 2),
            # full height
            ((4, 6), [[0, 3], [2, 4]], None, 2),
            # window = entire array
            ((3, 4), [[0, 2], [0, 3]], None, 0),
            # single element at center
            ((3, 3), [[1, 1], [1, 1]], None, 4),
            # single full row
            ((5, 4), [[2, 2], [0, 3]], None, 2),
            # top edge, not touching sides
            ((4, 6), [[0, 1], [2, 4]], None, 3),
            # 1x1 array
            ((1, 1), [[0, 0], [0, 0]], None, 0),
            # --- 3D ---
            # centre : 6 regions
            ((5, 6, 7), [[1, 3], [2, 4], [1, 5]], None, 6),
            # corner
            ((4, 5, 6), [[0, 1], [0, 2], [0, 3]], None, 3),
            # entire array
            ((3, 4, 5), [[0, 2], [0, 3], [0, 4]], None, 0),
            # single element
            ((3, 3, 3), [[1, 1], [1, 1], [1, 1]], None, 6),
            # full slab on 2 axes
            ((4, 5, 6), [[1, 2], [0, 4], [0, 5]], None, 2),
            # --- with axes ---
            # 3D, single axis
            ((4, 5, 6), [[1, 2], [1, 3], [2, 4]], 0, 2),
            # 3D, two axes
            ((4, 5, 6), [[1, 2], [1, 3], [2, 4]], (0, 2), 4),
            # 2D, rows only
            ((5, 4), [[1, 3], [1, 2]], 0, 2),
            # 2D, columns only
            ((5, 4), [[1, 3], [1, 2]], 1, 2),
        ],
    )
    def test_complementary_window_indices_coverage_and_region_count(
        self, shape, win, axes, expected_regions
    ):
        win = np.array(win)
        comp = complementary_window_indices(win, shape, axes=axes)
        assert len(comp) == expected_regions
        assert_complement_covers(shape, win, axes=axes)

    # --- complementary_window_indices : free axes remain unconstrained ---

    @pytest.mark.parametrize(
        "axes, free_axes",
        [
            (0, [1, 2]),
            ((0, 2), [1]),
            (1, [0, 2]),
        ],
    )
    def test_complementary_window_indices_free_axes_unconstrained(self, axes, free_axes):
        shape = (4, 5, 6)
        win = np.array([[1, 2], [1, 3], [2, 4]])
        comp = complementary_window_indices(win, shape, axes=axes)
        for s in comp:
            for ax in free_axes:
                assert s[ax] == slice(None)

    # --- complementary_window_indices : sum and element count integrity ---

    @pytest.mark.parametrize(
        "shape, win",
        [
            ((4, 5, 3), [[1, 2], [1, 3], [0, 1]]),
            ((5, 6), [[1, 3], [2, 4]]),
            ((3, 3, 3), [[1, 1], [1, 1], [1, 1]]),
        ],
    )
    def test_complementary_window_indices_sum_and_count(self, shape, win):
        win = np.array(win)
        arr = np.arange(np.prod(shape)).reshape(shape)

        inside = window_indices(win)
        comp = complementary_window_indices(win, shape)

        inside_sum = arr[inside].sum()
        comp_sum = sum(arr[s].sum() for s in comp)
        assert inside_sum + comp_sum == arr.sum()

        inside_count = arr[inside].size
        comp_count = sum(arr[s].size for s in comp)
        assert inside_count + comp_count == arr.size

    @pytest.mark.parametrize(
        "data, expected",
        [
            (([(2, 3), (3, 6)], None), (2, 4)),
            (([(2, 3), (3, 6)], [0]), (2, None)),
            (([(2, 3), (3, 6)], [1]), (None, 4)),
        ],
    )
    def test_window_shape(self, data, expected):
        """Test window_apply method"""
        win, axes = data
        try:
            shape = window_shape(win, axes)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except Exception:
                pass

            assert shape == expected

    @pytest.mark.parametrize(
        "data, expected",
        [
            (np.array([[2, 3], [3, 6]]), rasterio.windows.Window(3, 2, 4, 2)),
        ],
    )
    def test_as_rio_window(self, data, expected):
        """Test window_apply method"""
        win = data
        try:
            win_rio = as_rio_window(win)
        except Exception as e:
            if isinstance(e, expected):
                pass
            else:
                raise
        else:
            try:
                if issubclass(expected, BaseException):
                    raise Exception(f"The test should have raised an exceptionof type {expected}")
            except Exception:
                pass

            assert win_rio == expected

    @pytest.mark.parametrize(
        "data, expected, testing_decimal",
        [
            ((ARRAY_00, WIN_00_01, AXES_00_01), (WIN_ARRAY_00_01, NOCHECK_EXPECT_00_01), 6),
            ((ARRAY_00, WIN_00_02, AXES_00_02), (WIN_ARRAY_00_02, NOCHECK_EXPECT_00_02), 6),
            ((ARRAY_00, WIN_00_03, AXES_00_03), (WIN_ARRAY_00_03, NOCHECK_EXPECT_00_03), 6),
            ((ARRAY_00, WIN_00_04, AXES_00_04), (WIN_ARRAY_00_04, NOCHECK_EXPECT_00_04), 6),
            ((ARRAY_00, WIN_00_05, AXES_00_05), (WIN_ARRAY_00_05, NOCHECK_EXPECT_00_05), 6),
            ((ARRAY_00, WIN_00_06, AXES_00_06), (WIN_ARRAY_00_06, NOCHECK_EXPECT_00_06), 6),
            ((ARRAY_00, WIN_00_07, AXES_00_07), (WIN_ARRAY_00_07, NOCHECK_EXPECT_00_07), 6),
        ],
    )
    @pytest.mark.parametrize("check", [True, False])
    def test_window_apply(self, data, expected, check, testing_decimal):
        """Test window_apply method"""
        array, win, axes = data
        expected_array, expected_nocheck = expected
        try:

            win_array = window_apply(array, win, axes, check=check)
        except Exception as e:
            if check:
                if isinstance(e, expected_array):
                    pass
            elif not check:
                if isinstance(e, expected_nocheck):
                    pass
            else:
                raise
        else:
            if check:
                try:
                    if issubclass(expected_array, BaseException):
                        raise Exception(
                            f"The test should have raised an exceptionof type {expected_array}"
                        )
                except TypeError:
                    pass

                try:
                    np.testing.assert_array_almost_equal(
                        win_array, expected_array, decimal=testing_decimal
                    )
                except TypeError as err:
                    raise Exception(f"Type error on {expected_array}") from err
            else:
                try:
                    if issubclass(expected_nocheck, BaseException):
                        raise Exception(
                            f"The test should have raised an exceptionof type {expected_nocheck}. "
                            f"Instead it returns {win_array}"
                        )
                except TypeError:
                    pass

                try:
                    np.testing.assert_array_almost_equal(
                        win_array, expected_nocheck, decimal=testing_decimal
                    )
                except TypeError as err:
                    raise Exception(f"Type error on {expected_nocheck}") from err

    def test_window_check(self):
        """Test the window_check method"""
        # test bidimensional array
        nrow, ncol = 4, 7
        data2d = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))

        # Test dimensions
        try:
            window_check(data2d, win=None, axes=None)
        except ValueError:
            # It is expected because win is considered as scalar.
            pass
        else:
            raise Exception("check should not have passed because win is scalar")

        try:
            window_check(data2d[0], win=[(0, 3), (0, 6)], axes=None)
        except ValueError:
            # It is expected because win has more elements (2) than the number
            # of dimension of data2d[0] (1)
            pass
        else:
            raise Exception("check should not have passed")

        try:
            window_check(data2d, win=[(0, 3)], axes=None)
        except ValueError:
            # It is expected because win has less elements (1) than the number
            # of dimension of data2d (2)
            pass
        else:
            raise Exception("check should not have passed")

        assert window_check(data2d[0], win=[(0, 3)], axes=None)

        # Test window

        assert window_check(data2d, win=[(0, 3), (0, 6)], axes=None)
        assert window_check(data2d, win=[(1, 2), (3, 3)], axes=None)
        assert ~window_check(data2d, win=[(1, 2), (3, 7)], axes=None)  # overflow on axe 1
        assert window_check(data2d, win=[(1, 2), (3, 7)], axes=0)  # check only on axe 0
        assert window_check(data2d, win=[(1, 2), (3, 7)], axes=(0,))  # check only on axe 0
        assert ~window_check(data2d, win=[(1, 2), (3, 7)], axes=1)  # check only on axe 1

        # test empty arrays
        assert ~window_check(np.empty((0, 0)), win=([0, 0]), axes=None)  # empty array
        assert ~window_check(np.empty(1), win=[[]], axes=None)  # empty window

        # test window order
        try:
            assert ~window_check(data2d, win=[(0, 3), (6, 0)], axes=None)
        except Exception:
            # It is expected here because the second dimension order has been reversed
            pass
        else:
            raise Exception("window order check should not have passed")

    def test_window_extent(self):
        """Test the window_extent method"""
        # Test outer extent
        np.testing.assert_equal(
            window_extend(win=[(0, 30), (0, 60)], extent=[[1, 2], [3, 4]]), [(-1, 32), (-3, 64)]
        )
        # Test inner extent
        np.testing.assert_equal(
            window_extend(win=[(0, 30), (0, 60)], extent=[[1, 2], [3, 4]], reverse=True),
            [(1, 28), (3, 56)],
        )

    def test_window_overflow(self):
        """Test the window_overflow method"""
        # test a bidimensional array
        nrow, ncol = 4, 7
        data2d = np.arange(nrow * ncol, dtype=np.float32).reshape((nrow, ncol))
        # test case : window covers all data
        np.testing.assert_equal(
            window_overflow(arr=data2d, win=[(0, 3), (0, 6)], axes=None), [(0, 0), (0, 0)]
        )
        # test case : window 1st dimension is greater (right) than the data 1st dimension
        np.testing.assert_equal(
            window_overflow(arr=data2d, win=[(0, 4), (0, 6)], axes=None), [(0, 1), (0, 0)]
        )
        # test case :
        #    - window 1st dimension is greater (right) than the data 1st dimension and
        #    - window 2nd dimension is greater (left and right) than the data 2nd dimension
        #    - test performed on all axes
        np.testing.assert_equal(
            window_overflow(arr=data2d, win=[(0, 4), (-4, 15)], axes=None), [(0, 1), (4, 9)]
        )
        # test case :
        #    - window 1st dimension is greater (right) than the data 1st dimension and
        #    - window 2nd dimension is greater (left and right) than the data 2nd dimension
        #    - test performed on 1st axe only => expect 0 overflow on second axe.
        np.testing.assert_equal(
            window_overflow(arr=data2d, win=[(0, 4), (-4, 15)], axes=0), [[0, 1], [0, 0]]
        )

    
    # -------------------------------------------------------------------------
    # Test compose_slice
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "N, outer, inner",
        [
            # --- simple / nominal cases ---
            (10, slice(1, None, 2), slice(0, None, 3)), # pure decimation
            (10, slice(2, 8), slice(1, None, 2)), # windowing + decimation
            (10, slice(None), slice(None)), # two no-op slices
            (10, slice(None), slice(2, None, 2)), # no-op outer
            (10, slice(2, 8), slice(None)), # no-op inner

            # --- N == 0 / empty lists ---
            (0, slice(None), slice(None)),
            (0, slice(1, 5), slice(0, None, 2)),

            # --- N == 1 (negative steps) ---
            (1, slice(None), slice(None)),
            (1, slice(1, -20, -2), slice(-20, 20, None)),
            (1, slice(0, 1), slice(0, None, -1)),

            # --- outer/inner that yield an empty result ---
            (10, slice(5, 5), slice(None)), # outer already empty
            (10, slice(2, 8), slice(3, 3)), # empty inner
            (10, slice(8, 2), slice(None)), # empty outer (reversed bounds, +step)
            (10, slice(None), slice(5, 2)), # empty inner (reversed bounds, +step)

            # --- out-of-bounds offset/factor ---
            (10, slice(100, None), slice(None)), # outer start > N
            (10, slice(None, None, 3), slice(100, None)), # inner start > L_o
            (10, slice(-100, -1), slice(None)), # outer start very negative

            # --- combined negative steps ---
            (10, slice(None, None, -1), slice(None, None, -1)), # double reversal
            (10, slice(8, 2, -1), slice(1, None, 2)),
            (10, slice(None), slice(None, None, -1)),
            (5, slice(-2, -20, -2), slice(0, None, 1)),
            (3, slice(2, -100, -2), slice(-100, 2, 1)),

            # --- combined steps producing a large step_c ---
            (100, slice(1, None, 3), slice(2, None, 5)),

            # --- explicit None everywhere ---
            (10, slice(None, None, None), slice(None, None, None)),
        ],
    )
    def test_compose_slice_matches_chained_indexing(self, N, outer, inner):
        arr = list(range(N))
        expected = arr[outer][inner]
        combined = compose_slice(outer, inner, N)
        assert arr[combined] == expected


    def test_compose_slice_random_exhaustive_domain_positive_and_negative(self):
        """Large random sweep, including negative and out-of-bounds offsets/steps."""
        random.seed(42)
        Ns = [0, 1, 2, 3, 5, 10]
        vals = [None, -100, -20, -7, -3, -2, -1, 0, 1, 2, 3, 7, 20, 100]
        steps = [None, -5, -2, -1, 1, 2, 5]

        for _ in range(20000):
            N = random.choice(Ns)
            arr = list(range(N))

            outer = slice(random.choice(vals), random.choice(vals), random.choice(steps))
            expected_after_outer = arr[outer]

            inner = slice(random.choice(vals), random.choice(vals), random.choice(steps))
            expected = expected_after_outer[inner]

            combined = compose_slice(outer, inner, N)
            got = arr[combined]

            assert got == expected, (
                f"N={N}, outer={outer}, inner={inner} "
                f"-> expected={expected}, got={got}, combined={combined}"
            )


class TestWindowNormalize:

    def test_none_is_the_whole_array(self):
        assert window_normalize(None, (50, 60)).tolist() == [[0, 49], [0, 59]]

    def test_bounds_are_inclusive_and_integer(self):
        window = window_normalize(((10, 20), (30, 40)), (50, 60))
        assert window.tolist() == [[10, 20], [30, 40]]
        assert np.issubdtype(window.dtype, np.integer)

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4])
    def test_any_rank(self, ndim):
        shape = tuple(range(10, 10 + ndim))
        assert window_normalize(None, shape).shape == (ndim, 2)

    @pytest.mark.parametrize(
        ("win", "match"),
        [
            (((10, 20),), "shape"),
            (((10, 20), (30, 40), (0, 1)), "shape"),
            (((20, 10), (30, 40)), "empty"),
            (((-1, 20), (30, 40)), "not contained"),
            (((10, 50), (30, 40)), "not contained"),
            (((10, 20), (30, 60)), "not contained"),
        ],
        ids=["too-few", "too-many", "empty", "negative", "past-rows", "past-cols"],
    )
    def test_malformed(self, win, match):
        with pytest.raises(ValueError, match=match):
            window_normalize(win, (50, 60))

    @pytest.mark.parametrize(
        "win", [((10.0, 20.0), (30.0, 40.0)), ((None, None), (30, 40))], ids=["float", "none"]
    )
    def test_non_integer_bounds_are_a_type_error(self, win):
        with pytest.raises(TypeError, match="integers only"):
            window_normalize(win, (50, 60))