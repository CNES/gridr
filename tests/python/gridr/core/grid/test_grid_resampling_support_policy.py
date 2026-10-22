#
# Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests of the ``support_policy`` parameter of ``array_grid_resampling``.

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/core/grid/test_grid_resampling_support_policy.py
"""  # noqa: E501

import numpy as np
import pytest

from gridr.core.grid.grid_mask import Validity
from gridr.core.grid.grid_resampling import array_grid_resampling

NODATA_OUT = -9999.0
NROW, NCOL = 6, 7
MASKED_PIXEL = (2, 3)


def _input() -> tuple[np.ndarray, np.ndarray]:
    """Build the input image and its mask.

    Returns
    -------
    tuple of numpy.ndarray
        The ``(NROW, NCOL)`` float64 image ``10 * row + col`` and its uint8
        mask, valid everywhere except at ``MASKED_PIXEL``.
    """
    rows, cols = np.mgrid[0:NROW, 0:NCOL]
    array_in = (10.0 * rows + cols).astype(np.float64)
    mask = np.full((NROW, NCOL), Validity.VALID, dtype=np.uint8)
    mask[MASKED_PIXEL] = Validity.INVALID
    return array_in, mask


def _resample(
    grid_row: np.ndarray, grid_col: np.ndarray, interp: str, support_policy: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the test input on a full-resolution grid.

    Parameters
    ----------
    grid_row, grid_col : numpy.ndarray
        Target coordinates, in the index space of the input image.
    interp : str
        Interpolator identifier.
    support_policy : str
        Support policy passed to ``array_grid_resampling``.

    Returns
    -------
    tuple of numpy.ndarray
        The output image and its validity mask.
    """
    array_in, mask = _input()
    return array_grid_resampling(
        interp=interp,
        array_in=array_in,
        grid_row=grid_row,
        grid_col=grid_col,
        grid_resolution=(1, 1),
        array_out=None,
        nodata_out=NODATA_OUT,
        array_in_mask=mask,
        array_out_mask=True,
        standalone=False,
        support_policy=support_policy,
    )


def _identity_grid() -> tuple[np.ndarray, np.ndarray]:
    """Build the identity grid: every target falls on an input sample.

    Returns
    -------
    tuple of numpy.ndarray
        Row and column coordinates, float64, shape ``(NROW, NCOL)``.
    """
    rows, cols = np.mgrid[0:NROW, 0:NCOL]
    return rows.astype(np.float64), cols.astype(np.float64)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_effective_identity_keeps_valid_samples(interp):
    """On the identity grid, an interpolating kernel only uses the target
    sample under the effective support policy: the output equals the input
    wherever the input is valid, up to the border."""
    array_in, mask = _input()
    array_out, mask_out = _resample(*_identity_grid(), interp, "effective")
    np.testing.assert_array_equal(mask_out, mask)
    valid = mask == Validity.VALID
    np.testing.assert_allclose(array_out[valid], array_in[valid], rtol=0, atol=1e-12)
    assert np.all(array_out[~valid] == NODATA_OUT)


def test_full_identity_linear():
    """On the identity grid, the full support policy requires the whole 2 x 2
    linear stencil (rows r..r+1, columns c..c+1) to be inside the image and
    valid."""
    _, mask = _input()
    _, mask_out = _resample(*_identity_grid(), "linear", "full")
    expected = np.full((NROW, NCOL), Validity.INVALID, dtype=np.uint8)
    for r in range(NROW - 1):
        for c in range(NCOL - 1):
            if np.all(mask[r : r + 2, c : c + 2] == Validity.VALID):
                expected[r, c] = Validity.VALID
    np.testing.assert_array_equal(mask_out, expected)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_full_is_a_subset_of_effective(interp):
    """On sub-pixel targets, every output valid under the full support policy is
    valid under the effective one, with the same value."""
    rng = np.random.default_rng(12)
    grid_row = rng.uniform(-0.5, NROW - 0.5, size=(NROW, NCOL))
    grid_col = rng.uniform(-0.5, NCOL - 0.5, size=(NROW, NCOL))
    grid_row[0, :] = np.round(grid_row[0, :])
    out_e, mask_e = _resample(grid_row, grid_col, interp, "effective")
    out_f, mask_f = _resample(grid_row, grid_col, interp, "full")
    full_valid = mask_f == Validity.VALID
    assert np.all(mask_e[full_valid] == Validity.VALID)
    np.testing.assert_array_equal(out_f[full_valid], out_e[full_valid])
    assert np.all(out_f[~full_valid] == NODATA_OUT)


def test_unknown_policy_raises():
    """An unknown policy name is rejected."""
    with pytest.raises(ValueError, match="support policy"):
        _resample(*_identity_grid(), "linear", "loose")
