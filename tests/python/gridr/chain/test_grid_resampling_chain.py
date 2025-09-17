# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://gitlab.cnes.fr/gridr/gridr).
#
#
"""
Tests for the gridr.chain.grid_resampling_chain

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/chain/test_grid_resampling_chain.py
"""
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely

from gridr.chain.grid_resampling_chain import GEOMETRY_RASTERIZE_KWARGS, basic_grid_resampling_chain
from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.core.grid.grid_mask import Validity, build_mask
from gridr.core.grid.grid_resampling import array_grid_resampling
from gridr.core.utils.array_utils import array_replace
from gridr.io.common import GridRIOMode, safe_raster_open
from gridr.misc.mandrill import mandrill

UNMASKED_VALUE = Validity.VALID
MASKED_VALUE = Validity.INVALID


def create_grid(
    nrow, ncol, origin_pos, origin_node, v_row_y, v_row_x, v_col_y, v_col_x, grid_dtype
):
    """ """
    x = np.arange(0, ncol, dtype=grid_dtype)
    y = np.arange(0, nrow, dtype=grid_dtype)
    xx, yy = np.meshgrid(x, y)

    xx -= origin_pos[0]
    yy -= origin_pos[1]

    yyy = origin_node[0] + yy * v_row_y + xx * v_col_y
    xxx = origin_node[1] + yy * v_row_x + xx * v_col_x

    return yyy, xxx


def write_array(array, dtype, fileout):
    """ """
    if array.ndim == 3:
        with rasterio.open(
            fileout,
            "w",
            driver="GTiff",
            dtype=dtype,
            height=array.shape[1],
            width=array.shape[2],
            count=array.shape[0],
        ) as array_in_ds:
            for band in range(array.shape[0]):
                array_in_ds.write(array[band].astype(dtype), band + 1)
            array_in_ds = None
    elif array.ndim == 2:
        with rasterio.open(
            fileout,
            "w",
            driver="GTiff",
            dtype=dtype,
            height=array.shape[0],
            width=array.shape[1],
            count=1,
        ) as array_in_ds:
            array_in_ds.write(array.astype(dtype), 1)
            array_in_ds = None


def shape2(array):
    if array.ndim == 3:
        return (array.shape[1], array.shape[2])
    else:
        return array.shape


class TestGridResamplingChain:
    """Test class"""

    def _generic_test_build_mask_chain(
        self,
        request,
        interp,
        array_in,
        array_in_mask_positions,
        array_in_mask_validity_pair,
        array_in_mask_type,
        array_in_bands,
        array_in_dtype,
        grid_vec_row,
        grid_vec_col,
        grid_origin_pos,
        grid_origin_node,
        grid_dtype,
        grid_mask,
        grid_shape,
        grid_resolution,
        window,
        array_in_geometry_origin,
        array_in_geometry_pair,
        mask_out,
        io_strip_size,
        io_strip_size_target,
        tile_shape,
    ):
        """Test the grid_resampling_chain - compare results with results in the core method"""
        test_id = request.node.nodeid.split("::")[-1].replace("[", "-").replace("]", "")
        test_id = hashlib.md5(test_id.encode("utf-8")).hexdigest()[:8]

        output_dir = tempfile.mkdtemp(suffix=test_id, prefix=None, dir=None)
        array_in_path = Path(output_dir) / "array_in.tif"
        grid_in_path = Path(output_dir) / "grid_in.tif"
        array_out_path = Path(output_dir) / "array_out.tif"
        grid_mask_in_path = None
        array_mask_in_path = None
        array_mask_out_path = None

        # computation_dtype = np.float64
        output_dtype = np.float64

        try:
            # Write input array
            write_array(array_in, array_in_dtype, array_in_path)

            # Write input grid
            grid_row, grid_col = create_grid(
                nrow=grid_shape[0],
                ncol=grid_shape[1],
                origin_pos=np.asarray(grid_origin_pos),
                origin_node=np.asarray(grid_origin_node),
                v_row_y=grid_vec_row[0],
                v_row_x=grid_vec_row[1],
                v_col_y=grid_vec_col[0],
                v_col_x=grid_vec_col[1],
                grid_dtype=grid_dtype,
            )
            write_array(np.array([grid_row, grid_col]), grid_dtype, grid_in_path)

            # Create a grid mask if True
            if grid_mask:
                grid_mask_in_path = Path(output_dir) / "grid_mask_in.tif"
                grid_mask_array = np.ones(grid_row.shape, dtype=np.uint8) * UNMASKED_VALUE
                grid_mask_array[
                    grid_row.shape[0] // 8 : grid_row.shape[0] // 8 + 3,
                    grid_row.shape[1] // 10 : grid_row.shape[1] // 10 + 2,
                ] = MASKED_VALUE
                write_array(grid_mask_array, np.uint8, grid_mask_in_path)

            # Create array mask
            if array_in_mask_positions is not None:
                array_mask_in_path = Path(output_dir) / "array_mask_in.tif"
                array_mask = np.full(
                    shape2(array_in), array_in_mask_validity_pair[0], dtype=array_in_mask_type
                )
                for pos in array_in_mask_positions:
                    array_mask[pos[0], pos[1]] = array_in_mask_validity_pair[1]
                write_array(array_mask, array_in_mask_type, array_mask_in_path)

            # compute window and output shape
            full_output_shape = grid_full_resolution_shape(
                shape=grid_row.shape, resolution=grid_resolution
            )
            if window is None:
                window = np.array(((0, full_output_shape[0] - 1), (0, full_output_shape[1] - 1)))
                output_shape = full_output_shape
            else:
                output_shape = window[:, 1] - window[:, 0] + 1

            array_out_open_kwargs = {
                "driver": "GTiff",
                "dtype": output_dtype,
                "height": output_shape[0],
                "width": output_shape[1],
                "count": len(array_in_bands),
            }

            # Check if we want a mask out
            mask_out_open_kwargs = {}
            if mask_out:
                array_mask_out_path = Path(output_dir) / "array_mask_out.tif"
                mask_out_open_kwargs = {
                    "driver": "GTiff",
                    "dtype": np.uint8,
                    "height": output_shape[0],
                    "width": output_shape[1],
                    "count": 1,
                    "nbits": 1,
                }

            array_out_validate, mask_out_validate = None, None
            #F = 1
            with (
                rasterio.open(grid_in_path, "r") as grid_in_ds,
                rasterio.open(array_in_path, "r") as array_in_ds,
                rasterio.open(array_out_path, "w", **array_out_open_kwargs) as array_out_ds,
                safe_raster_open(grid_mask_in_path) as grid_mask_in_ds,
                safe_raster_open(array_mask_in_path) as array_mask_in_ds,
                safe_raster_open(
                    array_mask_out_path, "w", **mask_out_open_kwargs
                ) as array_mask_out_ds,
            ):

                basic_grid_resampling_chain(
                    grid_ds=grid_in_ds,
                    grid_col_ds=grid_in_ds,
                    grid_row_coords_band=1,
                    grid_col_coords_band=2,
                    grid_resolution=grid_resolution,
                    array_src_ds=array_in_ds,
                    array_src_bands=array_in_bands,
                    array_src_mask_ds=array_mask_in_ds,
                    array_src_mask_band=1,
                    array_src_mask_validity_pair=array_in_mask_validity_pair,
                    array_out_ds=array_out_ds,
                    interp=interp,
                    nodata_out=0,
                    win=window,
                    mask_out_ds=array_mask_out_ds,
                    grid_mask_in_ds=grid_mask_in_ds,
                    grid_mask_in_unmasked_value=UNMASKED_VALUE,
                    grid_mask_in_band=1,
                    # computation_dtype = computation_dtype,
                    array_src_geometry_origin=array_in_geometry_origin,
                    array_src_geometry_pair=array_in_geometry_pair,
                    # grid_geometry_origin = (0, 0),
                    io_strip_size=io_strip_size,  # 10000,
                    io_strip_size_target=io_strip_size_target,
                    tile_shape=tile_shape,
                )

                # Compute the same thing with the core method

                # If array_src_geometry_pair is not None we have to compute
                # a raster mask
                validate_array_in_mask = array_mask_in_ds.read(1) if array_mask_in_ds else None
                
                if validate_array_in_mask is not None:
                    array_replace(
                        validate_array_in_mask,
                        array_in_mask_validity_pair[0],
                        Validity.VALID,
                        Validity.INVALID,
                    )
                    validate_array_in_mask = validate_array_in_mask.view(np.uint8)

                if array_in_geometry_pair is not None and (
                    array_in_geometry_pair[0] is not None or array_in_geometry_pair[1] is not None
                ):
                    # rasterize mask on full input array domain
                    geometry_mask = build_mask(
                        shape=(array_in_ds.height, array_in_ds.width),
                        resolution=(1, 1),
                        out=None,
                        geometry_origin=array_in_geometry_origin,
                        geometry_pair=array_in_geometry_pair,
                        mask_in=None,  # The current mask is passed as `out`
                        mask_in_target_win=None,
                        mask_in_resolution=None,
                        oversampling_dtype=None,
                        mask_in_binary_threshold=None,
                        rasterize_kwargs=GEOMETRY_RASTERIZE_KWARGS,
                        init_out=False,
                    )

                    if validate_array_in_mask is not None:
                        validate_array_in_mask &= geometry_mask
                    else:
                        validate_array_in_mask = geometry_mask

                array_out_validate, mask_out_validate = array_grid_resampling(
                    interp=interp,
                    array_in=np.asarray(
                        [array_in_ds.read(b).astype(np.float64) for b in array_in_bands]
                    ),
                    grid_row=grid_in_ds.read(1),
                    grid_col=grid_in_ds.read(2),
                    grid_resolution=grid_resolution,
                    array_out=None,
                    array_out_win=None,
                    nodata_out=0,
                    array_in_origin=None,
                    win=window,
                    array_in_mask=validate_array_in_mask,
                    grid_mask=grid_mask_in_ds.read(1) if grid_mask_in_ds else None,
                    grid_mask_valid_value=UNMASKED_VALUE,
                    grid_nodata=None,
                    array_out_mask=mask_out,
                    check_boundaries=True,
                )

            with (
                rasterio.open(array_out_path, "r") as array_out_ds,
                safe_raster_open(array_mask_out_path, "r") as array_mask_out_ds,
            ):

                np.testing.assert_allclose(
                    np.squeeze(array_out_ds.read()),
                    array_out_validate,
                    rtol=1e-7,
                    atol=1e-8,
                    err_msg="Output image computed with the basic_grid_resampling_chain differs"
                    "from the image computed with the corresponding core method",
                )

                if mask_out:
                    np.testing.assert_array_equal(
                        np.squeeze(array_mask_out_ds.read()),
                        mask_out_validate,
                        err_msg="Output mask computed with the basic_grid_resampling_chain differs"
                        "from the mask computed with the corresponding core method",
                    )

        finally:
            # Clear
            os.unlink(array_in_path)
            os.unlink(grid_in_path)
            os.unlink(array_out_path)
            if grid_mask_in_path:
                os.unlink(grid_mask_in_path)
            if array_mask_in_path:
                os.unlink(array_mask_in_path)
            if array_mask_out_path:
                os.unlink(array_mask_out_path)
            os.rmdir(output_dir)

    @pytest.mark.parametrize("interp", ["cubic"])
    @pytest.mark.parametrize(
        "array_in, array_in_mask_positions, array_in_bands",
        [
            (
                mandrill[0],
                None,
                [
                    1,
                ],
            ),
            (mandrill, None, [1, 2, 3]),
            (
                mandrill[0],
                [
                    (60, 160),
                ],
                [
                    1,
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("array_in_dtype", [np.float64])
    @pytest.mark.parametrize(
        "grid_vec_row, grid_vec_col, grid_origin_pos, grid_origin_node, grid_dtype, grid_mask",
        [
            ((5.2, 1.2), (-2.7, 7.1), (0.3, 0.2), (0.0, 0.0), np.float64, True),
            ((5.2, 1.2), (-2.7, 7.1), (0.3, 0.2), (0.0, 0.0), np.float64, False),
        ],
    )
    @pytest.mark.parametrize(
        "grid_shape, grid_resolution, window",
        [
            ((50, 40), (10, 10), None),
        ],
    )
    @pytest.mark.parametrize(
        "array_in_geometry_origin, array_in_geometry_pair",
        [
            (None, None),
            (
                (0.5, 0.5),
                (
                    None,
                    shapely.geometry.Polygon(
                        [(10.5, 12.5), (30.5, 12.5), (30.5, 40.5), (10.5, 40.5)]
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("mask_out", [True, False])
    # @pytest.mark.parametrize("grid_resolution", [(3, 4),])
    @pytest.mark.parametrize(
        "io_strip_size, io_strip_size_target, tile_shape",
        [
            (100, GridRIOMode.OUTPUT, None),
            (100, GridRIOMode.OUTPUT, (80, 200)),
            (100, GridRIOMode.INPUT, (80, 200)),
            (100, GridRIOMode.OUTPUT, (100, 200)),
            (100, GridRIOMode.OUTPUT, (102, 200)),
            (1000, GridRIOMode.OUTPUT, (1000, 200)),
            (1000, GridRIOMode.OUTPUT, (1000, 1000)),
        ],
    )
    # @pytest.mark.parametrize("ncpu", [1, 2])
    # @pytest.mark.parametrize("cpu_tile_shape", [(1000,1000),])
    # @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    def test_build_mask_chain(
        self,
        request,
        interp,
        array_in,
        array_in_mask_positions,
        array_in_bands,
        array_in_dtype,
        grid_vec_row,
        grid_vec_col,
        grid_origin_pos,
        grid_origin_node,
        grid_dtype,
        grid_mask,
        grid_shape,
        grid_resolution,
        window,
        array_in_geometry_origin,
        array_in_geometry_pair,
        mask_out,
        io_strip_size,
        io_strip_size_target,
        tile_shape,
    ):
        """Test the grid_resampling_chain - compare results with results in the core method"""
        self._generic_test_build_mask_chain(
            request=request,
            interp=interp,
            array_in=array_in,
            array_in_mask_positions=array_in_mask_positions,
            array_in_mask_validity_pair=(Validity.VALID, Validity.INVALID),
            array_in_mask_type=np.uint8,
            array_in_bands=array_in_bands,
            array_in_dtype=array_in_dtype,
            grid_vec_row=grid_vec_row,
            grid_vec_col=grid_vec_col,
            grid_origin_pos=grid_origin_pos,
            grid_origin_node=grid_origin_node,
            grid_dtype=grid_dtype,
            grid_mask=grid_mask,
            grid_shape=grid_shape,
            grid_resolution=grid_resolution,
            window=window,
            array_in_geometry_origin=array_in_geometry_origin,
            array_in_geometry_pair=array_in_geometry_pair,
            mask_out=mask_out,
            io_strip_size=io_strip_size,
            io_strip_size_target=io_strip_size_target,
            tile_shape=tile_shape,
        )

    @pytest.mark.parametrize("interp", ["cubic"])
    @pytest.mark.parametrize(
        "array_in, array_in_mask_positions, array_in_bands",
        [
            (
                mandrill[0],
                [
                    (60, 160),
                ],
                [
                    1,
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("array_in_dtype", [np.float64])
    @pytest.mark.parametrize(
        "grid_vec_row, grid_vec_col, grid_origin_pos, grid_origin_node, grid_dtype, grid_mask",
        [((5.2, 1.2), (-2.7, 7.1), (0.3, 0.2), (0.0, 0.0), np.float64, False)],
    )
    @pytest.mark.parametrize(
        "grid_shape, grid_resolution, window",
        [
            ((50, 40), (10, 10), None),
        ],
    )
    @pytest.mark.parametrize(
        "array_in_geometry_origin, array_in_geometry_pair",
        [
            (None, None),
            (
                (0.5, 0.5),
                (
                    None,
                    shapely.geometry.Polygon(
                        [(10.5, 12.5), (30.5, 12.5), (30.5, 40.5), (10.5, 40.5)]
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("mask_out", [True])
    # @pytest.mark.parametrize("grid_resolution", [(3, 4),])
    @pytest.mark.parametrize(
        "io_strip_size, io_strip_size_target, tile_shape",
        [
            (100, GridRIOMode.OUTPUT, (80, 200)),
        ],
    )
    @pytest.mark.parametrize(
        "array_in_mask_validity_pair, array_in_mask_type",
        [
            ((Validity.VALID, Validity.INVALID), np.uint8),
            ((Validity.VALID, Validity.INVALID), np.int8),
            ((0, -1), np.int8),
            ((1, -127), np.int8),
            ((-6, 8), np.int8),
            ((Validity.VALID, 5), np.uint8),
        ],
    )
    # @pytest.mark.parametrize("ncpu", [1, 2])
    # @pytest.mark.parametrize("cpu_tile_shape", [(1000,1000),])
    # @pytest.mark.parametrize("computation_dtype", [DTYPE_00,])
    def test_build_mask_chain_array_mask_values_and_dtype(
        self,
        request,
        interp,
        array_in,
        array_in_mask_positions,
        array_in_mask_validity_pair,
        array_in_mask_type,
        array_in_bands,
        array_in_dtype,
        grid_vec_row,
        grid_vec_col,
        grid_origin_pos,
        grid_origin_node,
        grid_dtype,
        grid_mask,
        grid_shape,
        grid_resolution,
        window,
        array_in_geometry_origin,
        array_in_geometry_pair,
        mask_out,
        io_strip_size,
        io_strip_size_target,
        tile_shape,
    ):
        """Test the grid_resampling_chain - compare results with results in the core method"""
        self._generic_test_build_mask_chain(
            request=request,
            interp=interp,
            array_in=array_in,
            array_in_mask_positions=array_in_mask_positions,
            array_in_mask_validity_pair=array_in_mask_validity_pair,
            array_in_mask_type=array_in_mask_type,
            array_in_bands=array_in_bands,
            array_in_dtype=array_in_dtype,
            grid_vec_row=grid_vec_row,
            grid_vec_col=grid_vec_col,
            grid_origin_pos=grid_origin_pos,
            grid_origin_node=grid_origin_node,
            grid_dtype=grid_dtype,
            grid_mask=grid_mask,
            grid_shape=grid_shape,
            grid_resolution=grid_resolution,
            window=window,
            array_in_geometry_origin=array_in_geometry_origin,
            array_in_geometry_pair=array_in_geometry_pair,
            mask_out=mask_out,
            io_strip_size=io_strip_size,
            io_strip_size_target=io_strip_size_target,
            tile_shape=tile_shape,
        )
