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
"""
Regression tests for the gridr.chain.grid_resampling_chain

Command to run test :
PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest -s \
tests/python/regression/gridr/chain/test_regression_grid_resampling_chain.py

Note : you may have to run multiple times with the reference code in order to
generate reference data
Reference data will be stored in :
tests/python/regression/_regression_data/gridr/chain/test_regression_grid_resampling_chain
"""
import copy
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
import rasterio

from gridr.chain.grid_resampling_chain import basic_grid_resampling_chain
from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.core.grid.grid_mask import Validity
from gridr.io.common import safe_raster_open
from gridr.misc.mandrill import mandrill

UNMASKED_VALUE = Validity.VALID
MASKED_VALUE = Validity.INVALID


CANONICAL_PARAMS = {
    "method": [
        "nearest",
        "linear",
        "cubic",
        "bspline3",
        "bspline5",
        "bspline7",
        "bspline9",
        "bspline11",
    ],
}

PARAM_MAP = {
    "gridr.chain.grid_resampling_chain.basic_grid_resampling_chain": {
        "method": {
            "nearest": {"interp": "nearest"},
            "linear": {
                "interp": "linear",
            },
            "cubic": {
                "interp": "cubic",
            },
            "bspline3": {
                "interp": "bspline3",
                "interp_kwargs": {
                    "epsilon": 1e-4,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline5": {
                "interp": "bspline5",
                "interp_kwargs": {
                    "epsilon": 1e-4,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline7": {
                "interp": "bspline7",
                "interp_kwargs": {
                    "epsilon": 1e-4,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline9": {
                "interp": "bspline9",
                "interp_kwargs": {
                    "epsilon": 1e-4,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline11": {
                "interp": "bspline11",
                "interp_kwargs": {
                    "epsilon": 1e-4,
                    "mask_influence_threshold": 1,
                },
            },
        }
    },
}

RESOLUTIONS = [
    (1, 1),
    (5, 7),
]


def write_array(array, dtype, fileout):
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


def create_grid_generic(
    nrow,
    ncol,
    origin_pos,
    origin_node,
    v_row_y,
    v_row_x,
    v_col_y,
    v_col_x,
    random_seed,
    random_std,
    dtype,
):
    """ """
    x = np.arange(0, ncol, dtype=dtype)
    y = np.arange(0, nrow, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    xx -= origin_pos[0]
    yy -= origin_pos[1]
    yyy = origin_node[0] + yy * v_row_y + xx * v_col_y
    xxx = origin_node[1] + yy * v_row_x + xx * v_col_x

    if random_seed is not None:
        rng = np.random.default_rng(seed=random_seed)
        shift_x = rng.standard_normal(xxx.size).reshape(xxx.shape) * random_std
        shift_y = rng.standard_normal(xxx.size).reshape(xxx.shape) * random_std
        xxx += shift_x
        yyy += shift_y
    return yyy, xxx


def shape2(array):
    if array.ndim == 3:
        return (array.shape[1], array.shape[2])
    else:
        return array.shape


@pytest.fixture(scope="session")
def input_data_001_mandrill_grid_f64(tmp_path_factory):
    """Create input data"""
    print("\n --> create input data : input_data_001_mandrill_grid_f64")
    input_raster_dtype = mandrill.dtype
    input_grid_dtype = np.float64

    tmp_dir = tmp_path_factory.mktemp("raster")

    ret = {
        "raster_in_path": tmp_dir / "test_raster_in.tif",
        "raster_in_nbands": mandrill.shape[0],
        "raster_mask_in_path": tmp_dir / "test_raster_mask_in.tif",
        "raster_mask_in_band": 1,
        "grid_in_path": {"tif": tmp_dir / "test_grid_in.tif"},
        "grid_in_shape": (50, 40),
        "grid_mask_flag": True,
        "grid_mask_in_path": tmp_dir / "test_grid_mask_in.tif",
        "grid_mask_in_band": 1,
        "grid_mask_in_unmasked_value": 1,
        "grid_in_nodata": None,
    }

    # write input raster as tif
    write_array(mandrill, dtype=input_raster_dtype, fileout=ret["raster_in_path"])

    # create grid
    grid_row, grid_col = create_grid_generic(
        nrow=ret["grid_in_shape"][0],
        ncol=ret["grid_in_shape"][1],
        origin_pos=np.array((0.3, 0.2)),
        origin_node=np.array((0.0, 0.0)),
        v_row_y=5.2,
        v_row_x=1.2,
        v_col_y=-2.7,
        v_col_x=7.1,
        random_seed=1,
        random_std=0.01,
        dtype=input_grid_dtype,
    )

    # create input raster mask
    mask_in = np.ones((mandrill.shape[1], mandrill.shape[2]), dtype=np.uint8)
    rows_idx = np.array([grid_row[10, 10], grid_row[10, 12], grid_row[13, 12], grid_row[13, 10]])
    cols_idx = np.array([grid_col[10, 10], grid_col[10, 12], grid_col[13, 12], grid_col[13, 10]])
    row_min = int(np.max((0, np.floor(np.min(rows_idx)))))
    row_max = int(np.min((mandrill.shape[1], np.ceil(np.max(rows_idx)))))
    col_min = int(np.max((0, np.floor(np.min(cols_idx)))))
    col_max = int(np.min((mandrill.shape[2], np.ceil(np.max(cols_idx)))))
    mask_in[row_min:row_max, col_min:col_max] = 0
    print(
        "----> input data mask in invalid window : "
        f"rows: {row_min} -> {row_max} // cols : {col_min} -> {col_max}"
    )

    write_array(mask_in, dtype=np.uint8, fileout=ret["raster_mask_in_path"])

    # write grid as tif
    write_array(
        np.array([grid_row, grid_col]), dtype=input_grid_dtype, fileout=ret["grid_in_path"]["tif"]
    )

    # create grid mask
    grid_mask = np.ones(grid_row.shape, dtype=np.uint8)
    grid_mask[1, 2] = 0
    # write mask
    write_array(grid_mask, dtype=grid_mask.dtype, fileout=ret["grid_mask_in_path"])

    return ret


@pytest.fixture(scope="session")
def input_data(request):
    """Create the requested input data"""
    return request.getfixturevalue(request.param)


class GridrAdapter(Protocol):
    name = "gridr.chain.grid_resampling_chain.basic_grid_resampling_chain"

    def __init__(self):
        pass

    def prepare(self, request, params: dict[str, Any], work_dir):
        """ """
        pmap = PARAM_MAP[self.name]
        self._output_dir = work_dir

        self._kwargs = copy.deepcopy(pmap["method"][params["method"]])

        inputs = params["input_data"]
        self._input_image_path = inputs["raster_in_path"]
        self._input_image_nbands = inputs["raster_in_nbands"]

        if params["mask_in"]:
            self._input_image_mask_path = inputs["raster_mask_in_path"]
            self._kwargs["array_src_mask_band"] = inputs["raster_mask_in_band"]
            self._kwargs["array_src_mask_validity_pair"] = (1, 0)
        else:
            self._input_image_mask_path = None

        self._grid_in_path_dict = inputs["grid_in_path"]
        self._grid_in_shape = inputs["grid_in_shape"]

        if params["grid_mask_in"]:
            self._grid_mask_in_path = inputs["grid_mask_in_path"]
            self._kwargs["grid_mask_in_unmasked_value"] = inputs["grid_mask_in_unmasked_value"]
            self._kwargs["grid_mask_in_band"] = inputs["grid_mask_in_band"]
        else:
            self._grid_mask_in_path = None
        
        self._boundary_condition = params["boundary_condition"]

        # self._grid_mask_flag = inputs["grid_mask_flag"]
        # self._grid_mask_in_masked_value = inputs["grid_mask_in_masked_value"]
        # self._grid_in_nodata = inputs["grid_in_nodata"]

        self._output_image_path = self._output_dir / "image_out.tif"
        self._output_mask_path = None

        self._resolution = params["resolution"]
        self._output_shape = grid_full_resolution_shape(
            shape=self._grid_in_shape,
            resolution=self._resolution,
        )

        count = 1
        if not params["multi_bands"]:
            self._kwargs["array_src_bands"] = 1
            count = 1
        else:
            self._kwargs["array_src_bands"] = list(range(1, 1 + self._input_image_nbands))
            count = self._input_image_nbands

        self._output_image_profile = {
            "driver": "GTiff",
            "dtype": np.float64,
            "height": self._output_shape[0],
            "width": self._output_shape[1],
            "count": count,
        }
        self._output_mask_profile = {
            "driver": "GTiff",
            "dtype": np.uint8,
            "height": self._output_shape[0],
            "width": self._output_shape[1],
            "count": 1,
            "nbit": 1,
        }
        if params["mask_out"]:
            self._output_mask_path = Path(self._output_dir) / "mask_out.tif"

        # For GridR we use the TIF input grid
        self._grid_in_path = self._grid_in_path_dict["tif"]

        self._logger = None

    def run(self, ndarrays_regression, request):
        test_id = request.node.name
        with (
            rasterio.open(self._grid_in_path, "r") as grid_in_ds,
            rasterio.open(self._input_image_path, "r") as array_src_ds,
            rasterio.open(
                self._output_image_path, "w", **self._output_image_profile
            ) as array_out_ds,
            safe_raster_open(
                self._output_mask_path, "w", **self._output_mask_profile
            ) as mask_out_ds,
            safe_raster_open(self._grid_mask_in_path, "r") as grid_mask_in_ds,
            safe_raster_open(self._input_image_mask_path, "r") as array_src_mask_ds,
        ):
            # if grid_mask_in_band is not None:
            #    grid_mask_in_ds = grid_in_ds
            _ = basic_grid_resampling_chain(
                grid_ds=grid_in_ds,
                grid_row_coords_band=1,
                grid_col_coords_band=2,
                grid_resolution=self._resolution,
                array_src_ds=array_src_ds,
                # array_src_bands=1,
                array_out_ds=array_out_ds,
                boundary_condition=self._boundary_condition,
                nodata_out=0,
                win=None,
                mask_out_ds=mask_out_ds,
                grid_mask_in_ds=grid_mask_in_ds,
                # grid_mask_in_unmasked_value=grid_mask_in_unmasked_value,
                # grid_mask_in_band=grid_mask_in_band,
                array_src_mask_ds=array_src_mask_ds,
                # array_src_mask_band=1,
                # array_src_mask_validity_pair=(
                #    array_src_mask_validity_valid,
                #    array_src_mask_validity_invalid,
                # ),
                # array_src_geometry_origin=array_src_geometry_origin,
                # array_src_geometry_pair=[geometry_valid, geometry_invalid],
                # io_strip_size=processing_ressources["io_strip_size"],
                # io_strip_size_target=GridRIOMode.OUTPUT,
                # tile_shape=processing_ressources["tile_shape"],
                logger=self._logger,
                **self._kwargs,
            )

        with (
            rasterio.open(self._output_image_path, "r") as array_out_ds,
            safe_raster_open(self._output_mask_path, "r") as mask_out_ds,
        ):
            if mask_out_ds is not None:
                ndarrays_regression.check(
                    {
                        "array_out": array_out_ds.read(),
                    },
                    basename=f"{test_id}_array_out",
                    default_tolerance={"atol": 1e-10, "rtol": 1e-8},
                )
                ndarrays_regression.check(
                    {
                        "mask_out": mask_out_ds.read(),
                    },
                    basename=f"{test_id}_mask_out",
                    default_tolerance={"atol": 0, "rtol": 0},
                )
            else:
                ndarrays_regression.check(
                    {
                        "array_out": array_out_ds.read(),
                    },
                    basename=f"{test_id}_array_out",
                    default_tolerance={"atol": 1e-10, "rtol": 1e-8},
                )


@pytest.mark.parametrize(
    "adapter",
    [
        GridrAdapter(),
    ],
    ids=[
        0,
    ],
)
@pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
@pytest.mark.parametrize(
    "resolution", RESOLUTIONS, ids=[f"res{i}" for i in range(len(RESOLUTIONS))]
)
@pytest.mark.parametrize(
    "input_data",
    [
        "input_data_001_mandrill_grid_f64",
    ],
    ids=[
        f"data{0}",
    ],
    indirect=True,
)
@pytest.mark.parametrize("mask_in", [False, True], ids=["mi0", "mi1"])
@pytest.mark.parametrize("grid_mask_in", [False, True], ids=["gmi0", "gmi1"])
@pytest.mark.parametrize("mask_out", [False, True], ids=["mo0", "mo1"])
@pytest.mark.parametrize("multi_bands", [True, False], ids=["mul1", "mul0"])
@pytest.mark.parametrize("boundary_condition", [None, "reflect",],)
def test_grid_resampling(
    ndarrays_regression,
    request,
    adapter,
    method,
    input_data,
    resolution,
    mask_out,
    multi_bands,
    grid_mask_in,
    mask_in,
    boundary_condition,
    tmp_path,
):
    """ """
    adapter.prepare(
        request,
        params={
            "method": method,
            "mask_out": mask_out,
            "resolution": resolution,
            "input_data": input_data,
            "multi_bands": multi_bands,
            "mask_in": mask_in,
            "grid_mask_in": grid_mask_in,
            "boundary_condition": boundary_condition,
        },
        work_dir=tmp_path,
    )
    adapter.run(ndarrays_regression, request)
