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
GridR Time Benchmark for chain array_grid_resampling

Command :
ORION_BIN_PATH=\
/work/ARTEMIS/cots_delivery/icc_co3d/qtispack_minipack_from_sif/bin/orion.sh \
ORION_INIT_ONLY_PATH=\
/work/ARTEMIS/cots_delivery/icc_co3d/qtispack_minipack_from_sif/bin/orion_init_only.sh \
PYTHONPATH=$PWD/python:$PYTHONPATH python \
-m pytest tests/python/benchmarks/time/test_time_chain_array_grid_resampling.py \
--benchmark-save="test_2" \
--benchmark-storage=./benchmarks/results/pytest/ --benchmark-group-by=group \
--benchmark-columns=min,median,mean,stddev,iqr,outliers,ops,rounds -v
"""
from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
import rasterio

from benchmarks.wrappers.grid_orion import grid_orion, grid_orion_init_only
from gridr.chain.grid_resampling_chain import basic_grid_resampling_chain
from gridr.core.grid.grid_commons import grid_full_resolution_shape
from gridr.io.common import safe_raster_open
from gridr.misc.mandrill import mandrill

USE_ORION = True
try:
    from artemis_io.formats import frmt_cnes_bsq
except ImportError:
    # Orion option will not be available
    USE_ORION = False
try:
    if os.environ.get("ORION_BIN_PATH") is None:
        USE_ORION = False
    if os.environ.get("ORION_INIT_ONLY_PATH") is None:
        USE_ORION = False
except KeyError:
    USE_ORION = False

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

NROUNDS = 1
NITERATIONS = 1
NWARMUP = 0
RESOLUTIONS = [
    (1, 1),
    (10, 10),
    (100, 100),
]

DO_TEST_001=False
DO_TEST_INIT_ORION=False
DO_TEST_002=False
DO_TEST_003=False # 12000x12000 size
DO_TEST_004=False # 4000x4000 size


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


def write_array_bsq(array, dtype, fileout, oversampling_row, oversampling_col):
    # Full write
    with frmt_cnes_bsq.CnesBsqDatasetWriter(
        filename=fileout,
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=dtype,
        header_info={"LABEL": "dummy", "PAS COL": oversampling_col, "PAS LIG": oversampling_row},
        byte_order="native",
    ) as writer:
        writer.write(array[0], indexes=1, window=None)
        writer.write(array[1], indexes=2, window=None)


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
    input_raster_dtype = mandrill.dtype
    input_grid_dtype = np.float64

    tmp_dir = tmp_path_factory.mktemp("raster")

    ret = {
        "raster_in_path": tmp_dir / "test_raster_in.tif",
        "raster_in_nbands": mandrill.shape[0],
        "raster_in_mask_band": None,
        "grid_in_path": {
            "tif": tmp_dir / "test_grid_in.tif",
            "bsq": {res: tmp_dir / f"test_grid_in.bsq_{res[0]}_{res[1]}.hd" for res in RESOLUTIONS},
        },
        "grid_in_shape": (50, 40),
        "grid_mask_flag": False,
        "grid_mask_in_path": None,
        "grid_mask_in_masked_value": None,
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
        random_std=0.05,
        dtype=input_grid_dtype,
    )

    # write grid as tif
    write_array(
        np.array([grid_row, grid_col]), dtype=input_grid_dtype, fileout=ret["grid_in_path"]["tif"]
    )
    if USE_ORION:
        # write grid as bsq - it requires to write header for each resolution
        # here we write full grid data for each resolution to make it simple
        for res in RESOLUTIONS:
            write_array_bsq(
                np.array([grid_row, grid_col]),
                dtype=input_grid_dtype,
                fileout=ret["grid_in_path"]["bsq"][res],
                oversampling_row=res[0],
                oversampling_col=res[1],
            )

    return ret




def input_data_002_mandrill_grid_f64(tmp_path_factory, res):
    """Create input data"""
    input_raster_dtype = mandrill.dtype
    input_grid_dtype = np.float64

    tmp_dir = tmp_path_factory.mktemp("raster")

    output_shape_target = (10000, 10000)
    grid_in_shape = (
        (output_shape_target[0] - 1) // res[0] + 1,
        (output_shape_target[1] - 1) // res[1] + 1
    )
    v_row_y = (mandrill.shape[0] / output_shape_target[0])
    v_row_x = (0.0002)
    v_col_y = (-0.0005)
    v_col_x = (mandrill.shape[1] / output_shape_target[1])
    

    ret = {
        "raster_in_path": tmp_dir / "test_raster_in.tif",
        "raster_in_nbands": mandrill.shape[0],
        "raster_in_mask_band": None,
        "grid_in_path": {
            "tif": tmp_dir / "test_grid_in.tif",
            "bsq": {res: tmp_dir / f"test_grid_in.bsq_{res[0]}_{res[1]}.hd"},
        },
        "grid_in_shape": grid_in_shape,
        "grid_mask_flag": False,
        "grid_mask_in_path": None,
        "grid_mask_in_masked_value": None,
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
        v_row_y=v_row_y,
        v_row_x=v_row_x,
        v_col_y=v_col_y,
        v_col_x=v_col_x,
        random_seed=1,
        random_std=0.05 * mandrill.shape[1] / output_shape_target[1],
        dtype=input_grid_dtype,
    )

    # write grid as tif
    write_array(
        np.array([grid_row, grid_col]), dtype=input_grid_dtype, fileout=ret["grid_in_path"]["tif"]
    )
    if USE_ORION:
        # write grid as bsq - it requires to write header for each resolution
        # here we write full grid data for each resolution to make it simple
        write_array_bsq(
                np.array([grid_row, grid_col]),
                dtype=input_grid_dtype,
                fileout=ret["grid_in_path"]["bsq"][res],
                oversampling_row=res[0],
                oversampling_col=res[1],
            )

    return ret



@pytest.fixture(scope="session")
def input_data_002_mandrill_grid_50_f64(tmp_path_factory):
    res = (50, 50)
    return input_data_002_mandrill_grid_f64(tmp_path_factory, res)

@pytest.fixture(scope="session")
def input_data_002_mandrill_grid_100_f64(tmp_path_factory):
    res = (100, 100)
    return input_data_002_mandrill_grid_f64(tmp_path_factory, res)

@pytest.fixture(scope="session")
def input_data_002_mandrill_grid_500_f64(tmp_path_factory):
    res = (500, 500)
    return input_data_002_mandrill_grid_f64(tmp_path_factory, res)



def input_data_003_12000_grid_f64(tmp_path_factory, res):
    """Create input data"""
    input_raster_dtype = np.float64
    input_grid_dtype = np.float64

    tmp_dir = tmp_path_factory.mktemp("raster")

    input_shape = (3, 12000, 12000)
    output_shape_target = (10000, 10000)
    grid_in_shape = (
        (output_shape_target[0] - 1) // res[0] + 1,
        (output_shape_target[1] - 1) // res[1] + 1
    )
    v_row_y = (input_shape[1] / output_shape_target[0])
    v_row_x = 0.0002 * res[1]
    v_col_y = 0.0005 * res[0]
    v_col_x = (input_shape[2] / output_shape_target[1])

    input_data = np.arange(input_shape[0]*input_shape[1]*input_shape[2], dtype=np.uint16) % 4000
    input_data = input_data.reshape(input_shape).astype(input_raster_dtype)

    ret = {
        "raster_in_path": tmp_dir / "test_raster_in.tif",
        "raster_in_nbands": input_data.shape[0],
        "raster_in_mask_band": None,
        "grid_in_path": {
            "tif": tmp_dir / "test_grid_in.tif",
            "bsq": {res: tmp_dir / f"test_grid_in.bsq_{res[0]}_{res[1]}.hd"},
        },
        "grid_in_shape": grid_in_shape,
        "grid_mask_flag": False,
        "grid_mask_in_path": None,
        "grid_mask_in_masked_value": None,
        "grid_in_nodata": None,
    }

    # write input raster as tif
    write_array(input_data, dtype=input_raster_dtype, fileout=ret["raster_in_path"])

    # create grid
    grid_row, grid_col = create_grid_generic(
        nrow=ret["grid_in_shape"][0],
        ncol=ret["grid_in_shape"][1],
        origin_pos=np.array((0.3, 0.2)),
        origin_node=np.array((0.0, 0.0)),
        v_row_y=v_row_y,
        v_row_x=v_row_x,
        v_col_y=v_col_y,
        v_col_x=v_col_x,
        random_seed=1,
        random_std=0.05 * input_data.shape[1] / output_shape_target[1],
        dtype=input_grid_dtype,
    )

    # write grid as tif
    write_array(
        np.array([grid_row, grid_col]), dtype=input_grid_dtype, fileout=ret["grid_in_path"]["tif"]
    )
    if USE_ORION:
        # write grid as bsq - it requires to write header for each resolution
        # here we write full grid data for each resolution to make it simple
        write_array_bsq(
                np.array([grid_row, grid_col]),
                dtype=input_grid_dtype,
                fileout=ret["grid_in_path"]["bsq"][res],
                oversampling_row=res[0],
                oversampling_col=res[1],
            )

    return ret

@pytest.fixture(scope="session")
def input_data_003_12000_grid_1_f64(tmp_path_factory):
    res = (1, 1)
    return input_data_003_12000_grid_f64(tmp_path_factory, res)
    
@pytest.fixture(scope="session")
def input_data_003_12000_grid_50_f64(tmp_path_factory):
    res = (50, 50)
    return input_data_003_12000_grid_f64(tmp_path_factory, res)

@pytest.fixture(scope="session")
def input_data_003_12000_grid_100_f64(tmp_path_factory):
    res = (100, 100)
    return input_data_003_12000_grid_f64(tmp_path_factory, res)

@pytest.fixture(scope="session")
def input_data_003_12000_grid_200_f64(tmp_path_factory):
    res = (200, 200)
    return input_data_003_12000_grid_f64(tmp_path_factory, res)


def input_data_004_4000_grid_f64(tmp_path_factory, res):
    """Create input data"""
    input_raster_dtype = np.float64
    input_grid_dtype = np.float64

    tmp_dir = tmp_path_factory.mktemp("raster")

    input_shape = (3, 4000, 4000)
    output_shape_target = (3900, 3900)
    grid_in_shape = (
        (output_shape_target[0] - 1) // res[0] + 1,
        (output_shape_target[1] - 1) // res[1] + 1
    )
    v_row_y = (input_shape[1] / output_shape_target[0])
    v_row_x = 0.0002 * res[1]
    v_col_y = 0.0005 * res[0]
    v_col_x = (input_shape[2] / output_shape_target[1])

    input_data = np.arange(input_shape[0]*input_shape[1]*input_shape[2], dtype=np.uint16) % 4000
    input_data = input_data.reshape(input_shape).astype(input_raster_dtype)

    ret = {
        "raster_in_path": tmp_dir / "test_raster_in.tif",
        "raster_in_nbands": input_data.shape[0],
        "raster_in_mask_band": None,
        "grid_in_path": {
            "tif": tmp_dir / "test_grid_in.tif",
            "bsq": {res: tmp_dir / f"test_grid_in.bsq_{res[0]}_{res[1]}.hd"},
        },
        "grid_in_shape": grid_in_shape,
        "grid_mask_flag": False,
        "grid_mask_in_path": None,
        "grid_mask_in_masked_value": None,
        "grid_in_nodata": None,
    }

    # write input raster as tif
    write_array(input_data, dtype=input_raster_dtype, fileout=ret["raster_in_path"])

    # create grid
    grid_row, grid_col = create_grid_generic(
        nrow=ret["grid_in_shape"][0],
        ncol=ret["grid_in_shape"][1],
        origin_pos=np.array((0.3, 0.2)),
        origin_node=np.array((0.0, 0.0)),
        v_row_y=v_row_y,
        v_row_x=v_row_x,
        v_col_y=v_col_y,
        v_col_x=v_col_x,
        random_seed=1,
        random_std=0.05 * input_data.shape[1] / output_shape_target[1],
        dtype=input_grid_dtype,
    )

    # write grid as tif
    write_array(
        np.array([grid_row, grid_col]), dtype=input_grid_dtype, fileout=ret["grid_in_path"]["tif"]
    )
    if USE_ORION:
        # write grid as bsq - it requires to write header for each resolution
        # here we write full grid data for each resolution to make it simple
        write_array_bsq(
                np.array([grid_row, grid_col]),
                dtype=input_grid_dtype,
                fileout=ret["grid_in_path"]["bsq"][res],
                oversampling_row=res[0],
                oversampling_col=res[1],
            )

    return ret

@pytest.fixture(scope="session")
def input_data_004_4000_grid_1_f64(tmp_path_factory):
    res = (1, 1)
    return input_data_004_4000_grid_f64(tmp_path_factory, res)
    
@pytest.fixture(scope="session")
def input_data_004_4000_grid_50_f64(tmp_path_factory):
    res = (50, 50)
    return input_data_004_4000_grid_f64(tmp_path_factory, res)

@pytest.fixture(scope="session")
def input_data_004_4000_grid_100_f64(tmp_path_factory):
    res = (100, 100)
    return input_data_004_4000_grid_f64(tmp_path_factory, res)

@pytest.fixture(scope="session")
def input_data_004_4000_grid_200_f64(tmp_path_factory):
    res = (200, 200)
    return input_data_004_4000_grid_f64(tmp_path_factory, res)



@pytest.fixture(scope="session")
def input_data(request):
    """Create the requested input data"""
    return request.getfixturevalue(request.param)


class BenchAdapter(Protocol):
    """Each tool implements that interface"""

    name: str

    def check_support(self, params):
        pmap = PARAM_MAP[self.name]
        try:
            self._kwargs = copy.deepcopy(pmap["method"][params["method"]])
        except KeyError as err:
            raise pytest.skip(f"{self.name} does not support '!r{params['method']}'") from err

    def prepare(self, benchmark, request, params: dict[str, Any], work_dir: Path):
        """ """
        self._output_dir = work_dir

        inputs = params["input_data"]
        self._input_image_path = inputs["raster_in_path"]
        self._input_image_nbands = inputs["raster_in_nbands"]
        self._input_image_mask_band = inputs["raster_in_mask_band"]

        self._grid_in_path_dict = inputs["grid_in_path"]
        self._grid_in_shape = inputs["grid_in_shape"]

        self._grid_mask_flag = inputs["grid_mask_flag"]
        self._grid_mask_in_path = inputs["grid_mask_in_path"]
        self._grid_mask_in_masked_value = inputs["grid_mask_in_masked_value"]
        self._grid_in_nodata = inputs["grid_in_nodata"]

        self._output_image_path = self._output_dir / "image_out.tif"
        self._output_mask_path = None

        self._resolution = params["resolution"]
        self._output_shape = grid_full_resolution_shape(
            shape=self._grid_in_shape,
            resolution=self._resolution,
        )

    def run(self) -> Any:
        """ """
        pass


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
    "orion.grid_orion": {
        "method": {
            "nearest": {
                "filter": "PPV",
            },
            "linear": {
                "filter": "BLN",
            },
            "cubic": {
                "filter": "BCO",
            },
            "bspline3": {
                "filter": "SPLINE",
                "spline_order": 3,
            },
            "bspline5": {
                "filter": "SPLINE",
                "spline_order": 5,
            },
            "bspline7": {
                "filter": "SPLINE",
                "spline_order": 7,
            },
        }
    },
}


class GridrAdapter(BenchAdapter):
    name = "gridr.chain.grid_resampling_chain.basic_grid_resampling_chain"

    def __init__(self):
        self._fn = basic_grid_resampling_chain

    def prepare(self, benchmark, request, params: dict[str, Any], work_dir):
        """ """
        super().prepare(benchmark, request, params, work_dir)

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
        self._output_mask_profile = {}
        if params["mask_out"]:
            self._output_mask_path = Path(self._output_dir) / "mask_out.tif"

        # For GridR we use the TIF input grid
        self._grid_in_path = self._grid_in_path_dict["tif"]
        
        with rasterio.open(self._input_image_path, "r") as array_src_ds:
            if array_src_ds.width > 1000:
                self._kwargs['io_strip_size'] = 2000
                self._kwargs['tile_shape'] = (2000, 2000)
        self._logger = None

    def run(self):
        with (
            rasterio.open(self._grid_in_path, "r") as grid_in_ds,
            rasterio.open(self._input_image_path, "r") as array_src_ds,
            rasterio.open(
                self._output_image_path, "w", **self._output_image_profile
            ) as array_out_ds,
            safe_raster_open(
                self._output_mask_path, "w", **self._output_mask_profile
            ) as mask_out_ds,
            # safe_raster_open(self._grid_mask_in_path, "r") as grid_mask_in_ds,
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
                # boundary_condition=boundary_condition,
                nodata_out=0,
                win=None,
                mask_out_ds=mask_out_ds,
                # grid_mask_in_ds=grid_mask_in_ds,
                # grid_mask_in_unmasked_value=grid_mask_in_unmasked_value,
                # grid_mask_in_band=grid_mask_in_band,
                # array_src_mask_ds=array_src_mask_ds,
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


class OrionAdapter(BenchAdapter):
    name = "orion.grid_orion"

    def __init__(self):
        self._fn = grid_orion

    def prepare(self, benchmark, request, params: dict[str, Any], work_dir):
        """ """
        if not USE_ORION:
            raise pytest.skip(f"{self.name} not supported : orion is not available")
        super().prepare(benchmark, request, params, work_dir)

        if not params["multi_bands"]:
            self._kwargs["num_canal_in"] = 1
        
        self._kwargs['mode_read'] = "IMAGE"
        with rasterio.open(self._input_image_path, "r") as array_src_ds:
            if array_src_ds.width > 1000:
                self._kwargs['mode_read'] = "TUILE"
                self._kwargs['largeur_imagette'] = 2000
                self._kwargs['hauteur_imagette'] = 2000
                #self._kwargs['tile_orion_auto'] = True

        # For Orion we use the BSQ input grid for the corresponding resolution
        self._grid_in_path = self._grid_in_path_dict["bsq"][params["resolution"]]

    def run(self):
        grid_orion(
            input_image=Path(self._input_image_path).resolve().as_posix(),
            output_image=Path(self._output_image_path).as_posix(),
            grid_orion=Path(self._grid_in_path).resolve().as_posix(),
            surech_row=self._resolution[0],
            surech_col=self._resolution[1],
            origin_row=0,
            origin_col=0,
            nb_row_out=self._output_shape[0],
            nb_col_out=self._output_shape[1],
            format_grid="BSQ",
            edges_management="EXACTE",
            format_in="AUTO",
            format_out=None,
            type_in=None,
            type_out=None,
            slope=1.0,
            bias=0.0,
            no_data=0.0,
            # complex_image: bool = False,
            # num_canal_in = 1,
            # tile_orion_auto: bool = False,
            # largeur_imagette: Optional[int] = None,
            # hauteur_imagette: Optional[int] = None,
            # taille_cache_image: Optional[int] = None,
            # zu_orion_target_in: Optional[str] = None,
            # gene_grid_dense: Optional[str] = None,
            # use_invalid_data: bool = False,
            # invalid_data_val: Optional[float] = None,
            # alpha_filter: Optional[float] = None,
            # name_filter: Optional[str] = None,
            # center_filter_row: Optional[int] = None,
            # center_filter_col: Optional[int] = None,
            # a_filter: Optional[int] = None,
            # sigma_filter: Optional[float] = None,
            # precision_filter: Optional[int] = None,
            # spline_order: Optional[int] = None,
            # antialiasing: bool = False,
            # sensor: bool = False,
            # name_sensor: Optional[str] = None,
            # pente: Optional[float] = None,
            # repli: Optional[float] = None,
            # nb_processus: int = 1,
            # trace: bool = False,
            # dry_run: bool = False,
            **self._kwargs,
        )


ADAPTERS = [
    GridrAdapter(),
    OrionAdapter(),
]

if DO_TEST_001:
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", RESOLUTIONS)
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_001_mandrill_grid_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )

if DO_TEST_002:
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((50,50),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_002_mandrill_grid_50_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_50(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )


    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((100,100),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_002_mandrill_grid_100_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_100(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
        

    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((500,500),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_002_mandrill_grid_500_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_500(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )



if DO_TEST_003:
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((1,1),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_003_12000_grid_1_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_003_12000_1(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((50,50),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_003_12000_grid_50_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_003_12000_50(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((100,100),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_003_12000_grid_100_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_003_12000_100(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((200,200),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_003_12000_grid_200_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_003_12000_200(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )


if DO_TEST_004:
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((1,1),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_004_4000_grid_1_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_004_4000_1(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((50,50),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_004_4000_grid_50_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_004_4000_50(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((100,100),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_004_4000_grid_100_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_004_4000_100(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
    @pytest.mark.benchmark(group="time:gridr.chain.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("resolution", ((200,200),))
    @pytest.mark.parametrize(
        "input_data",
        [
            "input_data_004_4000_grid_200_f64",
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "mask_out",
        [
            False,
        ],
    )
    @pytest.mark.parametrize("multi_bands", [True, False])
    def test_grid_resampling_004_4000_200(
        benchmark,
        request,
        adapter,
        method,
        input_data,
        resolution,
        mask_out,
        multi_bands,
        tmp_path,
    ):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                benchmark,
                request,
                params={
                    "method": method,
                    "mask_out": mask_out,
                    "resolution": resolution,
                    "input_data": input_data,
                    "multi_bands": multi_bands,
                },
                work_dir=tmp_path,
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )


if DO_TEST_INIT_ORION:
    @pytest.mark.benchmark(group="time:orion_init_only")
    def test_orion_init_only(benchmark):
        """ """
        if USE_ORION:
            benchmark.pedantic(
                grid_orion_init_only,
                warmup_rounds=NWARMUP,
                rounds=NROUNDS,
                iterations=NITERATIONS,
            )
