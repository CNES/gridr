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
GridR Time Benchmark for core array_grid_resampling

Command : PYTHONPATH=$PWD/python:$PYTHONPATH python \
-m pytest tests/python/benchmarks/time --benchmark-save="test_1" \
--benchmark-storage=./benchmarks/results/pytest/ --benchmark-group-by=group \
--benchmark-columns=min,median,mean,stddev,iqr,outliers,ops,rounds -v
"""
from __future__ import annotations

import copy
from typing import Any, Protocol

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from gridr.core.grid.grid_resampling import array_grid_resampling

NROUNDS = 20
NITERATIONS = 1
NWARMUP = 3
NSIZES = [
    100,
    500,
    1000,
    2000,
    4000,
]

DO_BENCH = False

class BenchAdapter(Protocol):
    """Each tool implements that interface"""

    name: str

    def check_support(self, params):
        pmap = PARAM_MAP[self.name]
        try:
            self._kwargs = copy.deepcopy(pmap["method"][params["method"]])
        except KeyError as err:
            raise pytest.skip(f"{self.name} does not support '!r{params['method']}'") from err

    def prepare(self, n: int, params: dict[str, Any]):
        """ """
        shape = (n, n)
        rng = np.random.default_rng(seed=0)
        shift_x = rng.standard_normal(shape[0] * shape[1]).reshape(shape) * 0.1
        shift_y = rng.standard_normal(shape[0] * shape[1]).reshape(shape) * 0.1
        self._data = np.arange(shape[0] * shape[1]).astype(np.float64).reshape(shape)
        self._grid_row, self._grid_col = np.meshgrid(
            np.arange(shape[0], dtype=np.float64), np.arange(shape[1], dtype=np.float64)
        )
        self._grid_row += shift_y
        self._grid_col += shift_x

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
    "gridr.core.grid.grid_resampling.array_grid_resampling": {
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
                    "epsilon": 1e-6,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline5": {
                "interp": "bspline5",
                "interp_kwargs": {
                    "epsilon": 1e-6,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline7": {
                "interp": "bspline7",
                "interp_kwargs": {
                    "epsilon": 1e-6,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline9": {
                "interp": "bspline9",
                "interp_kwargs": {
                    "epsilon": 1e-6,
                    "mask_influence_threshold": 1,
                },
            },
            "bspline11": {
                "interp": "bspline11",
                "interp_kwargs": {
                    "epsilon": 1e-6,
                    "mask_influence_threshold": 1,
                },
            },
        }
    },
    "scipy.map_coordinates": {
        "method": {
            "nearest": {
                "order": 0,
                "prefilter": True,
            },
            "linear": {
                "order": 1,
                "prefilter": True,
            },
            "bspline3": {
                "order": 3,
                "prefilter": True,
            },
            "bspline5": {
                "order": 5,
                "prefilter": True,
            },
        }
    },
}


class GridrAdapter(BenchAdapter):
    name = "gridr.core.grid.grid_resampling.array_grid_resampling"

    def __init__(self):
        self._fn = array_grid_resampling

    def run(self):
        self._fn(
            # interp = self._kwargs["interp"],
            array_in=self._data.astype(np.float64),
            grid_row=self._grid_row,
            grid_col=self._grid_col,
            grid_resolution=(1, 1),
            array_out=None,
            array_out_win=None,
            nodata_out=0,
            check_boundaries=True,
            standalone=True,
            boundary_condition=None,
            **self._kwargs,
        )


class ScipyMapCoordinateAdapter(BenchAdapter):
    name = "scipy.map_coordinates"

    def __init__(self):
        self._fn = map_coordinates

    def run(self):
        self._fn(
            self._data,
            [self._grid_row.flatten(), self._grid_col.flatten()],
            **self._kwargs,
        )


ADAPTERS = [GridrAdapter(), ScipyMapCoordinateAdapter()]

if DO_BENCH:
    @pytest.mark.benchmark(group="time:gridr.core.grid.grid_resampling.array_grid_resampling")
    @pytest.mark.parametrize("adapter", ADAPTERS, ids=[a.name for a in ADAPTERS])
    @pytest.mark.parametrize("method", CANONICAL_PARAMS["method"])
    @pytest.mark.parametrize("n", NSIZES)
    def test_grid_resampling(benchmark, adapter, method, n):
        """ """
        adapter.check_support(
            params={"method": method},
        )

        def setup():
            adapter.prepare(
                n=n,
                params={"method": method},
            )

        benchmark.pedantic(
            adapter.run,
            setup=setup,
            warmup_rounds=NWARMUP,
            rounds=NROUNDS,
            iterations=NITERATIONS,
        )
