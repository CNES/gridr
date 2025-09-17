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
from ._libgridr import (  # from py_grid_geometry
    PyArrayWindow2,
    PyGeometryBoundsF64,
    PyGeometryBoundsUsize,
    PyGridGeometriesMetricsF64,
    PyGridTransitionMatrix,
    PyInterpolatorType,
    py_array1_compute_resampling_grid_geometries_f64_f64,
    py_array1_grid_resampling_f64,
    py_array1_replace_f32_i8,
    py_array1_replace_f32_u8,
    py_array1_replace_f64_i8,
    py_array1_replace_f64_u8,
    py_array1_replace_i8,
    py_array1_replace_u8,
)

__all__ = [
    "PyArrayWindow2",
    "py_array1_replace_i8",
    "py_array1_replace_f32_i8",
    "py_array1_replace_f64_i8",
    "py_array1_replace_u8",
    "py_array1_replace_f32_u8",
    "py_array1_replace_f64_u8",
    "PyInterpolatorType",
    "py_array1_grid_resampling_f64",
    "PyGridTransitionMatrix",
    "PyGeometryBoundsUsize",
    "PyGeometryBoundsF64",
    "PyGridGeometriesMetricsF64",
    "py_array1_compute_resampling_grid_geometries_f64_f64",
]
