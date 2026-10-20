// Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
//
// This file is part of GRIDR
// (see https://github.com/CNES/gridr).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![warn(missing_docs)]
//! Crate doc
pub mod gx_array_view_interp;
#[cfg(test)]
pub mod gx_array_view_interp_test;
pub mod gx_optimized_bicubic_kernel;
pub mod gx_nearest_kernel;
pub mod gx_linear_kernel;
pub mod gx_bspline_kernel;
pub mod gx_bspline_prefiltering;
#[cfg(test)]
mod gx_kernel_test_utils;
