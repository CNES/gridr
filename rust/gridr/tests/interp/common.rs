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

//! Helpers shared by the interpolation integration tests.

use gridr::interp::gx_array_view_interp::GxArrayViewInterpolator;
use gridr::interp::gx_bspline_kernel::{GxBSplineInterpolator, GxBSplineInterpolatorArgs};

/// Builds and initialises a B-spline interpolator of order `N`.
pub fn bspline<const N: usize>() -> GxBSplineInterpolator<N> {
    let args = GxBSplineInterpolatorArgs { epsilon: 1e-6, mask_influence_threshold: 1e-3 };
    let mut interp = GxBSplineInterpolator::<N>::new(&args);
    interp.initialize().expect("B-spline initialisation failed");
    interp
}
