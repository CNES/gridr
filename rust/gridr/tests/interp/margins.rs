// Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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

//! Integration tests of `GxArrayViewInterpolator::total_margins`.

use gridr::interp::gx_array_view_interp::{GxArrayViewInterpolator, GxArrayViewInterpolatorNoArgs};
use gridr::interp::gx_linear_kernel::GxLinearInterpolator;
use gridr::interp::gx_optimized_bicubic_kernel::GxOptimizedBicubicInterpolator;

use crate::common::bspline;

/// `total_margins` must cover the stencil read by the `NoBoundsCheck` path
/// for any target position inside the footprint of the region of interest,
/// i.e. `floor(K / 2)` samples on each side for a stencil of `K` taps.
#[test]
fn margins_cover_stencil() {
    fn check<I: GxArrayViewInterpolator>(name: &str, interp: &I) {
        let m = interp.total_margins().expect("margins");
        let need_r = interp.kernel_row_size() / 2;
        let need_c = interp.kernel_col_size() / 2;
        assert!(
            m[0] >= need_r && m[1] >= need_r && m[2] >= need_c && m[3] >= need_c,
            "{name}: margins {m:?} < stencil half sizes ({need_r}, {need_c})"
        );
    }
    check("linear", &GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs {}));
    check("cubic", &GxOptimizedBicubicInterpolator::new(&GxArrayViewInterpolatorNoArgs {}));
    check("bspline3", &bspline::<3>());
    check("bspline5", &bspline::<5>());
    check("bspline7", &bspline::<7>());
    check("bspline9", &bspline::<9>());
    check("bspline11", &bspline::<11>());
}
