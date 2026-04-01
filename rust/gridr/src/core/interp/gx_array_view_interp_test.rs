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

//! Unit tests for [`GxArrayViewInterpolatorCore`].
//!
//! # Test strategy
//!
//! All five methods of the trait are exercised:
//!
//! | Module                    | Methods under test                       |
//! |---------------------------|------------------------------------------|
//! | `mock`                    | Mock interpolator setup (shared)         |
//! | `nomask_unchecked`        | `interpolate_nomask_unchecked`           |
//! | `nomask_partial`          | `interpolate_nomask_partial`             |
//! | `masked_unchecked`        | `interpolate_masked_unchecked`           |
//! | `masked_partial`          | `interpolate_masked_partial`             |
//! | `separable_core`          | `array1_interp2_separable_core`          |
//!
//! # Mock interpolator
//!
//! [`MockInterp`] is a minimal 3×3 nearest-neighbour-like interpolator whose
//! `compute_weights` sets the centre weight to `1.0` and all others to `0.0`
//! when `x == 0.0`, and distributes weight linearly between the two
//! neighbouring pixels otherwise.  This keeps expected values easy to compute
//! by hand while still exercising all code paths.
//!
//! Kernel size: 3×3 -> `KROWS = 3`, `KCOLS = 3`, `KSIZE = 9`.
//!
//! # Coverage
//!
//! For each method the following scenarios are tested:
//! - **Nominal** — interior pixel, sub-pixel position, correct output value.
//! - **Identity** — integer position, output equals source value exactly.
//! - **Multi-variable** — `nvar > 1`, each plane interpolated independently.
//! - **Boundary** — stencil exactly on the first/last valid row or column.
//! - **Out-of-bounds** — pixel whose stencil (partially) exits the array.
//! - **Masked input** — one pixel in the stencil masked -> `nodata` output.
//! - **All-valid mask** — full-ones mask gives the same result as no mask.
//! - **Output mask** — output mask value is `1` on success, `0` on failure.

#[cfg(test)]
mod mock {
    //! Shared mock interpolator used by all test modules.

    use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
    use crate::core::gx_errors::GxError;
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolationContextTrait,
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolator,
        GxArrayViewInterpolatorArgs,
    };

    // ------------------------------------------------------------------
    // Weight function
    // ------------------------------------------------------------------

    /// Linear-interpolation weights for a 3-tap kernel.
    ///
    /// - `x = 0`  -> `[0, 1, 0]` (identity).
    /// - `x > 0`  -> weight split between centre (`w[1]`) and right (`w[2]`).
    /// - `x < 0`  -> weight split between left (`w[0]`) and centre (`w[1]`).
    ///
    /// The weight indexing convention of `GxArrayViewInterpolatorCore` places
    /// index `KCOLS - 1 - icol` at position `icol`, so for a 3-tap kernel:
    /// - `weights[2]` = weight of the left  pixel (icol = 0).
    /// - `weights[1]` = weight of the centre pixel (icol = 1).
    /// - `weights[0]` = weight of the right pixel  (icol = 2).
    pub fn linear_weights_3(x: f64, weights: &mut [f64]) {
        assert!(weights.len() >= 3);
        if x >= 0.0 {
            weights[2] = 0.0;           // left
            weights[1] = 1.0 - x;      // centre
            weights[0] = x;            // right
        } else {
            weights[2] = -x;           // left
            weights[1] = 1.0 + x;     // centre
            weights[0] = 0.0;          // right
        }
    }

    // ------------------------------------------------------------------
    // Mock struct
    // ------------------------------------------------------------------

    /// Minimal 3×3 separable interpolator used as a test fixture.
    pub struct MockInterp;

    impl GxArrayViewInterpolatorCore<3, 3, 9> for MockInterp {
        fn compute_weights(&self, x: f64, weights: &mut [f64]) {
            linear_weights_3(x, weights);
        }
    }

    /// Trivial `GxArrayViewInterpolator` impl so `MockInterp` compiles as
    /// a complete type.  Most methods panic — only those exercised in tests
    /// matter.
    impl GxArrayViewInterpolator for MockInterp {
        fn new(_args: &dyn GxArrayViewInterpolatorArgs) -> Self {
            MockInterp
        }
        fn shortname(&self) -> String {
            "mock".to_string()
        }
        fn initialize(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
            vec![0.0f64; 3 + 3].into_boxed_slice()
        }
        fn kernel_row_size(&self) -> usize { 3 }
        fn kernel_col_size(&self) -> usize { 3 }
        fn total_margins(&self) -> Result<[usize; 4], GxError> {
            Ok([1, 1, 1, 1])
        }
        fn array1_interp2<T, V, IC>(
            &self,
            weights_buffer: &mut [f64],
            target_row_pos: f64,
            target_col_pos: f64,
            out_idx: usize,
            array_in: &GxArrayView<'_, T>,
            array_out: &mut GxArrayViewMut<'_, V>,
            nodata_out: V,
            context: &mut IC,
        ) -> Result<(), String>
        where
            T: Copy + PartialEq
                + std::ops::Mul<f64, Output = f64>
                + Into<f64>,
            V: Copy + PartialEq + From<f64>,
            IC: GxArrayViewInterpolationContextTrait,
        {
            self.array1_interp2_separable_core(
                weights_buffer,
                target_row_pos,
                target_col_pos,
                out_idx,
                array_in,
                array_out,
                nodata_out,
                context,
            )
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Tolerance for floating-point comparisons.
    pub const TOL: f64 = 1e-10;

    pub fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL
    }

    /// Builds a 1-variable `nrow × ncol` array from a flat slice.
    pub fn arr<'a>(
        data: &'a [f64],
        nrow: usize,
        ncol: usize,
    ) -> GxArrayView<'a, f64> {
        GxArrayView::new(data, 1, nrow, ncol)
    }

    /// Builds a 1-variable mutable array from a flat slice.
    pub fn arr_mut<'a>(
        data: &'a mut [f64],
        nrow: usize,
        ncol: usize,
    ) -> GxArrayViewMut<'a, f64> {
        GxArrayViewMut::new(data, 1, nrow, ncol)
    }

    /// Builds a `nvar`-variable `nrow × ncol` array.
    pub fn arr_nvar<'a>(
        data: &'a [f64],
        nvar: usize,
        nrow: usize,
        ncol: usize,
    ) -> GxArrayView<'a, f64> {
        GxArrayView::new(data, nvar, nrow, ncol)
    }

    pub fn arr_mut_nvar<'a>(
        data: &'a mut [f64],
        nvar: usize,
        nrow: usize,
        ncol: usize,
    ) -> GxArrayViewMut<'a, f64> {
        GxArrayViewMut::new(data, nvar, nrow, ncol)
    }
}


// =============================================================================
// interpolate_nomask_unchecked
// =============================================================================

#[cfg(test)]
mod nomask_unchecked {
    use super::mock::*;
    use crate::core::interp::gx_array_view_interp::GxArrayViewInterpolatorCore;

    // Flat 5×5 input — all values equal to `v` -> any position returns `v`.
    fn uniform(v: f64) -> [f64; 25] {
        [v; 25]
    }

    /// Integer centre (x = 0): identity weights -> output equals source pixel.
    #[test]
    fn identity_at_integer_position() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 7., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // Weights: identity at offset 0
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2,
        );
        assert!(approx(out[0], 7.0), "got {}", out[0]);
    }

    /// Sub-pixel position x = 0.5: weight split equally between centre and
    /// right neighbour.
    #[test]
    fn subpixel_half_step() {
        // Row 2 of a 5×5 array: [0, 0, 4, 8, 0]
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 4., 8., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // x = 0.5 -> weights: [0.5, 0.5, 0.0] (right=0.5, centre=0.5, left=0)
        let w_row = [0.0, 1.0, 0.0f64]; // no row interpolation
        let w_col = [0.5, 0.5, 0.0f64]; // col: 0.5 right + 0.5 centre
        // Centre at (2, 2), col offset +0.5 -> mixes col 2 (4) and col 3 (8)
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 2, 2,
        );
        // Expected: 0.5 * 4 + 0.5 * 8 = 6
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    /// Uniform array: result must equal the uniform value regardless of
    /// position.
    #[test]
    fn uniform_array_any_position() {
        let data = uniform(3.5);
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.25, 0.5, 0.25f64]; // sums to 1
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2,
        );
        assert!(approx(out[0], 3.5), "got {}", out[0]);
    }

    /// Two variable planes are interpolated independently.
    #[test]
    fn multivar_two_planes() {
        // plane 0: impulse 10 at (2,2); plane 1: impulse 20 at (2,2)
        let mut data = [0.0f64; 50];
        data[2 * 5 + 2]      = 10.0;
        data[25 + 2 * 5 + 2] = 20.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2,
        );
        assert!(approx(out[0], 10.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 20.0), "plane1: {}", out[1]);
    }

    /// Stencil centred on the last valid interior row/col: result is the
    /// source value (identity weights, single impulse at the corner).
    #[test]
    fn stencil_at_last_interior_row_col() {
        // 5×5 array, impulse at (3, 3) — the last position where a 3-tap
        // stencil centred there still reaches row/col 2 and 4, both in bounds.
        let mut data = [0.0f64; 25];
        data[3 * 5 + 3] = 5.0;
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64]; // identity
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, 3, 3,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Negative sub-pixel offset (x = −0.5): weight split between left
    /// neighbour and centre.
    #[test]
    fn subpixel_negative_half_step() {
        // Row 2: [0, 4, 8, 0, 0]
        let mut data = [0.0f64; 25];
        data[2 * 5 + 1] = 4.0;
        data[2 * 5 + 2] = 8.0;
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        // x = −0.5 -> weights: left 0.5 (w[2]), centre 0.5 (w[1]), right 0
        let w_col = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 2, 2,
        );
        // 0.5 * 8 + 0.5 * 4 = 6
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }
    
    /// Sub-pixel interpolation with non-zero offset on both row and col.
    #[test]
    fn subpixel_both_axes() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 2., 4., 0., 0.,
            0., 6., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // row: left=0.5, centre=0.5; col: left=0.5, centre=0.5
        let w = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2,
        );
        // Bilinear: 0.5*0.5*2 + 0.5*0.5*4 + 0.5*0.5*6 + 0.5*0.5*8 = 5.0
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Writing to a non-zero out_idx within the output array.
    #[test]
    fn nonzero_out_idx() {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let mut out = [-9.0f64; 3];
        let mut arr_out = arr_mut(&mut out, 1, 3);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 2, 2, 2,
        );
        assert!(approx(out[2], 1.0), "out[2]={}", out[2]);
        assert_eq!(out[0], -9.0, "out[0] should be untouched");
        assert_eq!(out[1], -9.0, "out[1] should be untouched");
    }

    /// Multivar with non-zero out_idx: each plane at the correct offset.
    #[test]
    fn multivar_nonzero_out_idx() {
        let mut data = [0.0f64; 50]; // 2 vars, 5×5
        data[2 * 5 + 2]      = 10.0;
        data[25 + 2 * 5 + 2] = 20.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        // Output: 2 vars, 1×3 -> var_size=3, total 6.
        let mut out = [-9.0f64; 6];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 3);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 1, 2, 2,
        );
        assert!(approx(out[1], 10.0), "plane0 at idx 1: {}", out[1]);
        assert!(approx(out[4], 20.0), "plane1 at idx 4: {}", out[4]);
    }
}


// =============================================================================
// interpolate_nomask_partial
// =============================================================================

#[cfg(test)]
mod nomask_partial {
    use super::mock::*;
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        NoOutputMask,
    };

    fn no_mask() -> NoOutputMask {
        NoOutputMask::default()
    }

    /// Interior pixel: same result as unchecked.
    #[test]
    fn interior_matches_unchecked() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 9., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, &mut no_mask(), 0, 2, 2, -1.0,
        );
        assert!(approx(out[0], 9.0), "got {}", out[0]);
    }

    /// Centre at (0, 2): the stencil needs row -1, which is out of bounds.
    /// With identity column weights, the left/right columns have zero weight.
    /// The out-of-bounds row also has zero weight (identity row) so the result
    /// should be valid and equal to data[0][2] = 5.
    #[test]
    fn top_border_zero_weight_oob_row_succeeds() {
        #[rustfmt::skip]
        let data = [
            0., 0., 5., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 3, 5);
        let mut out = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // Row weights: identity -> only centre row (row 0) contributes.
        // Column weights: identity -> only col 2 contributes.
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, &mut no_mask(), 0, 0, 2, -9.0,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Centre at (1, 1) with col weight 0.5 right — col 2 is in bounds, so
    /// interpolation succeeds and returns a weighted mix.
    #[test]
    fn right_col_in_bounds_partial_succeeds() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0.,
            0., 4., 8.,
            0., 0., 0.,
        ];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.5, 0.5, 0.0f64]; // right=0.5, centre=0.5
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 1, 1, -9.0,
        );
        // 0.5 * 4 + 0.5 * 8 = 6
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    /// Stencil position with non-zero weight is out of bounds -> nodata.
    #[test]
    fn nonzero_weight_oob_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // Centre at col 0, with non-zero weight for col -1 (out of bounds)
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64]; // left has weight 0.5
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 1, 0, -9.0,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// Output mask is set to 1 on success and 0 on failure.
    #[test]
    fn output_mask_set_correctly() {
        use crate::core::gx_array::GxArrayViewMut;
        use crate::core::interp::gx_array_view_interp::BinaryOutputMask;

        let data = [1.0f64; 9];
        let arr_in = arr(&data, 3, 3);

        // Success case
        let mut out1 = [0.0f64; 1];
        let mut mask1_data = [255u8; 1];
        let mut arr_out1    = arr_mut(&mut out1, 1, 1);
        let mut arr_mask1 =
            GxArrayViewMut::new(&mut mask1_data, 1, 1, 1);
        let mut om1 = BinaryOutputMask { mask: &mut arr_mask1 };
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out1, &mut om1, 0, 1, 1, -9.0,
        );
        assert_eq!(mask1_data[0], 1, "success: mask should be 1");

        // Failure case — non-zero weight at col -1
        let mut out2 = [42.0f64; 1];
        let mut mask2_data = [255u8; 1];
        let mut arr_out2  = arr_mut(&mut out2, 1, 1);
        let mut arr_mask2 =
            GxArrayViewMut::new(&mut mask2_data, 1, 1, 1);
        let mut om2 = BinaryOutputMask { mask: &mut arr_mask2 };
        let w_col = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w_col, &arr_in, &mut arr_out2, &mut om2, 0, 1, 0, -9.0,
        );
        assert_eq!(mask2_data[0], 0, "failure: mask should be 0");
    }

    /// Two variable planes: each interpolated independently.
    #[test]
    fn multivar_two_planes() {
        let mut data = [0.0f64; 18]; // 2 vars, 3×3
        data[1 * 3 + 1]     = 5.0;  // plane 0, centre
        data[9 + 1 * 3 + 1] = 15.0; // plane 1, centre
        let arr_in = arr_nvar(&data, 2, 3, 3);
        let mut out = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, &mut no_mask(), 0, 1, 1, -9.0,
        );
        assert!(approx(out[0], 5.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 15.0), "plane1: {}", out[1]);
    }

    /// Multi-variable OOB: nodata written to all planes.
    #[test]
    fn multivar_oob_all_planes_nodata() {
        let data = [1.0f64; 18]; // 2 vars, 3×3
        let arr_in = arr_nvar(&data, 2, 3, 3);
        let mut out = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64]; // left weight at col 0 -> OOB
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 1, 0, -9.0,
        );
        assert_eq!(out[0], -9.0, "plane0 expected nodata");
        assert_eq!(out[1], -9.0, "plane1 expected nodata");
    }

    /// OOB on rows: centre at row 0 with non-zero upward weight -> nodata.
    #[test]
    fn oob_row_top_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // Row weights: left=0.3 -> needs row -1 which is OOB.
        let w_row = [0.0, 0.7, 0.3f64];
        let w_col = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 0, 1, -9.0,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB row");
    }

    /// OOB on rows: centre at last row with non-zero downward weight -> nodata.
    #[test]
    fn oob_row_bottom_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // Row weights: right=0.3 -> needs row 3 (nrow=3) which is OOB.
        let w_row = [0.3, 0.7, 0.0f64];
        let w_col = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 2, 1, -9.0,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB row bottom");
    }

    /// Sub-pixel interpolation with non-zero offset on both row and col.
    #[test]
    fn subpixel_both_axes() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 2., 4., 0., 0.,
            0., 6., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 2, 2, -9.0,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Non-zero out_idx.
    #[test]
    fn nonzero_out_idx() {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let mut out = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out,
            &mut no_mask(), 2, 2, 2, -9.0,
        );
        assert!(approx(out[2], 1.0), "out[2]={}", out[2]);
        assert_eq!(out[0], -9.0, "out[0] untouched");
    }

    /// OOB on right column: col = ncol-1 with right weight -> OOB.
    #[test]
    fn oob_col_right_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        // right weight = 0.5 -> needs col 3 (ncol=3) which is OOB.
        let w_col = [0.5, 0.5, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, 1, 2, -9.0,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB col right");
    }
}


// =============================================================================
// interpolate_masked_unchecked
// =============================================================================

#[cfg(test)]
mod masked_unchecked {
    use super::mock::*;
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        NoOutputMask,
        BoundsCheck,
    };
    use crate::core::gx_array::GxArrayView;

    fn ctx_with_mask<'a>(
        mask: &'a GxArrayView<'a, u8>,
    ) -> GxArrayViewInterpolationContext<
        'a,
        BinaryInputMask<'a>,
        NoOutputMask,
        BoundsCheck,
    > {
        GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask },
            NoOutputMask::default(),
            BoundsCheck,
        )
    }

    /// All-valid mask -> same result as nomask variant.
    #[test]
    fn all_valid_mask_same_as_nomask() {
        #[rustfmt::skip]
        let data: [f64; 25] = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 6., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let mask_data = [1u8; 25];
        let arr_in   = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out  = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2, -1.0, &mut ctx,
        );
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    /// The centre pixel is masked -> nodata written.
    #[test]
    fn centre_pixel_masked_writes_nodata() {
        #[rustfmt::skip]
        let data: [f64; 25] = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 6., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let mut mask_data = [1u8; 25];
        mask_data[2 * 5 + 2] = 0; // mask the centre
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// A pixel with zero weight is masked but the result is still valid
    /// because its contribution is zero.
    #[test]
    fn zero_weight_masked_pixel_is_ignored() {
        #[rustfmt::skip]
        let data: [f64; 25] = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 6., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let mut mask_data = [1u8; 25];
        // Mask a pixel that the identity weight ignores
        mask_data[2 * 5 + 3] = 0; // right neighbour of centre
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64]; // only centre has non-zero weight
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    /// Two variable planes: if the shared mask is valid, both are
    /// interpolated independently.
    #[test]
    fn multivar_both_planes_interpolated() {
        let mut data = [0.0f64; 50];
        data[2 * 5 + 2]      = 3.0;
        data[25 + 2 * 5 + 2] = 7.0;
        let mask_data = [1u8; 25];
        let arr_in   = arr_nvar(&data, 2, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out  = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2, -1.0, &mut ctx,
        );
        assert!(approx(out[0], 3.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 7.0), "plane1: {}", out[1]);
    }

    /// Output mask is set to 1 when all inputs are valid, 0 when any is masked.
    #[test]
    fn output_mask_valid_and_invalid() {
        use crate::core::gx_array::GxArrayViewMut;
        use crate::core::interp::gx_array_view_interp::{
            GxArrayViewInterpolationContext,
            BinaryInputMask,
            BinaryOutputMask,
            BoundsCheck,
        };
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];

        // --- Valid path ---
        let all_valid = [1u8; 25];
        let mask_view_v = GxArrayView::new(&all_valid, 1, 5, 5);
        let mut out1 = [0.0f64; 1];
        let mut mask1_data = [255u8; 1];
        let mut arr_out1 = arr_mut(&mut out1, 1, 1);
        let mut arr_mask1 = GxArrayViewMut::new(&mut mask1_data, 1, 1, 1);
        let mut ctx1 = GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask: &mask_view_v },
            BinaryOutputMask { mask: &mut arr_mask1 },
            BoundsCheck,
        );
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out1, 0, 2, 2, -9.0, &mut ctx1,
        );
        assert_eq!(mask1_data[0], 1, "valid: mask should be 1");

        // --- Invalid path: centre pixel masked ---
        let mut masked_data = [1u8; 25];
        masked_data[2 * 5 + 2] = 0;
        let mask_view_i = GxArrayView::new(&masked_data, 1, 5, 5);
        let mut out2 = [42.0f64; 1];
        let mut mask2_data = [255u8; 1];
        let mut arr_out2 = arr_mut(&mut out2, 1, 1);
        let mut arr_mask2 = GxArrayViewMut::new(&mut mask2_data, 1, 1, 1);
        let mut ctx2 = GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask: &mask_view_i },
            BinaryOutputMask { mask: &mut arr_mask2 },
            BoundsCheck,
        );
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out2, 0, 2, 2, -9.0, &mut ctx2,
        );
        assert_eq!(mask2_data[0], 0, "invalid: mask should be 0");
        assert_eq!(out2[0], -9.0, "invalid: output should be nodata");
    }

    /// Sub-pixel both axes with all-valid mask.
    #[test]
    fn subpixel_both_axes_all_valid() {
        #[rustfmt::skip]
        let data: [f64; 25] = [
            0., 0., 0., 0., 0.,
            0., 2., 4., 0., 0.,
            0., 6., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let mask_data = [1u8; 25];
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 0.5, 0.5f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Non-zero out_idx with mask.
    #[test]
    fn nonzero_out_idx() {
        let data = [1.0f64; 25];
        let mask_data = [1u8; 25];
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 2, 2, 2, -9.0, &mut ctx,
        );
        assert!(approx(out[2], 1.0), "out[2]={}", out[2]);
        assert_eq!(out[0], -9.0, "out[0] untouched");
    }

    /// Masked pixel that is a neighbour (not centre) with non-zero weight ->
    /// nodata.
    #[test]
    fn neighbour_masked_nonzero_weight_writes_nodata() {
        let data = [1.0f64; 25];
        let mut mask_data = [1u8; 25];
        // Mask the left neighbour of centre (2,1)
        mask_data[2 * 5 + 1] = 0;
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // left weight = 0.5 -> accesses (2,1) which is masked.
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 2, 2, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata");
    }
}


// =============================================================================
// interpolate_masked_partial
// =============================================================================

#[cfg(test)]
mod masked_partial {
    use super::mock::*;
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        NoOutputMask,
        BoundsCheck,
    };
    use crate::core::gx_array::GxArrayView;

    fn ctx_with_mask<'a>(
        mask: &'a GxArrayView<'a, u8>,
    ) -> GxArrayViewInterpolationContext<
        'a,
        BinaryInputMask<'a>,
        NoOutputMask,
        BoundsCheck,
    > {
        GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask },
            NoOutputMask::default(),
            BoundsCheck,
        )
    }

    /// Interior pixel, all valid -> correct interpolated value.
    #[test]
    fn interior_all_valid_correct_value() {
        #[rustfmt::skip]
        let data: [f64; 9] = [
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
        ];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 1, 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Centre at boundary row 0 with identity weights (zero-weight OOB row)
    /// and all-valid mask -> succeeds.
    #[test]
    fn top_border_zero_weight_oob_row_valid_mask_succeeds() {
        #[rustfmt::skip]
        let data: [f64; 9] = [
            0., 5., 0.,
            0., 0., 0.,
            0., 0., 0.,
        ];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 0, 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Masked pixel with non-zero weight -> nodata.
    #[test]
    fn masked_active_pixel_writes_nodata() {
        #[rustfmt::skip]
        let data: [f64; 9] = [
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
        ];
        let mut mask_data = [1u8; 9];
        mask_data[1 * 3 + 1] = 0; // mask centre
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 1, 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// Non-zero weight pointing outside the array -> nodata even if mask is
    /// all-valid.
    #[test]
    fn nonzero_weight_oob_writes_nodata_regardless_of_mask() {
        let data = [1.0f64; 9];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64]; // left pixel -> col -1
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 1, 0, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// All-valid mask gives same result as the nomask_partial variant.
    #[test]
    fn all_valid_mask_agrees_with_nomask() {
        use crate::core::interp::gx_array_view_interp::NoOutputMask;
        #[rustfmt::skip]
        let data: [f64; 9] = [
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
        ];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.5, 0.5, 0.0f64];

        // masked_partial
        let mut out_masked = [0.0f64; 1];
        let mut arr_out1   = arr_mut(&mut out_masked, 1, 1);
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out1, 0, 1, 1, -9.0, &mut ctx,
        );

        // nomask_partial
        let mut out_nm   = [0.0f64; 1];
        let mut arr_out2 = arr_mut(&mut out_nm, 1, 1);
        let mut nm = NoOutputMask::default();
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out2, &mut nm, 0, 1, 1, -9.0,
        );

        assert!(
            approx(out_masked[0], out_nm[0]),
            "masked={} vs nomask={}",
            out_masked[0], out_nm[0]
        );
    }

    /// Masked pixel with zero weight does not invalidate the result.
    #[test]
    fn zero_weight_masked_pixel_ignored() {
        #[rustfmt::skip]
        let data: [f64; 9] = [
            0., 0., 0.,
            0., 5., 0.,
            0., 0., 0.,
        ];
        let mut mask_data = [1u8; 9];
        // Mask the right neighbour of centre — but its weight is 0
        mask_data[1 * 3 + 2] = 0;
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // Identity weights: only centre (col 1) has non-zero weight
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 1, 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Output mask is 1 on success, 0 when a masked pixel has non-zero weight.
    #[test]
    fn output_mask_set_on_success_and_failure() {
        use crate::core::gx_array::GxArrayViewMut;
        use crate::core::interp::gx_array_view_interp::{
            GxArrayViewInterpolationContext,
            BinaryInputMask,
            BinaryOutputMask,
            BoundsCheck,
        };
        let data = [1.0f64; 9];
        let arr_in = arr(&data, 3, 3);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];

        // Success
        let valid_mask = [1u8; 9];
        let mask_view_v = GxArrayView::new(&valid_mask, 1, 3, 3);
        let mut out1 = [0.0f64; 1];
        let mut mdata1 = [255u8; 1];
        let mut arr_out1 = arr_mut(&mut out1, 1, 1);
        let mut amask1 = GxArrayViewMut::new(&mut mdata1, 1, 1, 1);
        let mut ctx1 = GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask: &mask_view_v },
            BinaryOutputMask { mask: &mut amask1 },
            BoundsCheck,
        );
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out1, 0, 1, 1, -9.0, &mut ctx1,
        );
        assert_eq!(mdata1[0], 1, "success: mask should be 1");

        // Failure: centre masked
        let mut bad_mask = [1u8; 9];
        bad_mask[1 * 3 + 1] = 0;
        let mask_view_i = GxArrayView::new(&bad_mask, 1, 3, 3);
        let mut out2 = [42.0f64; 1];
        let mut mdata2 = [255u8; 1];
        let mut arr_out2 = arr_mut(&mut out2, 1, 1);
        let mut amask2 = GxArrayViewMut::new(&mut mdata2, 1, 1, 1);
        let mut ctx2 = GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask: &mask_view_i },
            BinaryOutputMask { mask: &mut amask2 },
            BoundsCheck,
        );
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out2, 0, 1, 1, -9.0, &mut ctx2,
        );
        assert_eq!(mdata2[0], 0, "failure: mask should be 0");
        assert_eq!(out2[0], -9.0, "failure: output should be nodata");
    }

    /// OOB on rows with non-zero weight -> nodata (row direction).
    #[test]
    fn oob_row_top_nonzero_weight_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // left row weight = 0.3 -> needs row -1 (OOB)
        let w_row = [0.0, 0.7, 0.3f64];
        let w_col = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 0, 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB row");
    }

    /// OOB on bottom row with non-zero weight -> nodata.
    #[test]
    fn oob_row_bottom_nonzero_weight_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        // right row weight = 0.3 -> needs row 3 (OOB)
        let w_row = [0.3, 0.7, 0.0f64];
        let w_col = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 2, 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB row bottom");
    }

    /// Multi-variable with all-valid mask and border: both planes correct.
    #[test]
    fn multivar_border_all_valid() {
        let mut data = [0.0f64; 18]; // 2 vars, 3×3
        data[1 * 3 + 1]     = 5.0;  // plane 0
        data[9 + 1 * 3 + 1] = 15.0; // plane 1
        let mask_data = [1u8; 9];
        let arr_in    = arr_nvar(&data, 2, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 1, 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 15.0), "plane1: {}", out[1]);
    }

    /// Multi-variable OOB: nodata written to all planes.
    #[test]
    fn multivar_oob_all_planes_nodata() {
        let data = [1.0f64; 18]; // 2 vars, 3×3
        let mask_data = [1u8; 9];
        let arr_in    = arr_nvar(&data, 2, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64]; // left -> col -1
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, 1, 0, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "plane0 expected nodata");
        assert_eq!(out[1], -9.0, "plane1 expected nodata");
    }

    /// Multi-variable masked: nodata written to all planes.
    #[test]
    fn multivar_masked_all_planes_nodata() {
        let data = [1.0f64; 18]; // 2 vars, 3×3
        let mut mask_data = [1u8; 9];
        mask_data[4] = 0; // centre
        let arr_in    = arr_nvar(&data, 2, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 1, 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "plane0 expected nodata");
        assert_eq!(out[1], -9.0, "plane1 expected nodata");
    }

    /// Sub-pixel both axes with all-valid mask -> correct value.
    #[test]
    fn subpixel_both_axes_all_valid() {
        #[rustfmt::skip]
        let data: [f64; 25] = [
            0., 0., 0., 0., 0.,
            0., 2., 4., 0., 0.,
            0., 6., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let mask_data = [1u8; 25];
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let w = [0.0, 0.5, 0.5f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, 2, 2, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Non-zero out_idx.
    #[test]
    fn nonzero_out_idx() {
        let data = [1.0f64; 25];
        let mask_data = [1u8; 25];
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockInterp;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 2, 2, 2, -9.0, &mut ctx,
        );
        assert!(approx(out[2], 1.0), "out[2]={}", out[2]);
        assert_eq!(out[0], -9.0, "out[0] untouched");
    }
}


// =============================================================================
// array1_interp2_separable_core
// =============================================================================

#[cfg(test)]
mod separable_core {
    use super::mock::*;
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolator,
        GxArrayViewInterpolationContext,
        DefaultCtx,
        BinaryInputMask,
        BinaryOutputMask,
        BoundsCheck,
        NoBoundsCheck,
        NoInputMask,
        NoOutputMask,
    };
    use crate::core::gx_array::{GxArrayView, GxArrayViewMut};

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_buf() -> Box<[f64]> {
        MockInterp.allocate_kernel_buffer()
    }

    // ------------------------------------------------------------------
    // Interior path (full stencil in bounds)
    // ------------------------------------------------------------------

    /// Integer position in the interior -> identity.
    #[test]
    fn interior_integer_position_identity() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 8.0), "got {}", out[0]);
    }

    /// Sub-pixel position: linear mix of two pixels.
    #[test]
    fn interior_subpixel_linear_mix() {
        // Row 2: [0, 4, 8, 0, 0]
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 4., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        // target_col = 1.5 -> centre at col 2 (rounded), rel = -0.5
        // weights: left=-0.5 weight -> w[2]=0.5, centre w[1]=0.5, right w[0]=0
        // -> 0.5*4 + 0.5*8 = 6
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 1.5, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Border path (centre in bounds, stencil may cross border)
    // ------------------------------------------------------------------

    /// Centre at row 0, col 2: the stencil would need row -1 but identity
    /// row weights give it zero weight -> result is data[0][2].
    #[test]
    fn border_top_identity_row_weights() {
        #[rustfmt::skip]
        let data = [
            0., 0., 5., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 3, 5);
        let mut out = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        // target at exactly (0.0, 2.0): centre_row=0, rel_row=0 -> identity
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 0.0, 2.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Outside path (centre outside the array)
    // ------------------------------------------------------------------

    /// Centre at col -1 -> entirely outside -> nodata.
    #[test]
    fn outside_centre_writes_nodata() {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.0, -1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// Centre at row `nrow` (below last row) -> nodata.
    #[test]
    fn outside_below_array_writes_nodata() {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 3.0, 1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Output mask
    // ------------------------------------------------------------------

    /// Output mask = 1 for a valid pixel, 0 for an outside pixel.
    #[test]
    fn output_mask_valid_and_invalid() {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);

        let mut out      = [0.0f64; 2];
        let mut mask_out = [255u8; 2];
        let mut arr_out  = arr_mut_nvar(&mut out,      1, 1, 2);

        let interp = MockInterp;
        let mut buf = make_buf();

        // Valid pixel at (2, 2)
        {
            let mut arr_mask = GxArrayViewMut::new(&mut mask_out, 1, 1, 2);
            let mut ctx = GxArrayViewInterpolationContext::new(
                NoInputMask::default(),
                BinaryOutputMask { mask: &mut arr_mask },
                BoundsCheck,
            );
            interp.array1_interp2::<f64, f64, _>(
                &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -9.0,
                &mut ctx,
            ).unwrap();
        }
        // Invalid pixel (outside)
        {
            let mut arr_mask = GxArrayViewMut::new(&mut mask_out, 1, 1, 2);
            let mut ctx = GxArrayViewInterpolationContext::new(
                NoInputMask::default(),
                BinaryOutputMask { mask: &mut arr_mask },
                BoundsCheck,
            );
            interp.array1_interp2::<f64, f64, _>(
                &mut buf, 1.0, -1.0, 1, &arr_in, &mut arr_out, -9.0,
                &mut ctx,
            ).unwrap();
        }

        assert_eq!(mask_out[0], 1, "valid pixel: mask should be 1");
        assert_eq!(mask_out[1], 0, "invalid pixel: mask should be 0");
    }

    // ------------------------------------------------------------------
    // Masked input
    // ------------------------------------------------------------------

    /// A masked centre pixel with active weight -> nodata via masked path.
    #[test]
    fn masked_centre_pixel_produces_nodata() {
        #[rustfmt::skip]
        let data: [f64; 25] = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 9., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let mut mask_data = [1u8; 25];
        mask_data[2 * 5 + 2] = 0;
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);

        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask: &mask_view },
            NoOutputMask::default(),
            BoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // NoBoundsCheck path
    // ------------------------------------------------------------------

    /// With `NoBoundsCheck`, an interior pixel returns the correct value.
    #[test]
    fn no_bounds_check_interior_correct_value() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 4., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            NoInputMask::default(),
            NoOutputMask::default(),
            NoBoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 4.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Multi-variable
    // ------------------------------------------------------------------

    /// Two variable planes are interpolated independently.
    #[test]
    fn multivar_planes_independent() {
        let mut data = [0.0f64; 50];
        data[2 * 5 + 2]      = 11.0;
        data[25 + 2 * 5 + 2] = 22.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 11.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 22.0), "plane1: {}", out[1]);
    }

    // ------------------------------------------------------------------
    // Return value
    // ------------------------------------------------------------------

    #[test]
    fn returns_ok_on_success() {
        let data = [1.0f64; 25];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        let result = interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        );
        assert!(result.is_ok());
    }
    // ------------------------------------------------------------------
    // Border path — non-zero weight OOB -> nodata
    // ------------------------------------------------------------------

    /// Centre at col 0 with non-zero left weight: stencil accesses col -1
    /// which is out of bounds -> nodata via border path.
    #[test]
    fn border_nonzero_weight_oob_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target_col = -0.3 -> centre at col 0, rel = -0.3
        // linear_weights_3(-0.3): left weight = 0.3 -> accesses col -1
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.0, -0.3, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0,
            "border OOB with non-zero weight should produce nodata");
    }

    /// Centre at row 0, col 1 with negative row offset: the stencil row -1
    /// is out of bounds but its weight is 0 -> result is data[0][1].
    #[test]
    fn border_top_zero_weight_row_identity_succeeds() {
        #[rustfmt::skip]
        let data = [
            0., 7., 0.,
            0., 0., 0.,
            0., 0., 0.,
        ];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target at exactly (0.0, 1.0): centre_row=0, rel_row=0 -> identity
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 0.0, 1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 7.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // NoBoundsCheck + active input mask
    // ------------------------------------------------------------------

    /// `NoBoundsCheck` with a masked centre pixel -> nodata (mask path still
    /// active even when bounds checking is disabled).
    #[test]
    fn no_bounds_check_masked_centre_writes_nodata() {
        let data = [1.0f64; 25];
        let mut mask_data = [1u8; 25];
        mask_data[2 * 5 + 2] = 0; // mask centre
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask: &mask_view },
            NoOutputMask::default(),
            NoBoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0,
            "NoBoundsCheck + masked centre should produce nodata");
    }

    // ------------------------------------------------------------------
    // Exact pixel positions at array corners
    // ------------------------------------------------------------------

    /// Pixel exactly at (0, 0) with identity weights: only the top-left
    /// pixel contributes; the stencil reaches (-1,-1) but with zero weight.
    #[test]
    fn exact_top_left_corner_identity() {
        let mut data = [0.0f64; 25];
        data[0] = 13.0; // top-left
        let arr_in  = arr(&data, 5, 5);
        let mut out = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 0.0, 0.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 13.0), "got {}", out[0]);
    }

    /// Pixel exactly at `(nrow-1, ncol-1)` with identity weights: only the
    /// bottom-right pixel contributes.
    #[test]
    fn exact_bottom_right_corner_identity() {
        let mut data = [0.0f64; 25];
        data[24] = 17.0; // bottom-right of 5×5
        let arr_in  = arr(&data, 5, 5);
        let mut out = [-9.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 4.0, 4.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 17.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Sub-pixel offset in both row AND col simultaneously
    // ------------------------------------------------------------------

    /// Interpolation with non-zero sub-pixel offset in both directions at once.
    /// This validates that the separable decomposition (row ⊗ col) works
    /// correctly when neither axis is at identity.
    #[test]
    fn subpixel_both_axes_simultaneously() {
        // 5×5 array with a 2×2 block of non-zero values at (1,1)..(2,2):
        //   row 1, col 1 = 2.0    row 1, col 2 = 4.0
        //   row 2, col 1 = 6.0    row 2, col 2 = 8.0
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 2., 4., 0., 0.,
            0., 6., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        // target at (1.5, 1.5):
        //   centre_row = round(1.5+0.5) = 2,  rel_row = 1.5 - 2 = -0.5
        //   centre_col = round(1.5+0.5) = 2,  rel_col = 1.5 - 2 = -0.5
        //   row weights: left=0.5, centre=0.5, right=0
        //   col weights: left=0.5, centre=0.5, right=0
        //
        //   row 1 (irow=0, weight=w_row[2]=0.5):
        //     col 1 (icol=0, weight=w_col[2]=0.5): 2.0*0.5 = 1.0
        //     col 2 (icol=1, weight=w_col[1]=0.5): 4.0*0.5 = 2.0
        //     acc_col = 3.0 -> acc += 0.5 * 3.0 = 1.5
        //   row 2 (irow=1, weight=w_row[1]=0.5):
        //     col 1 (icol=0, weight=w_col[2]=0.5): 6.0*0.5 = 3.0
        //     col 2 (icol=1, weight=w_col[1]=0.5): 8.0*0.5 = 4.0
        //     acc_col = 7.0 -> acc += 0.5 * 7.0 = 3.5
        //   total = 1.5 + 3.5 = 5.0
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.5, 1.5, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Non-zero out_idx
    // ------------------------------------------------------------------

    /// Writing to a non-zero `out_idx` places the value at the correct offset.
    #[test]
    fn nonzero_out_idx_writes_at_correct_offset() {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        // Output array has 4 pixels (1×4), we write at index 2.
        let mut out = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 2, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        // Index 2 should have the interpolated value; others untouched.
        assert!(approx(out[2], 1.0), "out[2]={}", out[2]);
        assert_eq!(out[0], -9.0, "out[0] should be untouched");
        assert_eq!(out[1], -9.0, "out[1] should be untouched");
        assert_eq!(out[3], -9.0, "out[3] should be untouched");
    }

    /// Non-zero out_idx with multivar: each plane writes at the correct offset.
    #[test]
    fn nonzero_out_idx_multivar() {
        let mut data = [0.0f64; 50]; // 2 vars, 5×5
        data[2 * 5 + 2]      = 11.0; // plane 0
        data[25 + 2 * 5 + 2] = 22.0; // plane 1
        let arr_in = arr_nvar(&data, 2, 5, 5);
        // Output: 2 vars, 1×4 = 4 pixels per plane, total 8 elements.
        let mut out = [-9.0f64; 8];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 4);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 3, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        // plane 0 at out_idx=3
        assert!(approx(out[3], 11.0), "plane0 out[3]={}", out[3]);
        // plane 1 at out_idx=3 + var_size(4) = 7
        assert!(approx(out[7], 22.0), "plane1 out[7]={}", out[7]);
    }

    // ------------------------------------------------------------------
    // Outside path — all four sides
    // ------------------------------------------------------------------

    /// Centre at row -1 -> outside above -> nodata.
    #[test]
    fn outside_above_array_writes_nodata() {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, -1.0, 1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// Centre at col = ncol -> outside right -> nodata.
    #[test]
    fn outside_right_of_array_writes_nodata() {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.0, 3.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Multivar nodata — all planes receive nodata on outside path
    // ------------------------------------------------------------------

    /// When the centre is outside, nodata is written to all variable planes.
    #[test]
    fn outside_multivar_all_planes_nodata() {
        let data = [1.0f64; 50]; // 2 vars, 5×5
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, -1.0, 2.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "plane0 expected nodata");
        assert_eq!(out[1], -9.0, "plane1 expected nodata");
    }

    // ------------------------------------------------------------------
    // Border path — non-zero weight OOB on rows
    // ------------------------------------------------------------------

    /// Centre at row 0 with non-zero upward weight: stencil accesses row -1
    /// -> nodata via border path.
    #[test]
    fn border_nonzero_weight_oob_row_top_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target_row = -0.3 -> centre at row 0, rel = -0.3
        // linear_weights_3(-0.3): left weight = 0.3 -> accesses row -1
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, -0.3, 1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0,
            "border OOB row with non-zero weight should produce nodata");
    }

    /// Centre at last row with non-zero downward weight: stencil accesses
    /// row = nrow -> nodata via border path.
    #[test]
    fn border_nonzero_weight_oob_row_bottom_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target_row = 2.3 -> centre at row 2 (last row of 3×3), rel = 0.3
        // linear_weights_3(0.3): right weight = 0.3 -> accesses row 3 (OOB)
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.3, 1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0,
            "border OOB row bottom with non-zero weight should produce nodata");
    }

    // ------------------------------------------------------------------
    // Multivar border path — nodata written to all planes
    // ------------------------------------------------------------------

    /// Border OOB with multivar: nodata is written to every plane.
    #[test]
    fn border_nonzero_weight_oob_multivar_nodata() {
        let data = [1.0f64; 18]; // 2 vars, 3×3
        let arr_in = arr_nvar(&data, 2, 3, 3);
        let mut out = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target_col = -0.3 -> centre at col 0, rel = -0.3, left weight 0.3
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.0, -0.3, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "plane0 expected nodata");
        assert_eq!(out[1], -9.0, "plane1 expected nodata");
    }

    // ------------------------------------------------------------------
    // NoBoundsCheck + multivar
    // ------------------------------------------------------------------

    /// `NoBoundsCheck` with multivar: both planes interpolated correctly.
    #[test]
    fn no_bounds_check_multivar() {
        let mut data = [0.0f64; 50]; // 2 vars, 5×5
        data[2 * 5 + 2]      = 3.0;
        data[25 + 2 * 5 + 2] = 7.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            NoInputMask::default(),
            NoOutputMask::default(),
            NoBoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 3.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 7.0), "plane1: {}", out[1]);
    }

    // ------------------------------------------------------------------
    // NoBoundsCheck with sub-pixel on both axes
    // ------------------------------------------------------------------

    /// `NoBoundsCheck` with sub-pixel offset on both row and col.
    #[test]
    fn no_bounds_check_subpixel_both_axes() {
        #[rustfmt::skip]
        let data = [
            0., 0., 0., 0., 0.,
            0., 2., 4., 0., 0.,
            0., 6., 8., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
        ];
        let arr_in  = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            NoInputMask::default(),
            NoOutputMask::default(),
            NoBoundsCheck,
        );
        // Same as subpixel_both_axes_simultaneously: target (1.5, 1.5) -> 5.0
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.5, 1.5, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Output mask on border OOB
    // ------------------------------------------------------------------

    /// Output mask is set to 0 when border path produces nodata.
    #[test]
    fn output_mask_zero_on_border_oob() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut mask_out = [255u8; 1];
        let mut arr_out  = arr_mut(&mut out, 1, 1);
        let mut arr_mask = GxArrayViewMut::new(&mut mask_out, 1, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target_col = -0.3 -> border OOB with non-zero weight
        let mut ctx = GxArrayViewInterpolationContext::new(
            NoInputMask::default(),
            BinaryOutputMask { mask: &mut arr_mask },
            BoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.0, -0.3, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(mask_out[0], 0, "border OOB: mask should be 0");
    }
}


// =============================================================================
// is_valid_weighted_window — direct unit tests
// =============================================================================

#[cfg(test)]
mod is_valid_weighted_window_tests {
    use crate::core::gx_array::GxArrayView;
    use crate::core::interp::gx_array_view_interp::{
        BinaryInputMask,
        GxArrayViewInterpolatorInputMaskStrategy,
    };

    /// All-valid mask with identity weights -> returns 1.
    #[test]
    fn all_valid_returns_one() {
        let mask_data = [1u8; 9]; // 3×3
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        let w = [0.0, 1.0, 0.0f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 1);
    }

    /// Centre pixel masked with non-zero weight -> returns 0.
    #[test]
    fn centre_masked_returns_zero() {
        let mut mask_data = [1u8; 9];
        mask_data[4] = 0; // centre of 3×3
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        let w = [0.0, 1.0, 0.0f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 0);
    }

    /// Masked pixel at a position with zero weight -> returns 1 (ignored).
    #[test]
    fn masked_zero_weight_returns_one() {
        let mut mask_data = [1u8; 9];
        mask_data[5] = 0; // (row 1, col 2) — right of centre
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        // Identity weights: only centre has non-zero weight.
        let w = [0.0, 1.0, 0.0f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 1);
    }

    /// Row with zero weight is entirely skipped even if it contains masked
    /// pixels.
    #[test]
    fn zero_weight_row_entirely_skipped() {
        let mut mask_data = [1u8; 9];
        // Mask every pixel in the first row (row 0).
        mask_data[0] = 0;
        mask_data[1] = 0;
        mask_data[2] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        // Row weights: w[2]=0 (row 0 has zero weight), w[1]=1, w[0]=0.
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 1.0, 0.0f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w_row, &w_col, &mut cache,
        );
        assert_eq!(result, 1, "zero-weight row should be skipped");
    }

    /// Column with zero weight is skipped even if masked.
    #[test]
    fn zero_weight_col_skipped() {
        let mut mask_data = [1u8; 9];
        // Mask all pixels in column 0.
        mask_data[0] = 0;
        mask_data[3] = 0;
        mask_data[6] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        // Col weights: w[2]=0 (col 0 zero weight), w[1]=1, w[0]=0.
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 1.0, 0.0f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w_row, &w_col, &mut cache,
        );
        assert_eq!(result, 1, "zero-weight column should be skipped");
    }

    /// With non-identity weights where all active positions are valid -> 1.
    #[test]
    fn all_active_valid_nonidentity_weights() {
        let mask_data = [1u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        let w = [0.25, 0.5, 0.25f64]; // all non-zero
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 1);
    }

    /// Corner pixel masked with all non-zero weights -> 0.
    #[test]
    fn corner_masked_all_nonzero_weights() {
        let mut mask_data = [1u8; 9];
        mask_data[0] = 0; // top-left
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        let w = [0.25, 0.5, 0.25f64]; // all non-zero
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 0);
    }

    /// Window at a non-zero start_idx (windowing into a larger array).
    #[test]
    fn nonzero_start_idx() {
        // 5×5 mask, window the central 3×3 starting at flat index 6 (row 1, col 1).
        let mask_data = [1u8; 25];
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        let w = [0.0, 1.0, 0.0f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            6, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 1);
    }
}


// =============================================================================
// NoInputMask — direct unit tests
// =============================================================================

#[cfg(test)]
mod no_input_mask_tests {
    use crate::core::interp::gx_array_view_interp::{
        NoInputMask,
        GxArrayViewInterpolatorInputMaskStrategy,
    };

    /// `NoInputMask::is_valid` always returns 1 regardless of index.
    #[test]
    fn is_valid_always_one() {
        let mask = NoInputMask;
        assert_eq!(mask.is_valid(0), 1);
        assert_eq!(mask.is_valid(999), 1);
    }

    /// `NoInputMask::is_enabled` returns false.
    #[test]
    fn is_enabled_false() {
        let mask = NoInputMask;
        assert!(!mask.is_enabled());
    }

    /// `NoInputMask::is_valid_weighted_window` always returns 1.
    #[test]
    fn weighted_window_always_one() {
        let mask = NoInputMask;
        let w = [0.25, 0.5, 0.25f64];
        let mut cache = [0u8; 9];
        let result = mask.is_valid_weighted_window(
            0, 3, 3, &w, &w, &mut cache,
        );
        assert_eq!(result, 1);
    }
}


// =============================================================================
// NoOutputMask — direct unit tests
// =============================================================================

#[cfg(test)]
mod no_output_mask_tests {
    use crate::core::interp::gx_array_view_interp::{
        NoOutputMask,
        GxArrayViewInterpolatorOutputMaskStrategy,
    };

    /// `NoOutputMask::is_enabled` returns false.
    #[test]
    fn is_enabled_false() {
        let mask = NoOutputMask;
        assert!(!mask.is_enabled());
    }

    /// `NoOutputMask::set_value` is a no-op and does not panic.
    #[test]
    fn set_value_is_noop() {
        let mut mask = NoOutputMask;
        // Should not panic even with arbitrary indices.
        mask.set_value(0, 1);
        mask.set_value(9999, 0);
    }
}


// =============================================================================
// BoundsCheck / NoBoundsCheck — strategy tests
// =============================================================================

#[cfg(test)]
mod bounds_check_strategy_tests {
    use crate::core::interp::gx_array_view_interp::{
        BoundsCheck,
        NoBoundsCheck,
        GxArrayViewInterpolatorBoundsCheckStrategy,
    };

    #[test]
    fn bounds_check_returns_true() {
        assert!(BoundsCheck::do_check());
    }

    #[test]
    fn no_bounds_check_returns_false() {
        assert!(!NoBoundsCheck::do_check());
    }
}

