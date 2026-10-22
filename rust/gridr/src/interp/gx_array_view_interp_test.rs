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
//! Kernel size: 3×3 → `KROWS = 3`, `KCOLS = 3`
//!
//! # Coverage
//!
//! For each method the following scenarios are tested:
//! - **Nominal** — interior pixel, sub-pixel position, correct output value.
//! - **Identity** — integer position, output equals source value exactly.
//! - **Multi-variable** — `nvar > 1`, each plane interpolated independently.
//! - **Boundary** — stencil exactly on the first/last valid row or column.
//! - **Out-of-bounds** — pixel whose stencil (partially) exits the array.
//! - **Masked input** — one pixel in the stencil masked → `nodata` output.
//! - **All-valid mask** — full-ones mask gives the same result as no mask.
//! - **Output mask** — output mask value is `1` on success, `0` on failure.

#[cfg(test)]
mod mock {
    //! Shared mock interpolator used by all test modules.

    use crate::gx_array::{GxArrayView, GxArrayViewMut};
    use crate::gx_errors::GxError;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolationContextTrait,
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolator,
        GxArrayViewInterpolatorArgs,
    };

    // ------------------------------------------------------------------
    // Weight function
    // ------------------------------------------------------------------

    /// Native linear weights (2 taps, anchor = floor, phase `t` in `[0, 1)`).
    /// Inverted order: `weights[0]` = tap `anchor + 1`, `weights[1]` = anchor.
    pub fn linear_weights_2(t: f64, weights: &mut [f64]) {
        weights[0] = t;
        weights[1] = 1.0 - t;
    }

    /// Quadratic B-spline weights (3 taps, odd support, anchor = nearest
    /// integer, phase `t` in `[-0.5, 0.5]`). Inverted order.
    /// Note : only here for test purpose - only odd order BSpline are 
    /// implemented as interpolators
    pub fn quadratic_bspline_weights_3(t: f64, weights: &mut [f64]) {
        weights[0] = 0.5 * (0.5 + t) * (0.5 + t);
        weights[1] = 0.75 - t * t;
        weights[2] = 0.5 * (0.5 - t) * (0.5 - t);
    }

    /// Even-support fixture (native linear) used through
    /// `array1_interp2_separable_core`.
    pub struct MockInterp;

    impl GxArrayViewInterpolatorCore<2, 2> for MockInterp {
        fn compute_weights(&self, t: f64, weights: &mut [f64]) {
            linear_weights_2(t, weights);
        }
    }

    /// Odd-support fixture (3 taps). The variant tests pass their weights
    /// explicitly; `compute_weights` is a genuine odd kernel so that the odd
    /// anchoring branch of the core stays covered.
    pub struct MockOdd3;

    impl GxArrayViewInterpolatorCore<3, 3> for MockOdd3 {
        fn compute_weights(&self, t: f64, weights: &mut [f64]) {
            quadratic_bspline_weights_3(t, weights);
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
            vec![0.0f64; 2 + 2].into_boxed_slice()
        }
        fn kernel_row_size(&self) -> usize { 2 }
        fn kernel_col_size(&self) -> usize { 2 }
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

    /// Trivial `GxArrayViewInterpolator` impl so `MockInterp` compiles as
    /// a complete type.  Most methods panic — only those exercised in tests
    /// matter.
    impl GxArrayViewInterpolator for MockOdd3 {
        fn new(_args: &dyn GxArrayViewInterpolatorArgs) -> Self {
            MockOdd3
        }
        fn shortname(&self) -> String {
            "mock_odd3".to_string()
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
    use crate::interp::gx_array_view_interp::GxArrayViewInterpolatorCore;

    // Flat 5×5 input — all values equal to `v` → any position returns `v`.
    fn uniform(v: f64) -> [f64; 25] {
        [v; 25]
    }

    /// Integer centre (x = 0): identity weights → output equals source pixel.
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
        let interp = MockOdd3;
        // Weights: identity at offset 0
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1,
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
        let interp = MockOdd3;
        // x = 0.5 - weights: [0.5, 0.5, 0.0] (right=0.5, centre=0.5, left=0)
        let w_row = [0.0, 1.0, 0.0f64]; // no row interpolation
        let w_col = [0.5, 0.5, 0.0f64]; // col: 0.5 right + 0.5 centre
        // Centre at (2, 2), col offset +0.5 → mixes col 2 (4) and col 3 (8)
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1,
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
        let interp = MockOdd3;
        let w = [0.25, 0.5, 0.25f64]; // sums to 1
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1,
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1,
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64]; // identity
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, (3) - 1, (3) - 1,
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
        let interp = MockOdd3;
        let w_row = [0.0, 1.0, 0.0f64];
        // x = −0.5 → weights: left 0.5 (w[2]), centre 0.5 (w[1]), right 0
        let w_col = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1,
        );
        // 0.5 * 8 + 0.5 * 4 = 6
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }
}


// =============================================================================
// interpolate_nomask_partial
// =============================================================================

#[cfg(test)]
mod nomask_partial {
    use super::mock::*;
    use crate::interp::gx_array_view_interp::{
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, &mut no_mask(), 0, (2) - 1, (2) - 1, -1.0,
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
        let interp = MockOdd3;
        // Row weights: identity → only centre row (row 0) contributes.
        // Column weights: identity → only col 2 contributes.
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, &mut no_mask(), 0, (0) - 1, (2) - 1, -9.0,
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
        let interp = MockOdd3;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.5, 0.5, 0.0f64]; // right=0.5, centre=0.5
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, (1) - 1, (1) - 1, -9.0,
        );
        // 0.5 * 4 + 0.5 * 8 = 6
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    /// Stencil position with non-zero weight is out of bounds → nodata.
    #[test]
    fn nonzero_weight_oob_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockOdd3;
        // Centre at col 0, with non-zero weight for col -1 (out of bounds)
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64]; // left has weight 0.5
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, (1) - 1, (0) - 1, -9.0,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// Output mask is set to 1 on success and 0 on failure.
    #[test]
    fn output_mask_set_correctly() {
        use crate::gx_array::GxArrayViewMut;
        use crate::interp::gx_array_view_interp::BinaryOutputMask;

        let data = [1.0f64; 9];
        let arr_in = arr(&data, 3, 3);

        // Success case
        let mut out1 = [0.0f64; 1];
        let mut mask1_data = [255u8; 1];
        let mut arr_out1    = arr_mut(&mut out1, 1, 1);
        let mut arr_mask1 =
            GxArrayViewMut::new(&mut mask1_data, 1, 1, 1);
        let mut om1 = BinaryOutputMask { mask: &mut arr_mask1 };
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out1, &mut om1, 0, (1) - 1, (1) - 1, -9.0,
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
            &w, &w_col, &arr_in, &mut arr_out2, &mut om2, 0, (1) - 1, (0) - 1, -9.0,
        );
        assert_eq!(mask2_data[0], 0, "failure: mask should be 0");
    }
}


// =============================================================================
// interpolate_masked_unchecked
// =============================================================================

#[cfg(test)]
mod masked_unchecked {
    use super::mock::*;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        NoOutputMask,
        BoundsCheck,
    };
    use crate::gx_array::GxArrayView;

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

    /// All-valid mask → same result as nomask variant.
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -1.0, &mut ctx,
        );
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    /// The centre pixel is masked → nodata written.
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -9.0, &mut ctx,
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64]; // only centre has non-zero weight
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -9.0, &mut ctx,
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -1.0, &mut ctx,
        );
        assert!(approx(out[0], 3.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 7.0), "plane1: {}", out[1]);
    }

    /// Output mask is set to 1 when all inputs are valid, 0 when any is masked.
    #[test]
    fn output_mask_valid_and_invalid() {
        use crate::gx_array::GxArrayViewMut;
        use crate::interp::gx_array_view_interp::{
            GxArrayViewInterpolationContext,
            BinaryInputMask,
            BinaryOutputMask,
            BoundsCheck,
        };
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let interp = MockOdd3;
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
            &w, &w, &arr_in, &mut arr_out1, 0, (2) - 1, (2) - 1, -9.0, &mut ctx1,
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
            &w, &w, &arr_in, &mut arr_out2, 0, (2) - 1, (2) - 1, -9.0, &mut ctx2,
        );
        assert_eq!(mask2_data[0], 0, "invalid: mask should be 0");
        assert_eq!(out2[0], -9.0, "invalid: output should be nodata");
    }
}


// =============================================================================
// interpolate_masked_partial
// =============================================================================

#[cfg(test)]
mod masked_partial {
    use super::mock::*;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        NoOutputMask,
        BoundsCheck,
    };
    use crate::gx_array::GxArrayView;

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

    /// Interior pixel, all valid → correct interpolated value.
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (1) - 1, (1) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Centre at boundary row 0 with identity weights (zero-weight OOB row)
    /// and all-valid mask → succeeds.
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (0) - 1, (1) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Masked pixel with non-zero weight → nodata.
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
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (1) - 1, (1) - 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// Non-zero weight pointing outside the array → nodata even if mask is
    /// all-valid.
    #[test]
    fn nonzero_weight_oob_writes_nodata_regardless_of_mask() {
        let data = [1.0f64; 9];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockOdd3;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64]; // left pixel → col -1
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, (1) - 1, (0) - 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata, got {}", out[0]);
    }

    /// All-valid mask gives same result as the nomask_partial variant.
    #[test]
    fn all_valid_mask_agrees_with_nomask() {
        use crate::interp::gx_array_view_interp::NoOutputMask;
        #[rustfmt::skip]
        let data: [f64; 9] = [
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.,
        ];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let interp = MockOdd3;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.5, 0.5, 0.0f64];

        // masked_partial
        let mut out_masked = [0.0f64; 1];
        let mut arr_out1   = arr_mut(&mut out_masked, 1, 1);
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out1, 0, (1) - 1, (1) - 1, -9.0, &mut ctx,
        );

        // nomask_partial
        let mut out_nm   = [0.0f64; 1];
        let mut arr_out2 = arr_mut(&mut out_nm, 1, 1);
        let mut nm = NoOutputMask::default();
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out2, &mut nm, 0, (1) - 1, (1) - 1, -9.0,
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
        let interp = MockOdd3;
        // Identity weights: only centre (col 1) has non-zero weight
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (1) - 1, (1) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Output mask is 1 on success, 0 when a masked pixel has non-zero weight.
    #[test]
    fn output_mask_set_on_success_and_failure() {
        use crate::gx_array::GxArrayViewMut;
        use crate::interp::gx_array_view_interp::{
            GxArrayViewInterpolationContext,
            BinaryInputMask,
            BinaryOutputMask,
            BoundsCheck,
        };
        let data = [1.0f64; 9];
        let arr_in = arr(&data, 3, 3);
        let interp = MockOdd3;
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
            &w, &w, &arr_in, &mut arr_out1, 0, (1) - 1, (1) - 1, -9.0, &mut ctx1,
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
            &w, &w, &arr_in, &mut arr_out2, 0, (1) - 1, (1) - 1, -9.0, &mut ctx2,
        );
        assert_eq!(mdata2[0], 0, "failure: mask should be 0");
        assert_eq!(out2[0], -9.0, "failure: output should be nodata");
    }
}


// =============================================================================
// array1_interp2_separable_core
// =============================================================================

#[cfg(test)]
mod separable_core {
    use super::mock::*;
    use crate::interp::gx_array_view_interp::{
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
    use crate::gx_array::{GxArrayView, GxArrayViewMut};

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_buf() -> Box<[f64]> {
        MockInterp.allocate_kernel_buffer()
    }

    // ------------------------------------------------------------------
    // Interior path (full stencil in bounds)
    // ------------------------------------------------------------------

    /// Integer position in the interior → identity.
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
        // target_col = 1.5 → centre at col 2 (rounded), rel = -0.5
        // weights: left=-0.5 weight → w[2]=0.5, centre w[1]=0.5, right w[0]=0
        // → 0.5*4 + 0.5*8 = 6
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 1.5, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 6.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Border path (centre in bounds, stencil may cross border)
    // ------------------------------------------------------------------

    /// Centre at row 0, col 2: the stencil would need row -1 but identity
    /// row weights give it zero weight → result is data[0][2].
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
        // target at exactly (0.0, 2.0): centre_row=0, rel_row=0 → identity
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 0.0, 2.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // Outside path (centre outside the array)
    // ------------------------------------------------------------------

    /// Centre at col -1 → entirely outside → nodata.
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

    /// Centre at row `nrow` (below last row) → nodata.
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

    /// A masked centre pixel with active weight → nodata via masked path.
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

    /// Runs `MockInterp` with `NoBoundsCheck` at `(y, x)` on a 5 x 5 array.
    #[cfg(debug_assertions)]
    fn run_no_bounds_check(y: f64, x: f64) {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let mut out = [0.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            NoInputMask::default(),
            NoOutputMask::default(),
            NoBoundsCheck,
        );
        let _ = interp.array1_interp2::<f64, f64, _>(
            &mut buf, y, x, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        );
    }

    /// With `NoBoundsCheck`, a stencil starting before the array breaks the
    /// caller contract: debug builds stop on the assertion instead of
    /// reading out of bounds.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "NoBoundsCheck contract violated")]
    fn no_bounds_check_contract_negative() {
        run_no_bounds_check(2.0, -0.5);
    }

    /// The contract covers the full stencil: at `x = 4.0` the linear
    /// stencil reads columns 4 and 5 of a 5-column array, even though 
    /// the weight of column 5 is zero.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "NoBoundsCheck contract violated")]
    fn no_bounds_check_contract_last_column() {
        run_no_bounds_check(2.0, 4.0);
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
    // Border path — non-zero weight OOB → nodata
    // ------------------------------------------------------------------

    /// Centre at col 0 with non-zero left weight: stencil accesses col -1
    /// which is out of bounds → nodata via border path.
    #[test]
    fn border_nonzero_weight_oob_writes_nodata() {
        let data = [1.0f64; 9]; // 3×3
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        // target_col = -0.3 → centre at col 0, rel = -0.3
        // linear_weights_3(-0.3): left weight = 0.3 → accesses col -1
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.0, -0.3, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0,
            "border OOB with non-zero weight should produce nodata");
    }

    /// Centre at row 0, col 1 with negative row offset: the stencil row -1
    /// is out of bounds but its weight is 0 → result is data[0][1].
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
        // target at exactly (0.0, 1.0): centre_row=0, rel_row=0 → identity
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 0.0, 1.0, 0, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 7.0), "got {}", out[0]);
    }

    // ------------------------------------------------------------------
    // NoBoundsCheck + active input mask
    // ------------------------------------------------------------------

    /// `NoBoundsCheck` with a masked centre pixel → nodata (mask path still
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

    /// Sub-pixel offset on both axes simultaneously validates separable
    /// decomposition.
    #[test]
    fn subpixel_both_axes_simultaneously() {
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
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.5, 1.5, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Non-zero out_idx with multivar: each plane writes at the correct offset.
    #[test]
    fn nonzero_out_idx_multivar() {
        let mut data = [0.0f64; 50];
        data[2 * 5 + 2]      = 11.0;
        data[25 + 2 * 5 + 2] = 22.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [-9.0f64; 8];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 4);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 3, &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[3], 11.0), "plane0 out[3]={}", out[3]);
        assert!(approx(out[7], 22.0), "plane1 out[7]={}", out[7]);
    }

    /// Output mask is set to 0 when border path produces nodata.
    #[test]
    fn output_mask_zero_on_border_oob() {
        let data = [1.0f64; 9];
        let arr_in = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut mask_out = [255u8; 1];
        let mut arr_out  = arr_mut(&mut out, 1, 1);
        let mut arr_mask = GxArrayViewMut::new(&mut mask_out, 1, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
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
// Parametric tests (rstest) — additional coverage
// =============================================================================
//
// The following modules use `rstest` to cover many parameter combinations
// with minimal code duplication.  They complement (and do NOT replace) the
// hand-written tests above.
//
// Covered gaps:
//   - Outside path on all four sides
//   - Border OOB on rows (top & bottom)
//   - Non-zero out_idx (single-var & multivar)
//   - Sub-pixel on both axes for every method variant
//   - Multivar nodata propagation
//   - NoBoundsCheck combined with mask / multivar / sub-pixel
//   - Direct tests for `is_valid_weighted_window`, `write_nodata_all_vars`,
//     `NoInputMask`, `NoOutputMask`, `BoundsCheck`/`NoBoundsCheck` strategies.

// =============================================================================
// separable_core — parametric outside / border / nonzero-out_idx
// =============================================================================

#[cfg(test)]
mod separable_core_rstest {
    use super::mock::*;
    use rstest::rstest;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolator,
        GxArrayViewInterpolationContext,
        DefaultCtx,
        BinaryInputMask,
        NoBoundsCheck,
        NoInputMask,
        NoOutputMask,
    };
    use crate::gx_array::GxArrayView;

    fn make_buf() -> Box<[f64]> {
        MockInterp.allocate_kernel_buffer()
    }

    // ------------------------------------------------------------------
    // Outside path — centre entirely outside the array → nodata
    // ------------------------------------------------------------------

    /// The centre is outside the 3×3 array on the specified side → nodata.
    #[rstest]
    #[case::above(-1.0, 1.0)]
    #[case::below( 3.0, 1.0)]
    #[case::left(  1.0, -1.0)]
    #[case::right( 1.0, 3.0)]
    fn outside_centre_writes_nodata(
        #[case] target_row: f64,
        #[case] target_col: f64,
    ) {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, target_row, target_col, 0,
            &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "expected nodata for ({target_row}, {target_col})");
    }

    /// Outside with 2 variable planes: nodata must be written to both.
    #[rstest]
    #[case::above(-1.0, 2.0)]
    #[case::below( 5.0, 2.0)]
    #[case::left(  2.0, -1.0)]
    #[case::right( 2.0, 5.0)]
    fn outside_multivar_all_planes_nodata(
        #[case] target_row: f64,
        #[case] target_col: f64,
    ) {
        let data = [1.0f64; 50];
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, target_row, target_col, 0,
            &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "plane0");
        assert_eq!(out[1], -9.0, "plane1");
    }

    // ------------------------------------------------------------------
    // Border path — non-zero weight reaches outside → nodata
    // ------------------------------------------------------------------

    /// Stencil with non-zero weight reaching outside via border path.
    #[rstest]
    #[case::row_top(-0.3, 1.0)]
    #[case::row_bottom(2.3, 1.0)]
    #[case::col_left(1.0, -0.3)]
    #[case::col_right(1.0, 2.3)]
    fn border_nonzero_weight_oob_writes_nodata(
        #[case] target_row: f64,
        #[case] target_col: f64,
    ) {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, target_row, target_col, 0,
            &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0,
            "border OOB at ({target_row}, {target_col}) should produce nodata");
    }

    /// Border OOB with multivar: nodata on all planes.
    #[rstest]
    #[case::col_left(1.0, -0.3)]
    #[case::row_top(-0.3, 1.0)]
    fn border_oob_multivar_nodata(
        #[case] target_row: f64,
        #[case] target_col: f64,
    ) {
        let data = [1.0f64; 18];
        let arr_in = arr_nvar(&data, 2, 3, 3);
        let mut out = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, target_row, target_col, 0,
            &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert_eq!(out[0], -9.0, "plane0");
        assert_eq!(out[1], -9.0, "plane1");
    }

    // ------------------------------------------------------------------
    // Non-zero out_idx
    // ------------------------------------------------------------------

    /// Writing at different out_idx values places the result correctly.
    #[rstest]
    #[case::idx0(0)]
    #[case::idx1(1)]
    #[case::idx2(2)]
    #[case::idx3(3)]
    fn nonzero_out_idx(#[case] idx: usize) {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let mut out = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = DefaultCtx::default();
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, idx,
            &arr_in, &mut arr_out, -9.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[idx], 1.0), "out[{idx}]={}", out[idx]);
        for i in 0..4 {
            if i != idx {
                assert_eq!(out[i], -9.0, "out[{i}] should be untouched");
            }
        }
    }

    // ------------------------------------------------------------------
    // NoBoundsCheck paths
    // ------------------------------------------------------------------

    /// NoBoundsCheck: multivar at integer position.
    #[test]
    fn no_bounds_check_multivar() {
        let mut data = [0.0f64; 50];
        data[2 * 5 + 2]      = 3.0;
        data[25 + 2 * 5 + 2] = 7.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockInterp;
        let mut buf = make_buf();
        let mut ctx = GxArrayViewInterpolationContext::new(
            NoInputMask::default(), NoOutputMask::default(), NoBoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 2.0, 2.0, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 3.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 7.0), "plane1: {}", out[1]);
    }

    /// NoBoundsCheck: sub-pixel both axes.
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
            NoInputMask::default(), NoOutputMask::default(), NoBoundsCheck,
        );
        interp.array1_interp2::<f64, f64, _>(
            &mut buf, 1.5, 1.5, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
        ).unwrap();
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// NoBoundsCheck + masked centre pixel → nodata (mask path still active).
    #[test]
    fn no_bounds_check_masked_centre_writes_nodata() {
        let data = [1.0f64; 25];
        let mut mask_data = [1u8; 25];
        mask_data[2 * 5 + 2] = 0;
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
        assert_eq!(out[0], -9.0);
    }
}


// =============================================================================
// nomask_unchecked — parametric additional tests
// =============================================================================

#[cfg(test)]
mod nomask_unchecked_rstest {
    use super::mock::*;
    use rstest::rstest;
    use crate::interp::gx_array_view_interp::GxArrayViewInterpolatorCore;

    /// Non-zero out_idx: the interpolated value is written at the correct
    /// position in the output array.
    #[rstest]
    #[case::idx0(0)]
    #[case::idx1(1)]
    #[case::idx2(2)]
    fn nonzero_out_idx(#[case] idx: usize) {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let mut out = [-9.0f64; 3];
        let mut arr_out = arr_mut(&mut out, 1, 3);
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, idx, (2) - 1, (2) - 1,
        );
        assert!(approx(out[idx], 1.0), "out[{idx}]={}", out[idx]);
        for i in 0..3 {
            if i != idx {
                assert_eq!(out[i], -9.0, "out[{i}] untouched");
            }
        }
    }

    /// Sub-pixel both axes.
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
        let interp = MockOdd3;
        let w = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Multivar with non-zero out_idx.
    #[test]
    fn multivar_nonzero_out_idx() {
        let mut data = [0.0f64; 50];
        data[2 * 5 + 2]      = 10.0;
        data[25 + 2 * 5 + 2] = 20.0;
        let arr_in = arr_nvar(&data, 2, 5, 5);
        let mut out = [-9.0f64; 6];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 3);
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_unchecked::<f64, f64>(
            &w, &w, &arr_in, &mut arr_out, 1, (2) - 1, (2) - 1,
        );
        assert!(approx(out[1], 10.0), "plane0: {}", out[1]);
        assert!(approx(out[4], 20.0), "plane1: {}", out[4]);
    }
}


// =============================================================================
// nomask_partial — parametric OOB on all four sides + extras
// =============================================================================

#[cfg(test)]
mod nomask_partial_rstest {
    use super::mock::*;
    use rstest::rstest;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        NoOutputMask,
    };

    fn no_mask() -> NoOutputMask {
        NoOutputMask::default()
    }

    // Row weights [0.0, 1.0, 0.0] = identity row
    // Col weights [0.0, 0.5, 0.5] = left weight 0.5 at centre col
    //
    // (row_c, col_c) + side-specific weight → OOB on one side.

    /// Non-zero weight OOB on each side of a 3×3 array → nodata.
    #[rstest]
    #[case::row_top(
        [0.0, 0.7, 0.3f64], [0.0, 1.0, 0.0f64], 0, 1, "row -1"
    )]
    #[case::row_bottom(
        [0.3, 0.7, 0.0f64], [0.0, 1.0, 0.0f64], 2, 1, "row 3"
    )]
    #[case::col_left(
        [0.0, 1.0, 0.0f64], [0.0, 0.5, 0.5f64], 1, 0, "col -1"
    )]
    #[case::col_right(
        [0.0, 1.0, 0.0f64], [0.5, 0.5, 0.0f64], 1, 2, "col 3"
    )]
    fn oob_writes_nodata(
        #[case] w_row: [f64; 3],
        #[case] w_col: [f64; 3],
        #[case] row_c: i64,
        #[case] col_c: i64,
        #[case] label: &str,
    ) {
        let data = [1.0f64; 9];
        let arr_in  = arr(&data, 3, 3);
        let mut out = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockOdd3;
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, (row_c) - 1, (col_c) - 1, -9.0,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB {label}");
    }

    /// Multivar: each plane interpolated independently.
    #[test]
    fn multivar_two_planes() {
        let mut data = [0.0f64; 18];
        data[1 * 3 + 1]     = 5.0;
        data[9 + 1 * 3 + 1] = 15.0;
        let arr_in = arr_nvar(&data, 2, 3, 3);
        let mut out = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, &mut no_mask(), 0, (1) - 1, (1) - 1, -9.0,
        );
        assert!(approx(out[0], 5.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 15.0), "plane1: {}", out[1]);
    }

    /// Multivar OOB: nodata on all planes.
    #[test]
    fn multivar_oob_all_planes_nodata() {
        let data = [1.0f64; 18];
        let arr_in = arr_nvar(&data, 2, 3, 3);
        let mut out = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockOdd3;
        let w_row = [0.0, 1.0, 0.0f64];
        let w_col = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out,
            &mut no_mask(), 0, (1) - 1, (0) - 1, -9.0,
        );
        assert_eq!(out[0], -9.0, "plane0");
        assert_eq!(out[1], -9.0, "plane1");
    }

    /// Sub-pixel both axes.
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
        let interp = MockOdd3;
        let w = [0.0, 0.5, 0.5f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out,
            &mut no_mask(), 0, (2) - 1, (2) - 1, -9.0,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Non-zero out_idx.
    #[rstest]
    #[case::idx1(1)]
    #[case::idx2(2)]
    fn nonzero_out_idx(#[case] idx: usize) {
        let data = [1.0f64; 25];
        let arr_in = arr(&data, 5, 5);
        let mut out = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        interp.interpolate_nomask_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out,
            &mut no_mask(), idx, (2) - 1, (2) - 1, -9.0,
        );
        assert!(approx(out[idx], 1.0), "out[{idx}]={}", out[idx]);
    }
}


// =============================================================================
// masked_unchecked — parametric extra tests
// =============================================================================

#[cfg(test)]
mod masked_unchecked_rstest {
    use super::mock::*;
    use rstest::rstest;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        NoOutputMask,
        BoundsCheck,
    };
    use crate::gx_array::GxArrayView;

    fn ctx_with_mask<'a>(
        mask: &'a GxArrayView<'a, u8>,
    ) -> GxArrayViewInterpolationContext<
        'a, BinaryInputMask<'a>, NoOutputMask, BoundsCheck,
    > {
        GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask },
            NoOutputMask::default(),
            BoundsCheck,
        )
    }

    /// Masking different neighbours with non-zero weight → nodata.
    #[rstest]
    #[case::left_neighbour(2 * 5 + 1, [0.0, 1.0, 0.0f64], [0.0, 0.5, 0.5f64])]
    #[case::right_neighbour(2 * 5 + 3, [0.0, 1.0, 0.0f64], [0.5, 0.5, 0.0f64])]
    #[case::top_neighbour(1 * 5 + 2, [0.0, 0.5, 0.5f64], [0.0, 1.0, 0.0f64])]
    #[case::bottom_neighbour(3 * 5 + 2, [0.5, 0.5, 0.0f64], [0.0, 1.0, 0.0f64])]
    fn masked_neighbour_nonzero_weight_writes_nodata(
        #[case] masked_idx: usize,
        #[case] w_row: [f64; 3],
        #[case] w_col: [f64; 3],
    ) {
        let data = [1.0f64; 25];
        let mut mask_data = [1u8; 25];
        mask_data[masked_idx] = 0;
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockOdd3;
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0,
            "masked at flat idx {masked_idx} with non-zero weight should give nodata");
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
        let interp = MockOdd3;
        let w = [0.0, 0.5, 0.5f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Non-zero out_idx with mask.
    #[rstest]
    #[case::idx1(1)]
    #[case::idx2(2)]
    fn nonzero_out_idx(#[case] idx: usize) {
        let data = [1.0f64; 25];
        let mask_data = [1u8; 25];
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockOdd3;
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_unchecked::<f64, f64, _>(
            &[0.0, 1.0, 0.0f64], &[0.0, 1.0, 0.0f64],
            &arr_in, &mut arr_out, idx, (2) - 1, (2) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[idx], 1.0), "out[{idx}]={}", out[idx]);
    }
}


// =============================================================================
// masked_partial — parametric OOB + multivar + extras
// =============================================================================

#[cfg(test)]
mod masked_partial_rstest {
    use super::mock::*;
    use rstest::rstest;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorCore,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        NoOutputMask,
        BoundsCheck,
    };
    use crate::gx_array::GxArrayView;

    fn ctx_with_mask<'a>(
        mask: &'a GxArrayView<'a, u8>,
    ) -> GxArrayViewInterpolationContext<
        'a, BinaryInputMask<'a>, NoOutputMask, BoundsCheck,
    > {
        GxArrayViewInterpolationContext::new(
            BinaryInputMask { mask },
            NoOutputMask::default(),
            BoundsCheck,
        )
    }

    /// OOB on each side with non-zero weight → nodata (even if mask is valid).
    #[rstest]
    #[case::row_top(
        [0.0, 0.7, 0.3f64], [0.0, 1.0, 0.0f64], 0, 1, "row -1"
    )]
    #[case::row_bottom(
        [0.3, 0.7, 0.0f64], [0.0, 1.0, 0.0f64], 2, 1, "row 3"
    )]
    #[case::col_left(
        [0.0, 1.0, 0.0f64], [0.0, 0.5, 0.5f64], 1, 0, "col -1"
    )]
    #[case::col_right(
        [0.0, 1.0, 0.0f64], [0.5, 0.5, 0.0f64], 1, 2, "col 3"
    )]
    fn oob_writes_nodata(
        #[case] w_row: [f64; 3],
        #[case] w_col: [f64; 3],
        #[case] row_c: i64,
        #[case] col_c: i64,
        #[case] label: &str,
    ) {
        let data = [1.0f64; 9];
        let mask_data = [1u8; 9];
        let arr_in    = arr(&data, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 1];
        let mut arr_out = arr_mut(&mut out, 1, 1);
        let interp = MockOdd3;
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, (row_c) - 1, (col_c) - 1,
            -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "expected nodata for OOB {label}");
    }

    /// Multivar OOB: nodata on all planes.
    #[rstest]
    #[case::oob_col(
        [0.0, 1.0, 0.0f64], [0.0, 0.5, 0.5f64], 1, 0
    )]
    #[case::masked_centre(
        [0.0, 1.0, 0.0f64], [0.0, 1.0, 0.0f64], 1, 1
    )]
    fn multivar_nodata(
        #[case] w_row: [f64; 3],
        #[case] w_col: [f64; 3],
        #[case] row_c: i64,
        #[case] col_c: i64,
    ) {
        let data = [1.0f64; 18];
        let mut mask_data = [1u8; 9];
        // Mask centre for the "masked_centre" case; harmless for OOB case.
        if row_c == 1 && col_c == 1 {
            mask_data[4] = 0;
        }
        let arr_in    = arr_nvar(&data, 2, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [42.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockOdd3;
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w_row, &w_col, &arr_in, &mut arr_out, 0, (row_c) - 1, (col_c) - 1,
            -9.0, &mut ctx,
        );
        assert_eq!(out[0], -9.0, "plane0");
        assert_eq!(out[1], -9.0, "plane1");
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
        let interp = MockOdd3;
        let w = [0.0, 0.5, 0.5f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (2) - 1, (2) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "got {}", out[0]);
    }

    /// Multivar interior with all-valid mask.
    #[test]
    fn multivar_border_all_valid() {
        let mut data = [0.0f64; 18];
        data[1 * 3 + 1]     = 5.0;
        data[9 + 1 * 3 + 1] = 15.0;
        let mask_data = [1u8; 9];
        let arr_in    = arr_nvar(&data, 2, 3, 3);
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mut out   = [0.0f64; 2];
        let mut arr_out = arr_mut_nvar(&mut out, 2, 1, 1);
        let interp = MockOdd3;
        let w = [0.0, 1.0, 0.0f64];
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &w, &w, &arr_in, &mut arr_out, 0, (1) - 1, (1) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[0], 5.0), "plane0: {}", out[0]);
        assert!(approx(out[1], 15.0), "plane1: {}", out[1]);
    }

    /// Non-zero out_idx.
    #[rstest]
    #[case::idx1(1)]
    #[case::idx2(2)]
    fn nonzero_out_idx(#[case] idx: usize) {
        let data = [1.0f64; 25];
        let mask_data = [1u8; 25];
        let arr_in    = arr(&data, 5, 5);
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mut out   = [-9.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 1, 4);
        let interp = MockOdd3;
        let mut ctx = ctx_with_mask(&mask_view);
        interp.interpolate_masked_partial::<f64, f64, _>(
            &[0.0, 1.0, 0.0f64], &[0.0, 1.0, 0.0f64],
            &arr_in, &mut arr_out, idx, (2) - 1, (2) - 1, -9.0, &mut ctx,
        );
        assert!(approx(out[idx], 1.0), "out[{idx}]={}", out[idx]);
    }
}


// =============================================================================
// is_valid_weighted_window — parametric tests
// =============================================================================

#[cfg(test)]
mod is_valid_weighted_window_rstest {
    use rstest::rstest;
    use crate::gx_array::GxArrayView;
    use crate::interp::gx_array_view_interp::{
        BinaryInputMask,
        GxArrayViewInterpolatorInputMaskStrategy,
    };

    /// Parametric test: different mask / weight combinations.
    ///
    /// masked_idx: flat index to set to 0 in the 3×3 mask (None = all valid).
    /// w_row / w_col: weight slices for the 3-tap kernel.
    /// expected: 1 if valid, 0 if invalid.
    #[rstest]
    // All valid, identity weights.
    #[case::all_valid_identity(None, [0.0, 1.0, 0.0f64], [0.0, 1.0, 0.0f64], 1)]
    // Centre masked, identity weights → 0.
    #[case::centre_masked(Some(4), [0.0, 1.0, 0.0f64], [0.0, 1.0, 0.0f64], 0)]
    // Right-of-centre masked, but zero weight → 1 (ignored).
    #[case::right_masked_zero_weight(Some(5), [0.0, 1.0, 0.0f64], [0.0, 1.0, 0.0f64], 1)]
    // All valid, all non-zero weights → 1.
    #[case::all_valid_full_weights(None, [0.25, 0.5, 0.25f64], [0.25, 0.5, 0.25f64], 1)]
    // Top-left masked, all non-zero weights → 0.
    #[case::corner_masked_full_weights(Some(0), [0.25, 0.5, 0.25f64], [0.25, 0.5, 0.25f64], 0)]
    // Bottom-right masked, all non-zero weights → 0.
    #[case::bottom_right_masked(Some(8), [0.25, 0.5, 0.25f64], [0.25, 0.5, 0.25f64], 0)]
    // Row 0 fully masked but row weight = 0 → 1 (skipped).
    #[case::row0_masked_zero_weight(Some(0), [0.0, 1.0, 0.0f64], [0.25, 0.5, 0.25f64], 1)]
    fn parametric_window(
        #[case] masked_idx: Option<usize>,
        #[case] w_row: [f64; 3],
        #[case] w_col: [f64; 3],
        #[case] expected: u8,
    ) {
        let mut mask_data = [1u8; 9];
        if let Some(idx) = masked_idx {
            mask_data[idx] = 0;
        }
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        let result = mask.is_valid_weighted_window::<3, 3>(
            0, 0, 0, &w_row, &w_col,
        );
        assert_eq!(result, expected,
            "masked_idx={masked_idx:?} expected={expected} got={result}");
    }

    /// Non-zero start_idx (window into a larger array).
    #[test]
    fn nonzero_start_idx() {
        let mask_data = [1u8; 25];
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        let w = [0.0, 1.0, 0.0f64];
        let result = mask.is_valid_weighted_window::<3, 3>(
            6, 1, 1, &w, &w,
        );
        assert_eq!(result, 1);
    }
}


// =============================================================================
// write_nodata_all_vars — parametric tests
// =============================================================================

#[cfg(test)]
mod write_nodata_all_vars_rstest {
    use super::mock::*;
    use rstest::rstest;
    use crate::interp::gx_array_view_interp::{
        write_nodata_all_vars,
        NoOutputMask,
        BinaryOutputMask,
    };
    use crate::gx_array::GxArrayViewMut;

    /// Parametric: different nvar / out_idx combinations.
    #[rstest]
    #[case::single_var_idx0(1, 2, 2, 0)]
    #[case::single_var_idx1(1, 2, 2, 1)]
    #[case::two_vars_idx0(2, 2, 2, 0)]
    #[case::two_vars_idx1(2, 2, 2, 1)]
    fn writes_nodata_at_correct_positions(
        #[case] nvar: usize,
        #[case] nrow: usize,
        #[case] ncol: usize,
        #[case] out_idx: usize,
    ) {
        let var_size = nrow * ncol;
        let total = nvar * var_size;
        let mut out = vec![42.0f64; total];
        let mut arr_out = arr_mut_nvar(&mut out, nvar, nrow, ncol);
        let mut om = NoOutputMask::default();
        write_nodata_all_vars(
            &mut arr_out, out_idx, -9.0, &mut om, nvar, var_size,
        );
        for ivar in 0..nvar {
            let pos = out_idx + ivar * var_size;
            assert_eq!(out[pos], -9.0,
                "plane {ivar} at idx {pos} should be nodata");
        }
        // Check a non-target index is untouched.
        let other = if out_idx == 0 { 1 } else { 0 };
        assert_eq!(out[other], 42.0, "untouched index {other}");
    }

    /// Output mask set to 0.
    #[test]
    fn output_mask_set_to_zero() {
        let mut out = [1.0f64; 4];
        let mut arr_out = arr_mut(&mut out, 2, 2);
        let mut mask_data = [1u8; 4];
        let mut arr_mask = GxArrayViewMut::new(&mut mask_data, 1, 2, 2);
        let mut om = BinaryOutputMask { mask: &mut arr_mask };
        write_nodata_all_vars(&mut arr_out, 2, -9.0, &mut om, 1, 4);
        assert_eq!(mask_data[2], 0);
        assert_eq!(mask_data[0], 1);
    }
}


// =============================================================================
// Strategy types — unit tests
// =============================================================================

#[cfg(test)]
mod strategy_tests {
    use crate::interp::gx_array_view_interp::{
        NoInputMask, NoOutputMask,
        BoundsCheck, NoBoundsCheck,
        GxArrayViewInterpolatorInputMaskStrategy,
        GxArrayViewInterpolatorOutputMaskStrategy,
        GxArrayViewInterpolatorBoundsCheckStrategy,
    };

    #[test]
    fn no_input_mask_is_valid_always_one() {
        let m = NoInputMask;
        assert_eq!(m.is_valid(0), 1);
        assert_eq!(m.is_valid(9999), 1);
    }

    #[test]
    fn no_input_mask_is_disabled() {
        assert!(!NoInputMask.is_enabled());
    }

    #[test]
    fn no_input_mask_weighted_window_always_one() {
        let m = NoInputMask;
        let w = [0.25, 0.5, 0.25f64];
        assert_eq!(m.is_valid_weighted_window::<3, 3>(0, 0, 0, &w, &w), 1);
    }

    #[test]
    fn no_output_mask_is_disabled() {
        assert!(!NoOutputMask.is_enabled());
    }

    #[test]
    fn no_output_mask_set_value_is_noop() {
        let mut m = NoOutputMask;
        m.set_value(0, 1);
        m.set_value(9999, 0); // must not panic
    }

    #[test]
    fn bounds_check_returns_true() {
        assert!(BoundsCheck::do_check());
    }

    #[test]
    fn no_bounds_check_returns_false() {
        assert!(!NoBoundsCheck::do_check());
    }
}



// =============================================================================
// is_valid_window — parametric tests (NoInputMask + BinaryInputMask)
// =============================================================================

#[cfg(test)]
mod is_valid_window_rstest {
    use rstest::rstest;
    use crate::gx_array::GxArrayView;
    use crate::interp::gx_array_view_interp::{
        NoInputMask,
        BinaryInputMask,
        GxArrayViewInterpolatorInputMaskStrategy,
    };

    // ------------------------------------------------------------------
    // NoInputMask
    // ------------------------------------------------------------------

    /// NoInputMask always returns 1 regardless of window size.
    #[rstest]
    #[case::w1x1(1, 1)]
    #[case::w3x3(3, 3)]
    #[case::w5x5(5, 5)]
    fn no_input_mask_always_valid(
        #[case] _h: usize,
        #[case] _w: usize,
    ) {
        let m = NoInputMask;
        // H and W are const generics, so we test a few concrete sizes.
        assert_eq!(m.is_valid_window::<3, 3>(0, 0, 0), 1);
        assert_eq!(m.is_valid_window::<5, 5>(0, 0, 0), 1);
        assert_eq!(m.is_valid_window::<1, 1>(0, 0, 0), 1);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — 3x3 window on a 3x3 mask
    // ------------------------------------------------------------------

    /// All valid -> 1.
    #[test]
    fn binary_3x3_all_valid() {
        let mask_data = [1u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<3, 3>(0, 0, 0), 1);
    }

    /// Single masked pixel -> 0 (branchless AND).
    #[rstest]
    #[case::centre(4)]
    #[case::top_left(0)]
    #[case::bottom_right(8)]
    #[case::top_right(2)]
    #[case::bottom_left(6)]
    fn binary_3x3_one_masked(#[case] masked_idx: usize) {
        let mut mask_data = [1u8; 9];
        mask_data[masked_idx] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<3, 3>(0, 0, 0), 0,
            "masking idx {masked_idx} should invalidate the window");
    }

    /// All masked -> 0.
    #[test]
    fn binary_3x3_all_masked() {
        let mask_data = [0u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<3, 3>(0, 0, 0), 0);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — windowing into a larger array
    // ------------------------------------------------------------------

    /// 3x3 window at non-zero start_idx in a 5x5 array.
    #[test]
    fn binary_5x5_window_at_offset() {
        let mask_data = [1u8; 25];
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        // Window starting at (1,1) -> flat index 6.
        assert_eq!(mask.is_valid_window::<3, 3>(6, 1, 1), 1);
    }

    /// 3x3 window at offset with a masked pixel inside the window.
    #[test]
    fn binary_5x5_window_at_offset_with_masked() {
        let mut mask_data = [1u8; 25];
        // Mask (2,2) -> flat index 12, which is the centre of the
        // 3x3 window starting at (1,1).
        mask_data[12] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<3, 3>(6, 1, 1), 0);
    }

    /// Masked pixel outside the window does not affect the result.
    #[test]
    fn binary_5x5_masked_outside_window() {
        let mut mask_data = [1u8; 25];
        // Mask (0,0) -> flat index 0, outside the 3x3 window at (1,1).
        mask_data[0] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<3, 3>(6, 1, 1), 1);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — non-square windows
    // ------------------------------------------------------------------

    /// 3x5 window (H=3, W=5) all valid.
    #[test]
    fn binary_nonsquare_3x5_all_valid() {
        let mask_data = [1u8; 35]; // 7 cols, 5 rows
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 7);
        let mask = BinaryInputMask { mask: &mask_view };
        // Window starting at (1,1) -> flat index 8.
        assert_eq!(mask.is_valid_window::<3, 5>(8, 1, 1), 1);
    }

    /// 3x5 window with one masked pixel.
    #[test]
    fn binary_nonsquare_3x5_one_masked() {
        let mut mask_data = [1u8; 35];
        // Mask at (2,3) in the array -> flat index 2*7+3 = 17.
        // Inside the 3x5 window starting at (1,1).
        mask_data[17] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 7);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<3, 5>(8, 1, 1), 0);
    }

    /// 5x3 window (H=5, W=3) all valid.
    #[test]
    fn binary_nonsquare_5x3_all_valid() {
        let mask_data = [1u8; 35]; // 7 cols, 5 rows
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 7);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<5, 3>(0, 0, 0), 1);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — 1x1 window (degenerate)
    // ------------------------------------------------------------------

    /// 1x1 window on a valid pixel.
    #[test]
    fn binary_1x1_valid() {
        let mask_data = [1u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<1, 1>(4, 1, 1), 1);
    }

    /// 1x1 window on a masked pixel.
    #[test]
    fn binary_1x1_masked() {
        let mut mask_data = [1u8; 9];
        mask_data[4] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.is_valid_window::<1, 1>(4, 1, 1), 0);
    }
}


// =============================================================================
// count_valid_window — parametric tests (NoInputMask + BinaryInputMask)
// =============================================================================

#[cfg(test)]
mod count_valid_window_rstest {
    use rstest::rstest;
    use crate::gx_array::GxArrayView;
    use crate::interp::gx_array_view_interp::{
        NoInputMask,
        BinaryInputMask,
        GxArrayViewInterpolatorInputMaskStrategy,
    };

    // ------------------------------------------------------------------
    // NoInputMask
    // ------------------------------------------------------------------

    /// NoInputMask always returns H*W.
    #[test]
    fn no_input_mask_returns_full_count() {
        let m = NoInputMask;
        assert_eq!(m.count_valid_window::<3, 3>(0, 0, 0), 9);
        assert_eq!(m.count_valid_window::<5, 5>(0, 0, 0), 25);
        assert_eq!(m.count_valid_window::<1, 1>(0, 0, 0), 1);
        assert_eq!(m.count_valid_window::<3, 5>(0, 0, 0), 15);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — 3x3 window on a 3x3 mask
    // ------------------------------------------------------------------

    /// All valid -> count = 9.
    #[test]
    fn binary_3x3_all_valid() {
        let mask_data = [1u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 3>(0, 0, 0), 9);
    }

    /// All masked -> count = 0.
    #[test]
    fn binary_3x3_all_masked() {
        let mask_data = [0u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 3>(0, 0, 0), 0);
    }

    /// Parametric: mask N pixels in a 3x3 window.
    #[rstest]
    #[case::one_masked(vec![4], 8)]
    #[case::two_masked(vec![0, 8], 7)]
    #[case::three_masked(vec![0, 4, 8], 6)]
    #[case::full_row_masked(vec![0, 1, 2], 6)]
    #[case::full_col_masked(vec![0, 3, 6], 6)]
    #[case::corners_masked(vec![0, 2, 6, 8], 5)]
    #[case::only_centre_valid(vec![0, 1, 2, 3, 5, 6, 7, 8], 1)]
    fn binary_3x3_n_masked(
        #[case] masked_indices: Vec<usize>,
        #[case] expected: usize,
    ) {
        let mut mask_data = [1u8; 9];
        for &idx in &masked_indices {
            mask_data[idx] = 0;
        }
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 3>(0, 0, 0), expected,
            "masked={masked_indices:?} expected={expected}");
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — windowing into a larger array
    // ------------------------------------------------------------------

    /// 3x3 window at non-zero start_idx, all valid.
    #[test]
    fn binary_5x5_window_at_offset_all_valid() {
        let mask_data = [1u8; 25];
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 3>(6, 0, 0), 9);
    }

    /// 3x3 window at offset with 2 masked pixels inside the window.
    #[test]
    fn binary_5x5_window_at_offset_two_masked() {
        let mut mask_data = [1u8; 25];
        // Inside window (1,1)..(3,3): mask (1,2) and (2,1).
        mask_data[1 * 5 + 2] = 0; // flat 7
        mask_data[2 * 5 + 1] = 0; // flat 11
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 3>(6, 0, 0), 7);
    }

    /// Masked pixel outside the window does not affect the count.
    #[test]
    fn binary_5x5_masked_outside_window() {
        let mut mask_data = [1u8; 25];
        mask_data[0] = 0; // outside 3x3 window at (1,1)
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 5);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 3>(6, 0, 0), 9);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — non-square windows
    // ------------------------------------------------------------------

    /// 3x5 window all valid.
    #[test]
    fn binary_nonsquare_3x5_all_valid() {
        let mask_data = [1u8; 35]; // 7 cols, 5 rows
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 7);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 5>(8, 0, 0), 15);
    }

    /// 3x5 window with 3 masked.
    #[test]
    fn binary_nonsquare_3x5_three_masked() {
        let mut mask_data = [1u8; 35];
        // Window at (1,1) in a 5x7 array.
        // Mask (1,1), (1,3), (2,5) -> flat 8, 10, 19.
        mask_data[8] = 0;
        mask_data[10] = 0;
        mask_data[19] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 7);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<3, 5>(8, 0, 0), 12);
    }

    /// 5x3 window all valid.
    #[test]
    fn binary_nonsquare_5x3_all_valid() {
        let mask_data = [1u8; 35]; // 7 cols, 5 rows
        let mask_view = GxArrayView::new(&mask_data, 1, 5, 7);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<5, 3>(0, 0, 0), 15);
    }

    // ------------------------------------------------------------------
    // BinaryInputMask — 1x1 degenerate
    // ------------------------------------------------------------------

    /// 1x1 window valid -> 1.
    #[test]
    fn binary_1x1_valid() {
        let mask_data = [1u8; 9];
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<1, 1>(4, 0, 0), 1);
    }

    /// 1x1 window masked -> 0.
    #[test]
    fn binary_1x1_masked() {
        let mut mask_data = [1u8; 9];
        mask_data[4] = 0;
        let mask_view = GxArrayView::new(&mask_data, 1, 3, 3);
        let mask = BinaryInputMask { mask: &mask_view };
        assert_eq!(mask.count_valid_window::<1, 1>(4, 0, 0), 0);
    }

    // ------------------------------------------------------------------
    // Consistency: count == H*W iff is_valid_window == 1
    // ------------------------------------------------------------------

    /// For any mask, count_valid_window == H*W implies is_valid_window == 1,
    /// and count_valid_window < H*W implies is_valid_window == 0.
    #[rstest]
    #[case::all_valid(vec![], 3, 3)]
    #[case::one_masked(vec![4], 3, 3)]
    #[case::all_masked(vec![0,1,2,3,4,5,6,7,8], 3, 3)]
    #[case::corners(vec![0, 2, 6, 8], 3, 3)]
    fn count_consistent_with_is_valid(
        #[case] masked_indices: Vec<usize>,
        #[case] h: usize,
        #[case] w: usize,
    ) {
        let total = h * w;
        let mut mask_data = vec![1u8; total];
        for &idx in &masked_indices {
            mask_data[idx] = 0;
        }
        let mask_view = GxArrayView::new(&mask_data, 1, h, w);
        let mask = BinaryInputMask { mask: &mask_view };
        let count = mask.count_valid_window::<3, 3>(0, 0, 0);
        let valid = mask.is_valid_window::<3, 3>(0, 0, 0);

        if count == total {
            assert_eq!(valid, 1,
                "count=={total} but is_valid_window returned 0");
        } else {
            assert_eq!(valid, 0,
                "count=={count} < {total} but is_valid_window returned 1");
        }
    }
}

// =============================================================================
// is_valid_window_dyn_unsafe: consistency with the const-size variant
// =============================================================================

#[cfg(test)]
mod is_valid_window_dyn_rstest {
    use rstest::rstest;
    use crate::gx_array::GxArrayView;
    use crate::interp::gx_array_view_interp::{
        NoInputMask,
        BinaryInputMask,
        BinaryInputMaskWithSafeWindow,
        GxArrayViewInterpolatorInputMaskStrategy,
    };

    const NROW: usize = 6;
    const NCOL: usize = 7;

    /// Mask of `NROW x NCOL` samples with a few scattered invalid samples.
    fn mask_data(masked: &[usize]) -> Vec<u8> {
        let mut data = vec![1u8; NROW * NCOL];
        for &idx in masked {
            data[idx] = 0;
        }
        data
    }

    /// Compares the run-time size variant with `is_valid_window_unsafe` on
    /// every `H x W` window of the mask.
    fn check<IM: GxArrayViewInterpolatorInputMaskStrategy, const H: usize, const W: usize>(
        mask: &IM,
    ) {
        for r in 0..=NROW - H {
            for c in 0..=NCOL - W {
                let flat = r * NCOL + c;
                // SAFETY: the window lies inside the mask.
                let (fixed, dynamic) = unsafe {
                    (
                        mask.is_valid_window_unsafe::<H, W>(flat, r, c),
                        mask.is_valid_window_dyn_unsafe(flat, r, c, H, W),
                    )
                };
                assert_eq!(fixed, dynamic, "window {H}x{W} at ({r}, {c})");
            }
        }
    }

    /// Runs `check` on the window sizes used by the reduced windows.
    fn check_sizes<IM: GxArrayViewInterpolatorInputMaskStrategy>(mask: &IM) {
        check::<IM, 1, 1>(mask);
        check::<IM, 1, 4>(mask);
        check::<IM, 3, 1>(mask);
        check::<IM, 2, 3>(mask);
        check::<IM, 3, 3>(mask);
        check::<IM, 4, 5>(mask);
    }

    #[test]
    fn no_input_mask() {
        check_sizes(&NoInputMask);
    }

    #[rstest]
    #[case::all_valid(vec![])]
    #[case::scattered(vec![3, 11, 20, 24, 37])]
    #[case::all_masked((0..NROW * NCOL).collect())]
    fn binary(#[case] masked: Vec<usize>) {
        let data = mask_data(&masked);
        let view = GxArrayView::new(&data, 1, NROW, NCOL);
        check_sizes(&BinaryInputMask { mask: &view });
    }

    /// With a safe box consistent with the mask, the safe window variant
    /// gives the same result as its inner mask on every window.
    #[test]
    fn safe_window_matches_inner() {
        // Invalid samples outside the safe zone rows 1..=4, columns 1..=5.
        let data = mask_data(&[0, 6, 14, 41]);
        let view = GxArrayView::new(&data, 1, NROW, NCOL);
        let inner = BinaryInputMask { mask: &view };
        let safe = BinaryInputMaskWithSafeWindow {
            inner: BinaryInputMask { mask: &view },
            safe_min_row: 1, safe_max_row: 4, safe_min_col: 1, safe_max_col: 5,
            conv_safe_min_row: 1, conv_safe_max_row: 2,
            conv_safe_min_col: 1, conv_safe_max_col: 3,
        };
        for (h, w) in [(1, 1), (2, 3), (3, 3), (1, 5)] {
            for r in 0..=NROW - h {
                for c in 0..=NCOL - w {
                    let flat = r * NCOL + c;
                    // SAFETY: the window lies inside the mask.
                    let (a, b) = unsafe {
                        (
                            safe.is_valid_window_dyn_unsafe(flat, r, c, h, w),
                            inner.is_valid_window_dyn_unsafe(flat, r, c, h, w),
                        )
                    };
                    assert_eq!(a, b, "window {h}x{w} at ({r}, {c})");
                }
            }
        }
    }
}

// =============================================================================
// array1_interp2_separable_core: stencil anchoring (even and odd supports)
// =============================================================================

#[cfg(test)]
mod separable_core_anchoring {
    //! Tests of the convolution loop, without input mask.
    //!
    //! They check the anchoring rule of `array1_interp2_separable_core` for
    //! odd and even supports, the phase handed to `compute_weights`, the exact
    //! reproduction of polynomials in the interior, and the interior, border
    //! and outside paths. Fixtures are test-only kernels, independent from the
    //! production interpolators.
    use std::cell::RefCell;

    use crate::gx_array::{GxArrayView, GxArrayViewMut};
    use crate::interp::gx_array_view_interp::{
        support_left, BinaryOutputMask, BoundsCheck, Effective, DefaultCtx,
        GxArrayViewInterpolationContext, GxArrayViewInterpolatorBoundsCheckStrategy,
        GxArrayViewInterpolatorCore, GxArrayViewInterpolatorSupportPolicy, NoBoundsCheck,
        NoInputMask, Full, ZeroPattern,
    };

    // ------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------

    /// Puts the whole weight on the anchor tap and records every phase it
    /// receives (row phase, then column phase).
    struct AnchorProbe<const K: usize> {
        phases: RefCell<Vec<f64>>,
    }

    impl<const K: usize> AnchorProbe<K> {
        fn new() -> Self {
            Self { phases: RefCell::new(Vec::new()) }
        }
    }

    impl<const K: usize> GxArrayViewInterpolatorCore<K, K> for AnchorProbe<K> {
        fn compute_weights(&self, t: f64, weights: &mut [f64]) {
            self.phases.borrow_mut().push(t);
            weights[..K].fill(0.0);
            // Inverted order: tap `first + i` has weight `weights[K - 1 - i]`,
            // the anchor is `first + support_left(K)`.
            weights[K - 1 - support_left(K)] = 1.0;
        }
    }

    /// 4-tap cubic Lagrange interpolation on taps `anchor - 1 ..= anchor + 2`,
    /// phase `t` in `[0, 1)`. Reproduces cubic polynomials exactly.
    struct Lagrange4;

    impl GxArrayViewInterpolatorCore<4, 4> for Lagrange4 {
        fn compute_weights(&self, t: f64, w: &mut [f64]) {
            w[0] = (t + 1.0) * t * (t - 1.0) / 6.0;          // anchor + 2
            w[1] = -(t + 1.0) * t * (t - 2.0) / 2.0;         // anchor + 1
            w[2] = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0;  // anchor
            w[3] = -t * (t - 1.0) * (t - 2.0) / 6.0;         // anchor - 1
        }
    }

    /// 2-tap linear interpolation (native support), reproduces affine data.
    struct Linear2;

    impl GxArrayViewInterpolatorCore<2, 2> for Linear2 {
        fn compute_weights(&self, t: f64, w: &mut [f64]) {
            w[0] = t;
            w[1] = 1.0 - t;
        }
    }

    /// [`Lagrange4`] declaring its zeros: at `t = 0` only the anchor
    /// (offset 1) has a non-zero weight. Exercises the validity-first paths,
    /// while [`Lagrange4`] keeps the default `FromWeights` paths.
    struct Lagrange4Z;

    impl GxArrayViewInterpolatorCore<4, 4> for Lagrange4Z {
        const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: 0.0, start: 1, len: 1 };
        fn compute_weights(&self, t: f64, w: &mut [f64]) {
            Lagrange4.compute_weights(t, w);
        }
    }

    /// [`Linear2`] declaring its zeros: at `t = 0` only the anchor (offset 0)
    /// has a non-zero weight.
    struct Linear2Z;

    impl GxArrayViewInterpolatorCore<2, 2> for Linear2Z {
        const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: 0.0, start: 0, len: 1 };
        fn compute_weights(&self, t: f64, w: &mut [f64]) {
            Linear2.compute_weights(t, w);
        }
    }

    /// The test fixtures declare zero patterns that match their weights.
    #[test]
    fn fixtures_zero_pattern() {
        use crate::interp::gx_kernel_test_utils::check_zero_pattern;
        check_zero_pattern(|t, w| Lagrange4.compute_weights(t, w), 4,
            <Lagrange4Z as GxArrayViewInterpolatorCore<4, 4>>::ZERO_PATTERN);
        check_zero_pattern(|t, w| Linear2.compute_weights(t, w), 2,
            <Linear2Z as GxArrayViewInterpolatorCore<2, 2>>::ZERO_PATTERN);
    }

    fn grid(nrow: usize, ncol: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        (0..nrow * ncol)
            .map(|k| f((k / ncol) as f64, (k % ncol) as f64))
            .collect()
    }

    /// Deterministic positions in `[lo, hi)`, plus integers and half-integers.
    fn positions(lo: f64, hi: f64) -> Vec<f64> {
        let mut v: Vec<f64> = (0..97).map(|i| lo + (hi - lo) * (i as f64 * 0.618_034).fract()).collect();
        let mut x = lo.ceil();
        while x < hi {
            v.push(x);
            if x + 0.5 < hi {
                v.push(x + 0.5);
            }
            x += 1.0;
        }
        v
    }

    /// Interpolates one point, returns `(value, output mask)`.
    fn interp_one<I, BC, SP, const K: usize>(
        interp: &I, data: &[f64], nrow: usize, ncol: usize, y: f64, x: f64, bc: BC, sp: SP,
    ) -> (f64, u8)
    where
        I: GxArrayViewInterpolatorCore<K, K>,
        BC: GxArrayViewInterpolatorBoundsCheckStrategy,
        SP: GxArrayViewInterpolatorSupportPolicy,
    {
        let arr_in = GxArrayView::new(data, 1, nrow, ncol);
        let mut out = [f64::NAN; 1];
        let mut mask = [77u8; 1];
        {
            let mut arr_out = GxArrayViewMut::new(&mut out, 1, 1, 1);
            let mut mask_view = GxArrayViewMut::new(&mut mask, 1, 1, 1);
            let mut ctx = GxArrayViewInterpolationContext::with_policy(
                NoInputMask, BinaryOutputMask { mask: &mut mask_view }, bc, sp,
            );
            let mut buf = vec![0.0; 2 * K];
            interp
                .array1_interp2_separable_core(
                    &mut buf, y, x, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
                )
                .unwrap();
        }
        (out[0], mask[0])
    }

    // ------------------------------------------------------------------
    // Anchoring and phase
    // ------------------------------------------------------------------

    /// The output is the anchor sample and the phases are `pos - anchor`,
    /// with `anchor = floor(pos)` for an even support and `round(pos)` for an
    /// odd one.
    fn check_anchoring<const K: usize>() {
        let (nrow, ncol) = (14, 15);
        let data = grid(nrow, ncol, |i, j| 1000.0 * i + j);
        let margin = K as f64;
        for &y in &positions(margin, nrow as f64 - 1.0 - margin) {
            for &x in &positions(margin, ncol as f64 - 1.0 - margin) {
                let probe = AnchorProbe::<K>::new();
                let mut ctx = DefaultCtx::default();
                let arr_in = GxArrayView::new(&data, 1, nrow, ncol);
                let mut out = [f64::NAN; 1];
                let mut arr_out = GxArrayViewMut::new(&mut out, 1, 1, 1);
                let mut buf = vec![0.0; 2 * K];
                probe
                    .array1_interp2_separable_core(
                        &mut buf, y, x, 0, &arr_in, &mut arr_out, -1.0, &mut ctx,
                    )
                    .unwrap();

                // check anchor for each (y, x) positions
                let anchor = |p: f64| if K % 2 == 0 { p.floor() } else { (p + 0.5).floor() };
                let (ar, ac) = (anchor(y), anchor(x));
                assert_eq!(out[0], 1000.0 * ar + ac, "K={K} ({y}, {x}): wrong anchor");

                // compute_weights must be call 2 times only
                let phases = probe.phases.borrow();
                assert_eq!(phases.len(), 2, "K={K}: compute_weights called {} times", phases.len());

                // check that the phase is contained in the theoric interval depending
                // on K parity.
                for (&t, (p, a)) in phases.iter().zip([(y, ar), (x, ac)]) {
                    assert_eq!(t, p - a, "K={K} pos {p}: phase {t}");
                    if K % 2 == 0 {
                        assert!((0.0..1.0).contains(&t), "K={K} pos {p}: phase {t} not in [0, 1)");
                    } else {
                        assert!((-0.5..=0.5).contains(&t), "K={K} pos {p}: phase {t} not in [-0.5, 0.5]");
                    }
                }
            }
        }
    }

    #[test]
    fn anchoring_odd_k1() { check_anchoring::<1>(); }

    #[test]
    fn anchoring_even_k2() { check_anchoring::<2>(); }

    #[test]
    fn anchoring_odd_k3() { check_anchoring::<3>(); }

    #[test]
    fn anchoring_even_k4() { check_anchoring::<4>(); }

    #[test]
    fn anchoring_odd_k5() { check_anchoring::<5>(); }

    #[test]
    fn anchoring_even_k6() { check_anchoring::<6>(); }

    // ------------------------------------------------------------------
    // Polynomial reproduction in the interior
    // ------------------------------------------------------------------
    //
    // Principle
    //
    // A kernel reproduces the polynomials of degree at most m if, for every
    // such polynomial p and every target position x,
    //
    // sum_{k=0}^{K-1} w_k(t) * p(first + k) = p(x)
    //
    // where K is the support, first the first tap of the stencil, t the
    // phase and w_k the weight of tap first + k. Linear interpolation
    // reproduces degree 1 (affine functions). Four-point Lagrange
    // interpolation reproduces degree 3. In two dimensions, the product of
    // two 1D kernels reproduces every polynomial of degree at most m in i
    // and at most m in j, cross terms such as i * j included.
    //
    // If the input samples a polynomial, a[i, j] = p(i, j), the value
    // interpolated at (y, x) is therefore exactly p(y, x). The expected
    // value is a closed formula and needs no reference implementation.
    //
    // We use this property here to validate the convolution loop.
    //
    // Example: with p(i, j) = 3i - 2j + 5, linear interpolation at
    // (2.25, 4.5) must return 3 * 2.25 - 2 * 4.5 + 5 = 2.75.
    //
    // Only interior targets are used, where the whole stencil lies inside
    // the array
    fn cubic(i: f64, j: f64) -> f64 {
        0.5 * i * i * i - 2.0 * i * i + i + 3.0 + 0.25 * j * j * j - j * j + 0.5 * j * i
    }

    fn affine(i: f64, j: f64) -> f64 {
        3.0 * i - 2.0 * j + 5.0
    }

    #[test]
    fn lagrange4_reproduces_cubic_in_interior() {
        let (nrow, ncol) = (12, 13);
        let data = grid(nrow, ncol, cubic);
        // Interior for K = 4: floor(pos) - 1 >= 0 and floor(pos) + 2 <= n - 1.
        for &y in &positions(1.0, nrow as f64 - 3.0) {
            for &x in &positions(1.0, ncol as f64 - 3.0) {
                let expected = cubic(y, x);
                for (v, m) in [
                    interp_one::<_, _, _, 4>(&Lagrange4, &data, nrow, ncol, y, x, BoundsCheck, Effective),
                    interp_one::<_, _, _, 4>(&Lagrange4, &data, nrow, ncol, y, x, NoBoundsCheck, Effective),
                ] {
                    assert_eq!(m, 1, "({y}, {x}) flagged invalid");
                    assert!((v - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                        "({y}, {x}): got {v}, expected {expected}");
                }
            }
        }
    }

    #[test]
    fn linear2_reproduces_affine_in_interior() {
        let (nrow, ncol) = (9, 10);
        let data = grid(nrow, ncol, affine);
        for &y in &positions(0.0, nrow as f64 - 1.0) {
            for &x in &positions(0.0, ncol as f64 - 1.0) {
                let expected = affine(y, x);
                for (v, m) in [
                    interp_one::<_, _, _, 2>(&Linear2, &data, nrow, ncol, y, x, BoundsCheck, Effective),
                    interp_one::<_, _, _, 2>(&Linear2, &data, nrow, ncol, y, x, NoBoundsCheck, Effective),
                ] {
                    assert_eq!(m, 1, "({y}, {x}) flagged invalid");
                    assert!((v - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                        "({y}, {x}): got {v}, expected {expected}");
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Interior, border and outside paths
    // ------------------------------------------------------------------
    //
    // Principle
    //
    // For a target x, the core places a stencil of K taps starting at
    //
    // first = anchor - left, anchor = floor(x) (even K),
    // left = (K - 1) / 2
    //
    // Under the effective support policy, the output is valid if and only if every
    // tap with a non-zero weight lies inside the array. Taps with a zero weight may
    // lie outside the array: they contribute nothing and are never read.
    //
    // Under the full support policy, every tap of the stencil must lie inside the
    // array, whatever its weight: the border path never gives a valid output.
    //
    // Each fixture is run twice under the effective support policy: with the
    // default `FromWeights` pattern (zeros read from the weights) and with a
    // declared `AtPhase` pattern (validity decided first). Both must match.

    /// `(col, valid under Effective, valid under Full)` cases on one
    /// row; the row position is an interior integer so that only the column
    /// axis decides.
    fn check_paths<I, const K: usize>(
        interp: &I, cases: &[(f64, bool, bool)], value: fn(f64, f64) -> f64,
    )
    where
        I: GxArrayViewInterpolatorCore<K, K>,
    {
        let (nrow, ncol) = (9, 9);
        let data = grid(nrow, ncol, value);
        let y = 4.0;
        for &(x, valid_effective, valid_full) in cases {
            for (policy, valid, (v, m)) in [
                ("effective", valid_effective,
                 interp_one::<_, _, _, K>(interp, &data, nrow, ncol, y, x, BoundsCheck, Effective)),
                ("full", valid_full,
                 interp_one::<_, _, _, K>(interp, &data, nrow, ncol, y, x, BoundsCheck, Full)),
            ] {
                if valid {
                    assert_eq!(m, 1, "K={K} x={x} {policy}: expected valid");
                    let expected = value(y, x);
                    assert!((v - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                        "K={K} x={x} {policy}: got {v}, expected {expected}");
                } else {
                    assert_eq!((v, m), (-1.0, 0), "K={K} x={x} {policy}: expected nodata");
                }
            }
        }
    }

    /// ncol = 9. Taps are floor(x) - 1 ..= floor(x) + 2.
    const PATHS_K4: [(f64, bool, bool); 11] = [
        (1.0, true, true),     // first = 0: interior
        (1.25, true, true),    // interior
        (6.75, true, true),    // last interior anchor: taps 5..=8
        (0.0, true, false),    // border: tap -1 has a zero weight at t = 0
        (0.25, false, false),  // border: tap -1 has a non-zero weight
        (7.0, true, false),    // border: tap 9 has a zero weight at t = 0
        (7.5, false, false),   // border: tap 9 has a non-zero weight
        (8.0, true, false),    // border: last column, taps 9 and 10 have zero weights
        (8.5, false, false),   // border: taps 9 and 10 have non-zero weights
        (-0.25, false, false), // outside: anchor -1
        (9.0, false, false),   // outside: anchor 9
    ];

    /// ncol = 9. Taps are floor(x) ..= floor(x) + 1.
    const PATHS_K2: [(f64, bool, bool); 6] = [
        (0.0, true, true),     // interior, first = 0
        (7.5, true, true),     // interior, taps 7 and 8
        (8.0, true, false),    // border: tap 9 has a zero weight at t = 0
        (8.25, false, false),  // border: tap 9 has a non-zero weight
        (-0.0, true, true),    // negative zero is the anchor 0
        (-0.5, false, false),  // outside: anchor -1
    ];

    #[test]
    fn paths_even_k4() {
        check_paths::<_, 4>(&Lagrange4, &PATHS_K4, cubic);
        check_paths::<_, 4>(&Lagrange4Z, &PATHS_K4, cubic);
    }

    #[test]
    fn paths_even_k2() {
        check_paths::<_, 2>(&Linear2, &PATHS_K2, affine);
        check_paths::<_, 2>(&Linear2Z, &PATHS_K2, affine);
    }
}

#[cfg(test)]
mod support_policy {
    //! Tests of the support policies with an input mask.
    //!
    //! The effective support policy ignores masked samples whose weight is zero;
    //! the full support policy requires every sample of the stencil to be valid.
    //! Kernels declaring an `AtPhase` zero pattern decide the validity before
    //! computing their weights: they must give exactly the same outputs and
    //! output masks as the same kernels with the default `FromWeights`
    //! pattern, which read the zeros from the weights.
    use crate::gx_array::{GxArrayView, GxArrayViewMut};
    use crate::interp::gx_array_view_interp::{
        BinaryInputMask, BinaryInputMaskWithSafeWindow, BinaryOutputMask, BoundsCheck,
        Effective, GxArrayViewInterpolationContext, GxArrayViewInterpolatorBoundsCheckStrategy,
        GxArrayViewInterpolatorCore, GxArrayViewInterpolatorInputMaskStrategy,
        GxArrayViewInterpolatorSupportPolicy, NoBoundsCheck, NoInputMask, Full, ZeroPattern,
    };
    use super::mock::{linear_weights_2, quadratic_bspline_weights_3};

    // ------------------------------------------------------------------
    // Fixtures: one kernel, two declared patterns
    // ------------------------------------------------------------------

    /// Native linear kernel, zeros read from the weights.
    struct LinW;
    impl GxArrayViewInterpolatorCore<2, 2> for LinW {
        fn compute_weights(&self, t: f64, w: &mut [f64]) { linear_weights_2(t, w); }
    }

    /// Native linear kernel, zeros declared.
    struct LinZ;
    impl GxArrayViewInterpolatorCore<2, 2> for LinZ {
        const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: 0.0, start: 0, len: 1 };
        fn compute_weights(&self, t: f64, w: &mut [f64]) { linear_weights_2(t, w); }
    }

    /// Keys cubic kernel, zeros read from the weights.
    struct KeysW;
    impl GxArrayViewInterpolatorCore<4, 4> for KeysW {
        fn compute_weights(&self, t: f64, w: &mut [f64]) {
            crate::interp::gx_optimized_bicubic_kernel::bicubic_kernel_weights_native(t, w);
        }
    }

    /// Keys cubic kernel, zeros declared.
    struct KeysZ;
    impl GxArrayViewInterpolatorCore<4, 4> for KeysZ {
        const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: 0.0, start: 1, len: 1 };
        fn compute_weights(&self, t: f64, w: &mut [f64]) {
            crate::interp::gx_optimized_bicubic_kernel::bicubic_kernel_weights_native(t, w);
        }
    }

    /// Quadratic B-spline (odd support, phase in `[-0.5, 0.5)`), zeros read
    /// from the weights.
    struct QuadW;
    impl GxArrayViewInterpolatorCore<3, 3> for QuadW {
        fn compute_weights(&self, t: f64, w: &mut [f64]) { quadratic_bspline_weights_3(t, w); }
    }

    /// Quadratic B-spline, zeros declared: at `t = -0.5` the last tap lies
    /// on the edge of the support.
    struct QuadZ;
    impl GxArrayViewInterpolatorCore<3, 3> for QuadZ {
        const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: -0.5, start: 0, len: 2 };
        fn compute_weights(&self, t: f64, w: &mut [f64]) { quadratic_bspline_weights_3(t, w); }
    }

    /// The fixtures declare zero patterns that match their weights.
    #[test]
    fn fixtures_zero_pattern() {
        use crate::interp::gx_kernel_test_utils::check_zero_pattern;
        check_zero_pattern(linear_weights_2, 2,
            <LinZ as GxArrayViewInterpolatorCore<2, 2>>::ZERO_PATTERN);
        check_zero_pattern(
            crate::interp::gx_optimized_bicubic_kernel::bicubic_kernel_weights_native, 4,
            <KeysZ as GxArrayViewInterpolatorCore<4, 4>>::ZERO_PATTERN);
        check_zero_pattern(quadratic_bspline_weights_3, 3,
            <QuadZ as GxArrayViewInterpolatorCore<3, 3>>::ZERO_PATTERN);
    }

    // ------------------------------------------------------------------
    // Runner
    // ------------------------------------------------------------------

    const NROW: usize = 11;
    const NCOL: usize = 12;

    fn data() -> Vec<f64> {
        (0..2 * NROW * NCOL).map(|k| ((k * 7919) % 1013) as f64 - 500.0).collect()
    }

    /// About one invalid sample in five, deterministic.
    fn mask() -> Vec<u8> {
        (0..NROW * NCOL).map(|k| if (k * 2654435761usize) % 97 < 19 { 0 } else { 1 }).collect()
    }

    /// Positions covering random phases, integers, half-integers (rounding
    /// tie of the odd support), borders and outside.
    fn positions() -> Vec<(f64, f64)> {
        let mut pts = Vec::new();
        let frac = |i: usize, a: f64| (i as f64 * a).fract();
        for i in 0..400 {
            pts.push((-1.5 + (NROW as f64 + 2.0) * frac(i, 0.618_034),
                      -1.5 + (NCOL as f64 + 2.0) * frac(i, 0.414_214)));
        }
        for r in -1..=(NROW as i64) {
            for c in -1..=(NCOL as i64) {
                pts.push((r as f64, c as f64));
                pts.push((r as f64 + 0.5, c as f64));
                pts.push((r as f64, c as f64 - 0.5));
                pts.push((r as f64, c as f64 + 0.3));
            }
        }
        pts
    }

    /// Runs every position, returns `(values, output mask)`.
    fn run<I, IM, BC, SP, const K: usize>(
        interp: &I, pts: &[(f64, f64)], input_mask: IM, bc: BC, sp: SP,
    ) -> (Vec<f64>, Vec<u8>)
    where
        I: GxArrayViewInterpolatorCore<K, K>,
        IM: GxArrayViewInterpolatorInputMaskStrategy,
        BC: GxArrayViewInterpolatorBoundsCheckStrategy,
        SP: GxArrayViewInterpolatorSupportPolicy,
    {
        let d = data();
        let n = pts.len();
        let mut out = vec![f64::NAN; 2 * n];
        let mut mask_out = vec![77u8; n];
        {
            let arr_in = GxArrayView::new(&d, 2, NROW, NCOL);
            let mut arr_out = GxArrayViewMut::new(&mut out, 2, 1, n);
            let mut mask_view = GxArrayViewMut::new(&mut mask_out, 1, 1, n);
            let mut ctx = GxArrayViewInterpolationContext::with_policy(
                input_mask, BinaryOutputMask { mask: &mut mask_view }, bc, sp,
            );
            let mut buf = vec![0.0; 2 * K];
            for (k, &(y, x)) in pts.iter().enumerate() {
                interp
                    .array1_interp2_separable_core(&mut buf, y, x, k, &arr_in, &mut arr_out, -1.0, &mut ctx)
                    .unwrap();
            }
        }
        (out, mask_out)
    }

    /// Bit-exact comparison of two runs.
    fn assert_same(label: &str, pts: &[(f64, f64)], a: &(Vec<f64>, Vec<u8>), b: &(Vec<f64>, Vec<u8>)) {
        let n = pts.len();
        for (k, p) in pts.iter().enumerate() {
            assert_eq!(a.1[k], b.1[k], "{label} {p:?}: output mask differs");
            for v in 0..2 {
                assert_eq!(a.0[v * n + k].to_bits(), b.0[v * n + k].to_bits(),
                    "{label} {p:?} var {v}: {} vs {}", a.0[v * n + k], b.0[v * n + k]);
            }
        }
    }

    /// Runs one kernel with both declared patterns under one policy and
    /// compares the results, with and without input mask.
    fn check_same_policy<IW, IZ, SP, const K: usize>(label: &str, w: &IW, z: &IZ, sp: SP)
    where
        IW: GxArrayViewInterpolatorCore<K, K>,
        IZ: GxArrayViewInterpolatorCore<K, K>,
        SP: GxArrayViewInterpolatorSupportPolicy + Copy,
    {
        let pts = positions();
        let m = mask();
        let mv = GxArrayView::new(&m, 1, NROW, NCOL);
        assert_same(&format!("{label} no mask"), &pts,
            &run::<_, _, _, _, K>(w, &pts, NoInputMask, BoundsCheck, sp),
            &run::<_, _, _, _, K>(z, &pts, NoInputMask, BoundsCheck, sp));
        assert_same(&format!("{label} mask"), &pts,
            &run::<_, _, _, _, K>(w, &pts, BinaryInputMask { mask: &mv }, BoundsCheck, sp),
            &run::<_, _, _, _, K>(z, &pts, BinaryInputMask { mask: &mv }, BoundsCheck, sp));

        // The safe zone must hold only valid samples: the mask strategy
        // trusts it without reading the mask.
        let mut ms = m.clone();
        for r in 3..=7 {
            for c in 3..=8 {
                ms[r * NCOL + c] = 1;
            }
        }
        let msv = GxArrayView::new(&ms, 1, NROW, NCOL);
        let safe = || BinaryInputMaskWithSafeWindow {
            inner: BinaryInputMask { mask: &msv },
            safe_min_row: 3, safe_max_row: 7, safe_min_col: 3, safe_max_col: 8,
            conv_safe_min_row: 3, conv_safe_max_row: 7 - K, conv_safe_min_col: 3,
            conv_safe_max_col: 8 - K,
        };
        assert_same(&format!("{label} safe window"), &pts,
            &run::<_, _, _, _, K>(w, &pts, safe(), BoundsCheck, sp),
            &run::<_, _, _, _, K>(z, &pts, safe(), BoundsCheck, sp));
        // Without bounds check: interior positions only.
        let inner: Vec<(f64, f64)> = pts.iter().copied()
            .filter(|&(y, x)| y >= K as f64 && y <= (NROW - K) as f64 - 1.0
                && x >= K as f64 && x <= (NCOL - K) as f64 - 1.0)
            .collect();
        assert!(!inner.is_empty());
        assert_same(&format!("{label} mask, no bounds check"), &inner,
            &run::<_, _, _, _, K>(w, &inner, BinaryInputMask { mask: &mv }, NoBoundsCheck, sp),
            &run::<_, _, _, _, K>(z, &inner, BinaryInputMask { mask: &mv }, NoBoundsCheck, sp));
    }

    /// Declaring the zeros does not change any result, under both policies.
    #[test]
    fn declared_pattern_matches_weights() {
        check_same_policy::<_, _, _, 2>("linear effective", &LinW, &LinZ, Effective);
        check_same_policy::<_, _, _, 2>("linear full", &LinW, &LinZ, Full);
        check_same_policy::<_, _, _, 4>("keys effective", &KeysW, &KeysZ, Effective);
        check_same_policy::<_, _, _, 4>("keys full", &KeysW, &KeysZ, Full);
        check_same_policy::<_, _, _, 3>("quadratic effective", &QuadW, &QuadZ, Effective);
        check_same_policy::<_, _, _, 3>("quadratic full", &QuadW, &QuadZ, Full);
    }

    // ------------------------------------------------------------------
    // Policies on a hand-checked case
    // ------------------------------------------------------------------

    /// One masked sample next to the target, on a tap whose weight is zero
    /// at `t = 0`.
    fn one_masked(y: f64, x: f64, full_support: bool) -> (f64, u8) {
        let d: Vec<f64> = (0..25).map(|k| k as f64).collect();
        let mut m = vec![1u8; 25];
        m[2 * 5 + 3] = 0; // sample (2, 3)
        let mv = GxArrayView::new(&m, 1, 5, 5);
        let arr_in = GxArrayView::new(&d, 1, 5, 5);
        let mut out = [f64::NAN; 1];
        let mut mo = [77u8; 1];
        let mut arr_out = GxArrayViewMut::new(&mut out, 1, 1, 1);
        let mut mask_view = GxArrayViewMut::new(&mut mo, 1, 1, 1);
        let mut buf = [0.0; 4];
        if full_support {
            let mut ctx = GxArrayViewInterpolationContext::with_policy(
                BinaryInputMask { mask: &mv }, BinaryOutputMask { mask: &mut mask_view },
                BoundsCheck, Full);
            LinZ.array1_interp2_separable_core(&mut buf, y, x, 0, &arr_in, &mut arr_out, -1.0, &mut ctx).unwrap();
        } else {
            let mut ctx = GxArrayViewInterpolationContext::with_policy(
                BinaryInputMask { mask: &mv }, BinaryOutputMask { mask: &mut mask_view },
                BoundsCheck, Effective);
            LinZ.array1_interp2_separable_core(&mut buf, y, x, 0, &arr_in, &mut arr_out, -1.0, &mut ctx).unwrap();
        }
        (out[0], mo[0])
    }

    #[test]
    fn masked_zero_weight_tap() {
        // Target (2, 2): taps rows 2..=3, columns 2..=3; at t = 0 only (2, 2)
        // has a weight. The masked sample (2, 3) is ignored by the
        // effective support policy only.
        assert_eq!(one_masked(2.0, 2.0, false), (12.0, 1));
        assert_eq!(one_masked(2.0, 2.0, true), (-1.0, 0));
        // Target (2, 2.5): the weight of column 3 is no longer zero.
        assert_eq!(one_masked(2.0, 2.5, false), (-1.0, 0));
        assert_eq!(one_masked(2.0, 2.5, true), (-1.0, 0));
        // Target (2, 1.5): columns 1..=2, the masked sample is not in the
        // stencil.
        assert_eq!(one_masked(2.0, 1.5, false), (11.5, 1));
        assert_eq!(one_masked(2.0, 1.5, true), (11.5, 1));
    }
}
