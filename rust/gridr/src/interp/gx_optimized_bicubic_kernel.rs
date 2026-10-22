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
//! Implementation of GxArrayViewInterpolator for an optimized bicubic interpolator.
use crate::gx_array::{GxArrayView, GxArrayViewMut};
use crate::gx_errors::GxError;
use super::gx_array_view_interp::{
    GxArrayViewInterpolator,
    GxArrayViewInterpolatorArgs,
    GxArrayViewInterpolatorCore,
    GxArrayViewInterpolationContextTrait,
    ZeroPattern,
};

/// Optimized bicubic interpolator implementation.
/// 
/// This structure implements the `GxArrayViewInterpolator` trait for the
/// optimized bicubic interpolation operations.
#[derive(Clone, Debug)]
pub struct GxOptimizedBicubicInterpolator {
    /// The size of the kernel alongs the rows - it is set to 4 in the
    /// implemented new() method.
    kernel_row_size: usize,
    /// The size of the kernel alongs the columns - it is set to 4 in the
    /// implemented new() method.
    kernel_col_size: usize,
}

// Use generic implementation from GxArrayViewInterpolatorCore for
// KROWS = 4, NCOLS = 4
impl GxArrayViewInterpolatorCore<4, 4> for GxOptimizedBicubicInterpolator {
    /// Taps `anchor - 1` to `anchor + 2`
    /// At `t = 0` only the anchor (offset 1) has a non-zero weight
    const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: 0.0, start: 1, len: 1 };

    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        bicubic_kernel_weights_native(t, weights)
    }
}

/// Computes the Keys cubic convolution weights on the native 4-tap support.
///
/// # Continuous kernel
///
/// The cubic convolution kernel is defined on the distance `x` between the
/// target position and a sample:
///
/// ```latex
/// W(x) =
/// \begin{cases}
/// (a+2)|x|^3 - (a+3)|x|^2 + 1, & \text{if } |x| \leq 1 \\
/// a|x|^3 - 5a|x|^2 + 8a|x| - 4a, & \text{if } 1 < |x| < 2 \\
/// 0, & \text{otherwise}
/// \end{cases}
/// ```
///
/// with `a = -0.5`. Both pieces factor as
///
/// ```latex
/// W(x) = (|x| - 1)(1.5|x|^2 - |x| - 1) \quad |x| \leq 1 \\
/// W(x) = -0.5 (|x| - 1)(|x| - 2)^2 \quad 1 < |x| < 2
/// ```
///
/// # Stencil and phase
///
/// The support has 4 taps. The stencil is anchored on `floor(x)`; the phase
/// `t = x - floor(x)` lies in `[0, 1)` and `s = 1 - t`. The four taps sit
/// at the following distances from the target:
///
/// ```text
/// tap distance piece
/// anchor - 1 1 + t 1 < |x| < 2
/// anchor t |x| <= 1
/// anchor + 1 s |x| <= 1
/// anchor + 2 1 + s 1 < |x| < 2
/// ```
///
/// Substituting these distances in the factored pieces gives the weights
/// (in inverted order):
///
/// ```text
/// weights[0]  tap anchor + 2  distance 1 + s   -0.5 t^2 s
/// weights[1]  tap anchor + 1  distance s       -t (1.5 s^2 - s - 1)
/// weights[2]  tap anchor      distance t       -s (1.5 t^2 - t - 1)
/// weights[3]  tap anchor - 1  distance 1 + t   -0.5 t s^2
/// ```
///
/// Substituting again with u = t * s and h = -0.5 * u, we get:
///
/// ```text
/// weights[0]  tap anchor + 2  distance 1 + s   h t
/// weights[1]  tap anchor + 1  distance s       t + u (1 - 1.5 s)
/// weights[2]  tap anchor      distance t       s + u (1 - 1.5 t)
/// weights[3]  tap anchor - 1  distance 1 + t   h s

/// The weights sum to one for every `t`.
///
///
/// # Numerical properties
///
/// Every weight that vanishes at an end of `[0, 1)` carries the factor that
/// vanishes there (`t`, `s` or `u`). The weights are therefore exact zeros
/// at `t = 0` (`[0, 0, 1, 0]`), and stay accurate and non-zero for `t` in
/// `(0, 1)`, including next to `t = 1`. This matters for the mask and
/// bounds validation, which ignores exactly-zero weights only.
///
/// # Cost
///
/// 8 multiplications and 5 additions, no branch.
///
///
/// # Parameters
///
/// - `t`: phase in `[0, 1)`.
/// - `weights`: output slice of length at least 4.
///
/// # Example
///
/// ```
/// use gridr::interp::gx_optimized_bicubic_kernel::bicubic_kernel_weights_native;
///
/// let mut weights = [0.0; 4];
/// bicubic_kernel_weights_native(0.3, &mut weights);
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-15);
///
/// bicubic_kernel_weights_native(0.0, &mut weights);
/// assert_eq!(weights, [0.0, 0.0, 1.0, 0.0]);
/// ```
#[inline(always)]
pub fn bicubic_kernel_weights_native(t: f64, weights: &mut [f64])
{
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");   
    let s = 1.0 - t;
    let u = t * s;
    let h = -0.5 * u;
    weights[0] = h * t;
    weights[1] = t + u * (1.0 - 1.5 * s);
    weights[2] = s + u * (1.0 - 1.5 * t);
    weights[3] = h * s;
}

impl GxArrayViewInterpolator for GxOptimizedBicubicInterpolator
{
    fn new(_args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        Self {
            kernel_row_size: 4,
            kernel_col_size: 4,
        }
    }
    
    /// Get the short name of the interpolator
    ///
    /// # Returns
    /// A string representing the short name of the interpolator
    fn shortname(&self) -> String {
        "optimized_bicubic".to_string()
    }
    
    fn initialize(&mut self) -> Result<(), String> {
        Ok(())
    }
    
    fn kernel_row_size(&self) -> usize {
        self.kernel_row_size
    }
    
    fn kernel_col_size(&self) -> usize {
        self.kernel_col_size
    }
    
    #[inline(always)]
    fn total_margins(&self) -> Result<[usize; 4], GxError> {
        Ok([2, 2, 2, 2])
    }
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> =
            vec![0.0; self.kernel_row_size + self.kernel_col_size];
        buffer.into_boxed_slice()
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
        T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + PartialEq + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
        self.array1_interp2_separable_core(
            weights_buffer, target_row_pos, target_col_pos, out_idx,
            array_in, array_out, nodata_out, context
        )
    }
}


#[cfg(test)]
mod gx_optimized_bicubic_kernel_tests {
    use super::*;
    use crate::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorNoArgs,
        DefaultCtx
    };
       
    /// The declared zero pattern matches the computed weights.
    #[test]
    fn test_bicubic_zero_pattern() {
        use crate::interp::gx_kernel_test_utils::check_zero_pattern;
        check_zero_pattern(
            bicubic_kernel_weights_native, 4,
            <GxOptimizedBicubicInterpolator as GxArrayViewInterpolatorCore<4, 4>>::ZERO_PATTERN,
        );
    }

    /// Native weights against the continuous kernel.
    #[test]
    fn test_bicubic_kernel_weights_native() {
        use crate::interp::gx_kernel_test_utils::{check_native_weights, phi_keys};
        check_native_weights(bicubic_kernel_weights_native, 4, phi_keys, 1.0, 1e-15);
    }

    /// The weights are symmetric, `w(1 - t)` is `w(t)` reversed.
    #[test]
    fn test_bicubic_kernel_weights_native_symmetry() {
        let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
        for i in 1..100 {
            let t = i as f64 / 100.0;
            bicubic_kernel_weights_native(t, &mut a);
            bicubic_kernel_weights_native(1.0 - t, &mut b);
            for k in 0..4 {
                assert!((a[k] - b[3 - k]).abs() <= 1e-15, "t={t}: {a:?} vs {b:?}");
            }
        }
    }

    /// The expanded polynomials of the v0.7 positive branch, extended to
    /// `[0, 1)`, are rejected by the check: near `t -> 1` the weight
    /// of `anchor - 1` rounds to an exact zero.
    #[test]
    #[should_panic(expected = "exact zero inside the support")]
    fn test_expanded_polynomials_are_rejected() {
        fn expanded(y: f64, w: &mut [f64]) {
            let y2 = y * y;
            let y3 = y2 * y;
            w[0] = 0.5 * y3 - 0.5 * y2;
            w[1] = -1.5 * y3 + 2.0 * y2 + 0.5 * y;
            w[2] = 1.5 * y3 - 2.5 * y2 + 1.0;
            w[3] = -0.5 * y3 + y2 - 0.5 * y;
        }
        use crate::interp::gx_kernel_test_utils::{check_native_weights, phi_keys};
        check_native_weights(expanded, 4, phi_keys, 1.0, 1e-15);
    }

    #[test]
    fn test_array1_interp2_001() {
        let data_in = [ 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 4.0, 0.0, 0.0,
                        0.0, 0.0, 10.0, 2.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0 ];
        
        let mut data_out = [ -9.0, -9.0, -9.0 ];
        let array_in = GxArrayView::new(&data_in, 1, 5, 5);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, 1, 3);
        let interp = GxOptimizedBicubicInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

        let mut weights = interp.allocate_kernel_buffer();
        
        // Default context
        let mut context = DefaultCtx::default();
        
        // Test idendity
        // Expect : 10.
        let mut x = 2.;
        let mut y = 2.;
        let mut out_idx = 1;
        let _ = interp.array1_interp2::<f64, f64, DefaultCtx>(&mut weights, y, x, out_idx, &array_in, &mut array_out, 0., &mut context);
        assert_eq!(array_out.data, [-9., 10., -9.]);
        
        // Target x = 1.75 y = 2.5
        // Expect : 4.58203125
        x = 1.75;
        y = 2.5;
        out_idx = 0;
        let _ = interp.array1_interp2::<f64, f64, DefaultCtx>(&mut weights, y, x, out_idx, &array_in, &mut array_out, 0., &mut context);
        assert_eq!(array_out.data, [4.58203125, 10., -9.]);
    }
}
