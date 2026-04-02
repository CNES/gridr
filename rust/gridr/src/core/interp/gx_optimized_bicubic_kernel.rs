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

#![warn(missing_docs)]
//! Implementation of GxArrayViewInterpolator for an optimized bicubic interpolator.
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use crate::core::gx_errors::GxError;
use super::gx_array_view_interp::{
    GxArrayViewInterpolator,
    GxArrayViewInterpolatorArgs,
    GxArrayViewInterpolatorCore,
    GxArrayViewInterpolationContextTrait,
};

#[inline(always)]
fn optimized_bicubic_kernel_weights_compute_func1(x: f64) -> f64 {
    x * x * (1.5 * x - 2.5) + 1.0
}

#[inline(always)]
fn optimized_bicubic_kernel_weights_compute_func2(x: f64) -> f64 {
    x * (x * (-0.5 * x + 2.5) - 4.0) + 2.0
}

/// Computes the optimized bicubic interpolation kernel weights for a given position.
/// 
/// This function fills a mutable slice with the computed weights based on the input 
/// coordinate `x`. The weights are used in bicubic interpolation and follow a specific 
/// formula depending on the value of `x`.
///
/// # Mathematical Formula
///
/// The general interpolation kernel is defined as:
///
/// ```latex
/// W(x) =
/// \begin{cases} 
///     (a+2)|x|^3 - (a+3)|x|^2 + 1, & \text{if } |x| \leq 1 \\
///     a|x|^3 - 5a|x|^2 + 8a|x| - 4a, & \text{if } 1 < |x| < 2 \\
///     0, & \text{otherwise}
/// \end{cases}
/// ```
///
/// where `a` is typically set to -0.5 or -0.75.  
/// Notably, `W(0) = 1` and `W(n) = 0` for all nonzero integer values of `n`.
///
/// Here we set `a` to -0.5.
/// 
/// # Parameters
/// 
/// - `x`: The relative coordinate for which the kernel weights should be computed.
/// - `weights`: A mutable slice of length 5 where the computed weights will be stored.
/// 
/// # Panics
/// 
/// This function will panic if `weights` does not have a length of at least 5.
/// 
/// # Example
/// 
/// ```rust
/// let mut weights = [0.0; 5];
/// optimized_bicubic_kernel_weights(0.3, &mut weights);
/// println!("{:?}", weights);
/// ```
#[inline]
pub fn optimized_bicubic_kernel_weights(x: f64, weights: &mut [f64])
{
    if x < 0.0 && x > -1.0 {
        weights[0] = 0.0;
        // - instead of abs because we know x is negative
        weights[1] = optimized_bicubic_kernel_weights_compute_func2(-x + 1.0);
        weights[2] = optimized_bicubic_kernel_weights_compute_func1(-x);
        weights[3] = optimized_bicubic_kernel_weights_compute_func1(x + 1.0);
        weights[4] = optimized_bicubic_kernel_weights_compute_func2(x + 2.0);
        //weights[4] = 1. - weights[1] + weights[2] + weights[3];
    } else if x > 0.0 && x < 1.0 {
        // - instead of abs because we know x is positive
        weights[0] = optimized_bicubic_kernel_weights_compute_func2(-x + 2.0);
        weights[1] = optimized_bicubic_kernel_weights_compute_func1(-x + 1.0);
        weights[2] = optimized_bicubic_kernel_weights_compute_func1(x);
        weights[3] = optimized_bicubic_kernel_weights_compute_func2(x + 1.0);
        weights[4] = 0.0;
        //weights[3] = 1. - weights[1] + weights[2] + weights[0];
    } else if x == 0.0 {
        // Center pixel: interpolation is identity
        weights[0] = 0.0;
        weights[1] = 0.0;
        weights[2] = 1.0;
        weights[3] = 0.0;
        weights[4] = 0.0;
    } else {
        // Default formula
        for k in 0..=4 {
            weights[k] = 0.0;
            let xx = (x + k as f64 - 2.).abs();
            if xx < 1.0 {
                weights[k] = optimized_bicubic_kernel_weights_compute_func1(xx);
            } else if xx < 2.0 {
                weights[k] = optimized_bicubic_kernel_weights_compute_func2(xx);
            } else {
                weights[k] = 0.0;
            }
        }
    }
}

/// Calculates four cubic interpolation weights with factorisation of computation
/// between the weights.
///
/// These weights are derived from the evaluation of two distinct cubic 
/// polynomials, implemented in `optimized_bicubic_kernel_weights_compute_func1` and
/// `optimized_bicubic_kernel_weights_compute_func2`
/// at specific points that depend on `y`.
///
/// The base polynomials are defined as follows:
/// - $P_1(x) = x \cdot x \cdot (1.5 \cdot x - 2.5) + 1.0$
///   Expanded form: $P_1(x) = 1.5x^3 - 2.5x^2 + 1.0$
///
/// - $P_2(x) = x \cdot (x \cdot (-0.5 \cdot x + 2.5) - 4.0) + 2.0$
///   Expanded form: $P_2(x) = -0.5x^3 + 2.5x^2 - 4.0x + 2.0$
///
/// # Case x is positive
///
/// The weights to be calculated are assigned to the specified indices as follows:
/// - `w0 = P_2(2 - y)`
/// - `w1 = P_1(1 - y)`
/// - `w2 = P_1(y)`
/// - `w3 = P_2(y + 1)`
///
/// After substitution of `x` and algebraic simplification, the expressions for 
/// the weights in terms of `y` are:
///
/// $$
/// w_0 = 0.5y^3 - 0.5y^2 \\
/// w_1 = -1.5y^3 + 2.0y^2 + 0.5y \\
/// w_2 = 1.5y^3 - 2.5y^2 + 1.0 \\
/// w_3 = -0.5y^3 + 1.0y^2 - 0.5y \\
/// w_4 = 0
/// $$
///
/// 
/// # Case x is negative
///
/// The weights to be calculated are assigned to the specified indices as follows:
/// - `w1 = P_2(1 - y)`
/// - `w2 = P_1(-y)`
/// - `w3 = P_1(y + 1)`
/// - `w4 = P_2(y + 2)`
///
/// After substitution of `x` and algebraic simplification, the expressions for 
/// the weights in terms of `y` are:
///
/// $$
/// w_0 = 0 \\
/// w_1 = 0.5y^3 + 1.0y^2 + 0.5y \\
/// w_2 = -1.5y^3 - 2.5y^2 + 1.0 \\
/// w_3 = 1.5y^3 + 2.0y^2 - 0.5y \\
/// w_4 = -0.5y^3 - 0.5y^2
/// $$
///
/// # Optimization Strategy (Factorization)
///
/// To minimize computational cost, the calculation is optimized by:
/// 1.  Calculating powers of `y` ($y^2$ and $y^3$) only once.
/// 2.  Pre-calculating common scaled terms (e.g., $0.5y$, $1.5y^3$, etc.)
///     that are reused across multiple weight expressions.
/// 3.  Assembling the final weight results from these pre-calculated terms.
///
/// # Cost Analysis
///
/// Compared to a naive evaluation of each polynomial independently, this optimized 
/// method achieves:
/// - **8 Multiplications** (vs. 12 naive)
/// - **7 Additions/Subtractions** (vs. 10 naive)
/// - (2 Negations)
///
/// # Arguments
///
/// - `y`: The relative coordinate for which the kernel weights should be computed.
/// - `weights`: A mutable slice of length 5 where the computed weights will be stored.
///
/// ```
#[inline]
pub fn optimized_bicubic_kernel_weights_opt(y: f64, weights: &mut [f64])
{
    if y < 0.0 && y > -1.0 {
        // Calculate powers of y
        let y2 = y * y;
        let y3 = y2 * y;

        // Calculate common scaled terms
        let y_times_0_5 = 0.5 * y;
        let y2_times_0_5 = 0.5 * y2;
        let y3_times_0_5 = 0.5 * y3;

        let y3_times_1_5 = 1.5 * y3;
        let y2_times_2_0 = 2.0 * y2;
        let y2_times_2_5 = 2.5 * y2;

        // Assemble the final weight values
        weights[0] = 0.0;
        // w1 = 0.5y^3 + 1.0y^2 + 0.5y
        weights[1] = y3_times_0_5 + y2 + y_times_0_5;
        // w2 = -1.5y^3 - 2.5y^2 + 1.0
        weights[2] = -y3_times_1_5 - y2_times_2_5 + 1.0;
        // w3 = 1.5y^3 + 2.0y^2 - 0.5y
        weights[3] = y3_times_1_5 + y2_times_2_0 - y_times_0_5;
        // w4 = -0.5y^3 - 0.5y^2
        weights[4] = -y3_times_0_5 - y2_times_0_5;
        
    } else if y > 0.0 && y < 1.0 {
        // Calculate powers of y
        let y2 = y * y;
        let y3 = y2 * y;

        // Calculate common scaled terms
        let y_times_0_5 = 0.5 * y;
        let y2_times_0_5 = 0.5 * y2;
        let y3_times_0_5 = 0.5 * y3;

        let y3_times_1_5 = 1.5 * y3;
        let y2_times_2_0 = 2.0 * y2;
        let y2_times_2_5 = 2.5 * y2;

        // Assemble the final weight values
        // w0 = 0.5y^3 - 0.5y^2
        weights[0] = y3_times_0_5 - y2_times_0_5;
        // w1 = -1.5y^3 + 2.0y^2 + 0.5y
        weights[1] = -y3_times_1_5 + y2_times_2_0 + y_times_0_5;
        // w2 = 1.5y^3 - 2.5y^2 + 1.0
        weights[2] = y3_times_1_5 - y2_times_2_5 + 1.0;
        // w3 = -0.5y^3 + 1.0y^2 - 0.5y
        weights[3] = -y3_times_0_5 + y2 - y_times_0_5;
        weights[4] = 0.0;
        
    } else if y == 0.0 {
        // Center pixel: interpolation is identity
        weights[0] = 0.0;
        weights[1] = 0.0;
        weights[2] = 1.0;
        weights[3] = 0.0;
        weights[4] = 0.0;
    } else {
        // Default formula - fallback => should not pass here so no effort to 
        // optimize it.
        for k in 0..=4 {
            weights[k] = 0.0;
            let yy = (y + k as f64 - 2.).abs();
            if yy < 1.0 {
                weights[k] = optimized_bicubic_kernel_weights_compute_func1(yy);
            } else if yy < 2.0 {
                weights[k] = optimized_bicubic_kernel_weights_compute_func2(yy);
            } else {
                weights[k] = 0.0;
            }
        }
    }
}

/// Optimized bicubic interpolator implementation.
/// 
/// This structure implements the `GxArrayViewInterpolator` trait for the
/// optimized bicubic interpolation operations.
#[derive(Clone, Debug)]
pub struct GxOptimizedBicubicInterpolator {
    /// The size of the kernel alongs the rows - it is set to 5 in the
    /// implemented new() method.
    kernel_row_size: usize,
    /// The size of the kernel alongs the columns - it is set to 5 in the
    /// implemented new() method.
    kernel_col_size: usize,
}

// Use generic implementation from GxArrayViewInterpolatorCore for
// KROWS = 5, NCOLS = 5
impl GxArrayViewInterpolatorCore<5, 5> for GxOptimizedBicubicInterpolator {
        
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        optimized_bicubic_kernel_weights_opt(x, weights)
    }
}

impl GxArrayViewInterpolator for GxOptimizedBicubicInterpolator
{
    fn new(_args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        Self {
            kernel_row_size: 5,
            kernel_col_size: 5,
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
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorNoArgs,
        DefaultCtx
    };
    
    /// Checks if two slices of f64 values are approximately equal within a given tolerance.
    ///
    /// # Arguments
    /// * `a` - First slice of f64 values.
    /// * `b` - Second slice of f64 values.
    /// * `tol` - The allowed tolerance for differences.
    ///
    /// # Returns
    /// * `true` if all corresponding elements of `a` and `b` differ by at most `tol`, otherwise `false`.
    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (*x - *y).abs() <= tol)
    }

    /// Tests the optimized bicubic kernel weights function at the center (x = 0.0).
    #[test]
    fn test_optimized_bicubic_kernel_weights_at_center() {
        let mut weights = [0.0; 5];
        optimized_bicubic_kernel_weights(0.0, &mut weights);
        assert_eq!(weights, [0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    /// Tests the optimized bicubic kernel weights function at x = 0.5.
    /// Expected values are computed using the bicubic interpolation formula.
    #[test]
    fn test_optimized_bicubic_kernel_weights_at_halfway_positive() {
        let mut weights = [0.0; 5];
        optimized_bicubic_kernel_weights(0.5, &mut weights);
        // Expected values for x = 0.5 using bicubic interpolation formula
        let expected = [-0.0625, 0.5625, 0.5625, -0.0625, 0.];        
        assert!(approx_eq(&weights, &expected, 1e-6));
    }

    /// Tests the optimized bicubic kernel weights function at x = -0.5.
    /// Expected values are computed using the bicubic interpolation formula.
    #[test]
    fn test_optimized_bicubic_kernel_weights_at_halfway_negative() {
        let mut weights = [0.0; 5];
        optimized_bicubic_kernel_weights(-0.5, &mut weights);
        // Expected values for x = -0.5 using bicubic interpolation formula
        let expected = [0., -0.0625, 0.5625, 0.5625, -0.0625];        
        assert!(approx_eq(&weights, &expected, 1e-6));
    }

    /// Tests the optimized bicubic kernel weights function for values outside the valid range (|x| >= 2.0).
    /// Expected output: all weights should be zero.
    #[test]
    fn test_optimized_bicubic_kernel_weights_outside_range() {
        let mut weights = [0.0; 5];
        optimized_bicubic_kernel_weights(3.0, &mut weights);
        assert_eq!(weights, [0.0, 0.0, 0.0, 0.0, 0.0]);

        optimized_bicubic_kernel_weights(-3.0, &mut weights);
        assert_eq!(weights, [0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// Tests the optimized bicubic kernel weights function for values exactly at the bounds (x = ±1.0).
    #[test]
    fn test_optimized_bicubic_kernel_weights_exactly_on_bounds() {
        let mut weights = [0.0; 5];
        optimized_bicubic_kernel_weights(1.0, &mut weights);
        assert_eq!(weights, [0., 1., 0., 0., 0.]);
        
        
        optimized_bicubic_kernel_weights(-1.0, &mut weights);
        assert_eq!(weights, [0., 0., 0., 1., 0.]);
    }
    
    /// Tests the both implementations (naive and optimized) give the same results
    #[test]
    fn test_optimized_bicubic_kernel_weights_both_implementation() {
        let mut weights_naive = [0.0; 5];
        let mut weights_opt = [0.0; 5];
        
        let x_values: [f64; 13] = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999999, 1., 1.1, 1.3, 1.8, 2., 2.1];
        for &x in x_values.iter()  {
            optimized_bicubic_kernel_weights(x, &mut weights_naive);
            optimized_bicubic_kernel_weights_opt(x, &mut weights_opt);
            assert!(approx_eq(&weights_naive, &weights_opt, 1e-10));
            
            optimized_bicubic_kernel_weights(-x, &mut weights_naive);
            optimized_bicubic_kernel_weights_opt(-x, &mut weights_opt);
            assert!(approx_eq(&weights_naive, &weights_opt, 1e-10));
        }
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
