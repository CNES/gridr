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
use super::gx_array_view_interp::{GxArrayViewInterpolator, GxArrayViewInterpolationContextTrait, GxArrayViewInterpolatorBoundsCheckStrategy, GxArrayViewInterpolatorInputMaskStrategy, GxArrayViewInterpolatorOutputMaskStrategy};

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
/// This structure implements the `GxArrayViewInterpolator` trait for the optimized bicubic
/// interpolation operations.
pub struct GxOptimizedBicubicInterpolator {
    /// The size of the kernel alongs the rows - it is set to 5 in the implemented new() method.
    kernel_row_size: usize,
    /// The size of the kernel alongs the columns - it is set to 5 in the implemented new() method.
    kernel_col_size: usize,
}

impl GxOptimizedBicubicInterpolator {
    
    /// Performs a fast 2D interpolation without bounds checking or validity masking.
    ///
    /// This method computes an interpolated value at a given center coordinate `(row_c, col_c)`
    /// using separable 5×5 interpolation weights provided for rows and columns. The interpolation
    /// is applied independently across all variable planes (`nvar`) in the input array.
    ///
    /// The function assumes that all positions required for the 5×5 stencil (centered on
    /// `(row_c, col_c)`) are **within the bounds** of the input array. No checks are performed
    /// on the validity or boundaries of the input indices. If this assumption is violated,
    /// the behavior is undefined and may result in a panic or memory corruption.
    ///
    /// This method is designed for performance-critical inner loops where the caller guarantees
    /// safe access patterns, typically after a validity check has already been performed upstream.
    ///
    /// # Parameters
    /// - `weights_row`: Row-wise interpolation weights (length = 5, indexed as `weights_row[4 - irow]`).
    /// - `weights_col`: Column-wise interpolation weights (length = 5, indexed as `weights_col[4 - icol]`).
    /// - `array_in`: Input array view with shape `[nvar, nrow, ncol]`, flattened as a 1D array.
    /// - `array_out`: Output array view with shape `[nvar, nrow, ncol]`, flattened as a 1D array.
    /// - `out_idx`: Flat index in the output array where the interpolated result should be written,
    ///              for the current pixel (same for all variables).
    /// - `row_c`: Integer row coordinate of the interpolation center.
    /// - `col_c`: Integer column coordinate of the interpolation center.
    ///
    /// # Type Parameters
    /// - `T`: Scalar type of the input array, must support `Copy`, `Into<f64>`, and `f64` multiplication.
    /// - `V`: Scalar type of the output array, must support `Copy` and `From<f64>`.
    ///
    /// # Safety
    /// This method does not perform any boundary checks. The caller must ensure that all computed
    /// indices `row_c + irow - 2` and `col_c + icol - 2` are within the bounds of `array_in`.
    /// Violating this precondition results in undefined behavior.
    ///
    /// # Example
    /// ```ignore
    /// interpolator.interpolate_nomask_unchecked(
    ///     weights_row,
    ///     weights_col,
    ///     &input_array,
    ///     &mut output_array,
    ///     out_index,
    ///     row_center,
    ///     col_center,
    /// );
    /// ```
    ///
    /// # See Also
    /// - [`interpolate_nomask_partial`](Self::interpolate_nomask_partial): Safe version that performs bounds checking.
    /// - [`interpolate_masked`](...): Variant that handles a validity mask or nodata values.
    #[inline(always)]
    fn interpolate_nomask_unchecked<T, V>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        out_idx: usize,
        row_c: i64, 
        col_c: i64,
        )
    where
        T: Copy + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + From<f64>
    {
        let ncol = array_in.ncol;
        let array_in_var_size = array_in.var_size;
        let array_out_var_size = array_out.var_size;
        let mut arr_irow: usize;
        let mut arr_icol: usize;
        let mut arr_iflat: usize;
        let mut computed: f64;
        let mut computed_col: f64;
        let mut array_in_var_shift: usize = 0;
        let mut array_out_var_shift: usize = 0;
        
        // Loop on multipe variables in input array.
        for _ivar in 0..array_in.nvar {
            computed = 0.0;
            
            for irow in 0..=4 {
                computed_col = 0.0;
                arr_irow = (row_c + irow - 2) as usize;
                
                for icol in 0..=4 {
                    arr_icol = (col_c + icol - 2) as usize;
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + arr_irow * ncol + arr_icol;
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[4 - icol as usize];
                    
                }
                computed += weights_row[4 - irow as usize] * computed_col;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            array_in_var_shift += array_in_var_size;
            array_out_var_shift += array_out_var_size;
        }
    }
    
    /// Performs 2D interpolation without a validity mask, using a 5×5 weight window.
    ///
    /// This method computes an interpolated value at a given center coordinate `(row_c, col_c)`
    /// using separable row and column weights. It processes all variable planes simultaneously.
    ///
    /// If any weight in the 5×5 window is non-zero and refers to a location outside the bounds
    /// of `array_in`, then no interpolation is performed and the corresponding output values
    /// (for all variables) are set to `nodata`.
    ///
    /// # Parameters
    /// - `weights_row`: Row interpolation weights (length 5).
    /// - `weights_col`: Column interpolation weights (length 5).
    /// - `array_in`: Input array view (with `nvar` planes).
    /// - `array_out`: Output array view (with `nvar` planes), must be pre-allocated.
    /// - `out_idx`: Flat output index for the pixel to write to (same across all variables).
    /// - `row_c`: Center row coordinate for interpolation (integer).
    /// - `col_c`: Center column coordinate for interpolation (integer).
    /// - `nodata`: Value to assign in the output if the interpolation would access out-of-bounds data.
    ///
    /// # Type Parameters
    /// - `T`: Input scalar type (must support multiplication by `f64` and `Into<f64>`).
    /// - `V`: Output scalar type (must support conversion from `f64` and equality testing).
    #[inline(always)]
    fn interpolate_nomask_partial<T, V, N>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        output_mask_strategy: &mut N,
        out_idx: usize,
        row_c: i64, 
        col_c: i64,
        nodata: V,
        )
    where
        T: Copy + std::ops::Mul<f64, Output=f64> + Into<f64> + PartialEq,
        V: Copy + From<f64> + PartialEq,
        N: GxArrayViewInterpolatorOutputMaskStrategy,
    {
        let ncol = array_in.ncol;
        let array_in_var_size = array_in.var_size;
        let array_out_var_size = array_out.var_size;
        let mut arr_irow;
        let mut arr_icol;
        let mut arr_iflat: usize;
        let mut computed: f64;
        let mut computed_col: f64;
        let mut array_in_var_shift: usize = 0;
        let mut array_out_var_shift: usize = 0;
        
        // Pre check boundaries and weights value
        for irow in 0..=4 {
            if weights_row[4 - irow as usize] == 0.0 {
                continue;
            }
            arr_irow = row_c + irow - 2;
            
            for icol in 0..=4 {
                if weights_col[4 - icol as usize] == 0.0 {
                    continue;
                }
                
                arr_icol = col_c + icol - 2;
                
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 || arr_icol < 0 || arr_icol >= array_in.ncol_i64 {
                    // If we came here the weights are both positivs, we cant
                    // ignore that we go out of bounds as far as the validity is
                    // concerned.
                    for _ivar in 0..array_in.nvar {
                        array_out.data[out_idx + array_out_var_shift] = nodata;
                        array_out_var_shift += array_out_var_size;
                    }
                    output_mask_strategy.set_value(out_idx, 0);
                    return;
                }
                array_out_var_shift = 0;
            }
        }
        
        // Loop on multipe variables in input array.
        // If a iteration condition fails it is equivalent as considering a zero
        // for the corresponding row or column value.
        for _ivar in 0..array_in.nvar {
            computed = 0.0;
            
            for irow in 0..=4 {
                arr_irow = row_c + irow - 2;
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 {
                    continue;
                }
                computed_col = 0.0;
                
                for icol in 0..=4 {
                    arr_icol = col_c + icol - 2;
                    if arr_icol < 0 || arr_icol >= array_in.ncol_i64 {
                        continue;
                    }
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + (arr_irow as usize) * ncol + (arr_icol as usize);
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[4 - icol as usize];
                }
                computed += weights_row[4 - irow as usize] * computed_col;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            array_in_var_shift += array_in_var_size;
            array_out_var_shift += array_out_var_size;
        }
        output_mask_strategy.set_value(out_idx, 1);
    }
    
    /// Performs a fast 2D interpolation with a validity mask, without bounds checking.
    ///
    /// This method computes interpolated values at a given center coordinate `(row_c, col_c)`
    /// using separable 5×5 weights. It operates on multi-variable input data with an associated
    /// binary validity mask (`array_mask_in`). The interpolation is only performed if **all**
    /// relevant mask values under non-zero weights are valid (i.e., non-zero); otherwise, a
    /// `nodata` value is written to the output, and the corresponding output mask is cleared.
    ///
    /// This method assumes that all indices involved in the 5×5 interpolation stencil are within
    /// bounds. No bounds checking is performed. This function is designed for performance-critical
    /// code where the caller guarantees valid indexing, typically after pre-checking or clipping.
    ///
    /// # Parameters
    /// - `weights_row`: Row-direction interpolation weights (length = 5, accessed as `weights_row[4 - irow]`).
    /// - `weights_col`: Column-direction interpolation weights (length = 5, accessed as `weights_col[4 - icol]`).
    /// - `array_in`: Input data array with shape `[nvar, nrow, ncol]`, flattened to 1D.
    /// - `array_mask_in`: Binary mask array (`u8`, 0 = invalid, 1 = valid) of shape `[nrow, ncol]`.
    /// - `array_out`: Output array view of shape `[nvar, nrow, ncol]`, flattened to 1D.
    /// - `array_mask_out`: Output binary mask array (`u8`, same layout as `array_mask_in`).
    /// - `out_idx`: Flat index at which to write the output value(s) and mask.
    /// - `row_c`: Row index of the interpolation center.
    /// - `col_c`: Column index of the interpolation center.
    /// - `nodata`: Value to write to the output array if the mask is invalid (for all variables).
    ///
    /// # Type Parameters
    /// - `T`: Scalar type of the input array. Must implement `Copy`, `Into<f64>`, and `Mul<f64, Output=f64>`.
    /// - `V`: Scalar type of the output array. Must implement `Copy` and `From<f64>`.
    ///
    /// # Behavior
    /// - If all relevant mask values are `1`, interpolation is performed normally.
    /// - If any mask value under a non-zero weight is `0`, interpolation is skipped:
    ///     - `nodata` is written to all output variables at `out_idx`.
    ///     - The output mask at `out_idx` is set to `0`.
    ///
    /// # Safety
    /// This method does not perform bounds checks. The caller must ensure that all computed
    /// indices `row_c + irow - 2` and `col_c + icol - 2` are valid for both data and mask arrays.
    /// Accessing out-of-bounds data results in undefined behavior.
    ///
    /// # Example
    /// ```ignore
    /// interpolator.interpolate_masked_unchecked(
    ///     weights_row,
    ///     weights_col,
    ///     &input_array,
    ///     &input_mask,
    ///     &mut output_array,
    ///     &mut output_mask,
    ///     out_index,
    ///     row_center,
    ///     col_center,
    ///     nodata_value,
    /// );
    /// ```
    ///
    /// # See Also
    /// - [`interpolate_nomask_unchecked`](Self::interpolate_nomask_unchecked): Faster variant without a validity mask.
    /// - [`interpolate_masked_partial`](...): Safe version with bounds checking and masking.
    #[inline(always)]
    fn interpolate_masked_unchecked<T, V, IC>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        out_idx: usize,
        row_c: i64, 
        col_c: i64,
        nodata: V,
        context: &mut IC,
        )
    where
        T: Copy + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
        let ncol = array_in.ncol;
        let array_in_var_size = array_in.var_size;
        let array_out_var_size = array_out.var_size;
        let mut arr_iflat: usize;
        let mut computed: f64;
        let mut computed_col: f64;
        let mut array_out_var_shift: usize = 0;
        

        let row_c_m2 = (row_c - 2) as usize;
        let col_c_m2 = (col_c - 2) as usize;

        // Pre check mask
        // Ignore mask value where weights are zero
        let valid; // = 1u8;
        arr_iflat = row_c_m2 * ncol + col_c_m2;

        let local_cache: Vec<u8> = vec![0; 25];
        let mut local_cache = local_cache.into_boxed_slice();

        valid = context.input_mask().is_valid_weighted_window(arr_iflat, 5, 5, weights_row, weights_col, &mut local_cache); 
      
        if valid == 0 {
            
            for _ivar in 0..array_in.nvar {
                array_out.data[out_idx + array_out_var_shift] = nodata;
                array_out_var_shift += array_out_var_size;
            }
            context.output_mask().set_value(out_idx, 0);
            return;
        }

        // Loop on multipe variables in input array.      
        for _ivar in 0..array_in.nvar {

            computed = 0.0;
                        
            for irow in 0..=4 {
                computed_col = 0.0;
                
                for icol in 0..=4 {
                    
                    // flat 1d index computation
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[4 - icol as usize];
                    
                    arr_iflat += 1;
                }
                computed += weights_row[4 - irow as usize] * computed_col;
                
                arr_iflat += ncol - 5;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            arr_iflat += array_in_var_size - 5*ncol;
            array_out_var_shift += array_out_var_size;
        }

        context.output_mask().set_value(out_idx, 1);

    }

/*
    #[inline(always)]
    fn interpolate_masked_unchecked<T, V, IC>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        out_idx: usize,
        row_c: i64, 
        col_c: i64,
        nodata: V,
        context: &mut IC,
        )
    where
        T: Copy + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
        let ncol = array_in.ncol;
        let array_in_var_size = array_in.var_size;
        let array_out_var_size = array_out.var_size;
        let mut arr_irow: usize;
        let mut arr_icol: usize;
        let mut arr_iflat: usize;
        let mut computed: f64;
        let mut computed_col: f64;
        let mut array_in_var_shift: usize = 0;
        let mut array_out_var_shift: usize = 0;
        
        // Pre check mask
        // Ignore mask value where weights are zero
        let mut valid = 1u8;
        for irow in 0..=4 {
            if weights_row[4 - irow as usize] == 0.0 {
                continue;
            }
            
            arr_irow = (row_c + irow - 2) as usize;
            for icol in 0..=4 {
                if weights_col[4 - icol as usize] == 0.0 {
                    continue;
                }
                
                arr_icol = (col_c + icol - 2) as usize;
                arr_iflat = arr_irow * ncol + arr_icol;
                valid &= context.input_mask().is_valid(arr_iflat);
            }
        }
        
        if valid == 0 {
            for ivar in 0..array_in.nvar {
                array_out.data[out_idx + ivar * array_out_var_size] = nodata;
            }
            context.output_mask().set_value(out_idx, 0);
            return;
        }
        
        // Loop on multipe variables in input array.
        for ivar in 0..array_in.nvar {
            computed = 0.0;
            
            for irow in 0..=4 {
                computed_col = 0.0;
                arr_irow = (row_c + irow - 2) as usize;
                
                for icol in 0..=4 {
                    arr_icol = (col_c + icol - 2) as usize;
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + arr_irow * ncol + arr_icol;
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[4 - icol as usize];
                }
                computed += weights_row[4 - irow as usize] * computed_col;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            array_in_var_shift += array_in_var_size;
            array_out_var_shift += array_out_var_size;
        }
        context.output_mask().set_value(out_idx, 1);
    }
*/
    
    /// Performs a partial weighted interpolation on a 5×5 window centered at `(row_c, col_c)`
    /// within a multidimensional input array `array_in`, taking into account validity masks.
    ///
    /// # Description
    /// - Applies a 2D interpolation using the given row and column weight vectors (`weights_row` and `weights_col`), each of length 5.
    /// - Considers a validity mask `array_mask_in` for the input window.
    /// - Values outside the array bounds are treated as zero if their associated weight is zero (ignored).
    /// - If any out-of-bounds value has a non-zero weight, the interpolation is considered invalid,
    ///   and the output is set to `nodata` with the mask marked as invalid.
    /// - The interpolation is performed independently for each variable (dimension) in the input array.
    /// - The result is written into `array_out` at the index `out_idx`, and the output mask is updated accordingly.
    ///
    /// # Parameters
    /// - `weights_row`: Interpolation weights along the row dimension (length 5).
    /// - `weights_col`: Interpolation weights along the column dimension (length 5).
    /// - `array_in`: Input array view containing multiple variables and their data.
    /// - `array_mask_in`: Validity mask corresponding to `array_in` (1 = valid, 0 = invalid).
    /// - `array_out`: Mutable output array view to store interpolated results.
    /// - `array_mask_out`: Mutable mask array view for the output.
    /// - `out_idx`: 1D output index at which to write the interpolated value.
    /// - `row_c`, `col_c`: Coordinates of the center pixel around which the 5×5 window is defined.
    ///
    /// # Preconditions
    /// - Weight arrays correspond to a kernel centered on `(row_c, col_c)`.
    /// - Input and output arrays and masks have compatible dimensions.
    /// - The interpolation window size is fixed at 5×5.
    ///
    /// # Behavior
    /// 1. Checks that all points with non-zero weights lie within the input array bounds.
    ///    If any out-of-bounds point has a non-zero weight, interpolation is invalidated.
    /// 2. Verifies that all these points are marked valid in `array_mask_in`.
    /// 3. If valid, computes the weighted sum for each variable, ignoring out-of-bounds points or weights that are zero.
    /// 4. Writes the computed interpolated values to `array_out` and sets the output mask as valid.
    ///
    /// # Notes
    /// - The filter is not renormalized, meaning the sum of weights may be less than 1
    ///   if some weights correspond to out-of-bounds points and are ignored.
    /// - This effectively treats out-of-bounds data as zero unless their weight is non-zero,
    ///   in which case the interpolation is invalid.
    /// - Marked `#[inline(always)]` for optimization.
    ///
    /// # Example
    /// ```ignore
    /// self.interpolate_masked_partial(
    ///     &weights_row,
    ///     &weights_col,
    ///     &array_in,
    ///     &array_mask_in,
    ///     &mut array_out,
    ///     &mut array_mask_out,
    ///     out_idx,
    ///     row_c,
    ///     col_c,
    /// );
    /// ```
    #[inline(always)]
    fn interpolate_masked_partial<T, V, IC>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        //input_mask_strategy: &M,
        //array_mask_in: &GxArrayView<'_, u8>,
        array_out: &mut GxArrayViewMut<'_, V>,
        //output_mask_strategy: &mut N,
        //array_mask_out: &mut GxArrayViewMut<'_, u8>,
        out_idx: usize,
        row_c: i64, 
        col_c: i64,
        nodata: V,
        context: &mut IC,
        )
    where
        T: Copy + std::ops::Mul<f64, Output=f64> + Into<f64> + PartialEq,
        V: Copy + From<f64> + PartialEq,
        IC: GxArrayViewInterpolationContextTrait,
        //M: GxArrayViewInterpolatorInputMaskStrategy,
        //N: GxArrayViewInterpolatorOutputMaskStrategy,
    {
        let ncol = array_in.ncol;
        let array_in_var_size = array_in.var_size;
        let array_out_var_size = array_out.var_size;
        let mut arr_irow;
        let mut arr_icol;
        let mut arr_iflat: usize;
        let mut computed: f64;
        let mut computed_col: f64;
        let mut array_in_var_shift: usize = 0;
        let mut array_out_var_shift: usize = 0;
        
        // Pre check mask
        let mut valid = 1u8;
        //let mut partial_valid = 1u8;
        for irow in 0..=4 {
            if weights_row[4 - irow as usize] == 0.0 {
                continue;
            }
            arr_irow = row_c + irow - 2;
            
            for icol in 0..=4 {
                if weights_col[4 - icol as usize] == 0.0 {
                    continue;
                }
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 {
                    // If we came here the weights are both positivs, we cant
                    // ignore that we go out of bounds as far as the validity is
                    // concerned.
                    valid = 0;
                    break;
                }
                
                arr_icol = col_c + icol - 2;
                
                if arr_icol < 0 || arr_icol >= array_in.ncol_i64 {
                    // If we came here the weights are both positivs, we cant
                    // ignore that we go out of bounds as far as the validity is
                    // concerned.
                    valid = 0;
                    break;
                }
                arr_iflat = (arr_irow as usize) * ncol + (arr_icol as usize);
                valid &= context.input_mask().is_valid(arr_iflat);
            }
        }
        
        if valid == 0 {
            for _ivar in 0..array_in.nvar {
                array_out.data[out_idx + array_out_var_shift] = nodata;
                array_out_var_shift += array_out_var_size;
            }
            context.output_mask().set_value(out_idx, 0);
            //array_mask_out.data[out_idx] = 0;
            return;
        }
        
        // Loop on multipe variables in input array.
        // If a iteration condition fails it is equivalent as considering a zero
        // for the corresponding row or column value.
        for _ivar in 0..array_in.nvar {
            computed = 0.0;
            
            for irow in 0..=4 {
                arr_irow = row_c + irow - 2;
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 {
                    continue;
                }
                computed_col = 0.0;
                
                for icol in 0..=4 {
                    arr_icol = col_c + icol - 2;
                    if arr_icol < 0 || arr_icol >= array_in.ncol_i64 {
                        continue;
                    }
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + (arr_irow as usize) * ncol + (arr_icol as usize);
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[4 - icol as usize];
                }
                computed += weights_row[4 - irow as usize] * computed_col;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            array_in_var_shift += array_in_var_size;
            array_out_var_shift += array_out_var_size;
        }
        context.output_mask().set_value(out_idx, 1);
    }
}

impl GxArrayViewInterpolator for GxOptimizedBicubicInterpolator
{
    fn new() -> Self {
        Self {
            kernel_row_size: 5,
            kernel_col_size: 5,
        }
    }
    
    fn kernel_row_size(&self) -> usize {
        self.kernel_row_size
    }
    
    fn kernel_col_size(&self) -> usize {
        self.kernel_col_size
    }
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; self.kernel_row_size + self.kernel_col_size];
        buffer.into_boxed_slice()
    }
    
    /// weights_buffer : preallocated array of 10 elements
    /// todo : manage mask
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
            //array_mask_in: &GxArrayView<'_, u8>,
            //array_mask_out: &mut GxArrayViewMut<'_, u8>, 
            ) -> Result<(), String>
    where
        T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + PartialEq + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
        
        //<U as Mul<f64, Output=f64>>::Output: Add,
    {
        // Get the nearest corresponding index corresponding to the target position 
        let kernel_center_row: i64 = (target_row_pos + 0.5).floor() as i64;
        let kernel_center_col: i64 = (target_col_pos + 0.5).floor() as i64;
 
        //let array_in_var_size = array_in.nrow * array_in.ncol;
        let array_out_var_size = array_out.var_size;
        
        // Consider mask valid (if any)
        context.output_mask().set_value(out_idx, 1);
        
        // After compilation that test will have no cost in monomorphic created
        // method
        if IC::BoundsCheck::do_check() {
            // Check that all required data for interpolation is within the input
            // array shape - assuming here the radius is 2.
            // Here we do not need to check borders inside the inner loops.
            // That should be the most common path.
            if (kernel_center_row >=2)
                    && (kernel_center_row < array_in.nrow_i64-2)
                    && (kernel_center_col >= 2)
                    && (kernel_center_col < array_in.ncol_i64-2) {
                let rel_row: f64 = target_row_pos - kernel_center_row as f64;
                let rel_col: f64 = target_col_pos - kernel_center_col as f64;
                
                // Create slices to give to weight computation methods
                let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size);
                
                // from here - pass slice for weight computation
                // slices are used here in order to limit buffer allocation
                optimized_bicubic_kernel_weights_opt(rel_row, kernel_weights_row_slice);
                optimized_bicubic_kernel_weights_opt(rel_col, kernel_weights_col_slice);
                
                
                if context.input_mask().is_enabled() {
                    /* self.interpolate_masked_unchecked(kernel_weights_row_slice, kernel_weights_col_slice,
                            array_in, input_mask, array_out, output_mask, out_idx,
                            kernel_center_row, kernel_center_col, nodata_out); */
                    self.interpolate_masked_unchecked(kernel_weights_row_slice, kernel_weights_col_slice,
                            array_in, array_out, out_idx,
                            kernel_center_row, kernel_center_col, nodata_out, context);
                }
                else {
                    self.interpolate_nomask_unchecked(kernel_weights_row_slice, kernel_weights_col_slice,
                            array_in, array_out, out_idx, kernel_center_row, kernel_center_col);
                }
            }
            // Check the center is within the input array shape
            // The first test has not been passed : meaning at least one border is crossed.
            // We ensure here that the target point is within the input array shape.
            else if (kernel_center_row >=0)
                    && (kernel_center_row < array_in.nrow_i64)
                    && (kernel_center_col >= 0)
                    && (kernel_center_col < array_in.ncol_i64) {
                let rel_row: f64 = target_row_pos - kernel_center_row as f64;
                let rel_col: f64 = target_col_pos - kernel_center_col as f64;
                
                // Create slices to give to weight computation methods
                let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size);
                
                // from here - pass slice for weight computation
                // slices are used here in order to limit buffer allocation
                optimized_bicubic_kernel_weights_opt(rel_row, kernel_weights_row_slice);
                optimized_bicubic_kernel_weights_opt(rel_col, kernel_weights_col_slice);
                
                if context.input_mask().is_enabled() {
                    /* self.interpolate_masked_partial(kernel_weights_row_slice, kernel_weights_col_slice,
                            array_in, input_mask, array_out, output_mask, out_idx,
                            kernel_center_row, kernel_center_col, nodata_out); */
                    self.interpolate_masked_partial(kernel_weights_row_slice, kernel_weights_col_slice,
                            array_in, array_out, out_idx,
                            kernel_center_row, kernel_center_col, nodata_out, context);
                }
                else {
                    self.interpolate_nomask_partial(kernel_weights_row_slice, kernel_weights_col_slice,
                            array_in, array_out, context.output_mask(), out_idx,
                            kernel_center_row, kernel_center_col, nodata_out);
                }

            } else {
                for ivar in 0..array_in.nvar {                
                    // Write nodata value to output buffer
                    array_out.data[out_idx + ivar * array_out_var_size] = nodata_out;
                }
                context.output_mask().set_value(out_idx, 0);
            }
        } else {
            let rel_row: f64 = target_row_pos - kernel_center_row as f64;
            let rel_col: f64 = target_col_pos - kernel_center_col as f64;
            
            // Create slices to give to weight computation methods
            let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size);
            
            // from here - pass slice for weight computation
            // slices are used here in order to limit buffer allocation
            optimized_bicubic_kernel_weights_opt(rel_row, kernel_weights_row_slice);
            optimized_bicubic_kernel_weights_opt(rel_col, kernel_weights_col_slice);
            
            if context.input_mask().is_enabled() {
                self.interpolate_masked_unchecked(kernel_weights_row_slice, kernel_weights_col_slice,
                        array_in, array_out, out_idx,
                        kernel_center_row, kernel_center_col, nodata_out, context);
            }
            else {
                self.interpolate_nomask_unchecked(kernel_weights_row_slice, kernel_weights_col_slice,
                        array_in, array_out, out_idx, kernel_center_row, kernel_center_col);
            }
        }
        Ok(())
    }
    
    

}


#[cfg(test)]
mod gx_optimized_bicubic_kernel_tests {
    use super::*;
    //use crate::core::interp::gx_array_view_interp::{GxArrayViewInterpolationContext, DefaultCtx};
    use crate::core::interp::gx_array_view_interp::{DefaultCtx};
    
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
        let interp = GxOptimizedBicubicInterpolator::new();

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
