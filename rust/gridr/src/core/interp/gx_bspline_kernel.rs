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
//! Cardinal B-spline interpolators with compile-time order specialization.
//!
//! This module provides efficient B-spline interpolation implementations using Rust's 
//! const generics to specialize interpolators at compile time for different spline orders.
//!
//! # Core Components
//!
//! - **B-spline basis functions**: Optimized implementations for orders 3, 5, 7, 9, and 11
//! - **Generic interpolator**: `GxBSplineInterpolator<N>` with compile-time order parameter
//! - **Type aliases**: Convenience types (`GxBSpline3Interpolator`, `GxBSpline5Interpolator`, etc.)
//! - **Interpolation strategies**: Bounds checking and mask handling variants (unchecked/partial)
//!
//! # Trait Implementations
//!
//! ## `GxArrayViewInterpolator`
//! The main interpolation interface implemented by all B-spline interpolators. Provides:
//! - `array1_interp2()`: Core 2D interpolation method with configurable bounds checking and masking
//! - `kernel_row_size()`, `kernel_col_size()`: Kernel dimensions based on spline order
//! - `total_margins()`: Required margins for prefiltering and interpolation
//! - `allocate_kernel_buffer()`: Memory allocation for interpolation weights
//!
//! ## `GxBSplineInterpolatorTrait<N>`
//! B-spline-specific interface providing:
//! - `bspline()`: Evaluates the B-spline basis function at a given coordinate
//! - `bspline_kernel_weights()`: Computes interpolation weights for the kernel
//! - `array1_bspline_prefiltering_ext()`: Applies recursive prefiltering to input data
//!
//! # Supported Orders
//!
//! | Order | Kernel Radius | Type Alias                |
//! |-------|---------------|---------------------------|
//! | 3     | 2             | `GxBSpline3Interpolator`  |
//! | 5     | 3             | `GxBSpline5Interpolator`  |
//! | 7     | 4             | `GxBSpline7Interpolator`  |
//! | 9     | 5             | `GxBSpline9Interpolator`  |
//! | 11    | 6             | `GxBSpline11Interpolator` |
//!
//! # Performance Optimizations
//!
//! The implementation provides multiple interpolation paths optimized for different scenarios:
//! - **Unchecked**: No bounds checking, assumes valid indices (fastest)
//! - **Partial**: Bounds checking with graceful handling of edge cases
//! - **Masked**: Validity mask support with invalid pixel propagation
//!
//! # Usage Example
//!
//! ```ignore
//! use gridr::core::gx_bspline_interp::{GxBSpline5Interpolator, GxArrayViewInterpolator};
//!
//! let args = GxBSplineInterpolatorArgs {
//!     epsilon: 1e-3,
//!     mask_influence_threshold: 0.001,
//! };
//! let mut interp = GxBSpline5Interpolator::new(&args);
//! interp.initialize()?;
//!
//! // Prefilter input data
//! interp.array1_bspline_prefiltering_ext(&mut input_array, Some(&mut input_mask))?;
//!
//! // Perform interpolation
//! interp.array1_interp2(&mut weights, row_pos, col_pos, out_idx, 
//!                       &input_array, &mut output_array, nodata, &mut context)?;
//! ```
//!
//! # References
//!
//! Briand, T., & Monasse, P. (2018). Theory and Practice of Image B-Spline Interpolation.
//! *Image Processing On Line*, 8, 99-141. https://doi.org/10.5201/ipol.2018.221
use::std::format;
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use crate::core::gx_errors::GxError;
use super::gx_array_view_interp::{GxArrayViewInterpolator, GxArrayViewInterpolatorArgs, GxArrayViewInterpolationContextTrait, GxArrayViewInterpolatorBoundsCheckStrategy, GxArrayViewInterpolatorInputMaskStrategy, GxArrayViewInterpolatorOutputMaskStrategy};
use super::gx_bspline_prefiltering::{
    compute_2d_truncation_index,
    compute_2d_domain_extension_from_truncation_idx,
    array1_bspline_prefiltering_ext_gene,
    TRUNCATION_INDEX_BUFFER_MAX_SIZE,
    TRUNCATION_L_BUFFER_MAX_SIZE
};

/// Cubic B-spline function - radius 2
#[inline]
pub fn bspline3(x: f64) -> f64
{
    let x = x.abs();
    if x < 1.
    {
        return 4. + (-6. + 3.*x)*x*x;
    }
    else if x < 2.
    {
        let x = 2. - x;
        return x*x*x;
    }
    0.0
}

/// Quintic B-spline function - radius 3
#[inline]
pub fn bspline5(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return ((-10.*x + 30.)*x2 - 60.)*x2 + 66.;
    }
    else if x < 2.
    {
        let x = 2. - x;
        return 1. + (5. + (10. + (10. + (5. - 5.*x)*x)*x)*x)*x;
    }
    else if x < 3.
    {
        let x = 3. - x;
        let x2 = x*x;
        return x2*x2*x;
    }
    0.0
}

/// Septic B-spline function - radius 4
#[inline]
pub fn bspline7(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return (((35.*x - 140.)*x2 + 560.)*x2 - 1680.)*x2 + 2416.;
    }
    else if x < 2.
    {
        let x = 2. - x;
        return 120. + (392. + (504. + (280. + (-84. + (-42. + 21.*x)*x)*x*x)*x)*x)*x;
    }
    else if x < 3.
    {
        let x = 3. - x;
        return ((((((-7.*x + 7.)*x + 21.)*x + 35.)*x + 35.)*x + 21.)*x + 7.)*x + 1.;
    }
    else if x < 4.
    {
        let x = 4. - x;
        let x2 = x*x;
        return x2*x2*x2*x;
    }
    0.0
}

/// Nonic B-spline function - radius 5
#[inline]
pub fn bspline9(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return (((((-63.*x + 315.)*x2 - 2100.)*x2 + 11970.)*x2 - 44100.)*x2 + 78095.)*2.;
    }
    else if x <= 2.
    {
        let x = 2. - x;
        return 14608. + (36414. + (34272. + (11256. + (-4032. + (-4284. + (-672. + (504. + (252. - 84.*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x <= 3.
    {
        let x = 3. - x;
        return 502. + (2214. + (4248. + (4536. + (2772. + (756. + (-168. + (-216. + (-72. + 36.*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 4.
    {
        let x = 4. - x;
        return 1. + (9. + (36. + (84. + (126. + (126. + (84. + (36. + (9. - 9.*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 5.
    {
        let x = 5. - x;
        let x3 = x*x*x;
        return x3*x3*x3;
    }
    0.0
}

/// Eleventh order B-spline function - radius 6
#[inline]
pub fn bspline11(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return 15724248. + (-7475160. + (1718640. + (-255024. + (27720.
            + (-2772. + 462.*x)*x2)*x2)*x2)*x2)*x2;
    }
    else if x <= 2.
    {
        let x = 2. - x;
        return 2203488. + (4480872. + (3273600. + (574200. + (-538560.
            + (-299376. + (39600. + (7920. + (-2640. + (-1320.
            + 330.*x)*x)*x)*x)*x*x)*x)*x)*x)*x)*x;
    }
    else if x <= 3.
    {
        let x = 3. - x;
        return 152637. + (515097. + (748275. + (586575. + (236610. + (12474.
            + (-34650. + (-14850. + (-495. + (1485.
            + (495.-165.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x <= 4.
    {
        let x = 4. - x;
        return 2036. + (11132. + (27500. + (40260. + (38280. + (24024. + (9240.
            + (1320. + (-660. + (-440. + (-110.
            + 55.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 5.
    {
        let x = 5. - x;
        return 1. + (11. + (55. + (165. + (330. + (462. + (462. + (330. + (165.
            + (55. + (11. - 11.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 6.
    {
        let x = 6. - x;
        let x2 = x*x;
        let x4 = x2*x2;
        return x4*x4*x2*x;
    }
    0.0
}

/// Trait defining the interface for generic B-spline interpolators with configurable order.
/// 
/// This trait provides the core functionality for B-spline interpolation operations, including:
/// - Access to filter poles for prefiltering operations
/// - Evaluation of B-spline basis functions
/// - Pre-filtering operations to prepare input data
/// 
/// The trait is parameterized by one const generic:
/// - `N`: The order of the B-spline (must be odd: 3, 5, 7, 9, 11)
/// 
/// # Type Constraints
/// - `N` must be an odd integer (3, 5, 7, 9, or 11)
/// 
/// # Implementation Requirements
/// Implementors must provide:
/// 1. `get_poles()`: Returns the filter poles for the specific spline order
/// 2. `bspline()`: Evaluates the B-spline basis function at a given point
/// 3. `array1_bspline_prefiltering_ext2()`: Applies pre-filtering to input data
pub trait GxBSplineInterpolatorTrait<const N: usize>: GxArrayViewInterpolator
{
    /*
    /// Returns a slice containing the filter poles for the B-spline of order N.
    /// 
    /// These poles are used in the pre-filtering stage to prepare input data before interpolation.
    /// The returned slice has exactly N/2 elements.
    /// 
    /// # Returns
    /// - Slice of f64 values representing the filter poles
    #[inline(always)]
    fn get_poles(&self) -> &[f64];
    */
    
    /// Returns the number of poles
    ///
    /// # Returns
    /// - usize: The number of poles for the B-Spline.
    fn get_npoles(&self) -> usize;
    
    /// Evaluates the B-spline basis function of order N at the specified coordinate.
    /// 
    /// This method computes the value of the B-spline basis function at position `x`.
    /// The implementation varies based on the spline order N.
    /// 
    /// # Parameters
    /// - `x`: The coordinate at which to evaluate the B-spline function
    /// 
    /// # Returns
    /// - f64: The value of the B-spline basis function at position x
    fn bspline(&self, x: f64) -> f64;
    
    
    /// Evaluates the B-spline weights
    fn bspline_kernel_weights(&self, x: f64, weights: &mut [f64]);
    
    /// Applies pre-filtering to input data using the B-spline filter poles.
    /// 
    /// This operation prepares input data by applying the B-spline pre-filtering with the specified
    /// poles and truncation indices.
    /// 
    /// # Parameters
    /// - `ima_in`: Mutable reference to the input data array to be pre-filtered
    /// - `mask_in`: Optional mutable reference to the input mask
    /// 
    /// # Returns
    /// - `Ok(())` if pre-filtering completes successfully
    /// - `Err(GxError)` containing error information if pre-filtering fails
    fn array1_bspline_prefiltering_ext<'a>(
        &'a self,
        ima_in: &mut GxArrayViewMut<'_, f64>,
        mask_in: Option<&'a mut GxArrayViewMut<'a, u8>>,
    ) -> Result<(), GxError>;
}

/// Generic implementation of B-spline interpolators with compile-time order and pole configuration.
///
/// This struct serves as the concrete implementation for all supported B-spline orders (3, 5, 7, 9, 11).
/// It stores the order and number of poles as runtime parameters while utilizing compile-time
/// constants for the specific mathematical function implementations. The struct also contains
/// prefiltering parameters and associated buffers for efficient computation.
///
/// # Fields
/// - `order`: The order of the B-spline (N), determining the smoothness and support size
/// - `npoles`: The number of poles (N/2), used in the recursive filtering implementation
/// - `epsilon`: Precision parameter for the truncation index calculation. Defines the 
///   acceptable error when approximating the infinite sums. Smaller values require 
///   larger margins for prefiltering. The total required margin combines both the 
///   prefiltering margin (truncation index) and the interpolation kernel radius.
/// - `mask_influence_threshold`: Residual influence threshold $s$ used to compute the 
///   radius of the propagation of masked data. Required when `ima_mask_in` is provided.
///   Determines the acceptable relative contamination from invalid pixels.
/// - `truncation_index`: Buffer storing truncation index
/// - `domain_extension`: Buffer storing domain extension
#[derive(Clone, Debug)]
pub struct GxBSplineInterpolator<const N: usize> {
    /// The order of the B-spline (N), determining the smoothness and support size
    pub order: usize,
    /// The number of poles (N/2), used in the recursive filtering implementation
    pub npoles: usize,
    /// Precision parameter for the truncation index calculation
    pub epsilon: f64,
    /// Acceptable relative contamination from invalid pixels.
    pub mask_influence_threshold: f64,
    /// Buffer storing truncation index
    pub truncation_index: [usize; TRUNCATION_INDEX_BUFFER_MAX_SIZE],
    /// Buffer storing domain extension
    pub domain_extension: [usize; TRUNCATION_L_BUFFER_MAX_SIZE],
}

impl<const N: usize> GxBSplineInterpolatorTrait<N> for GxBSplineInterpolator<N> {
    
    /// Returns the number of poles
    ///
    /// # Returns
    /// - usize: The number of poles for the B-Spline.
    #[inline(always)]
    fn get_npoles(&self) -> usize {
        self.npoles
    }
    
    /// Evaluates the B-spline basis function of order N at the specified coordinate.
    /// 
    /// This implementation uses compile-time matching to select the appropriate B-spline
    /// evaluation function based on the order N. Each function is optimized for its specific
    /// order and provides accurate evaluation of the corresponding B-spline basis function.
    /// 
    /// # Parameters
    /// - `x`: The coordinate at which to evaluate the B-spline function
    /// 
    /// # Returns
    /// - f64: The value of the B-spline basis function at position x
    /// 
    /// # Panics
    /// - If the spline order is not supported (N not in {3,5,7,9,11})
    #[inline(always)]
    fn bspline(&self, x: f64) -> f64 {
        match N {
            3 => bspline3(x),
            5 => bspline5(x),
            7 => bspline7(x),
            9 => bspline9(x),
            11 => bspline11(x),
            _ => panic!("Unsupported spline order"),
        }
    }
    
    #[inline(always)]
    /// Non optimized B-spline weight computation
    fn bspline_kernel_weights(&self, x: f64, weights: &mut [f64]) 
    {
        let nt = N / 2 + 1;
        for k in 0..=2*nt {
            weights[k] = 0.0;
            let xx = (x + k as f64 - nt as f64).abs();
            weights[k] = self.bspline(xx)
        }
    }
    
    /// Applies pre-filtering to input data using the B-spline filter poles.
    /// 
    /// This method performs the pre-filtering operation using the specific implementation
    /// for each supported spline order. It applies the B-spline filter with the provided
    /// poles and truncation indices to the input data to prepare it for interpolation.
    /// 
    /// # Parameters
    /// - `ima_in`: Mutable reference to the input data array to be pre-filtered
    /// - `mask_in`: Optional mutable reference to the input mask
    /// 
    /// # Returns
    /// - `Ok(())` if pre-filtering completes successfully
    /// - `Err(GxError)` containing error information if pre-filtering fails
    /// 
    /// # Panics
    /// - If the spline order is not supported (N not in {3,5,7,9,11})
    #[inline]
    fn array1_bspline_prefiltering_ext<'a>(
        &'a self,
        ima_in: &mut GxArrayViewMut<'_, f64>,
        mask_in: Option<&'a mut GxArrayViewMut<'a, u8>>,
    ) -> Result<(), GxError>
    {
        match N {
            3 | 5 | 7 | 9 | 11 => {
                return array1_bspline_prefiltering_ext_gene(
                    self.order,
                    self.epsilon,
                    Some(&self.truncation_index),
                    ima_in,
                    mask_in,
                    Some(self.mask_influence_threshold),
                );
            },
            _ => {
                return Err(GxError::ErrMessage("Unsupported spline order".to_string()));
            },
        }
    }
}

impl<const N: usize> GxBSplineInterpolator<N>
{
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
        
        let kernel_radius: i64 = (N / 2 + 1) as i64;
        let kernel_support: i64 = 2 * kernel_radius;
        
        // Loop on multipe variables in input array.
        for _ivar in 0..array_in.nvar {
            computed = 0.0;
            
            for irow in 0..=kernel_support {
                computed_col = 0.0;
                arr_irow = (row_c + irow - kernel_radius) as usize;
                
                for icol in 0..=kernel_support {
                    arr_icol = (col_c + icol - kernel_radius) as usize;
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + arr_irow * ncol + arr_icol;
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[(kernel_support - icol) as usize];
                    
                }
                computed += weights_row[(kernel_support - irow) as usize] * computed_col;
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
    fn interpolate_nomask_partial<T, V, M>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        output_mask_strategy: &mut M,
        out_idx: usize,
        row_c: i64, 
        col_c: i64,
        nodata: V,
        )
    where
        T: Copy + std::ops::Mul<f64, Output=f64> + Into<f64> + PartialEq,
        V: Copy + From<f64> + PartialEq,
        M: GxArrayViewInterpolatorOutputMaskStrategy,
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
        
        let kernel_radius: i64 = (N / 2 + 1) as i64;
        let kernel_support: i64 = 2 * kernel_radius;
        
        // Pre check boundaries and weights value
        for irow in 0..=kernel_support {
            if weights_row[(kernel_support - irow) as usize] == 0.0 {
                continue;
            }
            arr_irow = row_c + irow - kernel_radius;
            
            for icol in 0..=kernel_support {
                if weights_col[(kernel_support - icol) as usize] == 0.0 {
                    continue;
                }
                
                arr_icol = col_c + icol - kernel_radius;
                
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
            
            for irow in 0..=kernel_support {
                arr_irow = row_c + irow - kernel_radius;
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 {
                    continue;
                }
                computed_col = 0.0;
                
                for icol in 0..=kernel_support {
                    arr_icol = col_c + icol - kernel_radius;
                    if arr_icol < 0 || arr_icol >= array_in.ncol_i64 {
                        continue;
                    }
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + (arr_irow as usize) * ncol + (arr_icol as usize);
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[(kernel_support - icol) as usize];
                }
                computed += weights_row[(kernel_support - irow) as usize] * computed_col;
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
        
        let kernel_radius: i64 = (N / 2 + 1) as i64;
        let kernel_support: i64 = 2 * kernel_radius;
        let kernel_support_p1_usize = (kernel_support + 1) as usize;

        let row_c_m2 = (row_c - kernel_radius) as usize;
        let col_c_m2 = (col_c - kernel_radius) as usize;

        // Pre check mask
        // Ignore mask value where weights are zero
        let valid; // = 1u8;
        arr_iflat = row_c_m2 * ncol + col_c_m2;

        let cache_size: usize = kernel_support_p1_usize * kernel_support_p1_usize;
        let local_cache: Vec<u8> = vec![0; cache_size];
        let mut local_cache = local_cache.into_boxed_slice();

        valid = context.input_mask().is_valid_weighted_window(arr_iflat, kernel_support_p1_usize, kernel_support_p1_usize, weights_row, weights_col, &mut local_cache); 
      
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
                        
            for irow in 0..=kernel_support {
                computed_col = 0.0;
                
                for icol in 0..=kernel_support {
                    
                    // flat 1d index computation
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[(kernel_support - icol) as usize];
                    
                    arr_iflat += 1;
                }
                computed += weights_row[(kernel_support - irow) as usize] * computed_col;
                
                arr_iflat += ncol - kernel_support_p1_usize;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            arr_iflat += array_in_var_size - kernel_support_p1_usize*ncol;
            array_out_var_shift += array_out_var_size;
        }

        context.output_mask().set_value(out_idx, 1);

    }
    
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
        
        let kernel_radius: i64 = (N / 2 + 1) as i64;
        let kernel_support: i64 = 2 * kernel_radius;
        
        // Pre check mask
        let mut valid = 1u8;
        //let mut partial_valid = 1u8;
        for irow in 0..=kernel_support {
            if weights_row[(kernel_support - irow) as usize] == 0.0 {
                continue;
            }
            arr_irow = row_c + irow - kernel_radius;
            
            for icol in 0..=kernel_support {
                if weights_col[(kernel_support - icol) as usize] == 0.0 {
                    continue;
                }
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 {
                    // If we came here the weights are both positivs, we cant
                    // ignore that we go out of bounds as far as the validity is
                    // concerned.
                    valid = 0;
                    break;
                }
                
                arr_icol = col_c + icol - kernel_radius;
                
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
            
            for irow in 0..=kernel_support {
                arr_irow = row_c + irow - kernel_radius;
                if arr_irow < 0 || arr_irow >= array_in.nrow_i64 {
                    continue;
                }
                computed_col = 0.0;
                
                for icol in 0..=kernel_support {
                    arr_icol = col_c + icol - kernel_radius;
                    if arr_icol < 0 || arr_icol >= array_in.ncol_i64 {
                        continue;
                    }
                    
                    // flat 1d index computation
                    arr_iflat = array_in_var_shift + (arr_irow as usize) * ncol + (arr_icol as usize);
                    // add current weighted product
                    computed_col += array_in.data[arr_iflat] * weights_col[(kernel_support - icol) as usize];
                }
                computed += weights_row[(kernel_support - irow) as usize] * computed_col;
            }
            // Write interpolated value to output buffer
            array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            
            array_in_var_shift += array_in_var_size;
            array_out_var_shift += array_out_var_size;
        }
        context.output_mask().set_value(out_idx, 1);
    }
}

/// A structure holding the parameters required to create a `GxBSplineInterpolator<N>`
///
/// This structure implements the `GxBSplineInterpolatorArgs` trait and contains
/// the essential configuration parameters for B-spline interpolation.
pub struct GxBSplineInterpolatorArgs {
    /// Precision parameter for truncation index calculation.
    pub epsilon: f64,
    /// Acceptable relative contamination from invalid pixels.
    pub mask_influence_threshold: f64,
}

/// Concrete implementation of `GxArrayViewInterpolatorArgs` trait for B-spline interpolators
///
/// This implementation provides the necessary interface for interpolators derived from
/// `GxBSplineInterpolator<N>`. It implements the `bspline_args` function to expose
/// the configured parameters for B-spline interpolation.
impl GxArrayViewInterpolatorArgs for GxBSplineInterpolatorArgs {
    fn bspline_args(&self) -> Option<(f64, f64)> {
        Some((self.epsilon, self.mask_influence_threshold))
    }
}

impl<const N: usize> GxArrayViewInterpolator for GxBSplineInterpolator<N> {
    /// Creates a new instance of the B-spline interpolator with the specified order and poles.
    /// 
    /// # Returns
    /// - New instance with order N and npoles N/2
    fn new(args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        GxBSplineInterpolator {
            order: N,
            npoles: N/2,
            epsilon: args.bspline_args().expect("GxBSplineInterpolator requires bspline_args. Check the GxArrayViewInterpolatorArgs argument that has been passed !").0,
            mask_influence_threshold: args.bspline_args().expect("GxBSplineInterpolator requires bspline_args. Check the GxArrayViewInterpolatorArgs argument that has been passed !").1,
            truncation_index: [0; TRUNCATION_INDEX_BUFFER_MAX_SIZE],
            domain_extension: [0; TRUNCATION_L_BUFFER_MAX_SIZE],
        }
    }
    
    /// Get the short name of the interpolator
    ///
    /// # Returns
    /// A string representing the short name of the interpolator
    fn shortname(&self) -> String {
        format!("bspline{}", self.order)
    }
    
    fn initialize(&mut self) -> Result<(), String> {
        self.truncation_index = compute_2d_truncation_index(self.order, self.epsilon);
        self.domain_extension = compute_2d_domain_extension_from_truncation_idx(self.order, &self.truncation_index);
        Ok(())
    }
    
    /// Returns the kernel size in rows for the B-spline interpolation.
    /// 
    /// The mathematic kernel size is determined by the spline order, specifically (order + 1).
    /// The radius is given by int(N/2) for an odd-order spline
    /// For implementation purpose we compute the kernel on 2 * radius + 1 values in order to avoid if branching
    /// 
    /// # Returns
    /// - usize: Number of rows in the interpolation kernel
    fn kernel_row_size(&self) -> usize {
        2 * (self.npoles + 1) + 1
    }

    /// Returns the kernel size in columns for the B-spline interpolation.
    /// 
    /// The kernel size is determined by the spline order, specifically (order + 1).
    /// The radius is given by int(N/2) + 1 for an odd-order spline
    /// For implementation purpose we compute the kernel on 2 * radius + 1 values in order to avoid if branching
    /// 
    /// # Returns
    /// - usize: Number of columns in the interpolation kernel
    fn kernel_col_size(&self) -> usize {
        2 * (self.npoles + 1) + 1
    }
    
    /// Computes and returns the total margins required on each side for the entire interpolation process.
    ///
    /// The margins are provided as a 4-element `usize` array representing the top, bottom, left, and right sides respectively.
    ///
    /// # Returns
    /// A 4-element `usize` array where each element corresponds to:
    /// - Index 0: Top margin
    /// - Index 1: Bottom margin
    /// - Index 2: Left margin
    /// - Index 3: Right margin
    #[inline(always)]
    fn total_margins(&self) -> Result<[usize; 4], GxError> {
        let margin = self.domain_extension[0]; // TODO : We may need to add 1 here in order to acount for the kernel_size 
        if margin == 0 {
            return Err(GxError::ErrMessage("GxBSplineInterpolator has not been initialized".to_string()));
        }
        Ok([margin, margin, margin, margin])
    }
    
    /// Allocates a buffer for storing kernel weights during interpolation operations.
    /// 
    /// The buffer size is determined by the kernel dimensions (order+1 × order+1) to accommodate
    /// the full support of the B-spline basis functions.
    /// 
    /// # Returns
    /// - Boxed slice of f64 values with size (order+1)²
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; self.kernel_row_size() + self.kernel_col_size()];
        buffer.into_boxed_slice()
    }
    
    /// Performs 2D interpolation at the specified target position using the B-spline basis functions.
    /// 
    /// This method evaluates the B-spline basis functions at the target coordinates and computes
    /// the interpolated value using the input data and pre-filtered coefficients.
    /// 
    /// # Parameters
    /// - `weights_buffer`: Mutable buffer to store computed kernel weights
    /// - `target_row_pos`: Target row coordinate (floating point)
    /// - `target_col_pos`: Target column coordinate (floating point)
    /// - `out_idx`: Flat output index to write the result
    /// - `array_in`: Input data array (flattened 3D view)
    /// - `array_out`: Output data array (mutable flattened 3D view)
    /// - `nodata_out`: Value to use for nodata output
    /// - `context`: Interpolation context controlling input/output masks and bounds checking
    /// 
    /// # Returns
    /// - `Ok(())` if interpolation succeeds
    /// - `Err(String)` with error message if interpolation fails
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
        // Get the nearest corresponding index corresponding to the target position 
        let kernel_center_row: i64 = (target_row_pos + 0.5).floor() as i64;
        let kernel_center_col: i64 = (target_col_pos + 0.5).floor() as i64;
 
        //let array_in_var_size = array_in.nrow * array_in.ncol;
        let array_out_var_size = array_out.var_size;
        
        // Get the radius
        let nt: i64 = (N / 2 + 1).try_into().unwrap();
        
        // Consider mask valid (if any)
        context.output_mask().set_value(out_idx, 1);
        
        // After compilation that test will have no cost in monomorphic created
        // method
        if IC::BoundsCheck::do_check() {
            // Check that all required data for interpolation is within the input
            // array shape - assuming here the radius is 2.
            // Here we do not need to check borders inside the inner loops.
            // That should be the most common path.
            if (kernel_center_row >=nt)
                    && (kernel_center_row < array_in.nrow_i64-nt)
                    && (kernel_center_col >= nt)
                    && (kernel_center_col < array_in.ncol_i64-nt) {
                let rel_row: f64 = target_row_pos - kernel_center_row as f64;
                let rel_col: f64 = target_col_pos - kernel_center_col as f64;
                
                // Create slices to give to weight computation methods
                let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size());
                
                // from here - pass slice for weight computation
                // slices are used here in order to limit buffer allocation
                self.bspline_kernel_weights(rel_row, kernel_weights_row_slice);
                self.bspline_kernel_weights(rel_col, kernel_weights_col_slice);
                
                
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
                let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size());
                
                // from here - pass slice for weight computation
                // slices are used here in order to limit buffer allocation
                self.bspline_kernel_weights(rel_row, kernel_weights_row_slice);
                self.bspline_kernel_weights(rel_col, kernel_weights_col_slice);
                
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
            let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size());
            
            // from here - pass slice for weight computation
            // slices are used here in order to limit buffer allocation
            self.bspline_kernel_weights(rel_row, kernel_weights_row_slice);
            self.bspline_kernel_weights(rel_col, kernel_weights_col_slice);
            
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

/// Alias type for cubic B-spline interpolator (3rd order, 1 pole)
pub type GxBSpline3Interpolator = GxBSplineInterpolator<3>;

/// Alias type for quintic B-spline interpolator (5th order, 2 poles)
pub type GxBSpline5Interpolator = GxBSplineInterpolator<5>;

/// Alias type for septic B-spline interpolator (7th order, 3 poles)
pub type GxBSpline7Interpolator = GxBSplineInterpolator<7>;

/// Alias type for nonic B-spline interpolator (9th order, 4 poles)
pub type GxBSpline9Interpolator = GxBSplineInterpolator<9>;

/// Alias type for eleventh-order B-spline interpolator (11th order, 5 poles)
pub type GxBSpline11Interpolator = GxBSplineInterpolator<11>;


