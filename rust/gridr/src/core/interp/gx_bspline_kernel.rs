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
//! This module provides efficient B-spline interpolation implementations using
//! Rust's const generics to specialize interpolators at compile time for 
//! different spline orders.
//!
//! # Core Components
//!
//! - **B-spline basis functions**: Optimized implementations for orders 3, 5,
//!   7, 9, and 11
//! - **Generic interpolator**: `GxBSplineInterpolator<N>` with compile-time
//!   order parameter
//! - **Type aliases**: Convenience types (`GxBSpline3Interpolator`, 
//!   `GxBSpline5Interpolator`, etc.)
//!
//! # Trait Implementations
//!
//! ## `GxArrayViewInterpolator`
//! The main interpolation interface implemented by all B-spline interpolators.
//!
//! ## `GxBSplineInterpolatorTrait<N>`
//! B-spline-specific interface providing:
//! - `bspline()`: Evaluates the B-spline basis function at a given coordinate
//! - `bspline_kernel_weights()`: Computes interpolation weights for the kernel
//! - `array1_bspline_prefiltering_ext()`: Applies recursive prefiltering to
//!   input data
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
//!
//! # Usage Example
//!
//! ```ignore
//! use gridr::core::gx_bspline_interp::{
//!     GxBSpline5Interpolator,
//!     GxArrayViewInterpolator
//! };
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
use super::gx_array_view_interp::{
    GxArrayViewInterpolator,
    GxArrayViewInterpolatorArgs,
    GxArrayViewInterpolatorCore,
    GxArrayViewInterpolationContextTrait,
};
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

/// Trait defining the interface for generic B-spline interpolators with
/// configurable order.
/// 
/// This trait provides the core functionality for B-spline interpolation
/// operations, including:
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
    /// Returns the number of poles
    ///
    /// # Returns
    /// - usize: The number of poles for the B-Spline.
    fn get_npoles(&self) -> usize;
    
    /// Evaluates the B-spline basis function of order N at the specified
    /// coordinate.
    /// 
    /// This method computes the value of the B-spline basis function at
    /// position `x`.
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
    /// This operation prepares input data by applying the B-spline 
    /// pre-filtering with the specified poles and truncation indices.
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

/// Generic implementation of B-spline interpolators with compile-time order and
/// pole configuration.
///
/// This struct serves as the concrete implementation for all supported B-spline
/// orders (3, 5, 7, 9, 11).
/// It stores the order and number of poles as runtime parameters while
/// utilizing compile-time
/// constants for the specific mathematical function implementations. The struct
/// also contains prefiltering parameters and associated buffers for efficient 
/// computation.
///
/// # Fields
/// - `order`: The order of the B-spline (N), determining the smoothness and 
///   support size
/// - `npoles`: The number of poles (N/2), used in the recursive filtering
///   implementation
/// - `epsilon`: Precision parameter for the truncation index calculation.
///   Defines the acceptable error when approximating the infinite sums. Smaller
///   values require larger margins for prefiltering. The total required margin
///   combines both the prefiltering margin (truncation index) and the 
///   interpolation kernel radius.
/// - `mask_influence_threshold`: Residual influence threshold $s$ used to
///   compute the radius of the propagation of masked data. Required when 
///   `ima_mask_in` is provided.
///   Determines the acceptable relative contamination from invalid pixels.
/// - `truncation_index`: Buffer storing truncation index
/// - `domain_extension`: Buffer storing domain extension
#[derive(Clone, Debug)]
pub struct GxBSplineInterpolator<const N: usize> {
    /// The order of the B-spline (N), determining the smoothness and support
    /// size
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
    
    #[inline(always)]
    fn get_npoles(&self) -> usize {
        self.npoles
    }
    
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
    fn bspline_kernel_weights(&self, x: f64, weights: &mut [f64]) 
    {
        let nt = N / 2 + 1;
        for k in 0..=2*nt {
            weights[k] = 0.0;
            let xx = (x + k as f64 - nt as f64).abs();
            weights[k] = self.bspline(xx)
        }
    }
    
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


impl<const N: usize> GxArrayViewInterpolatorCore<5, 5, 25> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<7, 7, 49> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<9, 9, 81> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<11, 11, 121> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<13, 13, 169> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}

impl<const N: usize> GxBSplineInterpolator<N>
{   
    #[inline]
    fn array1_interp2_separable_core<T, V, IC>(
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
       
        match N {
            3 => GxArrayViewInterpolatorCore::<5, 5, 25>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            5 => GxArrayViewInterpolatorCore::<7, 7, 49>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            7 => GxArrayViewInterpolatorCore::<9, 9, 81>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            9 => GxArrayViewInterpolatorCore::<11, 11, 121>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            11 => GxArrayViewInterpolatorCore::<13, 13, 169>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            _ => panic!("Unsupported spline order"),

        }
    }
}

/// A structure holding the parameters required to create a 
/// `GxBSplineInterpolator<N>`
///
/// This structure implements the `GxBSplineInterpolatorArgs` trait and contains
/// the essential configuration parameters for B-spline interpolation.
pub struct GxBSplineInterpolatorArgs {
    /// Precision parameter for truncation index calculation.
    pub epsilon: f64,
    /// Acceptable relative contamination from invalid pixels.
    pub mask_influence_threshold: f64,
}

/// Concrete implementation of `GxArrayViewInterpolatorArgs` trait for B-spline
/// interpolators
///
/// This implementation provides the necessary interface for interpolators
/// derived from `GxBSplineInterpolator<N>`. It implements the `bspline_args` 
/// function to expose the configured parameters for B-spline interpolation.
impl GxArrayViewInterpolatorArgs for GxBSplineInterpolatorArgs {
    fn bspline_args(&self) -> Option<(f64, f64)> {
        Some((self.epsilon, self.mask_influence_threshold))
    }
}

impl<const N: usize> GxArrayViewInterpolator for GxBSplineInterpolator<N> {
    /// Creates a new instance of the B-spline interpolator with the specified
    /// order and poles.
    /// 
    /// # Returns
    /// - New instance with order N and npoles N/2
    fn new(args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        GxBSplineInterpolator {
            order: N,
            npoles: N/2,
            epsilon: args.bspline_args().expect(
                "GxBSplineInterpolator requires bspline_args. Check the \
                GxArrayViewInterpolatorArgs argument that has been passed !").0,
            mask_influence_threshold: args.bspline_args().expect(
                "GxBSplineInterpolator requires bspline_args. Check the \
                GxArrayViewInterpolatorArgs argument that has been passed !").1,
            truncation_index: [0; TRUNCATION_INDEX_BUFFER_MAX_SIZE],
            domain_extension: [0; TRUNCATION_L_BUFFER_MAX_SIZE],
        }
    }

    fn shortname(&self) -> String {
        format!("bspline{}", self.order)
    }
    
    fn initialize(&mut self) -> Result<(), String> {
        self.truncation_index =
            compute_2d_truncation_index(self.order, self.epsilon);
        self.domain_extension =
            compute_2d_domain_extension_from_truncation_idx(
                self.order, &self.truncation_index
            );
        Ok(())
    }
    
    fn kernel_row_size(&self) -> usize {
        2 * (self.npoles + 1) + 1
    }

    fn kernel_col_size(&self) -> usize {
        2 * (self.npoles + 1) + 1
    }
    
    #[inline(always)]
    fn total_margins(&self) -> Result<[usize; 4], GxError> {
        let margin = self.domain_extension[0]; // TODO : We may need to add 1 here in order to acount for the kernel_size 
        if margin == 0 {
            return Err(GxError::ErrMessage(
                "GxBSplineInterpolator has not been initialized".to_string())
            );
        }
        Ok([margin, margin, margin, margin])
    }
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; self.kernel_row_size() + self.kernel_col_size()];
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


