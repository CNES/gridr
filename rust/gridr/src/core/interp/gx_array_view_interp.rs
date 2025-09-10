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
//! This module defines an interpolation architecture over 1D arrays (`GxArrayView`)
//! based on a flexible system of generic strategies to handle:
//!
//! 1. Input mask strategy (`InputMaskStrategy`): determines whether a given
//!    input index should be considered valid.
//!
//! 2. Output mask strategy (`OutputMaskStrategy`): marks or flags points
//!    written to the output.
//!
//! 3. Bounds check strategy (`BoundsCheckStrategy`): controls whether index
//!    boundary checks are performed.
//!
//! These strategies are combined within a generic context `GxArrayViewInterpolationContext`
//! allowing generic code without runtime dynamic dispatch, ensuring optimal
//! performance and easy adaptability to future SIMD or other low-level optimizations.
//!
//! Using traits and generics avoids runtime indirection overhead while
//! maintaining high flexibility in behavior.
//!
//! # Overall architecture
//! ```text
//! ┌─────────────────────────────────┐
//! │ GxArrayViewInterpolationContext │
//! │ ├─ InputMaskStrategy (trait)    │
//! │ ├─ OutputMaskStrategy (trait)   │
//! │ └─ BoundsCheckStrategy (trait)  │
//! └─────────────────────────────────┘
//! ```
//!
//! Each strategy can be replaced by a custom implementation (e.g., binary mask,
//! no mask, etc.). The context is passed to interpolation functions which adapt
//! their behavior accordingly.
//!
//! This design facilitates maintainability, modularity, and performance,
//! while preparing for future extensions such as SIMD or parallelization.
//!
use std::marker::PhantomData;
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};


/// Generic strategy trait to validate whether a given input point is valid.
///
/// This abstraction tests if a specific index in the input array should
/// be considered during interpolation.
///
/// - `is_valid(idx)` returns 1 if valid, 0 otherwise.
/// - `is_enabled()` indicates whether the mask strategy is active
///   (useful to avoid unnecessary checks).
pub trait GxArrayViewInterpolatorInputMaskStrategy {
    /// Returns whether the point at index `idx` is valid (1) or invalid (0).
    fn is_valid(&self, idx: usize) -> u8;
    
    /*
    /// Returns whether the all the points in a array are valid (1) or invalid (0).
    fn is_valid_window(&self, start_idx: usize, height: usize, width: usize, cache: &mut [u8]) -> u8;
    */
    
    /// Returns whether the all the points in a array are valid (1) or invalid (0).
    fn is_valid_weighted_window(&self, start_idx: usize, height: usize, width: usize, weights_row: &[f64], weights_col: &[f64], cache: &mut [u8]) -> u8;
    
    /// Returns true if the input mask strategy is enabled.
    fn is_enabled(&self) -> bool;
}

impl<T: GxArrayViewInterpolatorInputMaskStrategy> GxArrayViewInterpolatorInputMaskStrategy for &T {
    #[inline(always)]
    fn is_valid(&self, idx: usize) -> u8 {
        (*self).is_valid(idx)
    }
    
    /*
    #[inline(always)]
    fn is_valid_window(&self, start_idx: usize, height: usize, width: usize, cache: &mut [u8]) -> u8 {
        (*self).is_valid_window(start_idx, height, width, cache)
    }
    */
    
    #[inline(always)]
    fn is_valid_weighted_window(&self, start_idx: usize, height: usize, width: usize, weights_row: &[f64], weights_col: &[f64], cache: &mut [u8]) -> u8 {
        (*self).is_valid_weighted_window(start_idx, height, width, weights_row, weights_col, cache)
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        (*self).is_enabled()
    }
}

/// Default implementation: no input mask, all points considered valid.
#[derive(Default)]
pub struct NoInputMask;

impl GxArrayViewInterpolatorInputMaskStrategy for NoInputMask {
    #[inline(always)]
    fn is_valid(&self, _idx: usize) -> u8 {
        1
    }
    
    /*
    #[inline(always)]
    fn is_valid_window(&self, _start_idx: usize, _height: usize, _width: usize, _cache: &mut [u8]) -> u8 {
        1
    }
    */
    
    #[inline(always)]
    fn is_valid_weighted_window(&self, _start_idx: usize, _height: usize, _width: usize, _weights_row: &[f64], _weights_col: &[f64], _cache: &mut [u8]) -> u8 {
        1
    }
    
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        false
    }
}

/// Binary input mask based on a `u8` array, where 1 = valid, 0 = invalid.
pub struct BinaryInputMask<'a> {
    /// An immutable reference to a `GxArrayView<u8>` corresponding to the binary mask.
    pub mask: &'a GxArrayView<'a, u8>,
}

impl<'a> GxArrayViewInterpolatorInputMaskStrategy for BinaryInputMask<'a> {
    #[inline(always)]
    fn is_valid(&self, idx: usize) -> u8 {
        self.mask.data[idx]
    }
    
    /*
    #[inline(always)]
    fn is_valid_window(&self, _start_idx: usize, _height: usize, _width: usize, _cache: &mut [u8]) -> u8 {
        1
    }
    */
    
    #[inline(always)]
    fn is_valid_weighted_window(&self, start_idx: usize, height: usize, width: usize, weights_row: &[f64], weights_col: &[f64], cache: &mut [u8]) -> u8 {
        let mut arr_iflat = start_idx;
        let ncol = self.mask.ncol;
        let height_m1 = height - 1;
        let width_m1 = width - 1;
        let mut i: usize = 0;
        
        for irow in 0..height {
            if weights_row[height_m1 - irow as usize] == 0.0 {
                // Go to next row
                i += height;
                arr_iflat += ncol;
                continue;
            }
            for icol in 0..width {
                if weights_col[width_m1 - icol as usize] == 0.0 {
                    // Go to next col
                    arr_iflat += 1;
                    i += 1;
                    continue;
                }
                cache[i] = self.mask.data[arr_iflat];
                
                if cache[i] == 0 {
                    return 0;
                }
                arr_iflat += 1;
                i += 1;
            }
            arr_iflat += ncol - width;
        }
        1
    }
    
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Enum wrapping possible input mask strategies.
pub enum InputMaskStrategy<'a> {
    /// Enum item for BinaryInputMask strategy
    Binary(BinaryInputMask<'a>),
    /// Enum item for NoInputMask strategy
    None(NoInputMask),
}


/// Generic strategy trait for managing an output mask.
///
/// Allows marking points written to the output.
/// Methods:
/// - `is_enabled()` to query if the strategy is active.
/// - `set_value(idx, value)` to set a mask value at the given index.
pub trait GxArrayViewInterpolatorOutputMaskStrategy {   
    /// Returns true if the output mask strategy is enabled.
    fn is_enabled(&self) -> bool;
    
    /// Sets a mask value at the given index.
    fn set_value(&mut self, idx: usize, value: u8);
}

impl<T: GxArrayViewInterpolatorOutputMaskStrategy> GxArrayViewInterpolatorOutputMaskStrategy for &mut T {
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        (**self).is_enabled()
    }

    #[inline(always)]
    fn set_value(&mut self, idx: usize, value: u8) {
        (**self).set_value(idx, value)
    }
}

/// Default implementation: no output mask (no marking).
#[derive(Default)]
pub struct NoOutputMask;

impl GxArrayViewInterpolatorOutputMaskStrategy for NoOutputMask {
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        false
    }
    
    #[inline(always)]
    fn set_value(&mut self, _idx: usize, _value: u8) {}
}

/// Binary output mask stored in a mutable `u8` array.
pub struct BinaryOutputMask<'a> {
    /// A mutable reference to a `GxArrayViewMut<u8>` corresponding to the output binary mask.
    pub mask: &'a mut GxArrayViewMut<'a, u8>,
}

impl<'a> GxArrayViewInterpolatorOutputMaskStrategy for BinaryOutputMask<'a> {
    
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        true
    }
    
    #[inline(always)]
    fn set_value(&mut self, idx: usize, value: u8) {
        self.mask.data[idx] = value;
    }
}

/// Enum wrapping possible output mask strategies.
pub enum OutputMaskStrategy<'a> {
    /// Enum item for BinaryOutputMask strategy
    Binary(BinaryOutputMask<'a>),
    /// Enum item for NoOutputMask strategy
    None(NoOutputMask),
}


/// Strategy trait for controlling bounds checking.
///
/// This abstraction enables or disables index boundary checks,
/// which can impact performance.
///
/// The `do_check()` method is static, enabling the compiler to optimize away
/// the checks if disabled.
pub trait GxArrayViewInterpolatorBoundsCheckStrategy {
    /// Method that return either true or false depending on the target strategy implementation
    fn do_check() -> bool;
}

/// Bounds checking disabled.
#[derive(Default)]
pub struct NoBoundsCheck;
impl GxArrayViewInterpolatorBoundsCheckStrategy for NoBoundsCheck {
    #[inline(always)]
    fn do_check() -> bool { false }
}

/// Bounds checking enabled.
#[derive(Default)]
pub struct BoundsCheck;
impl GxArrayViewInterpolatorBoundsCheckStrategy for BoundsCheck {
    #[inline(always)]
    fn do_check() -> bool { true }
}


/// Interpolation context aggregating the three strategies.
///
/// The context is generic over the strategies, allowing compile-time
/// monomorphization and zero runtime overhead.
///
/// PhantomData markers are used to handle typing and lifetimes,
/// especially for the bounds check strategy which carries no state.
///
/// # Type parameters
/// - `IM`: input mask strategy.
/// - `OM`: output mask strategy.
/// - `BC`: bounds check strategy.
pub struct GxArrayViewInterpolationContext<'a, IM, OM, BC>
where
    IM: GxArrayViewInterpolatorInputMaskStrategy + 'a,
    OM: GxArrayViewInterpolatorOutputMaskStrategy + 'a,
    BC: GxArrayViewInterpolatorBoundsCheckStrategy + 'a,
{
    /// Input mask strategy.
    pub input_mask: IM,
    
    /// Output mask strategy.
    pub output_mask: OM,
    
    /// PhantomData for bounds check strategy.
    pub _phantom_bounds: PhantomData<BC>,
    
    /// PhantomData for lifetime management.
    pub _phantom_lifetime: PhantomData<&'a ()>,
}

/// Context trait to easily access the contained strategies.
///
/// Allows writing generic code that depends on the context.
pub trait GxArrayViewInterpolationContextTrait {
    /// Alias for the input mask strategy type.
    /// 
    /// This type defines how input masking is handled during interpolation operations.
    type InputMask: GxArrayViewInterpolatorInputMaskStrategy;
    
    /// Alias for the output mask strategy type.
    /// 
    /// This type defines how output masking is handled during interpolation operations.
    type OutputMask: GxArrayViewInterpolatorOutputMaskStrategy;
    
    /// Alias for the bounds check strategy type.
    /// 
    /// This type defines how bounds checking is performed during interpolation operations.
    type BoundsCheck: GxArrayViewInterpolatorBoundsCheckStrategy;

    /// Returns a reference to the input mask strategy.
    fn input_mask(&self) -> &Self::InputMask;
    
    /// Returns a mutable reference to the output mask strategy.
    fn output_mask(&mut self) -> &mut Self::OutputMask;
}

impl<'a, IM, OM, BC> GxArrayViewInterpolationContextTrait for GxArrayViewInterpolationContext<'a, IM, OM, BC>
where
    IM: GxArrayViewInterpolatorInputMaskStrategy,
    OM: GxArrayViewInterpolatorOutputMaskStrategy,
    BC: GxArrayViewInterpolatorBoundsCheckStrategy,
{
    type InputMask = IM;
    type OutputMask = OM;
    type BoundsCheck = BC;

    #[inline(always)]
    fn input_mask(&self) -> &Self::InputMask {
        &self.input_mask
    }
    
    #[inline(always)]
    fn output_mask(&mut self) -> &mut Self::OutputMask {
        &mut self.output_mask
    }
}

impl<'a, IM, OM, BC> GxArrayViewInterpolationContext<'a, IM, OM, BC>
where
    IM: GxArrayViewInterpolatorInputMaskStrategy + 'a,
    OM: GxArrayViewInterpolatorOutputMaskStrategy + 'a,
    BC: GxArrayViewInterpolatorBoundsCheckStrategy + 'a,
{
    /// Creates a new interpolation context with the given strategies.
    pub fn new(input_mask: IM, output_mask: OM, _bounds_check: BC) -> Self {
        Self {
            input_mask,
            output_mask,
            _phantom_bounds: PhantomData,
            _phantom_lifetime: PhantomData,
        }
    }
}


/// Default context type for array view interpolation.
/// 
/// This type provides a convenient default implementation using no masking strategies
/// and standard bounds checking.
pub type DefaultCtx<'a> = GxArrayViewInterpolationContext<'a, NoInputMask, NoOutputMask, BoundsCheck>;
impl<'a> DefaultCtx<'a> {
    /// Creates a new default context with no input masking, no output masking,
    /// and standard bounds checking.
    /// 
    /// # Returns
    /// A new instance of `DefaultCtx` with default strategies.
    pub fn default() -> Self {
        Self {
            input_mask: NoInputMask::default(),
            output_mask: NoOutputMask::default(),
            _phantom_bounds: PhantomData,
            _phantom_lifetime: PhantomData,
        }
    }
}

/// Trait defining the core 2D interpolation methods for resampling.
///
/// This trait provides a low-level abstraction for computing interpolated values at a given
/// integer center coordinate over a 2D input array. Implementations can support multiple
/// interpolation strategies, with or without input/output validity masks, and may optionally 
/// perform bounds checking depending on the variant.
///
/// The interpolator operates on flattened views of 3D arrays, where the first dimension
/// corresponds to multiple variables (or bands), and the last two dimensions correspond
/// to rows and columns.
///
/// # Coordinate Convention
/// The interpolation is performed at a discrete center coordinate `(row_c, col_c)`,
/// which defines the origin of a local neighborhood used for interpolation.
/// Implementations determine the neighborhood pattern and weight usage.
///
/// # Data Layout
/// All arrays follow a flattened 3D memory layout:
/// - `GxArrayView`: read-only input data with dimensions `[nvar, nrow, ncol]`
/// - `GxArrayViewMut`: mutable output data with the same layout
/// - `GxArrayView<u8>`: optional validity mask for input values embedded in the `context` parameter.
/// - `GxArrayViewMut<u8>`: optional mutable output mask embedded in the `context` parameter.
///
/// Flat indices must be computed manually using row and column sizes and variable offsets.
///
/// # Context Parameter
/// This trait integrates a generic interpolation context parameter (`context`) that
/// encapsulates the behavior for:
/// - Input validity masks (determining whether input points are considered valid)
/// - Output masks (enabling or disabling the production of an output mask)
/// - Bounds checking strategies (enabling or disabling bounds verification)
///
/// This design promotes flexibility, allowing different interpolation behaviors
/// without runtime overhead, by leveraging generic context implementations.
///
/// # Type Parameters
/// - `T`: Input scalar type, must support conversion to `f64` and multiplication with `f64`.
/// - `V`: Output scalar type, must be constructible from `f64`.
/// - `IC`: Interpolation context type implementing `GxArrayViewInterpolationContextTrait`,
///   which controls masking and bounds checking strategies.
///
/// # Related Types
/// - `GxArrayView`: View of the input 3D array
/// - `GxArrayViewMut`: Mutable view of the output 3D array
/// - `GxArrayViewInterpolationContextTrait`: Trait for generic interpolation context controlling masks and bounds
pub trait GxArrayViewInterpolator
{
    /// Constructs a new instance of the interpolator.
    fn new() -> Self;
    
    /// Allocates a buffer for kernel weights.
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]>;
    
    /// Performs 2D interpolation at the specified target position.
    ///
    /// # Parameters
    /// - `weights_buffer`: Mutable buffer to store computed kernel weights.
    /// - `target_row_pos`: Target row coordinate (floating point).
    /// - `target_col_pos`: Target column coordinate (floating point).
    /// - `out_idx`: Flat output index to write the result.
    /// - `array_in`: Input data array (flattened 3D view).
    /// - `array_out`: Output data array (mutable flattened 3D view).
    /// - `nodata_out`: Value to use for nodata output.
    /// - `context`: Interpolation context controlling input/output masks and bounds checking.
    ///
    /// # Returns
    /// - `Ok(())` if interpolation succeeded.
    /// - `Err(String)` with error message if interpolation failed (e.g., out of bounds).
    ///
    /// # Type constraints
    /// - `T`: Input type supporting copying, equality, multiplication with `f64`, and conversion to `f64`.
    /// - `V`: Output type supporting copying, equality, and construction from `f64`.
    /// - `IC`: Interpolation context implementing `GxArrayViewInterpolationContextTrait`.
    fn array1_interp2<T, V, IC> (
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
        IC: GxArrayViewInterpolationContextTrait;
    
    /// Returns the kernel size in rows.
    fn kernel_row_size(&self) -> usize;
    
    /// Returns the kernel size in rows.
    fn kernel_col_size(&self) -> usize;
}

