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
//! Interpolation architecture over 1D array views (`GxArrayView`).
//!
//! # Public API vs. internal computation trait
//!
//! This module draws a deliberate line between two concerns:
//!
//! * **Public API** — [`GxArrayViewInterpolator`] is the single stable
//!   interface that all concrete interpolators expose to callers.  It contains
//!   only the methods that external code needs: construction, metadata
//!   (`shortname`, `kernel_row_size`, `total_margins`, …), buffer allocation,
//!   and the top-level dispatch method `array1_interp2`.
//!
//! * **Internal computation** — [`GxArrayViewInterpolatorCore`] is an
//!   *implementation-only* trait, not part of the public API contract.  It
//!   provides the four separable kernel-convolution variants
//!   (`interpolate_nomask_unchecked`, `interpolate_nomask_partial`,
//!   `interpolate_masked_unchecked`, `interpolate_masked_partial`) and the
//!   unified dispatcher `array1_interp2_separable_core` as **default method
//!   implementations**.  A concrete interpolator only needs to provide
//!   `compute_weights`; all kernel logic is inherited automatically.
//!
//! This separation ensures that:
//! 1. Callers depend only on [`GxArrayViewInterpolator`] and are insulated
//!    from implementation details.
//! 2. New interpolators sharing the same separable-kernel structure (bicubic,
//!    B-spline, …) reuse the four variants without code duplication.
//! 3. Interpolators with non-separable kernels implement
//!    [`GxArrayViewInterpolator`] directly, bypassing
//!    [`GxArrayViewInterpolatorCore`] entirely.
//!
//! # Generic strategies
//!
//! Behaviour at array borders and on masked data is governed by three
//! orthogonal strategy types that are composed at compile time inside a
//! context object:
//!
//! 1. **Input mask** ([`GxArrayViewInterpolatorInputMaskStrategy`]):
//!    determines whether a given input index should be considered valid.
//!
//! 2. **Output mask** ([`GxArrayViewInterpolatorOutputMaskStrategy`]):
//!    marks or flags points written to the output.
//!
//! 3. **Bounds check** ([`GxArrayViewInterpolatorBoundsCheckStrategy`]):
//!    controls whether index boundary checks are performed.
//!
//! The strategies are assembled in a [`GxArrayViewInterpolationContext`] and
//! passed through all internal methods, enabling monomorphic code with zero
//! runtime dynamic dispatch.
//!
//! Each strategy can be replaced by a custom implementation (e.g., binary mask,
//! no mask, etc.). The context is passed to core interpolation functions 
//! implementation which adapt their behavior accordingly.
//!
//! # Module architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │              GxArrayViewInterpolationContext                 │
//! │  ├─ InputMaskStrategy  (NoInputMask | BinaryInputMask        │
//! |  │                       | BinaryInputMaskWithSafeWindow     │
//! │  ├─ OutputMaskStrategy (NoOutputMask | BinaryOutputMask)     │
//! │  └─ BoundsCheckStrategy (NoBoundsCheck | BoundsCheck)        │
//! └──────────────────────────────────────────────────────────────┘
//!
//!  ┌──────────────────────────────────────────────────────────────┐
//!  │  GxArrayViewInterpolator          ← PUBLIC API               │
//!  │  array1_interp2(), shortname(), total_margins(), …           │
//!  └──────────────────────────────────────────────────────────────┘
//!              ▲                              ▲
//!              │ impl                         │ impl (direct, non-separable)
//!              │                              │
//!  ┌─────────────────────────────┐   ┌─────────────────────────────────┐
//!  │ GxArrayViewInterpolatorCore │   │ (non-separable interpolators    │
//!  │ <KROWS, KCOLS>              │   │  implement GxArrayViewInterp-   │
//!  │                             │   │  olator directly)               │
//!  │ INTERNAL TRAIT              │   └─────────────────────────────────┘
//!  │ compute_weights()           │ ← only method to implement
//!  │ interpolate_nomask_*        │ ← default implementations
//!  │ interpolate_masked_*        │
//!  │ array1_interp2_             │
//!  │   separable_core()          │
//!  └─────────────────────────────┘
//!              ▲
//!              │ impl GxArrayViewInterpolatorCore<5,5>
//!              │
//!  ┌────────────────────────────────┐
//!  │ GxOptimizedBicubicInterpolator │
//!  │ GxBSplineInterpolator<N>       │
//!  └────────────────────────────────┘
//! ```
//!
//! # Design notes
//!
//! * **Zero-cost abstractions** — `do_check()` and `is_enabled()` return
//!   compile-time constants that LLVM folds away; the resulting machine code
//!   contains only the branch taken for the instantiated strategy type.
use std::marker::PhantomData;
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use crate::core::gx_errors::GxError;


// =============================================================================
// Input mask strategies
// =============================================================================

/// Strategy trait that determines whether a given input point is valid.
///
/// This abstraction is queried for each input index during interpolation to
/// decide whether the corresponding sample should contribute to the result.
///
/// # Contract
/// - [`is_valid`](Self::is_valid) returns `1` if the point is valid, `0`
///   otherwise.
/// - [`is_enabled`](Self::is_enabled) returns `false` when the strategy is a
///   no-op (all points valid), allowing callers to skip the validity check
///   entirely.
pub trait GxArrayViewInterpolatorInputMaskStrategy {
    /// Returns whether the point at index `idx` is valid (1) or invalid (0).
    fn is_valid(&self, idx: usize) -> u8;

    /// Returns whether the point at index `idx` is valid (1) or invalid (0).
    unsafe fn is_valid_unsafe(&self, idx: usize) -> u8;

    /// Returns `1` if all points in the window are valid, `0` oterhwise.
    ///
    /// # Parameters
    /// - `start_idx`: flat index of the top-left corner of the window
    /// - `start_row_idx`: row index of the top-left corner of the window
    /// - `start_col_idx`: column index of the top-left corner of the window
    ///
    /// # Const parameters
    /// - `H`: window's height.
    /// - `W`: widows's width.
    ///
    /// # Design Rationale
    /// This function accepts both a 1D flat index and its corresponding 2D 
    /// row/column indices for the window's top-left corner. This dual-parameter
    /// approach allows the implementation to chosse whichever coordinate 
    /// representation is more optimal, avoiding unnecessary index calculations
    /// that would be required to convert between 1D and 2D representations.
    fn is_valid_window<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> u8;

    /// Returns `1` if all points in the window are valid, `0` oterhwise.
    ///
    /// # Parameters
    /// - `start_idx`: flat index of the top-left corner of the window in the
    /// - `start_row_idx`: row index of the top-left corner of the window
    /// - `start_col_idx`: column index of the top-left corner of the window
    ///
    /// # Const parameters
    /// - `H`: window's height.
    /// - `W`: widows's width.
    ///
    /// # Design Rationale
    /// This function accepts both a 1D flat index and its corresponding 2D 
    /// row/column indices for the window's top-left corner. This dual-parameter
    /// approach allows the implementation to chosse whichever coordinate 
    /// representation is more optimal, avoiding unnecessary index calculations
    /// that would be required to convert between 1D and 2D representations.
    unsafe fn is_valid_window_unsafe<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> u8;

    /// Returns the number of valid points within the window
    ///
    /// # Parameters
    /// - `start_idx`: flat index of the top-left corner of the window in the
    /// - `start_row_idx`: row index of the top-left corner of the window
    /// - `start_col_idx`: column index of the top-left corner of the window
    ///
    /// # Const parameters
    /// - `H`: window's height.
    /// - `W`: widows's width.
    ///
    /// # Design Rationale
    /// This function accepts both a 1D flat index and its corresponding 2D 
    /// row/column indices for the window's top-left corner. This dual-parameter
    /// approach allows the implementation to chosse whichever coordinate 
    /// representation is more optimal, avoiding unnecessary index calculations
    /// that would be required to convert between 1D and 2D representations.
    fn count_valid_window<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> usize;
    
    /// Returns the number of valid points within the window
    ///
    /// # Parameters
    /// - `start_idx`: flat index of the top-left corner of the window in the
    /// - `start_row_idx`: row index of the top-left corner of the window
    /// - `start_col_idx`: column index of the top-left corner of the window
    ///
    /// # Const parameters
    /// - `H`: window's height.
    /// - `W`: widows's width.
    ///
    /// # Design Rationale
    /// This function accepts both a 1D flat index and its corresponding 2D 
    /// row/column indices for the window's top-left corner. This dual-parameter
    /// approach allows the implementation to chosse whichever coordinate 
    /// representation is more optimal, avoiding unnecessary index calculations
    /// that would be required to convert between 1D and 2D representations.
    unsafe fn count_valid_window_unsafe<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> usize;
    
    /// Returns `1` if all points with non-zero weights in the window are valid,
    /// `0` as soon as one invalid point is found.
    ///
    /// # Parameters
    /// - `start_idx`: flat index of the top-left corner of the window in the
    ///   input array.
    /// - `start_row_idx`: row index of the top-left corner of the window
    /// - `start_col_idx`: column index of the top-left corner of the window
    /// - `weights_row`: row-direction kernel weights (length `height`).
    /// - `weights_col`: column-direction kernel weights (length `width`).
    /// - `cache`: scratch buffer of length `height * width`; may be written
    ///   to by the implementation.
    ///
    /// # Design Rationale
    /// This function accepts both a 1D flat index and its corresponding 2D 
    /// row/column indices for the window's top-left corner. This dual-parameter
    /// approach allows the implementation to chosse whichever coordinate 
    /// representation is more optimal, avoiding unnecessary index calculations
    /// that would be required to convert between 1D and 2D representations.
    fn is_valid_weighted_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8;
    
    /// Returns `1` if all points with non-zero weights in the window are valid,
    /// `0` as soon as one invalid point is found.
    ///
    /// # Parameters
    /// - `start_idx`: flat index of the top-left corner of the window in the
    ///   input array.
    /// - `start_row_idx`: row index of the top-left corner of the window
    /// - `start_col_idx`: column index of the top-left corner of the window
    /// - `weights_row`: row-direction kernel weights (length `height`).
    /// - `weights_col`: column-direction kernel weights (length `width`).
    /// - `cache`: scratch buffer of length `height * width`; may be written
    ///   to by the implementation.
    ///
    /// # Design Rationale
    /// This function accepts both a 1D flat index and its corresponding 2D 
    /// row/column indices for the window's top-left corner. This dual-parameter
    /// approach allows the implementation to chosse whichever coordinate 
    /// representation is more optimal, avoiding unnecessary index calculations
    /// that would be required to convert between 1D and 2D representations.
    unsafe fn is_valid_weighted_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8;
    
    /// Returns `true` if this mask strategy is active.
    ///
    /// When `false`, callers may skip all validity checks, yielding a
    /// measurable performance gain on the hot path.
    fn is_enabled(&self) -> bool;
}

/// Blanket impl: a shared reference to a mask strategy is itself a mask
/// strategy, delegating all calls to the inner value.
impl<T: GxArrayViewInterpolatorInputMaskStrategy>
    GxArrayViewInterpolatorInputMaskStrategy for &T
{
    #[inline(always)]
    fn is_valid(&self, idx: usize) -> u8 {
        (*self).is_valid(idx)
    }
    
    #[inline(always)]
    unsafe fn is_valid_unsafe(&self, idx: usize) -> u8 {
        (*self).is_valid_unsafe(idx)
    }
    
    #[inline(always)]
    fn is_valid_window<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> u8 {
        (*self).is_valid_window::<H, W>(
            start_idx, start_row_idx, start_col_idx
        )
    }
    
    #[inline(always)]
    unsafe fn is_valid_window_unsafe<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> u8 {
        (*self).is_valid_window_unsafe::<H, W>(
            start_idx, start_row_idx, start_col_idx
        )
    }
    
    #[inline(always)]
    fn count_valid_window<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> usize {
        (*self).count_valid_window::<H, W>(
            start_idx,
            start_row_idx,
            start_col_idx,
        )
    }
    
    #[inline(always)]
    unsafe fn count_valid_window_unsafe<const H: usize, const W: usize>(
        &self, 
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> usize {
        (*self).count_valid_window_unsafe::<H, W>(
            start_idx,
            start_row_idx,
            start_col_idx,
        )
    }

    #[inline(always)]
    fn is_valid_weighted_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8 {
        (*self).is_valid_weighted_window::<H, W>(
            start_idx, start_row_idx, start_col_idx,
            weights_row, weights_col,
        )
    }
    
    #[inline(always)]
    unsafe fn is_valid_weighted_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8 {
        (*self).is_valid_weighted_window_unsafe::<H, W>(
            start_idx, start_row_idx, start_col_idx,
            weights_row, weights_col,
        )
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        (*self).is_enabled()
    }
}

/// No-op input mask: all points are unconditionally considered valid.
///
/// Use this when the input data is guaranteed to contain no invalid samples.
/// [`is_enabled`](GxArrayViewInterpolatorInputMaskStrategy::is_enabled) returns
/// `false`, allowing callers to elide the mask-check branch entirely.
#[derive(Default)]
pub struct NoInputMask;

impl GxArrayViewInterpolatorInputMaskStrategy for NoInputMask {
    #[inline(always)]
    fn is_valid(&self, _idx: usize) -> u8 {
        1
    }
    
    #[inline(always)]
    unsafe fn is_valid_unsafe(&self, _idx: usize) -> u8 {
       1
    }

    #[inline(always)]
    fn is_valid_window<const H: usize, const W: usize>(
        &self, 
        _start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> u8 {
        1
    }

    #[inline(always)]
    unsafe fn is_valid_window_unsafe<const H: usize, const W: usize>(
        &self, 
        _start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> u8 {
        1
    }

    #[inline(always)]
    fn count_valid_window<const H: usize, const W: usize>(
        &self, 
        _start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> usize {
        H*W
    }

    #[inline(always)]
    unsafe fn count_valid_window_unsafe<const H: usize, const W: usize>(
        &self, 
        _start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> usize {
        H*W
    }
    
    #[inline(always)]
    fn is_valid_weighted_window<const H: usize, const W: usize>(
        &self,
        _start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
        _weights_row: &[f64],
        _weights_col: &[f64],
    ) -> u8 {
        1
    }
    
    #[inline(always)]
    unsafe fn is_valid_weighted_window_unsafe<const H: usize, const W: usize>(
        &self,
        _start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
        _weights_row: &[f64],
        _weights_col: &[f64],
    ) -> u8 {
        1
    }

    #[inline(always)]
    fn is_enabled(&self) -> bool {
        false
    }
}

/// Binary input mask backed by a `u8` array, where `1` = valid, `0` = invalid.
///
/// Points whose mask value is `0` are excluded from interpolation; the output
/// is set to the caller-supplied `nodata` value instead.
#[derive(Debug)]
pub struct BinaryInputMask<'a> {
    /// Immutable view of the binary mask array.
    pub mask: &'a GxArrayView<'a, u8>,
}

impl<'a> GxArrayViewInterpolatorInputMaskStrategy for BinaryInputMask<'a> {

    #[inline(always)]
    fn is_valid(&self, idx: usize) -> u8 {
        self.mask.data[idx]
    }

    #[inline(always)]
    unsafe fn is_valid_unsafe(&self, idx: usize) -> u8 {
        // SAFETY: caller guarantees idx is within mask bounds.
        *self.mask.data.get_unchecked(idx)
    }

    /// Pre-slices one row at a time to eliminate bounds checks in the
    /// inner loop.  The iterator `fold` compiles to a tight AND chain
    /// that LLVM can vectorize.
    #[inline(always)]
    fn is_valid_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> u8 {
        let ncol = self.mask.ncol;
        let mut row_base = start_idx;
        let mut acc: u8 = 1;

        for _irow in 0..H {
            let row_slice = &self.mask.data[row_base..row_base + W];
            acc = row_slice.iter().fold(acc, |a, &v| a & v);
            row_base += ncol;
        }
        acc
    }

    #[inline(always)]
    unsafe fn is_valid_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> u8 {
        let ncol = self.mask.ncol;
        let data = &self.mask.data;
        let mut row_base = start_idx;
        let mut acc: u8 = 1;

        // SAFETY: the caller guarantees that
        // start_idx + irow * ncol + icol is within bounds for
        // all irow in 0..H and icol in 0..W.
        for _irow in 0..H {
            for icol in 0..W {
                acc &= *data.get_unchecked(row_base + icol);
            }
            row_base += ncol;
        }
        acc
    }

    /// Pre-slices one row at a time.  The `map(|&v| v as usize).sum()`
    /// pattern compiles to a tight add chain without bounds checks.
    #[inline(always)]
    fn count_valid_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> usize {
        let ncol = self.mask.ncol;
        let mut row_base = start_idx;
        let mut acc: usize = 0;

        for _irow in 0..H {
            let row_slice = &self.mask.data[row_base..row_base + W];
            acc += row_slice.iter().map(|&v| v as usize).sum::<usize>();
            row_base += ncol;
        }
        acc
    }

    #[inline(always)]
    unsafe fn count_valid_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
    ) -> usize {
        let ncol = self.mask.ncol;
        let data = &self.mask.data;
        let mut row_base = start_idx;
        let mut acc: usize = 0;

        // SAFETY: same as is_valid_window.
        for _irow in 0..H {
            for icol in 0..W {
                acc += *data.get_unchecked(row_base + icol) as usize;
            }
            row_base += ncol;
        }
        acc
    }

    /// For active rows, the inner loop uses a pre-sliced row and iterator to
    /// eliminate bounds checks.
    #[inline(always)]
    fn is_valid_weighted_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8 {
        let ncol = self.mask.ncol;
        let wr = &weights_row[..H];
        let wc = &weights_col[..W];
        let height_m1 = H - 1;
        let width_m1 = W - 1;
        let mut row_base = start_idx;

        for irow in 0..H {
            if wr[height_m1 - irow] == 0.0 {
                row_base += ncol;
                continue;
            }

            let row_slice = &self.mask.data[row_base..row_base + W];

            for (icol, &mask_val) in row_slice.iter().enumerate() {
                if wc[width_m1 - icol] == 0.0 {
                    continue;
                }
                if mask_val == 0 {
                    return 0;
                }
            }

            row_base += ncol;
        }
        1
    }

    #[inline(always)]
    unsafe fn is_valid_weighted_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        _start_row_idx: usize,
        _start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8 {
        let ncol = self.mask.ncol;
        let data = &self.mask.data;
        let height_m1 = H - 1;
        let width_m1 = W - 1;
        let mut row_base = start_idx;

        // SAFETY: the caller guarantees all indices are within bounds.
        // Weight slices have length >= H and >= W respectively.
        for irow in 0..H {
            if *weights_row.get_unchecked(height_m1 - irow) == 0.0 {
                row_base += ncol;
                continue;
            }
            for icol in 0..W {
                if *weights_col.get_unchecked(width_m1 - icol) == 0.0 {
                    continue;
                }
                if *data.get_unchecked(row_base + icol) == 0 {
                    return 0;
                }
            }
            row_base += ncol;
        }
        1
    }
    
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        true
    }
}

/// A binary input mask optimized with a safe zone to accelerate validity 
/// testing.
///
/// This structure stores a `u8` array where `1` indicates valid data and `0`
/// indicates invalid data. Within the defined safe zone, points are guaranteed
/// to have a valid mask value, enabling fast-path evaluation during processing.
///
/// # Behavior
/// - Points outside the safe zone follow standard validity checks
/// - Points within the safe zone skip validity validation (always valid)
///
/// # Coordinate System
/// All indices use the same 2D coordinate system (row, column) with origin at
/// the top-left corner of the mask. The safe zone and convolution bounds
/// share this common reference frame.
///
/// # Interpolator Dependencies
/// The `convolution_safe_bounds` must be computed based on the specific
/// interpolation window size used by the consuming interpolator. For a window
/// of height `H` and width `W`, the valid starting positions satisfy:
///
/// ```text
/// conv_safe_min_row    = safe_min_row
/// conv_safe_max_row    = safe_max_row    - (H - 1)
/// conv_safe_min_col    = safe_min_col
/// conv_safe_max_col    = safe_max_col    - (W - 1)
/// ```
///
/// This ensures that for any window starting within `convolution_safe_bounds`,
/// all points of the window remain inside the `safe_zone`, making validity
/// checks unnecessary and enabling performance optimizations.
///
/// # Fields
/// - `inner`: the underlying binary mask data
/// - `safe_zone`: rectangular region where validity is guaranteed (inclusive 
///   bounds)
/// - `convolution_safe_bounds`: the valid starting positions for a convolution
///   window within the safe zone, accounting for kernel boundaries. Must be
///   configured according to the window size of the interpolator that uses this
///   mask structure.
#[derive(Debug)]
pub struct BinaryInputMaskWithSafeWindow<'a> {
    /// Inner BinaryInputMask providing the actual validity data
    pub inner: BinaryInputMask<'a>,
    /// Inclusive minimum row index of the safe zone
    pub safe_min_row: usize,
    /// Inclusive maximum row index of the safe zone
    pub safe_max_row: usize,
    /// Inclusive minimum column index of the safe zone
    pub safe_min_col: usize,
    /// Inclusive maximum column index of the safe zone
    pub safe_max_col: usize,
    /// Inclusive minimum row index for safe convolution starting positions
    /// Uses the same coordinate system as `safe_*` fields
    pub conv_safe_min_row: usize,
    /// Inclusive maximum row index for safe convolution starting positions
    /// Uses the same coordinate system as `safe_*` fields
    pub conv_safe_max_row: usize,
    /// Inclusive minimum column index for safe convolution starting positions
    /// Uses the same coordinate system as `safe_*` fields
    pub conv_safe_min_col: usize,
    /// Inclusive maximum column index for safe convolution starting positions
    /// Uses the same coordinate system as `safe_*` fields
    pub conv_safe_max_col: usize,
}

impl<'a> BinaryInputMaskWithSafeWindow<'a> {
    
    /// A utility function to test if coordinates are with the convolution
    /// safe box.
    #[inline(always)]
    fn is_in_conv_safe_box(
        &self, 
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> bool {
        let test = start_row_idx >= self.conv_safe_min_row
            && start_row_idx <= self.conv_safe_max_row
            && start_col_idx >= self.conv_safe_min_col
            && start_col_idx <= self.conv_safe_max_col;
        //println!("row_idx={0} [{1} - {2}] | col_idx={3} [{4} <= {5}] = {6}", start_row_idx, self.conv_safe_min_row, self.conv_safe_max_row, start_col_idx, self.conv_safe_min_col, self.conv_safe_max_col, test);
        test
    }
}

impl<'a> GxArrayViewInterpolatorInputMaskStrategy
for BinaryInputMaskWithSafeWindow<'a> {

    #[inline(always)]
    fn is_valid(&self, idx: usize) -> u8 {
        self.inner.mask.data[idx]
    }

    #[inline(always)]
    unsafe fn is_valid_unsafe(&self, idx: usize) -> u8 {
        // SAFETY: caller guarantees idx is within mask bounds.
        *self.inner.mask.data.get_unchecked(idx)
    }

    /// Pre-slices one row at a time to eliminate bounds checks in the
    /// inner loop.  The iterator `fold` compiles to a tight AND chain
    /// that LLVM can vectorize.
    #[inline(always)]
    fn is_valid_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> u8 {
        if self.is_in_conv_safe_box(start_row_idx, start_col_idx)
        {
            1
        } else {
            self.inner.is_valid_window::<H, W>(
                start_idx, start_row_idx, start_col_idx
            )
        }
    }

    #[inline(always)]
    unsafe fn is_valid_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> u8 {
        if self.is_in_conv_safe_box(start_row_idx, start_col_idx)
        {
            1
        } else {
            self.inner.is_valid_window_unsafe::<H, W>(
                start_idx, start_row_idx, start_col_idx
            )
        }
    }

    /// Pre-slices one row at a time.  The `map(|&v| v as usize).sum()`
    /// pattern compiles to a tight add chain without bounds checks.
    #[inline(always)]
    fn count_valid_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> usize {
        // With a safe window we consider that if the index is considered
        // safe, ie. all points are valid within the window, and then the
        // number of valid point is H*W
        if self.is_in_conv_safe_box(start_row_idx, start_col_idx)
        {
            H*W
        } else {
            self.inner.count_valid_window::<H, W>(
                start_idx, start_row_idx, start_col_idx
            )
        }
    }

    #[inline(always)]
    unsafe fn count_valid_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
    ) -> usize {
        if self.is_in_conv_safe_box(start_row_idx, start_col_idx)
        {
            H*W
        } else {
            self.inner.count_valid_window_unsafe::<H, W>(
                start_idx, start_row_idx, start_col_idx
            )
        }
    }

    /// For active rows, the inner loop uses a pre-sliced row and iterator to
    /// eliminate bounds checks.
    #[inline(always)]
    fn is_valid_weighted_window<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8 {
        if self.is_in_conv_safe_box(start_row_idx, start_col_idx)
        {
            1
        } else {
            self.inner.is_valid_weighted_window::<H, W>(
                start_idx, start_row_idx, start_col_idx,
                weights_row, weights_col
            )
        }
    }

    #[inline(always)]
    unsafe fn is_valid_weighted_window_unsafe<const H: usize, const W: usize>(
        &self,
        start_idx: usize,
        start_row_idx: usize,
        start_col_idx: usize,
        weights_row: &[f64],
        weights_col: &[f64],
    ) -> u8 {
        if self.is_in_conv_safe_box(start_row_idx, start_col_idx)
        {
            1
        } else {
            self.inner.is_valid_weighted_window_unsafe::<H, W>(
                start_idx, start_row_idx, start_col_idx,
                weights_row, weights_col
            )
        }
    }
    
    #[inline(always)]
    fn is_enabled(&self) -> bool {
        true
    }
}




/// Convenience enum wrapping the two supported input mask strategies.
pub enum InputMaskStrategy<'a> {
    /// Delegate to a [`BinaryInputMask`].
    Binary(BinaryInputMask<'a>),
    /// Delegate to a [`BinaryInputMaskWithSafeWindow`].
    BinarySafeWindow(BinaryInputMaskWithSafeWindow<'a>),
    /// Delegate to [`NoInputMask`] (all points valid).
    None(NoInputMask),
}


// =============================================================================
// Output mask strategies
// =============================================================================

/// Strategy trait for writing to an output validity mask.
///
/// An output mask records which output pixels were successfully interpolated
/// (`1`) and which were set to `nodata` (`0`).
///
/// # Contract
/// - [`is_enabled`](Self::is_enabled) returns `false` for no-op
///   implementations; callers may skip the write in that case.
/// - [`set_value`](Self::set_value) writes the mask value at flat index `idx`.
pub trait GxArrayViewInterpolatorOutputMaskStrategy {   
    /// Returns `true` if this mask strategy is active.
    fn is_enabled(&self) -> bool;

    /// Writes `value` to the output mask at flat index `idx`.
    fn set_value(&mut self, idx: usize, value: u8);
}

/// Blanket impl: a mutable reference to an output mask strategy is itself an
/// output mask strategy, delegating all calls to the inner value.
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

/// No-op output mask: mask writes are discarded.
///
/// Use this when no output validity tracking is required.
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

/// Binary output mask backed by a mutable `u8` array.
///
/// Each call to [`set_value`](GxArrayViewInterpolatorOutputMaskStrategy::set_value)
/// stores the value directly in the underlying array.
pub struct BinaryOutputMask<'a> {
    /// Mutable view of the output binary mask array.
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

/// Convenience enum wrapping the two supported output mask strategies.
pub enum OutputMaskStrategy<'a> {
    /// Delegate to a [`BinaryOutputMask`].
    Binary(BinaryOutputMask<'a>),
    /// Delegate to [`NoOutputMask`] (writes discarded).
    None(NoOutputMask),
}


// =============================================================================
// Bounds check strategy
// =============================================================================

/// Strategy trait for controlling whether index boundary checks are performed.
///
/// The single static method [`do_check`](Self::do_check) allows the compiler
/// to eliminate the bounds-checking branch entirely when the strategy is
/// [`NoBoundsCheck`], yielding a zero-cost abstraction.
pub trait GxArrayViewInterpolatorBoundsCheckStrategy {
    /// Returns `true` if boundary checks should be performed, `false` if they
    /// can be skipped (caller guarantees valid indices).
    fn do_check() -> bool;
}

/// Bounds checking disabled.
///
/// The caller guarantees that all kernel indices are within the input array.
#[derive(Default)]
pub struct NoBoundsCheck;

impl GxArrayViewInterpolatorBoundsCheckStrategy for NoBoundsCheck {
    #[inline(always)]
    fn do_check() -> bool {
        false
    }
}

/// Bounds checking enabled (default, safe).
#[derive(Default)]
pub struct BoundsCheck;

impl GxArrayViewInterpolatorBoundsCheckStrategy for BoundsCheck {
    #[inline(always)]
    fn do_check() -> bool {
        true
    }
}


// =============================================================================
// Interpolation context
// =============================================================================

/// Interpolation context aggregating the three orthogonal strategies.
///
/// The context is generic over the strategies, enabling compile-time
/// monomorphisation with zero runtime overhead.  `PhantomData` markers are
/// used to carry the `BoundsCheck` type (which has no runtime state) and the
/// lifetime `'a`.
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
    /// Input mask strategy instance.
    pub input_mask: IM,

    /// Output mask strategy instance.
    pub output_mask: OM,

    /// Carries the bounds-check strategy type without storing a value.
    pub _phantom_bounds: PhantomData<BC>,

    /// Carries the lifetime `'a`.
    pub _phantom_lifetime: PhantomData<&'a ()>,
}

/// Trait providing type-erased access to the strategies stored in a context.
///
/// Implement this trait on a context type to allow generic interpolation code
/// to query the active strategies without knowing the concrete context type.
pub trait GxArrayViewInterpolationContextTrait {
    /// The concrete input mask strategy type.
    type InputMask: GxArrayViewInterpolatorInputMaskStrategy;

    /// The concrete output mask strategy type.
    type OutputMask: GxArrayViewInterpolatorOutputMaskStrategy;

    /// The concrete bounds-check strategy type.
    type BoundsCheck: GxArrayViewInterpolatorBoundsCheckStrategy;

    /// Returns a shared reference to the input mask strategy.
    fn input_mask(&self) -> &Self::InputMask;

    /// Returns a mutable reference to the output mask strategy.
    fn output_mask(&mut self) -> &mut Self::OutputMask;
}

impl<'a, IM, OM, BC> GxArrayViewInterpolationContextTrait
    for GxArrayViewInterpolationContext<'a, IM, OM, BC>
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
    ///
    /// The `_bounds_check` argument is consumed only to infer the `BC` type;
    /// no value is stored (the type carries all information).
    pub fn new(input_mask: IM, output_mask: OM, _bounds_check: BC) -> Self {
        Self {
            input_mask,
            output_mask,
            _phantom_bounds: PhantomData,
            _phantom_lifetime: PhantomData,
        }
    }
}

/// Default interpolation context: no input mask, no output mask, bounds
/// checking enabled.
pub type DefaultCtx<'a> =
    GxArrayViewInterpolationContext<'a, NoInputMask, NoOutputMask, BoundsCheck>;

impl<'a> DefaultCtx<'a> {
    /// Creates a [`DefaultCtx`] with no masking and standard bounds checking.
    pub fn default() -> Self {
        Self {
            input_mask:        NoInputMask::default(),
            output_mask:       NoOutputMask::default(),
            _phantom_bounds:   PhantomData,
            _phantom_lifetime: PhantomData,
        }
    }
}

// =============================================================================
// Interpolator arguments
// =============================================================================

/// Trait providing construction arguments for [`GxArrayViewInterpolator`]
/// instances.
///
/// Different interpolation algorithms require different parameters at
/// construction time.  This trait acts as an abstraction layer so that all
/// interpolators share the same `new(args: &dyn GxArrayViewInterpolatorArgs)`
/// signature while each can retrieve only the fields it needs.
///
/// Default implementations return sensible "not applicable" values, so that
/// simple interpolators (nearest-neighbour, linear, cubic) can leave the
/// B-spline accessor unimplemented.
///
/// # Supported interpolation methods
///
/// | Method          | Required args               |
/// |-----------------|-----------------------------|
/// | Nearest-neighbour | none                      |
/// | Linear          | none                        |
/// | Cubic           | none                        |
/// | B-spline        | `bspline_args` → `(ε, s)`   |
///
/// where `ε` is the prefiltering precision and `s` is the maximum acceptable
/// influence from masked pixels.
///
/// Implementation of this trait allows for compile-time verification of 
/// required parameters and provides a consistent interface for interpolation
/// configuration.
///
/// ## Design Considerations
///
/// While maintaining generality, the trait intentionally includes knowledge of
/// existing interpolator types to provide appropriate parameter exposure. This
/// design choice ensures that each interpolator receives only the parameters it
/// requires, preventing misuse and enabling compile-time validation.
pub trait GxArrayViewInterpolatorArgs {
    /// Returns `true` if the interpolator requires no arguments beyond the
    /// defaults.
    fn no_args(&self) -> bool {
        true
    }
    
    /// Returns the B-spline-specific arguments `(precision, influence)`.
    ///
    /// - `precision` (`ε`): accepted error for the infinite-sum approximation
    ///   in the prefiltering step.
    /// - `influence` (`s`): accepted relative contamination from masked invalid
    ///   pixels during prefiltering.
    ///
    /// # Returns
    /// - `Some((f64, f64))` for B-spline interpolators.
    /// - `None` for all other interpolators (default).
    fn bspline_args(&self) -> Option<(f64, f64)> {
        None
    }
}

/// Arguments for interpolators that require no additional configuration.
///
/// Implements [`GxArrayViewInterpolatorArgs`] with all-default behaviour,
/// suitable for nearest-neighbour, linear, and cubic interpolators.
pub struct GxArrayViewInterpolatorNoArgs;


impl GxArrayViewInterpolatorArgs for GxArrayViewInterpolatorNoArgs {
    // All methods use the trait defaults.
}


// =============================================================================
// Public interpolator trait
// =============================================================================

/// Public trait defining the 2D interpolation interface.
///
/// An implementor computes interpolated values at arbitrary floating-point
/// positions within a flattened 3D input array
/// `[nvar, nrow, ncol]` and writes the result to a matching output array.
///
/// # Coordinate Convention
/// The interpolation is performed at a discrete center coordinate `(row_c, 
/// col_c)`, which defines the origin of a local neighborhood used for 
/// interpolation.
///
/// # Data layout
///
/// All arrays follow a flattened row-major 3D layout:
///
/// - [`GxArrayView`]: read-only input, dimensions `[nvar, nrow, ncol]`.
/// - [`GxArrayViewMut`]: mutable output, same layout.
/// - Optional validity masks (`u8`) are embedded in the `context` parameter.
///
/// Flat indices are computed as `ivar * nrow * ncol + irow * ncol + icol`.
///
/// # Context parameter
///
/// The generic `context` argument of [`array1_interp2`](Self::array1_interp2)
/// encapsulates three orthogonal behaviours:
///
/// - **Input mask** — whether a given input sample is valid.
/// - **Output mask** — whether to record which output pixels were written.
/// - **Bounds check** — whether to verify that the interpolation stencil lies
///   fully within the input array.
///
/// # Thread safety
///
/// Implementors must be `Send + Sync` so that interpolators can be shared
/// across threads.
pub trait GxArrayViewInterpolator: Send + Sync {
    /// Constructs a new interpolator from the supplied arguments.
    fn new(args: &dyn GxArrayViewInterpolatorArgs) -> Self;
    
    /// Returns a short human-readable identifier for this interpolator.
    fn shortname(&self) -> String;
    
    /// Performs any pre-computation required before the first call to
    /// [`array1_interp2`](Self::array1_interp2).
    ///
    /// For simple methods (nearest-neighbour, linear, cubic) this is a no-op.
    /// For B-spline interpolators it computes the truncation indices and domain
    /// extensions.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(String)` with a descriptive message on failure.
    fn initialize(&mut self) -> Result<(), String>;
    
    /// Allocates a scratch buffer large enough to hold one row of kernel
    /// weights plus one column of kernel weights.
    ///
    /// The returned buffer is intended to be passed to
    /// [`array1_interp2`](Self::array1_interp2) as `weights_buffer` and reused
    /// across calls to avoid repeated allocation.
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]>;
    
    /// Performs 2D interpolation at the specified target position.
    ///
    /// # Parameters
    /// - `weights_buffer`: pre-allocated scratch buffer (see
    ///   [`allocate_kernel_buffer`](Self::allocate_kernel_buffer)); its
    ///   contents are overwritten on each call.
    /// - `target_row_pos`: target row coordinate (floating-point).
    /// - `target_col_pos`: target column coordinate (floating-point).
    /// - `out_idx`: flat index in `array_out` at which to write the result
    ///   (same offset for all variables).
    /// - `array_in`: input data array `[nvar, nrow, ncol]`, flattened.
    /// - `array_out`: output data array (same layout), written in place.
    /// - `nodata_out`: value written when interpolation cannot be performed
    ///   (e.g. out-of-bounds or masked input).
    /// - `context`: interpolation context controlling masking and bounds
    ///   checking.
    ///
    /// # Returns
    /// - `Ok(())` if interpolation succeeded.
    /// - `Err(String)` with an error message on failure.
    ///
    /// # Type parameters
    /// - `T`: input element type — must be `Copy + PartialEq + Into<f64>`
    ///   and support `Mul<f64, Output = f64>`.
    /// - `V`: output element type — must be `Copy + PartialEq + From<f64>`.
    /// - `IC`: context type implementing
    ///   [`GxArrayViewInterpolationContextTrait`].
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
    
    /// Returns the kernel height
    fn kernel_row_size(&self) -> usize;
    
    /// Returns the kernel width
    fn kernel_col_size(&self) -> usize;
    
    /// Returns the margins `[top, bottom, left, right]` (in pixels) that must 
    /// be available around the region of interest for the interpolation to be 
    /// valid.
    ///
    /// # Returns
    /// - `Ok([usize; 4])` where indices map to `[top, bottom, left, right]`.
    /// - `Err(GxError)` if the interpolator has not been initialised.
    fn total_margins(&self) -> Result<[usize; 4], GxError>;
}


// =============================================================================
// Helper
// =============================================================================

/// Writes `nodata` to every variable plane at `out_idx` and sets the output
/// mask to `0`.
///
/// This function is called in every branch where interpolation cannot be
/// performed (out-of-bounds centre, masked input pixel, etc.) to guarantee
/// that the output is always initialised.
///
/// # Parameters
/// - `array_out`: mutable output array (all variable planes are written).
/// - `out_idx`: flat pixel index within each plane.
/// - `nodata`: the sentinel value to write.
/// - `output_mask`: output mask strategy; `set_value(out_idx, 0)` is called.
/// - `nvar`: number of variable planes.
/// - `var_size`: stride between consecutive planes (`nrow * ncol`).
///
/// # Type parameters
/// - `V`: output element type, must be `Copy`.
/// - `OM`: output mask strategy, must implement
///   [`GxArrayViewInterpolatorOutputMaskStrategy`].
#[inline(always)]
pub fn write_nodata_all_vars<V, OM>(
    array_out: &mut GxArrayViewMut<'_, V>,
    out_idx: usize,
    nodata: V,
    output_mask: &mut OM,
    nvar: usize,
    var_size: usize,
) where
    V: Copy,
    OM: GxArrayViewInterpolatorOutputMaskStrategy,
{
    let mut shift = 0usize;
    for _ in 0..nvar {
        array_out.data[out_idx + shift] = nodata;
        shift += var_size;
    }
    output_mask.set_value(out_idx, 0);
}

// =============================================================================
// Internal interpolator trait
// =============================================================================

/// **Internal computation trait** for separable-kernel interpolators.
///
/// # Role and scope
///
/// [`GxArrayViewInterpolatorCore`] is *not* part of the public API: callers
/// always interact with [`GxArrayViewInterpolator`].  Its purpose is to factor
/// out the four separable kernel-convolution variants that would otherwise be
/// copy-pasted into every concrete interpolator:
///
/// * `interpolate_nomask_unchecked`
/// * `interpolate_nomask_partial`
/// * `interpolate_masked_unchecked`
/// * `interpolate_masked_partial`
/// * `array1_interp2_separable_core` (dispatcher, calls the four above)
///
/// A concrete interpolator implementing this trait inherits all five methods
/// for free; it only needs to provide
/// [`compute_weights`](Self::compute_weights).
///
/// # When to implement this trait
///
/// Implement [`GxArrayViewInterpolatorCore`] when:
/// - The stencil is **separable**.
/// - The kernel dimensions are **fixed at compile time** (`KROWS`, `KCOLS`).
///
/// Interpolators with non-separable kernels or runtime-determined stencil
/// sizes should implement [`GxArrayViewInterpolator`] directly and provide
/// their own convolution logic.
///
/// # Const parameters
///
/// | Parameter | Meaning                       | Bicubic example |
/// |-----------|-------------------------------|-----------------|
/// | `KROWS`   | Kernel height in rows         | `5`             |
/// | `KCOLS`   | Kernel width in columns       | `5`             |
///
/// # Provided methods
///
/// | Method                           | Input mask  | Bounds check |
/// |----------------------------------|-------------|--------------|
/// | `interpolate_nomask_unchecked`   | no          | no           |
/// | `interpolate_nomask_partial`     | no          | yes          |
/// | `interpolate_masked_unchecked`   | yes         | no           |
/// | `interpolate_masked_partial`     | yes         | yes          |
/// | `array1_interp2_separable_core`  | via context | via context  |
///
/// # Minimal implementation
///
/// ```ignore
/// // Only `compute_weights` is required; all other methods are inherited.
/// impl GxArrayViewInterpolatorCore<5, 5> for MyInterpolator {
///     fn compute_weights(&self, x: f64, weights: &mut [f64]) {
///         my_kernel_weights(x, weights);
///     }
/// }
/// ```
///
/// # Relationship with `GxArrayViewInterpolator`
///
/// A concrete type typically implements both traits.
/// `array1_interp2_separable_core` is a natural delegate for the required
/// `array1_interp2` method:
///
/// ```ignore
/// impl GxArrayViewInterpolator for MyInterpolator {
///     fn array1_interp2<T, V, IC>(
///         &self, buf, row, col, idx, arr_in, arr_out, nodata, ctx,
///     ) -> Result<(), String> {
///         self.array1_interp2_separable_core(
///             buf, row, col, idx, arr_in, arr_out, nodata, ctx)
///     }
///     // … other required methods …
/// }
/// ```
pub trait GxArrayViewInterpolatorCore<const KROWS: usize, const KCOLS: usize> {
    
    // =========================================================================
    // interpolate_nomask_unchecked
    // =========================================================================

    /// Separable 2D convolution with no input mask and no bounds checking.
    ///
    /// Computes the interpolated value at `(row_c, col_c)` using a
    /// `KROWS × KCOLS` separable stencil and writes it to `array_out` at
    /// `out_idx` for every variable plane.
    ///
    /// # Parameters
    /// - `weights_row`: row-direction kernel weights, length `KROWS`.
    /// - `weights_col`: column-direction kernel weights, length `KCOLS`.
    /// - `array_in`: input data array `[nvar, nrow, ncol]`, flattened.
    /// - `array_out`: output array (same layout), written in place.
    /// - `out_idx`: flat pixel index in the output (shared across all planes).
    /// - `row_c`: integer row coordinate of the interpolation centre.
    /// - `col_c`: integer column coordinate of the interpolation centre.
    ///
    /// # Safety
    /// The caller must guarantee that every index
    /// `row_c + irow - KROWS/2` and `col_c + icol - KCOLS/2`
    /// is within the bounds of `array_in`.  Violating this precondition
    /// results in a panic (index out of bounds) or, with unsafe indexing,
    /// undefined behaviour.
    ///
    /// # Type parameters
    /// - `T`: input element type.
    /// - `V`: output element type.
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
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64>,
        V: Copy + From<f64>,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        let row_start = (row_c - half_rows) as usize;
        let col_start = (col_c - half_cols) as usize;

        // Pre-slice weights: one bounds check each, inner loops are free.
        let wr = &weights_row[..KROWS];
        let wc = &weights_col[..KCOLS];

        let mut in_shift  = 0usize;
        let mut out_shift = 0usize;

        for _ivar in 0..array_in.nvar {
            let mut acc = 0.0f64;
            for irow in 0..KROWS {
                // Pre-slice one row of input data.
                let base = in_shift + (row_start + irow) * ncol + col_start;
                let row_slice = &array_in.data[base..base + KCOLS];

                // Iterator-based dot product: zero bounds checks.
                let acc_col: f64 = row_slice.iter()
                    .zip(wc.iter().rev())
                    .map(|(&d, &w)| d * w)
                    .sum();

                acc += wr[KROWS - 1 - irow] * acc_col;
            }
            array_out.data[out_idx + out_shift] = V::from(acc);
            in_shift  += in_var_sz;
            out_shift += out_var_sz;
        }
    }

    // =========================================================================
    // interpolate_nomask_partial
    // =========================================================================
    
    /// Separable 2D convolution with no input mask, with bounds checking.
    ///
    /// Identical to [`interpolate_nomask_unchecked`](Self::interpolate_nomask_unchecked)
    /// except that, before performing the convolution, every stencil position
    /// whose weight is non-zero is checked against the array bounds.  If any
    /// such position lies outside the array, the output at `out_idx` is set to
    /// `nodata` for all variable planes, the output mask is set to `0`, and the
    /// function returns early.
    ///
    /// Positions with zero weight that fall outside the array are silently
    /// ignored (they contribute nothing to the sum).
    ///
    /// # Parameters
    /// - `weights_row`: row-direction kernel weights, length `KROWS`.
    /// - `weights_col`: column-direction kernel weights, length `KCOLS`.
    /// - `array_in`: input data array `[nvar, nrow, ncol]`, flattened.
    /// - `array_out`: output array (same layout), written in place.
    /// - `output_mask`: output mask strategy; written once upon completion.
    /// - `out_idx`: flat pixel index in the output (shared across all planes).
    /// - `row_c`: integer row coordinate of the interpolation centre.
    /// - `col_c`: integer column coordinate of the interpolation centre.
    /// - `nodata`: sentinel value written when interpolation is invalid.
    ///
    /// # Type parameters
    /// - `T`: input element type.
    /// - `V`: output element type.
    /// - `OM`: output mask strategy implementing
    ///   [`GxArrayViewInterpolatorOutputMaskStrategy`].
    #[inline(always)]
    fn interpolate_nomask_partial<T, V, OM>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        output_mask: &mut OM,
        out_idx: usize,
        row_c: i64,
        col_c: i64,
        nodata: V,
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64>,
        V: Copy + From<f64>,
        OM: GxArrayViewInterpolatorOutputMaskStrategy,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        let wr = &weights_row[..KROWS];
        let wc = &weights_col[..KCOLS];

        // Pre-check: every stencil position with non-zero weight must
        // be in-bounds.
        for irow in 0..KROWS {
            if wr[KROWS - 1 - irow] == 0.0 {
                continue;
            }
            let r = row_c + irow as i64 - half_rows;
            for icol in 0..KCOLS {
                if wc[KCOLS - 1 - icol] == 0.0 {
                    continue;
                }
                let c = col_c + icol as i64 - half_cols;
                if r < 0 || r >= array_in.nrow_i64
                    || c < 0 || c >= array_in.ncol_i64
                {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        output_mask, array_in.nvar, out_var_sz,
                    );
                    return;
                }
            }
        }

        // Convolution.
        let mut in_shift  = 0usize;
        let mut out_shift = 0usize;

        for _ivar in 0..array_in.nvar {
            let mut acc = 0.0f64;
            for irow in 0..KROWS {
                let r = row_c + irow as i64 - half_rows;
                if r < 0 || r >= array_in.nrow_i64 {
                    continue;
                }
                let ru = r as usize;

                // Compute valid column range for this row.
                let c_first = col_c - half_cols;
                let c_last  = col_c + (KCOLS as i64 - 1) - half_cols;

                let col_lo = c_first.max(0) as usize;
                let col_hi = (c_last.min(array_in.ncol_i64 - 1) + 1) as usize;

                // Offset into the kernel for the first valid column.
                let icol_start = (col_lo as i64 - c_first) as usize;
                let icol_end   = icol_start + (col_hi - col_lo);

                // Pre-slice input row and matching weight segment.
                let row_base = in_shift + ru * ncol;
                let row_slice = &array_in.data[row_base + col_lo
                    ..row_base + col_hi];
                let wc_slice = &wc[KCOLS - icol_end..KCOLS - icol_start];

                let acc_col: f64 = row_slice.iter()
                    .zip(wc_slice.iter().rev())
                    .map(|(&d, &w)| d * w)
                    .sum();

                acc += wr[KROWS - 1 - irow] * acc_col;
            }
            array_out.data[out_idx + out_shift] = V::from(acc);
            in_shift  += in_var_sz;
            out_shift += out_var_sz;
        }
        output_mask.set_value(out_idx, 1);
    }

    // =========================================================================
    // interpolate_masked_unchecked
    // =========================================================================

    /// Separable 2D convolution with input mask validation, no bounds checking.
    ///
    /// # Design : Exact Interpolation Policy
    /// The mask validation operates on weighted support rather than on the 
    /// target coordinates alone. Stencil positions whose kernel weight is zero
    /// are excluded from the validity check, regardless of their mask value.
    /// This ensures that when the kernel reduces to a Dirac at the nearest grid
    /// node (all off-centre wieghts vanish), the method returns the source 
    /// value provided that single node is valid -- even if neighbouring samples
    /// in the convolution are masked.
    /// This is a design choice
    ///
    /// # Parameters
    /// - `weights_row`: row-direction kernel weights, length `KROWS`.
    /// - `weights_col`: column-direction kernel weights, length `KCOLS`.
    /// - `array_in`: input data array `[nvar, nrow, ncol]`, flattened.
    /// - `array_out`: output array (same layout), written in place.
    /// - `out_idx`: flat pixel index in the output (shared across all planes).
    /// - `row_c`: integer row coordinate of the interpolation centre.
    /// - `col_c`: integer column coordinate of the interpolation centre.
    /// - `nodata`: sentinel value written when the window contains masked pixels.
    /// - `context`: interpolation context providing the input and output mask
    ///   strategies.
    ///
    /// # Performance
    /// The mask validation follows a two-stage short-circuit strategy ordered
    /// by decreasing probability to minimise the cost of the common case.
    /// First, a branchless pass over the KROWS x KCOLS window accumulates the 
    /// number of valid samples. The resulting count feeds a short-circuit
    /// conditional :
    /// 1. count == KROWS x KSIZE (all valid) -- the dominant case
    /// 2. count == 0 (all masked) -- second most frequent and cheap
    /// 3. 0 < count < KROWS x KZIZE -- Both guards pass, triggering the full
    ///   [`is_valid_weighted_window()`](
    ///   GxArrayViewInterpolatorInputMaskStrategy::is_valid_weighted_window())
    ///   check required due to the exact policy.
    ///
    /// # Safety
    /// The caller must guarantee that every stencil index is within the bounds
    /// of both `array_in` and the mask array embedded in `context`.
    ///
    /// # Type parameters
    /// - `T`: input element type.
    /// - `V`: output element type.
    /// - `IM`: interpolation context
    ///   [`GxArrayViewInterpolationContext`].
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
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64>,
        V: Copy + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        let row_start = (row_c - half_rows) as usize;
        let col_start = (col_c - half_cols) as usize;
        let flat_base = row_start * ncol + col_start;

        let wr = &weights_row[..KROWS];
        let wc = &weights_col[..KCOLS];

        // Two-stage short-circuit mask validation.
        let count = context.input_mask()
            .count_valid_window::<KROWS, KCOLS>(flat_base, row_start, col_start);

        if count == KROWS * KCOLS
            || (count > 0
                && context.input_mask().is_valid_weighted_window::<KROWS, KCOLS>(
                    flat_base, row_start, col_start, weights_row, weights_col,
                ) == 1)
        {
            // Convolution with iterator-based inner loop.
            let mut out_shift = 0usize;
            for ivar in 0..array_in.nvar {
                let ain_base = flat_base + ivar * in_var_sz;
                let mut acc = 0.0f64;

                for irow in 0..KROWS {
                    let base = ain_base + irow * ncol;
                    let row_slice = &array_in.data[base..base + KCOLS];

                    let acc_col: f64 = row_slice.iter()
                        .zip(wc.iter().rev())
                        .map(|(&d, &w)| d * w)
                        .sum();

                    acc += wr[KROWS - 1 - irow] * acc_col;
                }
                array_out.data[out_idx + out_shift] = V::from(acc);
                out_shift += out_var_sz;
            }
            context.output_mask().set_value(out_idx, 1);
        } else {
            write_nodata_all_vars(
                array_out, out_idx, nodata,
                context.output_mask(), array_in.nvar, out_var_sz,
            );
        }
    }

    // =========================================================================
    // interpolate_masked_partial
    // =========================================================================

    /// Separable 2D convolution with input mask validation and bounds checking.
    ///
    /// Combines the bounds checking of
    /// [`interpolate_nomask_partial`](Self::interpolate_nomask_partial) with
    /// the mask validation of
    /// [`interpolate_masked_unchecked`](Self::interpolate_masked_unchecked).
    ///
    /// The pre-validation pass checks, for every stencil position with a
    /// non-zero weight:
    /// 1. That the position lies within the input array bounds.
    /// 2. That the corresponding input mask value is `1` (valid).
    ///
    /// If either check fails the function writes `nodata` to all output planes
    /// and returns early.  If both checks pass, the convolution is executed
    /// without further bound or mask testing (guaranteed safe by the pre-pass).
    ///
    /// Out-of-bounds or masked positions with zero weight are silently ignored.
    ///
    /// # Parameters
    /// - `weights_row`: row-direction kernel weights, length `KROWS`.
    /// - `weights_col`: column-direction kernel weights, length `KCOLS`.
    /// - `array_in`: input data array `[nvar, nrow, ncol]`, flattened.
    /// - `array_out`: output array (same layout), written in place.
    /// - `out_idx`: flat pixel index in the output (shared across all planes).
    /// - `row_c`: integer row coordinate of the interpolation centre.
    /// - `col_c`: integer column coordinate of the interpolation centre.
    /// - `nodata`: sentinel value written when interpolation is invalid.
    /// - `context`: interpolation context providing mask and bounds strategies.
    ///
    /// # See also
    /// - [`interpolate_masked_unchecked`](Self::interpolate_masked_unchecked):
    ///   faster variant assuming in-bounds indices.
    /// - [`interpolate_nomask_partial`](Self::interpolate_nomask_partial):
    ///   bounds-checked variant without mask testing.
    ///
    /// # Type parameters
    /// - `T`: input element type.
    /// - `V`: output element type.
    /// - `IM`: interpolation context
    ///   [`GxArrayViewInterpolationContext`].
    #[inline(always)]
    fn interpolate_masked_partial<T, V, IC>(
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
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64> + PartialEq,
        V: Copy + From<f64> + PartialEq,
        IC: GxArrayViewInterpolationContextTrait,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        let wr = &weights_row[..KROWS];
        let wc = &weights_col[..KCOLS];

        // Pre-validation: bounds + mask for all active stencil positions.
        for irow in 0..KROWS {
            if wr[KROWS - 1 - irow] == 0.0 {
                continue;
            }
            let r = row_c + irow as i64 - half_rows;
            for icol in 0..KCOLS {
                if wc[KCOLS - 1 - icol] == 0.0 {
                    continue;
                }
                if r < 0 || r >= array_in.nrow_i64 {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        context.output_mask(), array_in.nvar, out_var_sz,
                    );
                    return;
                }
                let c = col_c + icol as i64 - half_cols;
                if c < 0 || c >= array_in.ncol_i64 {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        context.output_mask(), array_in.nvar, out_var_sz,
                    );
                    return;
                }
                let flat = r as usize * ncol + c as usize;
                if context.input_mask().is_valid(flat) == 0 {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        context.output_mask(), array_in.nvar, out_var_sz,
                    );
                    return;
                }
            }
        }

        // Convolution -- bounds and mask guaranteed by pre-pass.
        let mut in_shift  = 0usize;
        let mut out_shift = 0usize;

        for _ivar in 0..array_in.nvar {
            let mut acc = 0.0f64;
            for irow in 0..KROWS {
                let r = row_c + irow as i64 - half_rows;
                if r < 0 || r >= array_in.nrow_i64 {
                    continue;
                }
                let ru = r as usize;

                // Valid column range.
                let c_first = col_c - half_cols;
                let c_last  = col_c + (KCOLS as i64 - 1) - half_cols;

                let col_lo = c_first.max(0) as usize;
                let col_hi = (c_last.min(array_in.ncol_i64 - 1) + 1) as usize;

                let icol_start = (col_lo as i64 - c_first) as usize;
                let icol_end   = icol_start + (col_hi - col_lo);

                let row_base = in_shift + ru * ncol;
                let row_slice = &array_in.data[row_base + col_lo
                    ..row_base + col_hi];
                let wc_slice = &wc[KCOLS - icol_end..KCOLS - icol_start];

                let acc_col: f64 = row_slice.iter()
                    .zip(wc_slice.iter().rev())
                    .map(|(&d, &w)| d * w)
                    .sum();

                acc += wr[KROWS - 1 - irow] * acc_col;
            }
            array_out.data[out_idx + out_shift] = V::from(acc);
            in_shift  += in_var_sz;
            out_shift += out_var_sz;
        }
        context.output_mask().set_value(out_idx, 1);
    }
    
// =============================================================================
// Interpolation variants — unsafe implementations
// =============================================================================

    // =========================================================================
    // interpolate_nomask_unchecked_unsafe
    // =========================================================================

    /// Unsafe variant for interpolate_nomask_unchecked
    /// See the safe variant for details
    #[inline(always)]
    unsafe fn interpolate_nomask_unchecked_unsafe<T, V>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        out_idx: usize,
        row_c: i64,
        col_c: i64,
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64>,
        V: Copy + From<f64>,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        let row_start = (row_c - half_rows) as usize;
        let col_start = (col_c - half_cols) as usize;

        let mut in_shift  = 0usize;
        let mut out_shift = 0usize;

        // SAFETY: the interior path guarantees that
        // row_start..row_start+KROWS and col_start..col_start+KCOLS
        // are within array bounds for all variable planes.
        for _ivar in 0..array_in.nvar {
            let mut acc = 0.0f64;
            for irow in 0..KROWS {
                let base = in_shift
                    + (row_start + irow) * ncol + col_start;
                let mut acc_col = 0.0f64;
                for icol in 0..KCOLS {
                    acc_col +=
                        *array_in.data.get_unchecked(base + icol)
                        * *weights_col.get_unchecked(KCOLS - 1 - icol);
                }
                acc += *weights_row.get_unchecked(KROWS - 1 - irow)
                    * acc_col;
            }
            *array_out.data.get_unchecked_mut(out_idx + out_shift) =
                V::from(acc);
            in_shift  += in_var_sz;
            out_shift += out_var_sz;
        }
    }

    // =========================================================================
    // interpolate_nomask_partial_unsafe
    // =========================================================================

    /// Unsafe variant for interpolate_nomask_partial_unsafe
    /// See the safe variant for details
    #[inline(always)]
    unsafe fn interpolate_nomask_partial_unsafe<T, V, OM>(
        &self,
        weights_row: &[f64],
        weights_col: &[f64],
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        output_mask: &mut OM,
        out_idx: usize,
        row_c: i64,
        col_c: i64,
        nodata: V,
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64>,
        V: Copy + From<f64>,
        OM: GxArrayViewInterpolatorOutputMaskStrategy,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        // Pre-check: use get_unchecked on weights only (indices are
        // const-generic derived, always valid).
        // SAFETY: KROWS - 1 - irow and KCOLS - 1 - icol are within
        // 0..KROWS and 0..KCOLS respectively.
        for irow in 0..KROWS {
            if *weights_row.get_unchecked(KROWS - 1 - irow) == 0.0 {
                continue;
            }
            let r = row_c + irow as i64 - half_rows;
            for icol in 0..KCOLS {
                if *weights_col.get_unchecked(KCOLS - 1 - icol) == 0.0 {
                    continue;
                }
                let c = col_c + icol as i64 - half_cols;
                if r < 0 || r >= array_in.nrow_i64
                    || c < 0 || c >= array_in.ncol_i64
                {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        output_mask, array_in.nvar, out_var_sz,
                    );
                    return;
                }
            }
        }

        // Convolution — positions validated above.
        let mut in_shift  = 0usize;
        let mut out_shift = 0usize;

        // SAFETY: the pre-check above guarantees that all accessed
        // positions with non-zero weight are within bounds.
        // Positions with zero weight may be out of bounds but are
        // skipped by the continue.
        for _ivar in 0..array_in.nvar {
            let mut acc = 0.0f64;
            for irow in 0..KROWS {
                let r = row_c + irow as i64 - half_rows;
                if r < 0 || r >= array_in.nrow_i64 {
                    continue;
                }
                let ru = r as usize;
                let mut acc_col = 0.0f64;
                for icol in 0..KCOLS {
                    let c = col_c + icol as i64 - half_cols;
                    if c < 0 || c >= array_in.ncol_i64 {
                        continue;
                    }
                    let flat = in_shift + ru * ncol + c as usize;
                    acc_col +=
                        *array_in.data.get_unchecked(flat)
                        * *weights_col.get_unchecked(KCOLS - 1 - icol);
                }
                acc += *weights_row.get_unchecked(KROWS - 1 - irow)
                    * acc_col;
            }
            *array_out.data.get_unchecked_mut(out_idx + out_shift) =
                V::from(acc);
            in_shift  += in_var_sz;
            out_shift += out_var_sz;
        }
        output_mask.set_value(out_idx, 1);
    }

    // =========================================================================
    // interpolate_masked_unchecked
    // =========================================================================

    /// Unsafe variant for interpolate_masked_unchecked
    /// See the safe variant for details
    #[inline(always)]
    unsafe fn interpolate_masked_unchecked_unsafe<T, V, IC>(
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
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64>,
        V: Copy + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        let row_start = (row_c - half_rows) as usize;
        let col_start = (col_c - half_cols) as usize;
        let flat_base = row_start * ncol + col_start;

        // Two-stage short-circuit mask validation.
        let count = context.input_mask()
            .count_valid_window::<KROWS, KCOLS>(flat_base, row_start, col_start);

        if count == KROWS * KCOLS
            || (count > 0
                && context.input_mask().is_valid_weighted_window_unsafe::<KROWS, KCOLS>(
                    flat_base, row_start, col_start, weights_row, weights_col,
                ) == 1)
        {
            // SAFETY: the interior path guarantees all stencil indices
            // are within bounds. Weights buffer has at least KROWS and
            // KCOLS elements respectively.
            let mut out_shift = 0usize;
            for ivar in 0..array_in.nvar {
                let ain_base = flat_base + ivar * in_var_sz;
                let mut acc = 0.0f64;

                for irow in 0..KROWS {
                    let base = ain_base + irow * ncol;
                    let mut acc_col = 0.0f64;
                    for icol in 0..KCOLS {
                        acc_col +=
                            *array_in.data.get_unchecked(base + icol)
                            * *weights_col.get_unchecked(
                                KCOLS - 1 - icol);
                    }
                    acc += *weights_row.get_unchecked(KROWS - 1 - irow)
                        * acc_col;
                }
                *array_out.data.get_unchecked_mut(
                    out_idx + out_shift) = V::from(acc);
                out_shift += out_var_sz;
            }
            context.output_mask().set_value(out_idx, 1);
        } else {
            write_nodata_all_vars(
                array_out, out_idx, nodata,
                context.output_mask(), array_in.nvar, out_var_sz,
            );
        }
    }

    // =========================================================================
    // interpolate_masked_partial
    // =========================================================================

    /// Unsafe variant for interpolate_masked_partial
    /// See the safe variant for details
    #[inline(always)]
    unsafe fn interpolate_masked_partial_unsafe<T, V, IC>(
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
    ) where
        T: Copy + std::ops::Mul<f64, Output = f64> + Into<f64> + PartialEq,
        V: Copy + From<f64> + PartialEq,
        IC: GxArrayViewInterpolationContextTrait,
    {
        let half_rows  = (KROWS / 2) as i64;
        let half_cols  = (KCOLS / 2) as i64;
        let ncol       = array_in.ncol;
        let in_var_sz  = array_in.var_size;
        let out_var_sz = array_out.var_size;

        // Pre-validation: bounds + mask.
        // SAFETY: weight indices are const-generic derived, always valid.
        for irow in 0..KROWS {
            if *weights_row.get_unchecked(KROWS - 1 - irow) == 0.0 {
                continue;
            }
            let r = row_c + irow as i64 - half_rows;
            for icol in 0..KCOLS {
                if *weights_col.get_unchecked(KCOLS - 1 - icol) == 0.0 {
                    continue;
                }
                if r < 0 || r >= array_in.nrow_i64 {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        context.output_mask(), array_in.nvar, out_var_sz,
                    );
                    return;
                }
                let c = col_c + icol as i64 - half_cols;
                if c < 0 || c >= array_in.ncol_i64 {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        context.output_mask(), array_in.nvar, out_var_sz,
                    );
                    return;
                }
                let flat = r as usize * ncol + c as usize;
                if context.input_mask().is_valid(flat) == 0 {
                    write_nodata_all_vars(
                        array_out, out_idx, nodata,
                        context.output_mask(), array_in.nvar, out_var_sz,
                    );
                    return;
                }
            }
        }

        // Convolution — bounds and mask guaranteed by pre-pass.
        let mut in_shift  = 0usize;
        let mut out_shift = 0usize;

        // SAFETY: the pre-validation above guarantees all accessed
        // positions with non-zero weight are within bounds and valid.
        for _ivar in 0..array_in.nvar {
            let mut acc = 0.0f64;
            for irow in 0..KROWS {
                let r = row_c + irow as i64 - half_rows;
                if r < 0 || r >= array_in.nrow_i64 {
                    continue;
                }
                let ru = r as usize;
                let mut acc_col = 0.0f64;
                for icol in 0..KCOLS {
                    let c = col_c + icol as i64 - half_cols;
                    if c < 0 || c >= array_in.ncol_i64 {
                        continue;
                    }
                    let flat = in_shift + ru * ncol + c as usize;
                    acc_col +=
                        *array_in.data.get_unchecked(flat)
                        * *weights_col.get_unchecked(KCOLS - 1 - icol);
                }
                acc += *weights_row.get_unchecked(KROWS - 1 - irow)
                    * acc_col;
            }
            *array_out.data.get_unchecked_mut(out_idx + out_shift) =
                V::from(acc);
            in_shift  += in_var_sz;
            out_shift += out_var_sz;
        }
        context.output_mask().set_value(out_idx, 1);
    }
    
    // =========================================================================
    // compute_weights  (required)
    // =========================================================================

    /// Computes the `KROWS` (or `KCOLS`) kernel weights for the given
    /// sub-pixel offset `x` and writes them into `weights`.
    ///
    /// This method is called twice per pixel — once for the row direction and
    /// once for the column direction — by
    /// [`array1_interp2_separable_core`](Self::array1_interp2_separable_core).
    ///
    /// # Parameters
    /// - `x`: sub-pixel offset
    /// - `weights`: output slice of length `KROWS` (or `KCOLS`); the
    ///   implementor writes one weight per stencil position.
    fn compute_weights(&self, x: f64, weights: &mut [f64]);
    
    // =========================================================================
    // array1_interp2_separable_core
    // =========================================================================

    /// Performs 2D interpolation at a floating-point target position using the
    /// separable kernel defined by [`compute_weights`](Self::compute_weights).
    ///
    /// This method implements the full interpolation pipeline:
    ///
    /// 1. Computes the nearest integer centre `(kernel_center_row,
    ///    kernel_center_col)` from the target position.
    /// 2. If bounds checking is enabled (`IC::BoundsCheck::do_check()`):
    ///    - **Interior path** — the entire `KROWS × KCOLS` stencil fits inside
    ///      the array: calls `interpolate_*_unchecked`.
    ///    - **Border path** — the centre is inside the array but the stencil
    ///      may cross a border: calls `interpolate_*_partial`.
    ///    - **Outside path** — the centre is outside the array: writes `nodata`
    ///      and sets the output mask to `0`.
    /// 3. If bounds checking is disabled: always calls `interpolate_*_unchecked`.
    ///
    /// The mask branch (`interpolate_masked_*` vs `interpolate_nomask_*`) is
    /// selected at compile time via `context.input_mask().is_enabled()`.
    ///
    /// Both branches compile to zero-cost monomorphic code: the bounds-check
    /// and mask-check conditionals are eliminated by the compiler when the
    /// corresponding strategy types implement constant-folding methods.
    ///
    /// # Parameters
    /// - `weights_buffer`: pre-allocated scratch buffer of length
    ///   `KROWS + KCOLS`; split at `KROWS` into row and column weight slices.
    /// - `target_row_pos`: target row in floating-point coordinates.
    /// - `target_col_pos`: target column in floating-point coordinates.
    /// - `out_idx`: flat output index (same across all variable planes).
    /// - `array_in`: input data array `[nvar, nrow, ncol]`, flattened.
    /// - `array_out`: output array (same layout), written in place.
    /// - `nodata_out`: sentinel value written for invalid output pixels.
    /// - `context`: interpolation context (masks + bounds check strategy).
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(String)` with an error message on failure.
    ///
    /// # Type parameters
    /// - `T`: input element type.
    /// - `V`: output element type.
    /// - `IC`: context type implementing
    ///   [`GxArrayViewInterpolationContextTrait`].
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
        let half_rows = (KROWS / 2) as i64;
        let half_cols = (KCOLS / 2) as i64;
        
        // Nearest integer centre for the interpolation stencil.
        let kernel_center_row: i64 = (target_row_pos + 0.5).floor() as i64;
        let kernel_center_col: i64 = (target_col_pos + 0.5).floor() as i64;
 
        let out_var_sz = array_out.var_size;
        
        // Initialise the output mask as valid; the interpolation variants will
        // overwrite it to 0 if the pixel turns out to be invalid.
        context.output_mask().set_value(out_idx, 1);
        
        if IC::BoundsCheck::do_check() {
            // Interior path: the full stencil fits strictly inside the array.
            // No per-element bounds checking is needed inside the inner loops.
            if (kernel_center_row >= half_rows)
                    && (kernel_center_row < array_in.nrow_i64-half_rows)
                    && (kernel_center_col >= half_cols)
                    && (kernel_center_col < array_in.ncol_i64-half_cols) {
                let rel_row: f64 = target_row_pos - kernel_center_row as f64;
                let rel_col: f64 = target_col_pos - kernel_center_col as f64;
                
                // Create slices to give to weight computation methods
                let (w_row, w_col) =
                    weights_buffer.split_at_mut(KROWS);
                self.compute_weights(rel_row, w_row);
                self.compute_weights(rel_col, w_col);

                if context.input_mask().is_enabled() {
                    // SAFETY : check has been performed in the previous if test
                    unsafe {
                        self.interpolate_masked_unchecked_unsafe(
                            w_row, w_col,
                            array_in, array_out, out_idx,
                            kernel_center_row, kernel_center_col,
                            nodata_out, context,
                        );
                    }
                } else {
                    // SAFETY : check has been performed in the previous if test
                    unsafe {
                        self.interpolate_nomask_unchecked_unsafe(
                            w_row, w_col,
                            array_in, array_out, out_idx,
                            kernel_center_row, kernel_center_col,
                        );
                    }
                }
            }
            // Border path: the centre is inside the array but the stencil may
            // cross one or more borders.
            else if (kernel_center_row >=0)
                    && (kernel_center_row < array_in.nrow_i64)
                    && (kernel_center_col >= 0)
                    && (kernel_center_col < array_in.ncol_i64) {
                let rel_row: f64 = target_row_pos - kernel_center_row as f64;
                let rel_col: f64 = target_col_pos - kernel_center_col as f64;
                
                let (w_row, w_col) =
                    weights_buffer.split_at_mut(KROWS);
                self.compute_weights(rel_row, w_row);
                self.compute_weights(rel_col, w_col);

                if context.input_mask().is_enabled() {
                    // SAFETY : check has been performed in the previous if test
                    // The partial variant ensures that out of bounds position
                    // are not used by checking the kernel weights
                    unsafe {
                        self.interpolate_masked_partial_unsafe(
                            w_row, w_col,
                            array_in, array_out, out_idx,
                            kernel_center_row, kernel_center_col,
                            nodata_out, context,
                        );
                    }
                } else {
                    // SAFETY : check has been performed in the previous if test
                    // The partial variant ensures that out of bounds position
                    // are not used by checking the kernel weights
                    unsafe {
                        self.interpolate_nomask_partial_unsafe(
                            w_row, w_col,
                            array_in, array_out,
                            context.output_mask(), out_idx,
                            kernel_center_row, kernel_center_col,
                            nodata_out,
                        );
                    }
                }
            } else {
                // Outside path: the centre is entirely outside the array.
                write_nodata_all_vars(
                    array_out, out_idx, nodata_out,
                    context.output_mask(), array_in.nvar, out_var_sz,
                );
            }
        } else {
            // No bounds check: the caller guarantees all indices are valid.
            let rel_row: f64 = target_row_pos - kernel_center_row as f64;
            let rel_col: f64 = target_col_pos - kernel_center_col as f64;
            
            let (w_row, w_col) = weights_buffer.split_at_mut(KROWS);
            self.compute_weights(rel_row, w_row);
            self.compute_weights(rel_col, w_col);

            if context.input_mask().is_enabled() {
                unsafe {
                    // SAFETY : check has been performed in the previous if test
                    self.interpolate_masked_unchecked_unsafe(
                        w_row, w_col,
                        array_in, array_out, out_idx,
                        kernel_center_row, kernel_center_col,
                        nodata_out, context,
                    );
                }
            } else {
                unsafe {
                    // SAFETY : check has been performed in the previous if test
                    self.interpolate_nomask_unchecked_unsafe(
                        w_row, w_col,
                        array_in, array_out, out_idx,
                        kernel_center_row, kernel_center_col,
                    );
                }
            }
        }
        Ok(())
    }
}
