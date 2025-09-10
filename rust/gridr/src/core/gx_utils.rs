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
//! # GridR utils definition
//!
//! This module provides utility traits and functions commonly used across the GridR
//! project. It includes numeric conversions, comparison helpers, and approximate equality
//! checks for tuples.
//!
//! # Contents
//! - `GxToF64` trait for converting various numeric types to `f64`.
//! - Generic min/max functions using `PartialOrd`.
//! - Approximate equality check for 2D tuples of numeric types.

/// Trait for converting various numeric types into `f64`.
///
/// This trait provides a unified method to convert values of different numeric types
/// (such as `f64`, `f32`, `usize`, and `u32`) into `f64` for consistent usage in
/// floating-point computations.
///
/// # Implementations
/// - For `f64`, returns the value unchanged.
/// - For `f32`, converts to `f64` via casting.
/// - For unsigned integer types like `usize` and `u32`, converts to `f64` via casting.
///
/// # Example
/// ```
/// let a: f32 = 3.14;
/// let b: f64 = a.to_f64();
/// let c: usize = 10;
/// let d: f64 = c.to_f64();
/// ```
pub trait GxToF64 {
    /// Converts the value to `f64`.
    ///
    /// # Returns
    /// The value converted to `f64`.    
    fn to_f64(self) -> f64;
}

impl GxToF64 for f64 {
    #[inline]
    /// Converts the value to `f64`.
    ///
    /// # Returns
    /// The value converted to `f64`.
    fn to_f64(self) -> f64 {
        self
    }
}

impl GxToF64 for f32 {
    #[inline]
    /// Converts the value to `f64`.
    ///
    /// # Returns
    /// The value converted to `f64`.
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl GxToF64 for usize {
    #[inline]
    /// Converts the value to `f64`.
    ///
    /// # Returns
    /// The value converted to `f64`.
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl GxToF64 for u32 {
    #[inline]
    /// Converts the value to `f64`.
    ///
    /// # Returns
    /// The value converted to `f64`.
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Returns the minimum of two values using `PartialOrd`.
///
/// # Arguments
///
/// * `a` - First value.
/// * `b` - Second value.
///
/// # Returns
///
/// The smaller of `a` and `b`. If both are equal, returns `a`.
///
/// # Notes
///
/// This function works with types that implement `PartialOrd + Copy`, such as `f64`, `f32`, etc.
/// It does not account for `NaN` behavior — comparisons follow standard `PartialOrd` logic.
pub fn min_partial<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

/// Returns the maximum of two values using `PartialOrd`.
///
/// # Arguments
///
/// * `a` - First value.
/// * `b` - Second value.
///
/// # Returns
///
/// The greater of `a` and `b`. If both are equal, returns `a`.
///
/// # Notes
///
/// Like `min_partial`, this does not handle `NaN` specifically and relies on standard `PartialOrd`.
pub fn max_partial<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// Compares two 2D tuples of numeric values for approximate equality.
///
/// Each component of the tuples `(a.0, a.1)` and `(b.0, b.1)` is compared using
/// absolute difference against the provided tolerance `epsilon`.
///
/// # Type Parameters
/// - `T`: A numeric type that can be converted into `f64` through the GxToF64 trait.
///
/// # Parameters
/// - `a`: The first tuple `(x, y)`.
/// - `b`: The second tuple `(x, y)`.
/// - `epsilon`: Maximum allowed absolute difference per component.
///
/// # Returns
/// - `true` if both components differ by less than or equal to `epsilon`.
/// - `false` otherwise.
///
/// # Examples
/// ```rust
/// let a = (1.0_f64, 2.0_f64);
/// let b = (1.000001_f64, 1.999999_f64);
/// assert!(approx_eq_tuple2(a, b, 1e-5));
/// ```
pub fn approx_eq_tuple2<T: GxToF64>(a: (T, T), b: (T, T), epsilon: f64) -> bool {
    (a.0.to_f64() - b.0.to_f64()).abs() <= epsilon &&
    (a.1.to_f64() - b.1.to_f64()).abs() <= epsilon
}



#[cfg(test)]
mod gx_utils_tests {
    use super::{min_partial, max_partial};

    #[test]
    fn test_min_partial_integers() {
        assert_eq!(min_partial(3i32, 7i32), 3);
        assert_eq!(min_partial(42u8, 5u8), 5);
        assert_eq!(min_partial(-10, -10), -10);
    }

    #[test]
    fn test_max_partial_integers() {
        assert_eq!(max_partial(3i32, 7i32), 7);
        assert_eq!(max_partial(42u8, 5u8), 42);
        assert_eq!(max_partial(-10, -10), -10);
    }

    #[test]
    fn test_min_partial_floats() {
        assert_eq!(min_partial(3.5f64, 7.2f64), 3.5);
        assert_eq!(min_partial(10.0, -4.0), -4.0);
        assert_eq!(min_partial(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_max_partial_floats() {
        assert_eq!(max_partial(3.5f64, 7.2f64), 7.2);
        assert_eq!(max_partial(10.0, -4.0), 10.0);
        assert_eq!(max_partial(0.0, 0.0), 0.0);
    }

}