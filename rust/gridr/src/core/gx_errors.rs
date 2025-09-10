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
//! GridR core error's definitions
use thiserror::Error;

/// Represents the various error types that can occur.
///
/// This enum is designed to capture a range of logical and structural issues that may arise,
/// such as invalid combinations of optional values, shape mismatches, or missing data. Each variant
/// is tailored to describe a specific class of error with contextual information to aid in debugging.
///
/// # Variants
///
/// - `OptionsMismatch`: Indicates that two optional fields are inconsistent — one is `Some` while the other is `None`. 
///   Both must be either `Some` or `None`. Contains the names of the mismatched fields.
///
/// - `ExclusiveOptionsMismatch`: Signals that two options are mutually exclusive, but both are present. 
///   Only one should be `Some`. Contains the names of the conflicting fields.
///
/// - `ShapesMismatch`: Raised when two arrays or data structures are expected to have the same shape but do not. 
///   Contains the names of the mismatched shape sources.
///
/// - `UnexpectedNone`: Triggered when a value was expected to be present (`Some`) but was actually `None`. 
///   Contains the name of the field or context where this occurred.
///
/// - `ZeroResolution`: Indicates that a resolution factor is zero or negative, while a strictly positive value was expected.
///
/// - `InsufficientGridCoverage`: Indicates that the grid does not have enough data to be interpolated
///
/// - `WindowOutOfBounds`: Raised if a window is out of bounds relative to the dimensions of the associated array.
///
/// - `ErrMessage`: A generic catch-all error containing a descriptive message.
#[derive(Debug, Error, PartialEq)]
pub enum GxError {
    /// Two options must either both be `Some` or both be `None`.
    ///
    /// This error occurs when an internal consistency requires two related fields
    /// or parameters to be jointly specified or jointly omitted, but they are in mismatched states.
    ///
    /// # Parameters
    /// - `field1`: Name of the first option field.
    /// - `field2`: Name of the second option field.
    #[error("Both fields `{field1}` and `{field2}` must be Some or both None.")]
    OptionsMismatch {
        /// Name of the first option field.
        field1: &'static str,
        /// Name of the second option field.
        field2: &'static str,
    },

    /// Two options must be mutually exclusive.
    ///
    /// This error occurs when two optional values are provided simultaneously, but the
    /// application logic dictates that only one should be specified at a time.
    ///
    /// # Parameters
    /// - `field1`: Name of the first conflicting option.
    /// - `field2`: Name of the second conflicting option.
    #[error("Fields `{field1}` and `{field2}` must be mutually exclusive.")]
    ExclusiveOptionsMismatch {
        /// Name of the first conflicting option.
        field1: &'static str,
        /// Name of the second conflicting option.
        field2: &'static str,
    },

    /// Two arrays or structures are expected to have identical shapes but do not.
    ///
    /// This typically occurs when performing element-wise operations or broadcasting
    /// assumptions that rely on shape compatibility.
    ///
    /// # Parameters
    /// - `field1`: Name of the first array or structure.
    /// - `field2`: Name of the second array or structure.
    #[error("Shapes mismatch between `{field1}` and `{field2}`.")]
    ShapesMismatch {
        /// Name of the first array or structure.
        field1: &'static str,
        /// Name of the second array or structure.
        field2: &'static str,
    },

    /// A required value was unexpectedly `None`.
    ///
    /// This error arises when a logic branch assumed the presence of a value,
    /// but the value was missing, often due to an earlier omission or precondition failure.
    ///
    /// # Parameters
    /// - `field1`: The field, context, or computation where the `None` occurred.
    #[error("Unexpected None encountered in `{field1}`.")]
    UnexpectedNone {
        /// The field, context, or computation where the `None` occurred.
        field1: &'static str,
    },

    /// A resolution or scaling factor was zero or negative.
    ///
    /// This typically indicates a configuration or computation error involving
    /// spatial resolution or scaling factors, which must be strictly positive
    /// for correct geometric transformations.
    #[error("Resolution must be non zero.")]
    ZeroResolution,

    /// The resolution is greater than 1 and there is not enough data to 
    /// interpolate the grid
    ///
    /// This typically indicates a too small grid that cant be interpolated for
    /// resolution greater than 1 : that's the case if there is not at least 
    /// two values along an axis for interpolation.
    #[error("InsufficientGridCoverage for `{field1}`.")]
    InsufficientGridCoverage{
        /// String identifier of the grid component that originates the error ('rows' or 'columns')
        field1: &'static str,
    },

    /// A window is out of bounds relative to the dimensions of the associated array.
    ///
    /// This error occurs when a [`GxArrayWindow`] references rows or columns
    /// that exceed the shape of the target array, typically during subsetting,
    /// view extraction, or window-based operations.
    ///
    /// # Parameters
    /// - `context`: The name of the array or structure being validated (e.g. `"input_grid"`).
    /// - `start_row`: First row index of the window.
    /// - `end_row`: Last row index of the window.
    /// - `start_col`: First column index of the window.
    /// - `end_col`: Last column index of the window.
    /// - `nrows`: Total number of rows of the data being validated.
    /// - `ncols`: Total number of columns of the data being validated.
    ///
    /// # Example
    /// Trying to apply a window like (start_row=4, end_row=10) on a 5-row array will trigger this error.
    #[error(
        "Window is out of bounds for `{context}`: \
         window=({start_row}:{end_row}, {start_col}:{end_col}), \
         shape=({nrows} rows × {ncols} cols)"
    )]
    WindowOutOfBounds {
        /// String identifier of the array or structure being validated (e.g. `"input_grid"`)
        context: &'static str,
        /// First row index of the window
        start_row: usize,
        /// Last row index of the window
        end_row: usize,
        /// First column index of the window
        start_col: usize,
        /// Last column index of the window
        end_col: usize,
        /// Total number of rows of the data being validated
        nrows: usize,
        /// Total number of columns of the data being validated
        ncols: usize,
    },

    /// A generic error containing a descriptive message.
    ///
    /// This variant is intended for use when no other variant applies,
    /// and a human-readable explanation is appropriate.
    ///
    /// # Parameters
    /// - `String`: A message describing the nature of the error.
    #[error("{0}")]
    ErrMessage(String),
}
