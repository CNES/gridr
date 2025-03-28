#![warn(missing_docs)]
//! # GxArray Module
//!
//! The `GxArray` module provides a set of structures for handling multi-dimensional arrays
//! stored in contiguous 1D slices. It enables efficient access, slicing, and modification 
//! of array data, while maintaining a structured view of dimensions.
//!
//! ## Overview
//!
//! This module defines four key components:
//!
//! - [`GxArrayShape`]: A trait that standardizes access to array dimensions (`nvar`, `nrow`, `ncol`).
//! - [`GxArrayWindow`]: Represents a sub-region (window) within an array, allowing for efficient operations on subsets of data.
//! - [`GxArrayView`]: An immutable view over an array, providing safe and read-only access to data.
//! - [`GxArrayViewMut`]: A mutable counterpart of `GxArrayView`, enabling in-place modifications of array data.
//!
//! ## Usage
//!
//! The module is designed for scenarios where large datasets need to be processed efficiently
//! without unnecessary memory allocations. It allows working with multi-dimensional data 
//! in a linear memory layout, optimizing cache locality and performance.
//!
//!

/// Trait defining the shape of a multi-dimensional array.
///
/// This trait allows extracting the dimensions of an array stored contiguously in memory,  
/// with a variable number of variables, rows, and columns.  
/// It is used to standardize access to the dimensions of structures handling data arrays.
///
/// # Methods
///
/// - [`Self::nvar()`]: Returns the number of variables (1st axis of the structure).
/// - [`Self::nrow()`]: Returns the number of rows (2nd axis of the structure).
/// - [`Self::ncol()`]: Returns the number of columns (3rd axis of the structure).


use crate::core::gx_errors::{GxError};

pub trait GxArrayShape {
    /// Returns the number of variables (1st axis of the structure).
    fn nvar(&self) -> usize;
    /// Returns the number of rows (2nd axis of the structure).
    fn nrow(&self) -> usize;
    /// Returns the number of columns (3rd axis of the structure).
    fn ncol(&self) -> usize;
}

/// A mutable view of a multi-dimensional array stored in a contiguous 1D slice.
///
/// This structure provides an efficient way to handle multi-dimensional arrays stored  
/// in a flat memory layout, allowing direct access to data while maintaining shape information.
/// It enables modifying the underlying data while ensuring correct indexing.
///
/// # Fields
///
/// - `data`: A mutable reference to a 1D slice storing the array data contiguously.
/// - `nvar`: The number of variables (1st axis).
/// - `nrow`: The number of rows (2nd axis).
/// - `ncol`: The number of columns (3rd axis).
///
/// # Trait Implementations
///
/// Implements [`GxArrayShape`] to expose the shape of the array.
///
/// # Example
///
/// ```rust
/// let mut buffer = vec![0.0; 2 * 3 * 4]; // (nvar=2, nrow=3, ncol=4)
/// let mut array_view = GxArrayViewMut {
///     data: &mut buffer,
///     nvar: 2,
///     nrow: 3,
///     ncol: 4,
/// };
///
/// assert_eq!(array_view.nvar(), 2);
/// assert_eq!(array_view.nrow(), 3);
/// assert_eq!(array_view.ncol(), 4);
/// ```
///
#[derive(Debug)]
pub struct GxArrayViewMut<'a, T> {
    /// Mutable slice storing the data contiguously.
    pub data: &'a mut [T],

    /// Number of variables (1st axis).
    pub nvar: usize,

    /// Number of rows (2nd axis).
    pub nrow: usize,

    /// Number of columns (3rd axis).
    pub ncol: usize,
    
    /// Number of rows as i64
    pub nrow_i64: i64,
    
    /// Number of cols as i64
    pub ncol_i64: i64,
}

impl<'a, T> GxArrayViewMut<'a, T> {
    /// Creates a new mutable array view from a contiguous data slice.
    ///
    /// # Arguments
    ///
    /// - `data`: A reference to a contiguous slice of elements.
    /// - `nvar`: Number of variables (first axis).
    /// - `nrow`: Number of rows (second axis).
    /// - `ncol`: Number of columns (third axis).
    ///
    /// # Returns
    ///
    /// A new `GxArrayViewMut` instance that provides a read-only view over the provided data.
    pub fn new(data: &'a mut [T], nvar: usize, nrow: usize, ncol: usize) -> Self {
        Self { data:data, nvar:nvar, nrow:nrow, ncol:ncol, nrow_i64:nrow as i64, ncol_i64:ncol as i64 }
    }
}

impl<T> GxArrayShape for GxArrayViewMut<'_, T> {
    /// Returns the number of variables (1st axis of the array).
    fn nvar(&self) -> usize {
        self.nvar
    }

    /// Returns the number of rows (2nd axis of the array).
    fn nrow(&self) -> usize {
        self.nrow
    }

    /// Returns the number of columns (3rd axis of the array).
    fn ncol(&self) -> usize {
        self.ncol
    }
}


/// An immutable view of a multi-dimensional array stored in a contiguous 1D slice.
///
/// This structure provides a lightweight way to access multi-dimensional data without  
/// requiring additional allocations. It maintains the shape information while allowing  
/// read-only access to the underlying data.
///
/// # Fields
///
/// - `data`: An immutable reference to a 1D slice storing the array data contiguously.
/// - `nvar`: The number of variables (1st axis).
/// - `nrow`: The number of rows (2nd axis).
/// - `ncol`: The number of columns (3rd axis).
///
/// # Trait Implementations
///
/// Implements [`GxArrayShape`] to expose the shape of the array.
///
/// # Example
///
/// ```rust
/// let buffer = vec![0.0; 2 * 3 * 4]; // (nvar=2, nrow=3, ncol=4)
/// let array_view = GxArrayView::new(&buffer, 2, 3, 4);
///
/// assert_eq!(array_view.nvar(), 2);
/// assert_eq!(array_view.nrow(), 3);
/// assert_eq!(array_view.ncol(), 4);
/// ```
///
#[derive(Debug)]
pub struct GxArrayView<'a, T> {
    /// Immutable slice storing the data contiguously.
    pub data: &'a [T],

    /// Number of variables (1st axis).
    pub nvar: usize,

    /// Number of rows (2nd axis).
    pub nrow: usize,

    /// Number of columns (3rd axis).
    pub ncol: usize,
    
    /// Number of rows as i64
    pub nrow_i64: i64,
    
    /// Number of cols as i64
    pub ncol_i64: i64,
}

impl<'a, T> GxArrayView<'a, T> {
    /// Creates a new immutable array view from a contiguous data slice.
    ///
    /// # Arguments
    ///
    /// - `data`: A reference to a contiguous slice of elements.
    /// - `nvar`: Number of variables (first axis).
    /// - `nrow`: Number of rows (second axis).
    /// - `ncol`: Number of columns (third axis).
    ///
    /// # Returns
    ///
    /// A new `GxArrayView` instance that provides a read-only view over the provided data.
    pub fn new(data: &'a [T], nvar: usize, nrow: usize, ncol: usize) -> Self {
        Self { data:data, nvar:nvar, nrow:nrow, ncol:ncol, nrow_i64:nrow as i64, ncol_i64:ncol as i64 }
    }
}

impl<T> GxArrayShape for GxArrayView<'_, T> {
    /// Returns the number of variables (1st axis of the array).
    fn nvar(&self) -> usize {
        self.nvar
    }

    /// Returns the number of rows (2nd axis of the array).
    fn nrow(&self) -> usize {
        self.nrow
    }

    /// Returns the number of columns (3rd axis of the array).
    fn ncol(&self) -> usize {
        self.ncol
    }
}


/// Represents a rectangular window within a multi-dimensional array.
///
/// This structure defines a sub-region of an array by specifying the start and  
/// end indices for both rows and columns. It is useful for operations that  
/// need to work on a specific subset of an array.
///
/// # Fields
///
/// - `start_row`: The starting row index (inclusive).
/// - `end_row`: The ending row index (inclusive).
/// - `start_col`: The starting column index (inclusive).
/// - `end_col`: The ending column index (inclusive).
///
/// # Example
///
/// ```rust
/// let window = GxArrayWindow {
///     start_row: 1,
///     end_row: 3,
///     start_col: 2,
///     end_col: 4,
/// };
///
/// assert_eq!(window.start_row, 1);
/// assert_eq!(window.end_col, 4);
/// ```
///
#[derive(Clone, Debug)]
pub struct GxArrayWindow {
    /// The starting row index (inclusive).
    pub start_row: usize,

    /// The ending row index (inclusive).
    pub end_row: usize,

    /// The starting column index (inclusive).
    pub start_col: usize,

    /// The ending column index (inclusive).
    pub end_col: usize,
}

impl GxArrayWindow {
    
    /// Returns the height (number of rows) of the window.
    pub fn height(&self) -> usize {
        self.end_row - self.start_row + 1
    }

    /// Returns the width (number of columns) of the window.
    pub fn width(&self) -> usize {
        self.end_col - self.start_col + 1
    }

    /// Returns the number of elements in the window.
    pub fn size(&self) -> usize {
        self.height() * self.width()
    }
    
    /// This method calculates and returns the required windows in a lower resolute coordinate system 
    /// to cover the full resolute target window given by the 'win' parameter.
    /// If the `resolution` is equal to 1 in both directions, the required window is directly equal to the input window.
    ///
    /// Args:
    ///     resolution : The grid's resolution factors for rows and columns (in that order).
    ///     win: The target full resolution window
    ///
    /// Returns:
    ///     A result containing the two windows: the source window and the relative window,
    ///     or an error if the resolution is invalid.
    pub fn get_wrapping_window_for_resolution(
        resolution: (usize, usize),
        win: &GxArrayWindow,
    ) -> Result<(GxArrayWindow, GxArrayWindow), GxError> {
        // Validate the resolution to avoid division by zero
        if resolution.0 == 0 || resolution.1 == 0 {
            return Err(GxError::ZeroResolution);
        }

        // Calculate the source window in the input grid
        let win_src = GxArrayWindow {
            start_row: win.start_row / resolution.0,
            start_col: win.start_col / resolution.1,
            end_row: win.end_row.div_ceil(resolution.0),
            end_col: win.end_col.div_ceil(resolution.1),
        };

        // Calculate the relative window inside the source window
        let win_rel = GxArrayWindow {
            start_row: win.start_row % resolution.0,
            start_col: win.start_col % resolution.1,
            end_row: win.start_row + (win.end_row - win.start_row),
            end_col: win.start_col + (win.end_col - win.start_col),
        };

        Ok((win_src, win_rel))
    }
    
    /// Validates the window against an array-like structure.
    ///
    /// Ensures that the window's row and column indices are within valid bounds  
    /// of the given array. The array must implement [`GxArrayShape`].
    ///
    /// # Arguments
    ///
    /// - `array`: A reference to an object implementing [`GxArrayShape`], representing  
    ///   the array whose dimensions will be checked.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the window is valid.
    /// - `Err(String)` if the window is out of bounds.
    ///
    /// # Errors
    ///
    /// This function returns an error in the following cases:
    /// - `start_row > end_row` or `end_row >= array.nrow()`
    /// - `start_col > end_col` or `end_col >= array.ncol()`
    ///
    /// # Example
    ///
    /// ```rust
    /// let array_shape = SomeArray { nvar: 1, nrow: 5, ncol: 5 }; // Implements GxArrayShape
    /// let window = GxArrayWindow { start_row: 1, end_row: 3, start_col: 0, end_col: 4 };
    ///
    /// assert!(window.validate_with_array(&array_shape).is_ok());
    ///
    /// let invalid_window = GxArrayWindow { start_row: 4, end_row: 6, start_col: 0, end_col: 4 };
    /// assert!(invalid_window.validate_with_array(&array_shape).is_err());
    /// ```
    ///
    pub fn validate_with_array<T: GxArrayShape>(&self, array: &T) -> Result<(), String> {
        if self.start_row > self.end_row || self.end_row >= array.nrow() {
            return Err(format!(
                "Invalid row indices: start_row={}, end_row={}, max_row={}",
                self.start_row, self.end_row, array.nrow() - 1
            ));
        }
        if self.start_col > self.end_col || self.end_col >= array.ncol() {
            return Err(format!(
                "Invalid column indices: start_col={}, end_col={}, max_col={}",
                self.start_col, self.end_col, array.ncol() - 1
            ));
        }
        Ok(())
    }
}

