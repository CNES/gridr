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
//! Crate doc
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};

/// Replaces values in a 1D array based on a conditional value or an optional condition array.
///
/// This function modifies an input slice in place by replacing elements according to the given conditions.
/// If an optional condition array (`array_cond`) and a corresponding conditional value (`array_cond_val`) 
/// are provided, replacement is performed based on the condition array. Otherwise, elements are replaced 
/// based on their direct comparison with `val_cond`.
///
/// # Parameters
/// - `array`: A mutable reference to a `GxArrayViewMut<T>`, which will be modified in place.
/// - `val_cond`: The value to be replaced when no `array_cond` is provided.
/// - `val_true`: The value that replaces elements matching the condition.
/// - `val_false`: An optional value used when the condition is not met. If `None`, the existing array value is preserved.
/// - `array_cond`: An optional reference to a condition array (`GxArrayView<C>`).
/// - `array_cond_val`: The value to match in `array_cond` when provided.
///
/// # Returns
/// - `Ok(())` if the operation is successful.
/// - `Err(String)` if `array` and `array_cond` have different sizes or if only one of `array_cond` or `array_cond_val` is provided.
///
/// # Behavior
/// - If `array_cond` is provided, values in `array` are updated where `array_cond` matches `array_cond_val`.
/// - If `array_cond` is not provided, values in `array` equal to `val_cond` are replaced.
/// - If `val_false` is provided, unmatched values are set to `val_false`. Otherwise, they remain unchanged.
///
/// # Example Usage
/// ```rust
/// let mut array = GxArrayViewMut::from(vec![1, 2, 3, 4]);
/// let condition = GxArrayView::from(vec![true, false, true, false]);
/// array1_replace(&mut array, 2, 99, Some(0), Some(&condition), Some(true));
/// // Result: array = [99, 2, 99, 4]
/// ```
///
/// # Performance Considerations
/// - Uses efficient iteration via `.iter_mut()` to minimize overhead.
/// - The function is marked as `#[inline(always)]` to encourage compiler optimization.
#[inline(always)]
pub fn array1_replace<T, C>(
        array: &mut GxArrayViewMut<'_, T>,
        val_cond: T,
        val_true: T,
        val_false: Option<T>,
        array_cond: Option<&GxArrayView<'_, C>>,
        array_cond_val: Option<C>,
        ) -> Result<(), String>
where
    T: Copy + PartialEq + Default,
    C: Copy + PartialEq,
{
    // Check that both array_cond and array_cond_val are given if one is given
    if array_cond.is_some() != array_cond_val.is_some() {
        return Err("Both array_cond and array_cond_val must be provided together or omitted together".to_string());
    }

    // Check if val_false is provided and set value to use for both cases
    let use_val_false = val_false.is_some();
    let default_false = val_false.unwrap_or_else(|| T::default());

    // Case of array_cond is provided
    if let Some(cond_array) = array_cond {
        if array.data.len() != cond_array.data.len() {
            return Err("Both array and array_cond must have same size.".to_string());
        }
        // Get array_cond_val value
        let cond_val = array_cond_val.unwrap();
        
        if use_val_false {
            // Browse both arrays
            for (elem, &elem_cond) in array.data.iter_mut().zip(cond_array.data.iter()) {
                *elem = if elem_cond == cond_val { val_true } else { default_false };
            }
        } else {
            for (elem, &elem_cond) in array.data.iter_mut().zip(cond_array.data.iter()) {
                if elem_cond == cond_val {
                    *elem = val_true;
                }
            }
        }
    }
    // Case of array_cond is omitted
    else {
        if use_val_false {
            for elem in array.data.iter_mut() {
                *elem = if *elem == val_cond { val_true } else { default_false };
            }
        } else {
            for elem in array.data.iter_mut() {
                if *elem == val_cond {
                    *elem = val_true;
                }
            }
        }       
    }
    Ok(())
}

/// Replaces elements in a 1D array based on a conditional value within a specified window.
///
/// This function operates on a mutable slice representing a 2D grid stored in a 1D array.
/// It modifies elements inside the window defined by `win` based on a conditional value.
/// 
/// # Type Parameters
/// - `T`: The element type of the main array. Must implement `Copy` and `PartialEq`.
/// - `C`: The element type of the optional condition array. Must implement `Copy` and `PartialEq`.
///
/// # Parameters
/// - `array`: A mutable reference to a `GxArrayViewMut<T>`, which will be modified in place.
/// - `win`: A `GxArrayWindow` defining the sub-region of `array` where modifications will be applied.
/// - `val_cond`: The value to be replaced when `array_cond` is not provided.
/// - `val_true`: The value that replaces elements matching the condition.
/// - `val_false`: An optional value used when the condition is not met. If `None`, the existing array value is preserved.
/// - `array_cond`: An optional reference to a condition array (`GxArrayView<C>`).
/// - `array_cond_val`: The value to match in `array_cond` when provided.
///
/// # Returns
/// - `Ok(())` if the operation is successful.
/// - `Err(String)` if `array` and `array_cond` have different sizes, 
///   if only one of `array_cond` or `array_cond_val` is provided, 
///   or if `win` is invalid for `array`.
///
/// # Behavior
/// - The function modifies only the elements within the window `win`.
/// - If `array_cond` is provided, values in `array` are updated where `array_cond` matches `array_cond_val`.
/// - If `array_cond` is not provided, values in `array` equal to `val_cond` are replaced.
/// - If `val_false` is provided, unmatched values are set to `val_false`. Otherwise, they remain unchanged.
///
/// # Example Usage
/// ```rust
/// let mut array = GxArrayViewMut::from(vec![
///     1, 2, 3, 4,
///     5, 6, 7, 8,
///     9, 10, 11, 12
/// ]);
/// let win = GxArrayWindow { start_row: 1, end_row: 2, start_col: 1, end_col: 2 };
/// let condition = GxArrayView::from(vec![
///     false, true, false, false,
///     false, true, true, false,
///     false, false, false, false
/// ]);
///
/// array1_replace_win2(&mut array, &win, 2, 99, Some(0), Some(&condition), Some(true));
/// // Result: array is updated within the window according to the condition.
/// ```
#[inline(always)]
pub fn array1_replace_win2<T, C>(
        array: &mut GxArrayViewMut<'_, T>,
        win: &GxArrayWindow,
        val_cond: T,
        val_true: T,
        val_false: Option<T>,
        array_cond: Option<&GxArrayView<'_, C>>,
        array_cond_val: Option<C>,
        ) -> Result<(), String>
where
    T: Copy + PartialEq + Default,
    C: Copy + PartialEq,
{
    // Check that both array_cond and array_cond_val are given if one is given
    if array_cond.is_some() != array_cond_val.is_some() {
        return Err("Both array_cond and array_cond_val must be provided together or omitted together".to_string());
    }
    
    if let Err(e) = win.validate_with_array(array) {
        return Err(e.to_string());
    }

    let mut row_offset = win.start_row * array.ncol;

    // Check if val_false is provided and set value to use for both cases
    let use_val_false = val_false.is_some();
    let default_false = val_false.unwrap_or_else(|| T::default());
    
    // Case of array_cond is provided
    if let Some(cond_array) = array_cond {
        if array.data.len() != cond_array.data.len() {
            return Err("Both array and array_cond must have same size.".to_string());
        }
        // Get array_cond_val value
        let cond_val = array_cond_val.unwrap();
        
        if use_val_false {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    array.data[i] = if cond_array.data[i] == cond_val { val_true } else { default_false };
                }
                row_offset += array.ncol;
            }
        } else {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    if cond_array.data[i] == cond_val {
                        array.data[i] = val_true;
                    }
                }
                row_offset += array.ncol;
            }
        }
    }
    // Case of array_cond is omitted
    else {
        if use_val_false {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    array.data[i] = if array.data[i] == val_cond { val_true } else { default_false };
                }
                row_offset += array.ncol;
            }
        } else {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    if array.data[i] == val_cond {
                        array.data[i] = val_true
                    }
                }
                row_offset += array.ncol;
            }
        }
    }
    Ok(())
}

/// Add a scalar to values in a 1D array based on a conditional value or an optional condition array.
///
/// This function modifies an input slice in place by adding a scalar value to elements that meet
/// specific conditions. The conditions can be based either on direct comparison with a value
/// or on matching elements in an optional condition array.
///
/// # Parameters
/// - `array`: A mutable reference to the array to be modified (`GxArrayViewMut<T>`)
/// - `val_cond`: The value to compare against when no condition array is provided
/// - `val_add`: The value to add to elements that meet the condition
/// - `add_on_true`: Determines whether to add when the condition is true (`true`) or false (`false`)
/// - `array_cond`: An optional reference to a condition array (`GxArrayView<C>`)
/// - `array_cond_val`: The value to match in the condition array when provided
///
/// # Returns
/// - `Ok(())` if the operation succeeds
/// - `Err(String)` if:
///   - The input array and condition array have different sizes
///   - Only one of `array_cond` or `array_cond_val` is provided
///
/// # Behavior
/// - When `add_on_true` is `true`:
///   - With `array_cond`: Adds to elements where `array_cond` matches `array_cond_val`
///   - Without `array_cond`: Adds to elements equal to `val_cond`
/// - When `add_on_true` is `false`:
///   - With `array_cond`: Adds to elements where `array_cond` does not match `array_cond_val`
///   - Without `array_cond`: Adds to elements not equal to `val_cond`
#[inline(always)]
pub fn array1_add<T, C>(
        array: &mut GxArrayViewMut<'_, T>,
        val_cond: T,
        val_add: T,
        add_on_true: bool,
        array_cond: Option<&GxArrayView<'_, C>>,
        array_cond_val: Option<C>,
        ) -> Result<(), String>
where
    T: Copy + PartialEq + Default + std::ops::Add<Output = T>,
    C: Copy + PartialEq,
{
    // Check that both array_cond and array_cond_val are given if one is given
    if array_cond.is_some() != array_cond_val.is_some() {
        return Err("Both array_cond and array_cond_val must be provided together or omitted together".to_string());
    }

    // Case of array_cond is provided
    if let Some(cond_array) = array_cond {
        if array.data.len() != cond_array.data.len() {
            return Err("Both array and array_cond must have same size.".to_string());
        }
        // Get array_cond_val value
        let cond_val = array_cond_val.unwrap();
        
        if add_on_true {
            // Browse both arrays
            for (elem, &elem_cond) in array.data.iter_mut().zip(cond_array.data.iter()) {
                if elem_cond == cond_val {
                    *elem = *elem + val_add;
                }
            }
        } else {
            // Browse both arrays
            for (elem, &elem_cond) in array.data.iter_mut().zip(cond_array.data.iter()) {
                if elem_cond != cond_val {
                    *elem = *elem + val_add;
                }
            }
        }
    }
    // Case of array_cond is omitted
    else {
        if add_on_true {
            for elem in array.data.iter_mut() {
                if *elem == val_cond {
                    *elem = *elem + val_add;
                }
            }
        } else {
            for elem in array.data.iter_mut() {
                if *elem != val_cond {
                    *elem = *elem + val_add;
                }
            }
        }
    }
    Ok(())
}

/// Add a scalar to values in a 1D array within a specified window, based on a conditional value or an optional condition array.
///
/// This function modifies elements in a 2D grid stored as a 1D array, operating only within the window
/// defined by `win`. The modifications are performed based on either direct comparison with a value
/// or on matching elements in an optional condition array.
/// 
/// # Type Parameters
/// - `T`: The element type of the main array. Must implement `Copy` and `PartialEq`.
/// - `C`: The element type of the optional condition array. Must implement `Copy` and `PartialEq`.
///
/// # Parameters
/// - `array`: A mutable reference to a `GxArrayViewMut<T>`, which will be modified in place.
/// - `win`: A `GxArrayWindow` defining the sub-region of `array` where modifications will be applied.
/// - `val_cond`: The value to compare against when no condition array is provided
/// - `val_add`: The value to add to elements that meet the condition
/// - `array_cond`: An optional reference to a condition array (`GxArrayView<C>`).
/// - `array_cond_val`: The value to match in `array_cond` when provided.
///
/// # Returns
/// - `Ok(())` if the operation is successful.
/// - `Err(String)` if `array` and `array_cond` have different sizes, 
///   if only one of `array_cond` or `array_cond_val` is provided, 
///   or if `win` is invalid for `array`.
///
/// # Behavior
/// - When `add_on_true` is `true`:
///   - With `array_cond`: Adds to elements where `array_cond` matches `array_cond_val`
///   - Without `array_cond`: Adds to elements equal to `val_cond`
/// - When `add_on_true` is `false`:
///   - With `array_cond`: Adds to elements where `array_cond` does not match `array_cond_val`
///   - Without `array_cond`: Adds to elements not equal to `val_cond`
#[inline(always)]
pub fn array1_add_win2<T, C>(
        array: &mut GxArrayViewMut<'_, T>,
        win: &GxArrayWindow,
        val_cond: T,
        val_add: T,
        add_on_true: bool,
        array_cond: Option<&GxArrayView<'_, C>>,
        array_cond_val: Option<C>,
        ) -> Result<(), String>
where
    T: Copy + PartialEq + Default + std::ops::Add<Output = T>,
    C: Copy + PartialEq,
{
    // Check that both array_cond and array_cond_val are given if one is given
    if array_cond.is_some() != array_cond_val.is_some() {
        return Err("Both array_cond and array_cond_val must be provided together or omitted together".to_string());
    }
    
    if let Err(e) = win.validate_with_array(array) {
        return Err(e.to_string());
    }

    let mut row_offset = win.start_row * array.ncol;
    
    // Case of array_cond is provided
    if let Some(cond_array) = array_cond {
        if array.data.len() != cond_array.data.len() {
            return Err("Both array and array_cond must have same size.".to_string());
        }
        // Get array_cond_val value
        let cond_val = array_cond_val.unwrap();
        
        if add_on_true {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    if cond_array.data[i] == cond_val {
                        array.data[i] = array.data[i] + val_add;
                    }
                }
                row_offset += array.ncol;
            }
        } else {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    if cond_array.data[i] != cond_val {
                        array.data[i] = array.data[i] + val_add;
                    }
                }
                row_offset += array.ncol;
            }
        }
    }
    // Case of array_cond is omitted
    else {
        if add_on_true {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    if array.data[i] == val_cond {
                        array.data[i] = array.data[i] + val_add;
                    }
                }
                row_offset += array.ncol;
            }
        } else {
            // Main loop
            for _c_row in win.start_row..=win.end_row {
                for c_col in win.start_col..=win.end_col {
                    let i = row_offset + c_col;
                    if array.data[i] != val_cond {
                        array.data[i] = array.data[i] + val_add;
                    }
                }
                row_offset += array.ncol;
            }
        }
    }
    Ok(())
}


/*
/// Function window_iter_1 - 1 variable optimized
/// Get an iterator on an 1 variable array given a window
#[inline(always)]
fn window_iter_1var<'a, T>(
    data: &'a [T],
    _nligne: usize,
    ncol: usize,
    l_start: usize,
    l_end: usize,
    c_start: usize,
    c_end: usize,
) -> impl Iterator<Item = &'a T> {
    (l_start..l_end).flat_map(move |l| {
        (c_start..c_end).map(move |c| {
            let start_idx = (l * ncol) + c;
            &data[start_idx]
        })
    })
}

/// Function window_iter_2 - 2 variable optimized
/// Get an iterator on an 2 variable array given a window
#[inline(always)]
fn window_iter_2var<'a, T>(
    data: &'a [T],
    nligne: usize,
    ncol: usize,
    l_start: usize,
    l_end: usize,
    c_start: usize,
    c_end: usize,
) -> impl Iterator<Item = (&'a T, &'a T)> {
    let stride = nligne * ncol;
    (l_start..l_end).flat_map(move |l| {
        (c_start..c_end).map(move |c| {
            let idx = c + l * ncol;
            (&data[idx], &data[idx + stride])
        })
    })
}
*/


#[cfg(test)]
mod gx_array_utils_tests {
    use super::*;
    use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};
    
    #[test]
    fn test_array1_add_with_condition_array() {
        // Test with condition array
        // Input array - it will be cloned for each test
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                        
        // Condition array
        let cond_data_in = [1, 0, 0,
                             0, 1, 0,
                             0, 0, 0];
        let cond_array_in = GxArrayView::new(&cond_data_in, 1, 3, 3);
        
        let val_add = 10.0;
        
        // expected_array_on_true
        let expected_data_on_true = [ 10.0, 1.0, 2.0,
                        10.0, 30.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                        
        // expected_array_on_false
        let expected_data_on_false = [ 0.0, 11.0, 12.0,
                        20.0, 20.0, 50.0,
                        110.0, 1010.0, 10010.0 ];
        
        // Test add_on_true = true
        { 
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);

            assert!(array1_add::<f64, u8>(
                &mut array_in,
                0., // val_cond (not used in this case)
                val_add,
                true,
                Some(&cond_array_in),
                Some(1)
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_true);
        }
        
        // Test add_on_true = false
        {
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);
            
            assert!(array1_add::<f64, u8>(
                &mut array_in,
                0., // val_cond (not used in this case)
                val_add,
                false,
                Some(&cond_array_in),
                Some(1)
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_false);
        }
    }

    #[test]
    fn test_array1_add_without_condition_array() {
        // Test with condition array
        // Input array - it will be cloned for each test
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        let val_cond = 1.0;        
        let val_add = 10.0;
        
        // expected_array_on_true
        let expected_data_on_true = [ 0.0, 11.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                        
        // expected_array_on_false
        let expected_data_on_false = [ 10.0, 1.0, 12.0,
                        20.0, 30.0, 50.0,
                        110.0, 1010.0, 10010.0 ];
        
        // Test add_on_true = true
        { 
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);

            assert!(array1_add::<f64, u8>(
                &mut array_in,
                val_cond,
                val_add,
                true,
                None,
                None
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_true);
        }
        
        // Test add_on_true = false
        {
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);
            
            assert!(array1_add::<f64, u8>(
                &mut array_in,
                val_cond,
                val_add,
                false,
                None,
                None
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_false);
        }
    }
    
    
    #[test]
    fn test_array1_add_win2_with_condition_array() {
        // Test with condition array
        // Input array - it will be cloned for each test
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                        
        // Condition array
        let cond_data_in = [0, 1, 0,
                             0, 0, 0,
                             0, 0, 0];
        let cond_array_in = GxArrayView::new(&cond_data_in, 1, 3, 3);
        
        let val_add = 10.0;
        
        let win = GxArrayWindow{ start_row: 0, end_row: 0, start_col: 1, end_col: 2 };
        
        // expected_array_on_true
        let expected_data_on_true = [ 0.0, 11.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                        
        // expected_array_on_false
        let expected_data_on_false = [ 0.0, 1.0, 12.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        
        // Test add_on_true = true
        { 
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);

            assert!(array1_add_win2::<f64, u8>(
                &mut array_in,
                &win,
                -9.,
                val_add,
                true,
                Some(&cond_array_in),
                Some(1)
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_true);
        }
        
        // Test add_on_true = false
        {
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);
            
            assert!(array1_add_win2::<f64, u8>(
                &mut array_in,
                &win,
                -9.,
                val_add,
                false,
                Some(&cond_array_in),
                Some(1)
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_false);
        }
    }

    #[test]
    fn test_array1_add_win2_without_condition_array() {
        // Test with condition array
        // Input array - it will be cloned for each test
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        let val_cond = 1.0;        
        let val_add = 10.0;
        
        let win = GxArrayWindow{ start_row: 0, end_row: 0, start_col: 1, end_col: 2 };
        
        // expected_array_on_true
        let expected_data_on_true = [ 0.0, 11.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                        
        // expected_array_on_false
        let expected_data_on_false = [ 0.0, 1.0, 12.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        
        // Test add_on_true = true
        { 
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);

            assert!(array1_add_win2::<f64, u8>(
                &mut array_in,
                &win,
                val_cond,
                val_add,
                true,
                None,
                None
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_true);
        }
        
        // Test add_on_true = false
        {
            let mut data_in_clone = data_in.clone();
            let mut array_in = GxArrayViewMut::new(&mut data_in_clone, 1, 3, 3);
            
            assert!(array1_add_win2::<f64, u8>(
                &mut array_in,
                &win,
                val_cond,
                val_add,
                false,
                None,
                None
            ).is_ok());

            assert_eq!(data_in_clone, expected_data_on_false);
        }
    }
}