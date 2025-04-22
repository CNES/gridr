#![warn(missing_docs)]
//! Crate doc
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use super::gx_array_view_interp::{GxArrayViewInterpolator};

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
    } else if x > 0.0 && x < 1.0 {
        // - instead of abs because we know x is positive
        weights[0] = optimized_bicubic_kernel_weights_compute_func2(-x + 2.0);
        weights[1] = optimized_bicubic_kernel_weights_compute_func1(-x + 1.0);
        weights[2] = optimized_bicubic_kernel_weights_compute_func1(x);
        weights[3] = optimized_bicubic_kernel_weights_compute_func2(x + 1.0);
        weights[4] = 0.0;
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


pub struct GxOptimizedBicubicInterpolator {
    kernel_row_size: usize,
    kernel_col_size: usize,
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
    fn array1_interp2<T, U, V>(
            &self,
            weights_buffer: &mut [f64],
            target_row_pos: f64,
            target_col_pos: f64,
            out_idx: usize,
            array_in: &GxArrayView<'_, T>,
            nodata_out: V,
            array_mask_in: Option<&GxArrayView<'_, U>>,
            array_out: &mut GxArrayViewMut<'_, V>,
            array_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
            ) -> Result<(), String>
    where
        T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
        U: Copy + PartialEq + Into<f64>,
        V: Copy + PartialEq + From<f64>,
        //<U as Mul<f64, Output=f64>>::Output: Add,
    {
        // Get the nearest corresponding index corresponding to the target position 
        let kernel_center_row: i64 = (target_row_pos + 0.5).floor() as i64;
        let kernel_center_col: i64 = (target_col_pos + 0.5).floor() as i64;
        let mut arr_irow;
        let mut arr_icol;
        let mut arr_iflat;
        let mut computed: f64;
        let array_in_var_size = array_in.nrow * array_in.ncol;
        let array_out_var_size = array_out.nrow * array_out.ncol;
        
        // Check that all required data for interpolation is within the input
        // array shape - assuming here the radius is 2.
        // Here we do not need to check borders inside the inner loops.
        // That should be the most common path.
        if (kernel_center_row >=2)
                && (kernel_center_row < array_in.nrow_i64-2)
                && (kernel_center_col >= 2)
                && (kernel_center_col < array_in.ncol_i64-2) {
            let kernel_center_row_as_usize = kernel_center_row as usize;
            let kernel_center_col_as_usize = kernel_center_col as usize;
            let rel_row: f64 = target_row_pos - kernel_center_row as f64;
            let rel_col: f64 = target_col_pos - kernel_center_col as f64;
            let mut computed_col: f64;
            let mut array_in_var_shift;
            let mut array_out_var_shift;
            
            // Create slices to give to weight computation methods
            //let kernel_weights_row_slice: &mut [f64] = &mut weights_buffer[0..5];
            //let kernel_weights_col_slice: &mut [f64] = &mut weights_buffer[5..10];
            let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size);
            
            // from here - pass slice for weight computation
            // slices are used here in order to limit buffer allocation
            optimized_bicubic_kernel_weights(rel_row, kernel_weights_row_slice);
            optimized_bicubic_kernel_weights(rel_col, kernel_weights_col_slice);
            
            // if let Some(mask) = array_mask_out.as_deref_mut() {
            //     mask.data[out_idx] = 1; // Example
            // }            
            // Loop on multipe variables in input array.
            for ivar in 0..array_in.nvar {
                computed = 0.0;
                array_in_var_shift = ivar * array_in_var_size;
                array_out_var_shift = ivar * array_out_var_size;
                
                for irow in 0..=4 {
                    computed_col = 0.0;
                    //arr_irow = (kernel_center_row_as_usize + irow) as i32 - 2;
                    // Given the condition on this branch we are sure that arr_irrow is positiv
                    arr_irow = kernel_center_row + irow - 2;
                    
                    for icol in 0..=4 {
                        //arr_icol = (kernel_center_col_as_usize + icol) as i32 - 2;
                        // Given the condition on this branch we are sure that arr_icol is positiv
                        arr_icol = kernel_center_col + icol - 2;
                        
                        // flat 1d index computation
                        arr_iflat = array_in_var_shift + (arr_irow as usize)* array_in.ncol + (arr_icol as usize);
                        // add current weighted product
                        computed_col += array_in.data[arr_iflat] * kernel_weights_col_slice[4 - icol as usize];
                    }
                    computed += kernel_weights_row_slice[4 - irow as usize] * computed_col;
                }
                // Write interpolated value to output buffer
                array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            }
        }
        // Check the center is within the input array shape
        // The first test has not been passed : meaning at least one border is crossed.
        // We ensure here that the target point is within the input array shape.
        else if (kernel_center_row >=0)
                && (kernel_center_row < array_in.nrow_i64)
                && (kernel_center_col >= 0)
                && (kernel_center_col < array_in.ncol_i64) {
            let kernel_center_row_as_usize = kernel_center_row as usize;
            let kernel_center_col_as_usize = kernel_center_col as usize;
            let rel_row: f64 = target_row_pos - kernel_center_row as f64;
            let rel_col: f64 = target_col_pos - kernel_center_col as f64;
            let mut computed_col: f64;
            let mut array_in_var_shift;
            let mut array_out_var_shift;
            
            // Create slices to give to weight computation methods
            //let kernel_weights_row_slice: &mut [f64] = &mut weights_buffer[0..5];
            //let kernel_weights_col_slice: &mut [f64] = &mut weights_buffer[5..10];
            let (kernel_weights_row_slice, kernel_weights_col_slice) = weights_buffer.split_at_mut(self.kernel_row_size);
            
            // from here - pass slice for weight computation
            // slices are used here in order to limit buffer allocation
            optimized_bicubic_kernel_weights(rel_row, kernel_weights_row_slice);
            optimized_bicubic_kernel_weights(rel_col, kernel_weights_col_slice);
            
            // if let Some(mask) = array_mask_out.as_deref_mut() {
            //     mask.data[out_idx] = 1; // Example
            // }
            
            
            // Loop on multipe variables in input array.
            for ivar in 0..array_in.nvar {
                computed = 0.0;
                array_in_var_shift = ivar * array_in_var_size;
                array_out_var_shift = ivar * array_out_var_size;
                
                for irow in 0..=4 {
                    computed_col = 0.0;
                    //arr_irow = (kernel_center_row_as_usize + irow) as i32 - 2;
                    arr_irow = kernel_center_row + irow - 2;
                    //arr_irow = arr_irow.clamp(0, array_in.nrow_i64 - 1);

                    // Check the data is available otherwise we do nothing.
                    // This is equivalent than adding zero to the computed variable.
                    if (arr_irow >= 0) && (arr_irow <= array_in.nrow_i64 - 1) {
                    
                        for icol in 0..=4 {
                            //arr_icol = (kernel_center_col_as_usize + icol) as i32 - 2;
                            arr_icol = kernel_center_col + icol - 2;
                            //arr_icol = arr_icol.clamp(0, array_in.ncol_i64 - 1);

                            if (arr_icol >= 0) && (arr_icol <= array_in.ncol_i64 - 1) {
                                // flat 1d index computation
                                arr_iflat = array_in_var_shift + (arr_irow as usize)* array_in.ncol + (arr_icol as usize);
                                // add current weighted product
                                computed_col += array_in.data[arr_iflat] * kernel_weights_col_slice[4 - icol as usize];
                            }
                        }
                        computed += kernel_weights_row_slice[4 - irow as usize] * computed_col;
                    }
                }
                // Write interpolated value to output buffer
                array_out.data[out_idx + array_out_var_shift] = V::from(computed);
            }
        } else {
            for ivar in 0..array_in.nvar {                
                // Write nodata value to output buffer
                array_out.data[out_idx + ivar * array_out_var_size] = nodata_out;
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod gx_optimized_bicubic_kernel_tests {
    use super::*;
    
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
        
        // Test idendity
        // Expect : 10.
        let mut x = 2.;
        let mut y = 2.;
        let mut out_idx = 1;
        let _ = interp.array1_interp2::<f64, i8, f64>(&mut weights, y, x, out_idx, &array_in, 0., None, &mut array_out, &mut None);
        assert_eq!(array_out.data, [-9., 10., -9.]);
        
        // Target x = 1.75 y = 2.5
        // Expect : 4.58203125
        x = 1.75;
        y = 2.5;
        out_idx = 0;
        let _ = interp.array1_interp2::<f64, i8, f64>(&mut weights, y, x, out_idx, &array_in, 0., None, &mut array_out, &mut None);
        assert_eq!(array_out.data, [4.58203125, 10., -9.]);
    }
}
