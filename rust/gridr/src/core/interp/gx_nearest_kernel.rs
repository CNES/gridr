#![warn(missing_docs)]
//! Crate doc
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use super::gx_array_view_interp::{GxArrayViewInterpolator, GxArrayViewInterpolationContextTrait, GxArrayViewInterpolatorBoundsCheckStrategy, GxArrayViewInterpolatorInputMaskStrategy, GxArrayViewInterpolatorOutputMaskStrategy};



pub struct GxNearestInterpolator {
    kernel_row_size: usize,
    kernel_col_size: usize,
}

impl GxArrayViewInterpolator for GxNearestInterpolator
{
    fn new() -> Self {
        Self {
            kernel_row_size: 1,
            kernel_col_size: 1,
        }
    }
    
    fn kernel_row_size(&self) -> usize {
        self.kernel_row_size
    }
    
    fn kernel_col_size(&self) -> usize {
        self.kernel_col_size
    }
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; 1];
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
        // Get the nearest corresponding index corresponding to the target position 
        let kernel_center_row: usize = (target_row_pos + 0.5).floor() as usize;
        let kernel_center_col: usize = (target_col_pos + 0.5).floor() as usize;
 
        let array_in_var_size: usize = array_in.nrow * array_in.ncol;
        let array_out_var_size: usize = array_out.nrow * array_out.ncol;
        
        let mut arr_iflat: usize = kernel_center_row * array_in.ncol + kernel_center_col;
        let mut out_idx_ivar: usize = out_idx;
        
        // Consider mask valid (if any)
        context.output_mask().set_value(out_idx, 1);
        
        // After compilation that test will have no cost in monomorphic created
        // method
        if IC::BoundsCheck::do_check() {
            // Check that the required data for interpolation is within the input
            // array shape
            // Here we do not need to check borders inside the inner loops.
            // That should be the most common path.
            if (kernel_center_row >=0)
                    && (kernel_center_row < array_in.nrow)
                    && (kernel_center_col >= 0)
                    && (kernel_center_col < array_in.ncol) {
                
                if context.input_mask().is_enabled() {
                    // There is a mask
                    if context.input_mask().is_valid(arr_iflat) == 1 {
                        // The mask is valid by default
                        // Set output
                        for _ivar in 0..array_in.nvar {
                            // Set output data
                            array_out.data[out_idx_ivar] = V::from((array_in.data[arr_iflat]).into());
                            
                            // Prepare indices for next iteration
                            arr_iflat += array_in_var_size;
                            out_idx_ivar += array_out_var_size;
                        }
                        // The mask is valid by default
                    }
                    else {
                        // Set output to nodata
                        for _ivar in 0..array_in.nvar {
                            // Set output data
                            array_out.data[out_idx_ivar] = V::from((array_in.data[arr_iflat]).into());
                            
                            // Prepare indices for next iteration
                            out_idx_ivar += array_out_var_size;
                        }
                        // Set output_mask
                        context.output_mask().set_value(out_idx, 0);
                    }
                }
                else {
                    // There is no mask - the boundary check has been performed
                    for _ivar in 0..array_in.nvar {
                        // Set output data
                        array_out.data[out_idx_ivar] = V::from((array_in.data[arr_iflat]).into());
                            
                        // Prepare indices for next iteration
                        arr_iflat += array_in_var_size;
                        out_idx_ivar += array_out_var_size;
                    }
                }
            }
            else {
                for _ivar in 0..array_in.nvar {
                    // Set output to nodata
                    array_out.data[out_idx_ivar] = nodata_out;
                            
                    // Prepare indices for next iteration
                    arr_iflat += array_in_var_size;
                    out_idx_ivar += array_out_var_size;
                }
                context.output_mask().set_value(out_idx, 0);
            }
        } else {
            // Here there is no boundary check - this code can panic !
            // This code is implemented for performance issue
            if context.input_mask().is_enabled() {
                // There is a mask
                if context.input_mask().is_valid(arr_iflat) == 1 {
                    // The mask is valid by default
                    // Set output
                    for _ivar in 0..array_in.nvar {
                        // Set output data
                        array_out.data[out_idx_ivar] = V::from((array_in.data[arr_iflat]).into());
                        
                        // Prepare indices for next iteration
                        arr_iflat += array_in_var_size;
                        out_idx_ivar += array_out_var_size;
                    }
                    // The mask is valid by default
                }
                else {
                    // Set output to nodata
                    for _ivar in 0..array_in.nvar {
                        // Set output data
                        array_out.data[out_idx_ivar] = V::from((array_in.data[arr_iflat]).into());
                        
                        // Prepare indices for next iteration
                        out_idx_ivar += array_out_var_size;
                    }
                    // Set output_mask
                    context.output_mask().set_value(out_idx, 0);
                }
            }
            else {
                // Set output
                for _ivar in 0..array_in.nvar {
                    // Set output data
                    array_out.data[out_idx_ivar] = V::from((array_in.data[arr_iflat]).into());
                    
                    // Prepare indices for next iteration
                    arr_iflat += array_in_var_size;
                    out_idx_ivar += array_out_var_size;
                }
            }
        }
        Ok(())
    }

}

