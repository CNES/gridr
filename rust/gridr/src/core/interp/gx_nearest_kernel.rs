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
//! Implementation of GxArrayViewInterpolator for a nearest neighbor
//! interpolator
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use crate::core::gx_errors::GxError;
use super::gx_array_view_interp::{
    GxArrayViewInterpolator,
    GxArrayViewInterpolatorArgs,
    GxArrayViewInterpolationContextTrait,
    GxArrayViewInterpolatorBoundsCheckStrategy,
    GxArrayViewInterpolatorInputMaskStrategy,
    GxArrayViewInterpolatorOutputMaskStrategy,
};


/// Nearest neighbor interpolator implementation
/// 
/// This structure implements the `GxArrayViewInterpolator` trait for nearest
/// neighbor interpolation operations.
#[derive(Clone, Debug)]
pub struct GxNearestInterpolator {
    /// The size of the kernel alongs the rows - it is set to 1 in the 
    /// implemented new() method.
    kernel_row_size: usize,
    /// The size of the kernel alongs the columns - it is set to 1 in the 
    ///implemented new() method.
    kernel_col_size: usize,
}

impl GxArrayViewInterpolator for GxNearestInterpolator
{
    fn new(_args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        Self {
            kernel_row_size: 1,
            kernel_col_size: 1,
        }
    }
    
    fn shortname(&self) -> String {
        "nearest".to_string()
    }
    
    fn initialize(&mut self) -> Result<(), String> {
        Ok(())
    }
    
    fn kernel_row_size(&self) -> usize {
        self.kernel_row_size
    }
    
    fn kernel_col_size(&self) -> usize {
        self.kernel_col_size
    }
    
    #[inline(always)]
    fn total_margins(&self) -> Result<[usize; 4], GxError> {
        Ok([1, 1, 1, 1])
    }
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; 1];
        buffer.into_boxed_slice()
    }
    
    /// Direct implementation without delegation to the 
    /// [`GxArrayViewInterpolatorCore`] trait : snaps directly to the nearest
    /// integer position if valid.
    fn array1_interp2<T, V, IC>(
            &self,
            _weights_buffer: &mut [f64],
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
        // Nearest integer centre for the interpolation stencil.
        let kernel_center_row: i64 = (target_row_pos + 0.5).floor() as i64;
        let kernel_center_col: i64 = (target_col_pos + 0.5).floor() as i64;
 
        let in_var_sz: usize = array_in.nrow * array_in.ncol;
        let out_var_sz: usize = array_out.nrow * array_out.ncol;
        
        let mut arr_iflat: usize =
            (kernel_center_row as usize) * array_in.ncol +
            (kernel_center_col as usize);
        let mut out_idx_ivar: usize = out_idx;
        
        // Initialise the output mask as valid; the interpolation variants will
        // overwrite it to 0 if the pixel turns out to be invalid.
        context.output_mask().set_value(out_idx, 1);
        
        if IC::BoundsCheck::do_check() {
            // Interior path: the full stencil fits strictly inside the array.
            // No per-element bounds checking is needed inside the inner loops.
            if (kernel_center_row >=0)
                    && (kernel_center_row < array_in.nrow_i64)
                    && (kernel_center_col >= 0)
                    && (kernel_center_col < array_in.ncol_i64) {
                
                if context.input_mask().is_enabled() {
                    if context.input_mask().is_valid(arr_iflat) == 1 {
                        // The mask is valid by default - set output data
                        for _ivar in 0..array_in.nvar {
                            array_out.data[out_idx_ivar] =
                                V::from((array_in.data[arr_iflat]).into());
                            
                            // Prepare indices for next iteration
                            arr_iflat += in_var_sz;
                            out_idx_ivar += out_var_sz;
                        }
                    }
                    else {
                        // Set output data as nodata
                        for _ivar in 0..array_in.nvar {
                            array_out.data[out_idx_ivar] = nodata_out;
                            
                            // Prepare indices for next iteration
                            out_idx_ivar += out_var_sz;
                        }
                        // Set output_mask as invalid
                        context.output_mask().set_value(out_idx, 0);
                    }
                }
                else {
                    // There is no mask - the boundary check has been performed
                    for _ivar in 0..array_in.nvar {
                        // Set output data
                        array_out.data[out_idx_ivar] =
                            V::from((array_in.data[arr_iflat]).into());
                            
                        // Prepare indices for next iteration
                        arr_iflat += in_var_sz;
                        out_idx_ivar += out_var_sz;
                    }
                }
            }
            else {
                for _ivar in 0..array_in.nvar {
                    // Set output to nodata
                    array_out.data[out_idx_ivar] = nodata_out;
                            
                    // Prepare indices for next iteration
                    out_idx_ivar += out_var_sz;
                }
                // Set output_mask as invalid
                context.output_mask().set_value(out_idx, 0);
            }
        } else {
            // No bounds check: the caller guarantees all indices are valid.
            if context.input_mask().is_enabled() {
                if context.input_mask().is_valid(arr_iflat) == 1 {
                    // The mask is valid by default - set output data
                    for _ivar in 0..array_in.nvar {
                        array_out.data[out_idx_ivar] =
                            V::from((array_in.data[arr_iflat]).into());
                        
                        // Prepare indices for next iteration
                        arr_iflat += in_var_sz;
                        out_idx_ivar += out_var_sz;
                    }
                }
                else {
                    // Set output data as nodata
                    for _ivar in 0..array_in.nvar {
                        array_out.data[out_idx_ivar] = nodata_out;
                        
                        // Prepare indices for next iteration
                        out_idx_ivar += out_var_sz;
                    }
                    // Set output_mask as invalid
                    context.output_mask().set_value(out_idx, 0);
                }
            }
            else {
                // Set output data
                for _ivar in 0..array_in.nvar {
                    array_out.data[out_idx_ivar] =
                        V::from((array_in.data[arr_iflat]).into());
                    
                    // Prepare indices for next iteration
                    arr_iflat += in_var_sz;
                    out_idx_ivar += out_var_sz;
                }
            }
        }
        Ok(())
    }

}


#[cfg(test)]
mod gx_linear_kernel_tests {
    use super::*;
    use crate::core::interp::gx_array_view_interp::{
        GxArrayViewInterpolatorNoArgs,
        GxArrayViewInterpolationContext,
        BinaryInputMask,
        BinaryOutputMask,
        BoundsCheck
    };

    #[test]
    fn test_array1_interp2_idendity_mask_full_valid() {
        
        // Input array
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        let array_in = GxArrayView::new(&data_in, 1, 3, 3);
        
        // Output array
        let mut data_out = [ -9.0, -9.0, -9.0 ];
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, 1, 3);
        
        // Input mask
        let mask_data_in: [u8; 9] = [1; 9];
        let array_mask_in = GxArrayView::new(&mask_data_in, 1, 3, 3);
        
        // Output mask
        let mut mask_data_out: [u8; 3] = [ 11; 3 ];
        let mut array_mask_out = GxArrayViewMut::new(&mut mask_data_out, 1, 1, 3);
        
        // Strategy context
        let mut context = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: &array_mask_in},
                BinaryOutputMask { mask: &mut array_mask_out },
                BoundsCheck {},
            );
        
        let interp = GxNearestInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

        let mut weights = interp.allocate_kernel_buffer();
        
        // Test idendity - with mask context - full valid mask
        let x = 1.;
        let y = 1.;
        let out_idx = 1;
        let expected = [-9., 20., -9.];
        let expected_mask = [11, 1, 11];
        let _ = interp.array1_interp2::<f64, f64, GxArrayViewInterpolationContext<_,_,_>>(&mut weights, y, x, out_idx, &array_in, &mut array_out, 0., &mut context);
        
        assert_eq!(array_out.data, expected);
        assert_eq!(mask_data_out, expected_mask);
    }
    
    #[test]
    fn test_array1_interp2_idendity_mask_full_invalid() {
        
        // Input array
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        let array_in = GxArrayView::new(&data_in, 1, 3, 3);
        
        // Output array
        let mut data_out = [ -9.0, -9.0, -9.0 ];
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, 1, 3);
        
        // Input mask
        let mask_data_in: [u8; 9] = [0; 9];
        let array_mask_in = GxArrayView::new(&mask_data_in, 1, 3, 3);
        
        // Output mask
        let mut mask_data_out: [u8; 3] = [ 11; 3 ];
        let mut array_mask_out = GxArrayViewMut::new(&mut mask_data_out, 1, 1, 3);
        
        // Strategy context
        let mut context = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: &array_mask_in},
                BinaryOutputMask { mask: &mut array_mask_out },
                BoundsCheck {},
            );
        
        let interp = GxNearestInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

        let mut weights = interp.allocate_kernel_buffer();
        
        // Test idendity - with mask context - full valid mask
        let x = 1.;
        let y = 1.;
        let out_idx = 1;
        let nodata_value = -7.;
        let expected = [-9., nodata_value, -9.];
        let expected_mask = [11, 0, 11];
        let _ = interp.array1_interp2::<f64, f64, GxArrayViewInterpolationContext<_,_,_>>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
        
        assert_eq!(array_out.data, expected);
        assert_eq!(mask_data_out, expected_mask);
    }
    
    #[test]
    fn test_array1_interp2_idendity_mask() {
        
        // Input array
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
        let array_in = GxArrayView::new(&data_in, 1, 3, 3);
        
        // Output array
        let mut data_out = [ -9.0, -9.0, -9.0 ];
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, 1, 3);
        
        // Input mask
        let mask_data_in: [u8; 9] = [1, 1, 1,
                                     0, 1, 1,
                                     1, 1, 1];
        let array_mask_in = GxArrayView::new(&mask_data_in, 1, 3, 3);
        
        // Output mask
        let mut mask_data_out: [u8; 3] = [ 11; 3 ];
                
        let interp = GxNearestInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

        let mut weights = interp.allocate_kernel_buffer();
        
        {
            let mut array_mask_out = GxArrayViewMut::new(&mut mask_data_out, 1, 1, 3);
            
            // Strategy context
            let mut context = GxArrayViewInterpolationContext::new(
                    BinaryInputMask { mask: &array_mask_in},
                    BinaryOutputMask { mask: &mut array_mask_out },
                    BoundsCheck {},
                );
            
            // Test idendity - with mask context - at (1.501, 1)
            let x = 1.501;
            let y = 1.;
            let out_idx = 1;
            let nodata_value = -7.;
            let expected = [-9., 40., -9.];
            let expected_mask = [11, 1, 11];
            let _ = interp.array1_interp2::<f64, f64, GxArrayViewInterpolationContext<_,_,_>>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
            assert_eq!(array_out.data, expected);
            assert_eq!(mask_data_out, expected_mask);
        }
        {
            let mut array_mask_out = GxArrayViewMut::new(&mut mask_data_out, 1, 1, 3);
            
            // Strategy context
            let mut context = GxArrayViewInterpolationContext::new(
                    BinaryInputMask { mask: &array_mask_in},
                    BinaryOutputMask { mask: &mut array_mask_out },
                    BoundsCheck {},
                );
            
            // Test idendity - with mask context - at (0.499, 1)
            let x = 0.499;
            let y = 1.;
            let out_idx = 1;
            let nodata_value = -7.;
            let expected = [-9., nodata_value, -9.];
            let expected_mask = [11, 0, 11];
            let _ = interp.array1_interp2::<f64, f64, GxArrayViewInterpolationContext<_,_,_>>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
            assert_eq!(array_out.data, expected);
            assert_eq!(mask_data_out, expected_mask);
        }
        {
            let mut array_mask_out = GxArrayViewMut::new(&mut mask_data_out, 1, 1, 3);
            
            // Strategy context
            let mut context = GxArrayViewInterpolationContext::new(
                    BinaryInputMask { mask: &array_mask_in},
                    BinaryOutputMask { mask: &mut array_mask_out },
                    BoundsCheck {},
                );
                
            // Test idendity - with mask context - at (0.499, 0.)
            let x = 0.499;
            let y = 0.;
            let out_idx = 1;
            let nodata_value = -7.;
            let expected = [-9., 0., -9.];
            let expected_mask = [11, 1, 11];
            let _ = interp.array1_interp2::<f64, f64, GxArrayViewInterpolationContext<_,_,_>>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
            assert_eq!(array_out.data, expected);
            assert_eq!(mask_data_out, expected_mask);
        }
    }
}