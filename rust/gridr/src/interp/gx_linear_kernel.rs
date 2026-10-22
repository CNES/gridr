// Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
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
//! Implementation of GxArrayViewInterpolator for a linear interpolator
use crate::gx_array::{GxArrayView, GxArrayViewMut};
use crate::gx_errors::GxError;
use super::gx_array_view_interp::{
    GxArrayViewInterpolator,
    GxArrayViewInterpolatorArgs,
    GxArrayViewInterpolatorCore,
    GxArrayViewInterpolationContextTrait,
    ZeroPattern,
};

/// Computes the linear kernel weights on a 2-tap support.
///
/// The stencil is anchored on `floor(x)`; `t = x - floor(x)` is the phase,
/// in `[0, 1)`. The weights are used through a convolution, their order is
/// inverted with respect to the data:
///
/// - `weights[0]`: tap `anchor + 1`, weight `t`
/// - `weights[1]`: tap `anchor`, weight `1 - t`
///
/// At `t = 0` the weights are exactly `[0, 1]`.
#[inline(always)]
pub fn linear_kernel_weights(t: f64, weights: &mut [f64])
{
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");
    weights[0] = t;
    weights[1] = 1.0 - t;
}

/// Linear interpolator implementation.
/// 
/// This structure implements the `GxArrayViewInterpolator` trait for linear
/// interpolation operations.
#[derive(Clone, Debug)]
pub struct GxLinearInterpolator {
    /// The size of the kernel alongs the rows - it is set to 2 in the
    /// implemented new() method.
    kernel_row_size: usize,
    /// The size of the kernel alongs the columns - it is set to 2 in the
    /// implemented new() method.
    kernel_col_size: usize,
}

// Use generic implementation from GxArrayViewInterpolatorCore for
// KROWS = 2, NCOLS = 2 (native support)
impl GxArrayViewInterpolatorCore<2, 2> for GxLinearInterpolator {
    /// Taps are limited to `anchor` and `anchor + 1`
    /// At `t = 0` only the anchor (offset 0) has a non-zero weight.
    const ZERO_PATTERN: ZeroPattern = ZeroPattern::AtPhase { phase: 0.0, start: 0, len: 1 };

    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        linear_kernel_weights(t, weights)
    }
}

impl GxArrayViewInterpolator for GxLinearInterpolator
{
    fn new(_args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        Self {
            kernel_row_size: 2,
            kernel_col_size: 2,
        }
    }
    
    fn shortname(&self) -> String {
        "linear".to_string()
    }
    
    fn initialize(&mut self) -> Result<(), String> {
        Ok(())
    }
    
    #[inline(always)]
    fn kernel_row_size(&self) -> usize {
        self.kernel_row_size
    }
    
    #[inline(always)]
    fn kernel_col_size(&self) -> usize {
        self.kernel_col_size
    }
    
    #[inline(always)]
    fn total_margins(&self) -> Result<[usize; 4], GxError> {
        Ok([1, 1, 1, 1])
    }
    
    #[inline(always)]
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; self.kernel_row_size + self.kernel_col_size];
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


#[cfg(test)]
mod gx_linear_kernel_tests {
    use super::*;
    use crate::interp::gx_array_view_interp::{GxArrayViewInterpolatorNoArgs, GxArrayViewInterpolationContext, DefaultCtx, BinaryInputMask, BinaryOutputMask, BoundsCheck};
    
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

    /// The declared zero pattern matches the computed weights.
    #[test]
    fn test_linear_zero_pattern() {
        use crate::interp::gx_kernel_test_utils::check_zero_pattern;
        check_zero_pattern(
            linear_kernel_weights, 2,
            <GxLinearInterpolator as GxArrayViewInterpolatorCore<2, 2>>::ZERO_PATTERN,
        );
    }

    /// native weights against the continuous linear kernel.
    #[test]
    fn test_linear_kernel_weights_native() {
        use crate::interp::gx_kernel_test_utils::{check_native_weights, phi_linear};
        check_native_weights(linear_kernel_weights, 2, phi_linear, 1.0, 1e-15);
    }

    /// Phase 0: identity, the anchor tap carries the full weight.
    #[test]
    fn test_linear_kernel_weights_phase_zero() {
        let mut weights = [-1.0; 2];
        linear_kernel_weights(0.0, &mut weights);
        assert_eq!(weights, [0.0, 1.0]);
    }

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
        
        let interp = GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

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
        
        let interp = GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

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
                
        let interp = GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

        let mut weights = interp.allocate_kernel_buffer();
        
        {
            let mut array_mask_out = GxArrayViewMut::new(&mut mask_data_out, 1, 1, 3);
            
            // Strategy context
            let mut context = GxArrayViewInterpolationContext::new(
                    BinaryInputMask { mask: &array_mask_in},
                    BinaryOutputMask { mask: &mut array_mask_out },
                    BoundsCheck {},
                );
            
            // Test idendity - with mask context - at (1.5, 1)
            let x = 1.5;
            let y = 1.;
            let out_idx = 1;
            let nodata_value = -7.;
            let expected = [-9., 30., -9.];
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
            
            // Test idendity - with mask context - at (0.5, 1)
            let x = 0.5;
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
                
            // Test idendity - with mask context - at (0.5, 0.)
            let x = 0.5;
            let y = 0.;
            let out_idx = 1;
            let nodata_value = -7.;
            let expected = [-9., 0.5, -9.];
            let expected_mask = [11, 1, 11];
            let _ = interp.array1_interp2::<f64, f64, GxArrayViewInterpolationContext<_,_,_>>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
            assert_eq!(array_out.data, expected);
            assert_eq!(mask_data_out, expected_mask);
        }
    }
    
    #[test]
    fn test_array1_interp2_001() {
        let data_in = [ 0.0, 1.0, 2.0,
                        10.0, 20.0, 40.0,
                        100.0, 1000.0, 10000.0 ];
                
        let mut data_out = [ -9.0, -9.0, -9.0 ];
        
        let array_in = GxArrayView::new(&data_in, 1, 3, 3);
        
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, 1, 3);
        let interp = GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs{});

        let mut weights = interp.allocate_kernel_buffer();
        
        // Default context
        let mut context = DefaultCtx::default();
                
        // Test idendity
        // Expect : 20.
        let mut x = 1.;
        let mut y = 1.;
        let mut out_idx = 1;
        let expected = [-9., 20., -9.];
        let _ = interp.array1_interp2::<f64, f64, DefaultCtx>(&mut weights, y, x, out_idx, &array_in, &mut array_out, 0., &mut context);
        assert_eq!(array_out.data, expected);
        
        
        // Target x = 1.75 y = 1.4
        // Expect : 0.6*(0.75*40 + 0.25*20) + 0.4 * (10000*0.75+1000*0.25) = 
        //          0.6 * 35 + 0.4 * 7750 = 3121
        x = 1.75;
        y = 1.4;
        out_idx = 0;
        let expected = [3121.0, 20., -9.];
        let _ = interp.array1_interp2::<f64, f64, DefaultCtx>(&mut weights, y, x, out_idx, &array_in, &mut array_out, 0., &mut context);
        assert!(approx_eq(&array_out.data, &expected, 1e-6));
        
        // Going outside with default ctx => set to nodata
        // Target x = 2.75 y = 1.4
        // Expect : 0.6*(0.75*40 + 0.25*20) + 0.4 * (10000*0.75+1000*0.25) = 
        //          0.6 * 35 + 0.4 * 7750 = 3121
        x = 2.75;
        y = 1.4;
        out_idx = 2;
        let nodata_value = 7.;
        let expected = [3121.0, 20., nodata_value];
        let _ = interp.array1_interp2::<f64, f64, DefaultCtx>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
        assert!(approx_eq(&array_out.data, &expected, 1e-6));
        
        x = 1.75;
        y = -2.4;
        out_idx = 1;
        let nodata_value = 7.;
        let expected = [3121.0, nodata_value, nodata_value];
        let _ = interp.array1_interp2::<f64, f64, DefaultCtx>(&mut weights, y, x, out_idx, &array_in, &mut array_out, nodata_value, &mut context);
        assert!(approx_eq(&array_out.data, &expected, 1e-6));
    }
}
