#![warn(missing_docs)]
//! Crate doc
//use crate::core::gx_array::{GxArrayView, GxArrayViewMut};

//use super::gx_array_view_interp::{GxArrayViewInterpolator};
//use super::gx_optimized_bicubic_kernel::{GxOptimizedBicubicInterpolator};

/* enum GxArrayInterpolator {
    OptimizeBicubic(GxOptimizedBicubicInterpolator),
}

impl GxArrayInterpolator {
    fn array1_interp2<T, U, V>(
        weights_buffer: &mut [f64],
        target_row_pos: f64,
        target_col_pos: f64,
        out_idx: usize,
        array_in: &GxArrayView<'_, T>,
        array_mask_in: Option<&GxArrayView<'_, U>>,
        array_out: &mut GxArrayViewMut<'_, V>,
        array_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
        nodata_out: Option<V>,
    ) -> Result<(), String> {
        match self {
            GxArrayInterpolator::OptimizeBicubic(arr_interp) => 
        }
    }
}
 */