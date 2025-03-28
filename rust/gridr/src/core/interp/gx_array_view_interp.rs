#![warn(missing_docs)]
//! Crate doc
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};

pub trait GxArrayViewInterpolator
{
    fn new() -> Self;
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]>;
    
    fn array1_interp2<T, U, V> (
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
        V: Copy + PartialEq + From<f64>;
    
    fn kernel_row_size(&self) -> usize;
    fn kernel_col_size(&self) -> usize;
}

