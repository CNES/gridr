#![warn(missing_docs)]
//! Crate doc
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};
//use crate::core::interp::gx_optimized_bicubic_kernel::{array1_optimized_bicubic_interp2};
use crate::core::interp::gx_array_view_interp::GxArrayViewInterpolator;
//use crate::core::interp::gx_optimized_bicubic_kernel::{GxOptimizedBicubicInterpolator};
use crate::{assert_options_match};
use crate::core::gx_errors::GxError;

/// A trait that standardize the the grid validity check at each grid position.
pub trait GridMeshValidator<W>
{
    fn validate<'a>(&self, mesh: &'a mut GridMesh, out_idx: &'a mut usize, grid_view: &GxArrayView<'a, W>) -> bool;
}

/// A structure to implement the `GridMeshValidator` trait with an always true validation. 
#[derive(Debug)]
pub struct NoCheckGridMeshValidator {
}

impl<W> GridMeshValidator<W> for NoCheckGridMeshValidator{
    
    #[inline]
    fn validate<'a>(&self, mesh: &'a mut GridMesh, out_idx: &'a mut usize, grid_view: &GxArrayView<'a, W>) -> bool
    {
        true
    }
}

/// A structure to implement the `GridMeshValidator` trait with a validity value directly set in the grid. 
#[derive(Debug)]
pub struct InvalidValueGridMeshValidator {
    pub invalid_value: f64,
    pub epsilon: f64,
}

impl<W> GridMeshValidator<W> for InvalidValueGridMeshValidator
where
    W: Into<f64> + Copy,
{
    
    #[inline]
    fn validate<'a>(&self, mesh: &'a mut GridMesh, out_idx: &'a mut usize, grid_view: &GxArrayView<'a, W>) -> bool
    {
        if (grid_view.data[mesh.node1].into() - self.invalid_value).abs() < self.epsilon {
            return false;
        } else if (grid_view.data[mesh.node2].into() - self.invalid_value).abs() < self.epsilon {
            return false;
        } else if (grid_view.data[mesh.node2].into() - self.invalid_value).abs() < self.epsilon {
            return false;
        } else if (grid_view.data[mesh.node2].into() - self.invalid_value).abs() < self.epsilon {
            return false;
        }
        true
    }
}


/// A structure to implement the `GridMeshValidator` trait with a validity state read from an auxiliary mask.
#[derive(Debug)]
pub struct MaskGridMeshValidator<'a> {
    pub mask_view: &'a GxArrayView<'a, u8>
}

impl<'a, W> GridMeshValidator<W> for MaskGridMeshValidator<'a>
where
    W: Copy,
{
    #[inline]
    fn validate(&self, mesh: &mut GridMesh, out_idx: &mut usize, grid_view: &GxArrayView<W>) -> bool
    {
        if self.mask_view.data[mesh.node1] != 0 {
            return false;
        } else if self.mask_view.data[mesh.node2] != 0 {
            return false;
        } else if self.mask_view.data[mesh.node3] != 0 {
            return false;
        } else if self.mask_view.data[mesh.node4] != 0 {
            return false;
        }
        true
    }
}


/// The GridMesh structure represents a grid mesh used for bilinear interpolation performed
/// in order to compute the full resolution grid's target coordinates.
/// The mesh is defined by its 4 corners ordered clockwise from the upper left corner to the bottom
/// left corner in a clockwise :
///
///     (node1 : upper left)  +--------------+ (node2: upper right)
///                           |              |
///                           |              |
///                           |              |
///                           |              |
///                           |              |
///    (node4 : bottom left)  +--------------+ (node3: bottom right)
///
/// # Fields
///
/// - `node1`: The mesh upper left corner.
/// - `node2`: The mesh upper right corner.
/// - `node3`: The mesh bottom right corner.
/// - `node4`: The mesh bottom left corner.
/// - `grid_nrow`: The number of rows of the parent low resolution grid.
/// - `grid_ncol`: The number of columns of the parent low resolution grid.
/// - `window_src`: The window applied on the parent low resolution grid to limit the computations.
///
#[derive(Debug)]
pub struct GridMesh<'a> {
    node1: usize,
    node2: usize,
    node3: usize,
    node4: usize,
    grid_nrow: usize,
    grid_ncol: usize,
    window_src: &'a GxArrayWindow, 
}

impl<'a> GridMesh<'a> {
    
    #[inline]
    pub fn new(grid_nrow: usize, grid_ncol: usize, window_src: &'a GxArrayWindow) -> Self {
        let node1 = window_src.start_row * grid_ncol + window_src.start_col;
        Self {
            node1: node1,
            node2: node1 + 1,
            node3: node1 + grid_ncol + 1,
            node4: node1 + grid_ncol,
            grid_nrow: grid_nrow,
            grid_ncol: grid_ncol,
            window_src: window_src }
    }
    
    #[inline]
    pub fn next_src_col(&mut self, grid_col_idx: usize) {
        self.node1 += 1;
        self.node2 += 1;
        self.node3 += 1;
        self.node4 += 1;
        
        // Manage the grid last point as node
        // We set to use the vertical 0-width mesh
        if grid_col_idx == self.grid_ncol - 1 {
            self.node2 = self.node1;
            self.node3 = self.node4;
        }
    }
    
    #[inline]
    pub fn next_src_row(&mut self, grid_row_idx: usize) {
        let node1 = grid_row_idx * self.grid_ncol + self.window_src.start_col;
        self.node1 = node1;
        self.node2 = node1 + 1;
        self.node4 = node1 + self.grid_ncol;
        self.node3 = self.node4 + 1;
        
        // Manage the grid last point as node
        // We set to use the vertical 0-width mesh
        if grid_row_idx == self.grid_nrow - 1 {
            self.node4 = self.node1;
            self.node3 = self.node2;
        }
    }
} 

/// Perform a bilinear grid interpolation with oversampling, computing interpolated
/// values for both grid rows and columns. The grid is iterated over in a row-major
/// order with oversampling applied to both rows and columns.
///
/// # Parameters
/// - `grid_row_array`: A structure holding the grid of row values to be interpolated.
/// - `grid_col_array`: A structure holding the grid of column values to be interpolated.
/// - `grid_row_oversampling`: The oversampling factor for the row dimension.
/// - `grid_col_oversampling`: The oversampling factor for the column dimension.
///
/// # Output
/// The function computes interpolated values for both grid rows and columns at each
/// output position. The grid is processed in a row-major order, with each output
/// position corresponding to an interpolated point on the grid.
///
/// # Process
/// - The output grid size is determined based on the oversampling factors and the
///   dimensions of the input grid.
/// - Bilinear interpolation is performed by determining the weights (`gmi_w1`, `gmi_w2`, `gmi_w3`, `gmi_w4`)
///   for each grid mesh and applying them to the neighboring nodes.
/// - The interpolation is done both on the grid columns and rows separately.
/// - The mesh index is updated iteratively, advancing over the grid and applying
///   the appropriate interpolation for each position.
///
/// # Flow
/// 1. The size of the output grid is computed based on oversampling factors.
/// 2. The algorithm loops over all output positions, applying bilinear interpolation.
/// 3. It tracks the current mesh position and advances through the grid as necessary.
/// 4. For each output position, interpolation weights are calculated, and the interpolated
///    values are stored.
/// 5. The loop advances to the next output column and row, updating mesh and output positions.
/// 6. When the row or column within the mesh is completed, the indices are reset, and the
///    algorithm moves to the next mesh position.
///
/// # Bilinear interpolation on grid mesh
/// Each interpolated point within a mesh uses the four neighbors in order to perform a bilinear 
/// interpolation.
///
///        (0,0) ----- (0,1)
///         |             |
///         |             |
///        (1,0) ----- (1,1)
///
/// Each grid mesh has 4 nodes :
/// - (0,0) ('gmi_node_1' in code)
/// - (0,1) ('gmi_node_2' in code)
/// - (1,1) ('gmi_node_3' in code)
/// - (1,0) ('gmi_node_4' in code)
///

pub fn array1_grid_resampling<T, U, V, W, I, C>(
        interp: &I,
        grid_validity_checker: &C,
        ima_in: &GxArrayView<'_, T>,
        grid_row_array: &GxArrayView<'_, W>,
        grid_col_array: &GxArrayView<'_, W>,
        grid_row_oversampling: usize,
        grid_col_oversampling: usize,
        ima_out: &mut GxArrayViewMut<'_, V>,
        nodata_val_out: V,
        ima_mask_in: Option<&GxArrayView<'_, U>>,
        grid_mask_array: Option<&GxArrayView<'_, u8>>,
        ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
        grid_win: Option<&GxArrayWindow>,
        ) -> Result<(), GxError>
where
    T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
    U: Copy + PartialEq + Into<f64>,
    V: Copy + PartialEq + From<f64>,
    W: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
    I: GxArrayViewInterpolator,
    C: GridMeshValidator<W>,
    //<U as Mul<f64, Output=f64>>::Output: Add,
{
    // Check that both grid_row_array and grid_col_array have same size
    if (grid_row_array.nrow != grid_col_array.nrow) || (grid_row_array.ncol != grid_col_array.ncol) {
        return Err(GxError::ShapesMismatch { field1:"grid_row_array", field2:"grid_col_array" })
    }
    
    // Check that both grid_mask_array and no_data_val_out are given if one is given
    //assert_options_match!(grid_mask_array, nodata_val_out,
    //        GxError::OptionsMismatch { field1:"grid_mask_array", field2:"nodata_val_out" });
    
    // Manage the optional grid_win
    // The grid_win contains production limit to apply on the full resolution grid.
    // If not given we init a window corresponding to the full grid
    let full_res_grid_window = match grid_win {
        Some(some_grid_win) => {
            some_grid_win
        }
        None =>  &GxArrayWindow {
            start_row: 0,
            end_row: (grid_row_array.nrow - 1) * grid_row_oversampling,
            start_col: 0,
            end_col: (grid_row_array.ncol - 1) * grid_col_oversampling,
        }, 
    };
    // Compute number of rows and columns for output
    let ncol_out = full_res_grid_window.width(); //full_res_grid_window.end_col - full_res_grid_window.start_col + 1;
    let size_out = full_res_grid_window.size(); //nrow_out * ncol_out;
    
    // Compute the grid interval containing the window
    // If no window was given it is directly the full grid but we still make the code generic
    // since it is performed only once before the loop
    // The interpolation nodes should be taken from grid_window_src, but the grid_window_rel is used
    // to set the start and stop indexes for columns and rows.
    let (grid_window_src, grid_window_rel) = GxArrayWindow::get_wrapping_window_for_resolution(
            (grid_row_oversampling, grid_col_oversampling), full_res_grid_window)?;

    // Current position in grid
    let mut grid_row_idx: usize = grid_window_src.start_row;
    let mut grid_col_idx: usize = grid_window_src.start_col;
    
    // Current output position
    let mut out_col_idx: usize = 0;
    
    // The 'gmi' prefix stands for grid mesh interpolation
    // It is used for all variable relative to the grid mesh bilinear interpolation
    // It defines the relative position in current grid mesh :
    // - gmi_{row|col}_idx can be set in [0, grid_{row|col}_oversampling[
    // - the init values (first iteration) are taken from the grid_window_rel
    
    // The init idx is given by the relative position in the source window.
    let mut gmi_row_idx: usize = grid_window_rel.start_row;
    let mut gmi_col_idx: usize = grid_window_rel.start_col;
    
    // Complement of the relative position in current grid mesh 
    let mut gmi_col_idx_t: usize = grid_col_oversampling - gmi_col_idx;
    let mut gmi_row_idx_t: usize = grid_row_oversampling - gmi_row_idx;
        
    // Init the mesh used for grid values bilinear interpolation.
    let mut gmi_mesh = GridMesh::new(grid_row_array.nrow, grid_row_array.ncol, &grid_window_src);
    
    // Mesh interpolation norm factor
    let gmi_norm_factor: f64 = (grid_col_oversampling * grid_row_oversampling) as f64;
    
    // Call the GxArrayViewInterpolator allocate_kernel_buffer to allocate
    // the buffer to store the kernel weights.
    // That buffer will be passed to the array1_interp2 method.
    let mut weights_buffer = interp.allocate_kernel_buffer();
    
    for mut out_idx in 0..size_out {
        
        // Here we call the validity checker for each oversampled index.
        // We may improve this loop by jumping to the next mesh directly
        if grid_validity_checker.validate(&mut gmi_mesh, &mut out_idx, &grid_row_array) {
        
            // Bilinear grid interpolation with oversampling
            let gmi_w1: f64 = (gmi_col_idx_t * gmi_row_idx_t) as f64;
            let gmi_w2: f64 = (gmi_col_idx * gmi_row_idx_t) as f64;
            let gmi_w3: f64 = (gmi_col_idx * gmi_row_idx) as f64;
            let gmi_w4: f64 = (gmi_col_idx_t * gmi_row_idx) as f64;
            
            // Perform the interpolation on column
            let grid_col_val: f64 = (
                    grid_col_array.data[gmi_mesh.node1] * gmi_w1 +
                    grid_col_array.data[gmi_mesh.node2] * gmi_w2 +
                    grid_col_array.data[gmi_mesh.node3] * gmi_w3 +
                    grid_col_array.data[gmi_mesh.node4] * gmi_w4
                    ) / gmi_norm_factor;
            
            // Perform the interpolation on rows
            let grid_row_val: f64 = (
                    grid_row_array.data[gmi_mesh.node1] * gmi_w1 +
                    grid_row_array.data[gmi_mesh.node2] * gmi_w2 +
                    grid_row_array.data[gmi_mesh.node3] * gmi_w3 +
                    grid_row_array.data[gmi_mesh.node4] * gmi_w4
                    ) / gmi_norm_factor;
            
            // Do grid interpolation here
            let _ = interp.array1_interp2(
                    &mut weights_buffer,
                    grid_row_val,
                    grid_col_val,
                    out_idx,
                    ima_in,
                    nodata_val_out,
                    ima_mask_in,
                    ima_out,
                    ima_mask_out, 
                    );
        } else {
            // do something
            for ivar in 0..ima_in.nvar {                
                // Write nodata value to output buffer
                ima_out.data[out_idx + ivar * size_out] = nodata_val_out;
            }
        }
        // Prepare next iteration
        
        // Global output : go next output column
        out_col_idx += 1;
        
        // Mesh interpolation : go next column in current mesh
        gmi_col_idx += 1;
        
        // Test if current row within current mesh is done, i.e all columns
        // have been passed through
        if gmi_col_idx == grid_col_oversampling {
            // Current row within mesh oversampling is done.
            // Warning : we cant go to next mesh on the same row if the current mesh is the last one
            // Go to next mesh on the same line
            // - reset relative mesh column index
            gmi_col_idx = 0;
            // - increment column index relative to input grid
            grid_col_idx += 1;
            // - shift all nodes to right - the method takes care of the last mesh border
            gmi_mesh.next_src_col(grid_col_idx);
        }
        
        // Test if current output row is done
        if out_col_idx == ncol_out {
            
            // Reset output column index to 0
            out_col_idx = 0;
            
            // Go next row
            // out_row_idx += 1;
            // Reset column index relative to input grid
            grid_col_idx = grid_window_src.start_col;
            
            // Reset relative mesh column index
            gmi_col_idx = grid_window_rel.start_col;
            
            // Increment relative mesh row index
            gmi_row_idx += 1;
            
            // Test if row oversampling in current mesh is done.
            if gmi_row_idx == grid_row_oversampling {
                // Reset mesh relative row index
                gmi_row_idx = 0;
                // Increment row index relative to input grid
                grid_row_idx += 1;
            }
            
            // Update interpolation nodes - going next src row
            gmi_mesh.next_src_row(grid_row_idx);
            
            // Update mesh relative row index complement
            gmi_row_idx_t = grid_row_oversampling - gmi_row_idx;
        }
        // Update mesh relative col index complement
        gmi_col_idx_t = grid_col_oversampling - gmi_col_idx;
    }
    Ok(())
}

#[cfg(test)]
mod gx_grid_resampling_test {
    use super::*;
    use crate::core::interp::gx_optimized_bicubic_kernel::{GxOptimizedBicubicInterpolator};
    
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
    
    /// Compares two 3D data windows for approximate equality, element by element.
    ///
    /// This function compares two subarrays (windows) of 3D data represented as 1D slices,
    /// allowing for differences in their shapes. It uses a tolerance value to check if 
    /// the corresponding elements in both arrays are approximately equal.
    ///
    /// # Arguments
    ///
    /// * `a` - A 1D slice representing the first 3D array. The data is stored contiguously.
    /// * `a_start` - A tuple `(nvar, nrow, ncol)` representing the starting indices in array `a` 
    ///   from which the comparison will begin.
    /// * `shape_a` - A tuple `(nvar, nrow, ncol)` representing the shape of array `a` in 3D.
    /// * `b` - A 1D slice representing the second 3D array. The data is stored contiguously.
    /// * `b_start` - A tuple `(nvar, nrow, ncol)` representing the starting indices in array `b` 
    ///   from which the comparison will begin.
    /// * `shape_b` - A tuple `(nvar, nrow, ncol)` representing the shape of array `b` in 3D.
    /// * `window_size` - A tuple `(nv, nr, nc)` representing the dimensions of the window to compare, 
    ///   which must be the same size in both `a` and `b`.
    /// * `tol` - A floating-point tolerance value used to determine if two elements are approximately equal. 
    ///   The function checks if the absolute difference between corresponding elements is less than or equal to `tol`.
    ///
    /// # Returns
    ///
    /// Returns `true` if all corresponding elements in the specified windows of `a` and `b` are approximately equal, 
    /// within the given tolerance `tol`. Returns `false` if any pair of elements differs by more than `tol`.
    ///
    /// # Example
    /// ```rust
    /// let shape_a = (2, 4, 4);
    /// let shape_b = (2, 3, 3);
    /// let a = vec![
    ///     1.0, 2.0, 3.0, 4.0,  
    ///     5.0, 6.0, 7.0, 8.0,  
    ///     9.0, 10.0, 11.0, 12.0,  
    ///     13.0, 14.0, 15.0, 16.0,  
    ///     17.0, 18.0, 19.0, 20.0,  
    ///     21.0, 22.0, 23.0, 24.0,  
    ///     25.0, 26.0, 27.0, 28.0,  
    ///     29.0, 30.0, 31.0, 32.0,  
    /// ];
    /// let b = vec![
    ///     6.0, 7.0, 8.0,  
    ///     10.0, 11.0, 12.0,  
    ///     14.0, 15.0, 16.0,  
    ///     22.0, 23.0, 24.0,  
    ///     26.0, 27.0, 28.0,  
    ///     30.0, 31.0, 32.0,  
    /// ];
    /// let tol = 0.1;
    /// let window_size = (2, 3, 3);
    /// let equal = approx_eq_window_3d(
    ///     &a, (0, 1, 1), shape_a, 
    ///     &b, (0, 0, 0), shape_b, 
    ///     window_size, tol
    /// );
    /// println!("The windows are equal: {}", equal);
    /// ```
    fn approx_eq_window_3d(
        a: &[f64], a_start: (usize, usize, usize), shape_a: (usize, usize, usize),
        b: &[f64], b_start: (usize, usize, usize), shape_b: (usize, usize, usize),
        window_size: (usize, usize, usize),
        tol: f64
    ) -> bool {
        let (nv, nr, nc) = window_size;
        let (nvar_a, nrow_a, ncol_a) = shape_a;
        let (nvar_b, nrow_b, ncol_b) = shape_b;

        (0..nv).all(|v| 
            (0..nr).all(|r| 
                (0..nc).all(|c| {
                    let a_idx = (a_start.0 + v) * nrow_a * ncol_a + (a_start.1 + r) * ncol_a + (a_start.2 + c);
                    let b_idx = (b_start.0 + v) * nrow_b * ncol_b + (b_start.1 + r) * ncol_b + (b_start.2 + c);
                    
                    (a[a_idx] - b[b_idx]).abs() <= tol
                })
            )
        )
    }

    
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let nrow_in = 15;
        let ncol_in = 10;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_out = nrow_in-2*margin;
        let ncol_out = ncol_in-2*margin;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_out * ncol_out];
        let mut grid_col = vec![0.0; nrow_out * ncol_out];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                grid_row[irow * ncol_out + icol] = irow as f64 + margin as f64;
                grid_col[irow * ncol_out + icol] = icol as f64 + margin as f64;
                data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = NoCheckGridMeshValidator{};
        
        let _ = array1_grid_resampling::<f64, i8, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                1, //grid_row_oversampling: usize,
                1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //grid_mask_array: Option<&GxArrayView<'_, u8>>,
                &mut None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                );
        
        assert!(approx_eq(&data_out, &data_expected, 1e-10));
    }
    
    
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_translate() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let dx = 10.5;
        let dy = -20.5;
        let nrow_in = 15;
        let ncol_in = 10;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_out = nrow_in-2*margin;
        let ncol_out = ncol_in-2*margin;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_out * ncol_out];
        let mut grid_col = vec![0.0; nrow_out * ncol_out];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                grid_row[irow * ncol_out + icol] = irow as f64 + margin as f64 + dy;
                grid_col[irow * ncol_out + icol] = icol as f64 + margin as f64 + dx;
                data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = NoCheckGridMeshValidator{};
        
        let _ = array1_grid_resampling::<f64, i8, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                1, //grid_row_oversampling: usize,
                1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //grid_mask_array: Option<&GxArrayView<'_, u8>>,
                &mut None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>,
                None, //grid_win: Option<&GxArrayWindow>,
                );
    }
    
    
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity_oversampling() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let nrow_in_decimated = 2;
        let ncol_in_decimated = 2;
        let oversampling_row = 6;
        let oversampling_col = 7;
        let nrow_in = (nrow_in_decimated - 1) * oversampling_row + 1;
        let ncol_in = (ncol_in_decimated - 1) * oversampling_col + 1;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
                
        // Here we apply an idendity geometric transformation => set margin to 0
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_grid = nrow_in_decimated;
        let ncol_grid = ncol_in_decimated;
        let nrow_out = nrow_in;
        let ncol_out = ncol_in;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_grid * ncol_grid];
        let mut grid_col = vec![0.0; nrow_grid * ncol_grid];
        
        // Init data_in values with a bicubic function -> we should be able
        // to find similar values by interpolation of a decimated array
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                let xf = icol as f64;
                let yf = irow as f64;
                data_in[irow * ncol_in + icol] = 1.0 + 2.0 * xf + 3.0 * yf + 4.0 * xf * yf
                        + 5.0 * xf.powi(2) + 6.0 * yf.powi(2)
                        + 7.0 * xf.powi(3) + 8.0 * yf.powi(3);
            }
        }
                
        // Init grid
        for irow in 0..nrow_grid {
            for icol in 0..ncol_grid {
                grid_row[irow * ncol_grid + icol] = (irow * oversampling_row) as f64;
                grid_col[irow * ncol_grid + icol] = (icol * oversampling_col) as f64;
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_grid, ncol_grid);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_grid, ncol_grid);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = NoCheckGridMeshValidator{};
        
        let _ = array1_grid_resampling::<f64, i8, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                oversampling_row, //grid_row_oversampling: usize,
                oversampling_col, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //grid_mask_array: Option<&GxArrayView<'_, u8>>,
                &mut None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                );
        assert!(approx_eq_window_3d(
            &array_out.data, (0, margin, margin), (array_out.nvar, array_out.nrow, array_out.ncol),
            &array_in.data, (0, margin, margin), (array_in.nvar, array_in.nrow, array_in.ncol),
            (1, array_in.nrow-2*margin, array_in.ncol-2*margin),
            1e-6));
        /*
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                let iflat = irow * ncol_in + icol;
                println!("{:?} {:?} : in = {:?}\t out = {:?}", irow, icol,  array_in.data[iflat], array_out.data[iflat]); 
            }
        }
        */
    }
}