#![warn(missing_docs)]
//! Crate doc
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use numpy::{PyArray1, PyArrayMethods, Element};
/// We tell here what module/functions we use from the pure rust library (lib.rs)
use crate::{assert_options_exclusive};
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};
use crate::core::gx_grid::{NoCheckGridNodeValidator, InvalidValueGridNodeValidator, MaskGridNodeValidator};
use crate::core::gx_grid_geometry::{GridTransitionMatrix, GeometryBounds, GridGeometriesMetrics, array1_compute_resampling_grid_geometries};
use crate::core::gx_const::{F64_TOLERANCE};
use crate::core::gx_utils::{GxToF64};

use super::py_array::{PyArrayWindow2};

/// A transition matrix defining a linear isomorphism between a canonical grid
/// (e.g., array indexing) and a transformed geometry grid.
///
/// This matrix defines a new basis using column and row generator vectors.
/// If either generator is `None`, the transformation is considered undefined.
#[pyclass(name = "PyGridTransitionMatrix")]
#[derive(Debug, Clone, Copy)]
pub struct PyGridTransitionMatrix {
    inner: GridTransitionMatrix,
}

#[pymethods]
impl PyGridTransitionMatrix {
    #[new]
    fn new(w1: Option<(f64, f64)>, w2: Option<(f64, f64)>) -> Self {
        Self {
            inner: GridTransitionMatrix::new(w1, w2),
        }
    }

    #[getter]
    fn w1(&self) -> Option<(f64, f64)> {
        self.inner.w1
    }

    #[setter]
    fn set_w1(&mut self, w1: Option<(f64, f64)>) {
        self.inner.w1 = w1;
    }

    #[getter]
    fn w2(&self) -> Option<(f64, f64)> {
        self.inner.w2
    }

    #[setter]
    fn set_w2(&mut self, w2: Option<(f64, f64)>) {
        self.inner.w2 = w2;
    }

    fn det_w(&self) -> Option<f64> {
        self.inner.det_w()
    }

    fn w(&self) -> Option<[f64; 4]> {
        self.inner.w()
    }

    fn __repr__(&self) -> String {
        format!("GridTransitionMatrix(w1={:?}, w2={:?})", self.inner.w1, self.inner.w2)
    }
}

impl From<GridTransitionMatrix> for PyGridTransitionMatrix {
    fn from(inner: GridTransitionMatrix) -> Self {
        Self { inner }
    }
}

impl From<PyGridTransitionMatrix> for GridTransitionMatrix {
    fn from(wrapper: PyGridTransitionMatrix) -> Self {
        wrapper.inner
    }
}

/// Axis-aligned rectangular bounds using usize coordinates.
#[pyclass(name = "PyGeometryBoundsUsize")]
#[derive(Debug, Clone)]
pub struct PyGeometryBoundsUsize {
    pub inner: GeometryBounds<usize>,
}

#[pymethods]
impl PyGeometryBoundsUsize {
    #[new]
    fn new(xmin: usize, xmax: usize, ymin: usize, ymax: usize) -> Self {
        Self {
            inner: GeometryBounds { xmin, xmax, ymin, ymax },
        }
    }

    #[getter]
    fn xmin(&self) -> usize {
        self.inner.xmin
    }

    #[getter]
    fn xmax(&self) -> usize {
        self.inner.xmax
    }

    #[getter]
    fn ymin(&self) -> usize {
        self.inner.ymin
    }

    #[getter]
    fn ymax(&self) -> usize {
        self.inner.ymax
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometryBoundsUsize(xmin={}, xmax={}, ymin={}, ymax={})",
            self.inner.xmin, self.inner.xmax, self.inner.ymin, self.inner.ymax
        )
    }
}

/// Axis-aligned rectangular bounds using floating-point coordinates.
#[pyclass(name = "PyGeometryBoundsF64")]
#[derive(Debug, Clone)]
pub struct PyGeometryBoundsF64 {
    pub inner: GeometryBounds<f64>,
}

#[pymethods]
impl PyGeometryBoundsF64 {
    #[new]
    fn new(xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> Self {
        Self {
            inner: GeometryBounds { xmin, xmax, ymin, ymax },
        }
    }

    #[getter]
    fn xmin(&self) -> f64 {
        self.inner.xmin
    }

    #[getter]
    fn xmax(&self) -> f64 {
        self.inner.xmax
    }

    #[getter]
    fn ymin(&self) -> f64 {
        self.inner.ymin
    }

    #[getter]
    fn ymax(&self) -> f64 {
        self.inner.ymax
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometryBoundsF64(xmin={}, xmax={}, ymin={}, ymax={})",
            self.inner.xmin, self.inner.xmax, self.inner.ymin, self.inner.ymax
        )
    }
}

/// Python wrapper for geometrical metrics between two grids (source and destination).
///
/// Includes bounding boxes, edge mappings, and the transition matrix that
/// defines the linear transformation between grids.
#[pyclass(name = "PyGridGeometriesMetricsF64")]
#[derive(Debug)]
pub struct PyGridGeometriesMetricsF64 {
    #[pyo3(get)]
    pub dst_bounds: PyGeometryBoundsUsize,

    #[pyo3(get)]
    pub src_bounds: PyGeometryBoundsF64,

    #[pyo3(get)]
    pub dst_row_edges: Vec<Option<(usize, usize)>>,

    #[pyo3(get)]
    pub dst_col_edges: Vec<Option<(usize, usize)>>,

    #[pyo3(get)]
    pub src_row_edges: (Vec<Option<(f64, f64)>>, Vec<Option<(f64, f64)>>),

    #[pyo3(get)]
    pub src_col_edges: (Vec<Option<(f64, f64)>>, Vec<Option<(f64, f64)>>),

    #[pyo3(get)]
    pub transition_matrix: PyGridTransitionMatrix,
}

#[pymethods]
impl PyGridGeometriesMetricsF64 {
    /// Creates a new GridGeometriesMetricsF64.
    ///
    /// Parameters:
    /// - dst_bounds: Bounding box of the destination grid (usize).
    /// - src_bounds: Bounding box of the source geometry (f64).
    /// - dst_row_edges: Optional per-row min/max in destination space.
    /// - dst_col_edges: Optional per-column min/max in destination space.
    /// - src_row_edges: Optional per-row min/max in source space (tuple of min and max vectors).
    /// - src_col_edges: Optional per-column min/max in source space (tuple of min and max vectors).
    /// - transition_matrix: Transformation matrix from canonical to physical space.
    #[new]
    fn new(
        dst_bounds: PyGeometryBoundsUsize,
        src_bounds: PyGeometryBoundsF64,
        dst_row_edges: Vec<Option<(usize, usize)>>,
        dst_col_edges: Vec<Option<(usize, usize)>>,
        src_row_edges: (Vec<Option<(f64, f64)>>, Vec<Option<(f64, f64)>>),
        src_col_edges: (Vec<Option<(f64, f64)>>, Vec<Option<(f64, f64)>>),
        transition_matrix: PyGridTransitionMatrix,
    ) -> Self {
        Self {
            dst_bounds,
            src_bounds,
            dst_row_edges,
            dst_col_edges,
            src_row_edges,
            src_col_edges,
            transition_matrix,
        }
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "GridGeometriesMetricsF64(dst_bounds={:?}, src_bounds={:?}, ...)",
            self.dst_bounds.inner, self.src_bounds.inner
        )
    }
}

impl From<GridGeometriesMetrics<f64>> for PyGridGeometriesMetricsF64 {
    fn from(native: GridGeometriesMetrics<f64>) -> Self {
        Self {
            dst_bounds: PyGeometryBoundsUsize {
                inner: native.dst_bounds
            },
            src_bounds: PyGeometryBoundsF64 {
                inner: native.src_bounds
            },
            dst_row_edges: native.dst_row_edges,
            dst_col_edges: native.dst_col_edges,
            src_row_edges: native.src_row_edges,
            src_col_edges: native.src_col_edges,
            transition_matrix: PyGridTransitionMatrix {
                inner: native.transition_matrix
            },
        }
    }
}

/// Computes resampling grid geometries metrics from given row and column grids.
///
/// This function analyzes the validity of grid points (nodes) along rows and columns to determine
/// bounding boxes and transition matrix for resampling operations on grids. It returns detailed
/// information about the source and destination grid bounds, as well as the transition matrix
/// describing the linear transformation between source and destination grids.
///
/// It takes 1D arrays representing the row and column coordinates of a grid,
/// along with the grid's shape and resolution parameters, and optionally a mask or nodata value
/// to validate grid nodes. It returns computed grid geometries metrics as `PyGridGeometriesMetricsF64`
/// or `None` if the computation yields no result.
///
/// It wraps the `gx_grid_geometry::array1_compute_resampling_grid_geometries(...)` function.
///
/// # Parameters
/// - `grid_row`: A bound immutable reference to a 1D NumPy array representing the row coordinates of the grid.
/// - `grid_col`: A bound immutable reference to a 1D NumPy array representing the column coordinates of the grid.
/// - `grid_shape`: A tuple `(rows, cols)` defining the shape of the grid.
/// - `grid_resolution`: A tuple `(row_resolution, col_resolution)` defining the oversampling resolution of the grid.
/// - `grid_mask`: An optional bound immutable reference to a 1D NumPy array mask, where nodes with a specific valid value are considered valid.
/// - `grid_mask_valid_value`: The valid value in the mask indicating valid grid nodes. Required if `grid_mask` is provided.
/// - `grid_nodata`: An optional nodata value indicating invalid grid nodes. Mutually exclusive with `grid_mask`.
///
/// # Returns
/// - `Ok(Some(PyGridGeometriesMetricsF64))` if metrics are successfully computed.
/// - `Ok(None)` if no metrics could be computed.
/// - `Err(PyErr)` if an error occurs (e.g., invalid parameters or internal computation failure).
///
/// # Constraints
/// The generic type `W` must implement:
/// - `Element` (NumPy element trait),
/// - `Ord`, `Copy`, `PartialEq`, `Default`,
/// - `GxToF64` (a trait converting to `f64`),
/// - `Into<f64>`.
///
/// # Behavior
/// - If neither `grid_mask` nor `grid_nodata` are provided, the function uses a `NoCheckGridNodeValidator` (no validation).
/// - If `grid_mask` is provided, a `MaskGridNodeValidator` is used, and `grid_mask_valid_value` must be specified.
/// - If `grid_nodata` is provided, an `InvalidValueGridNodeValidator` is used to exclude invalid nodes based on nodata value.
///
/// # Panics
/// - If both `grid_mask` and `grid_nodata` are provided simultaneously (mutually exclusive).
/// - If `grid_mask_valid_value` is not provided when `grid_mask` is given.
///
/// # Errors
/// Returns a Python `ValueError` (`PyValueError`) if parameter validation fails or if internal computations return errors.
///
/// # Example
/// ```ignore
/// let metrics = py_array1_compute_resampling_grid_geometries_f64(
///     &grid_row_pyarray,
///     &grid_col_pyarray,
///     (100, 200),
///     (2, 2),
///     Some(&grid_mask_pyarray),
///     Some(1u8),
///     None,
/// )?;
/// ```
fn py_array1_compute_resampling_grid_geometries_f64<W>(
    grid_row: &Bound<'_, PyArray1<W>>,
    grid_col: &Bound<'_, PyArray1<W>>,
    grid_shape: (usize, usize),
    grid_resolution: (usize, usize),
    grid_mask: Option<&Bound<'_, PyArray1<u8>>>,
    grid_mask_valid_value: Option<u8>,
    grid_nodata: Option<W>,
    grid_win: Option<PyArrayWindow2>,
    ) -> Result<Option<PyGridGeometriesMetricsF64>, PyErr>
where
    W: Element + PartialOrd + Copy + PartialEq + Default + GxToF64 + Into<f64>,
{
    // Create a safe immuable array_view in order to read from the grid - row - array
    let grid_row_view = grid_row.readonly();
    let grid_row_slice = grid_row_view.as_slice()?;
    let grid_row_arrayview = GxArrayView::new(grid_row_slice, 1, grid_shape.0, grid_shape.1);
    
    // Create a safe immuable array_view in order to read from the grid - col - array
    let grid_col_view = grid_col.readonly();
    let grid_col_slice = grid_col_view.as_slice()?;
    let grid_col_arrayview = GxArrayView::new(grid_col_slice, 1, grid_shape.0, grid_shape.1);
    
    // Manage the optional production window (in full resolution grid coordinates system)
    let rs_grid_win = grid_win.map(GxArrayWindow::from);
    
    // Manage the grid validator mode through a the `grid_validator_flag` variable.
    // - 0 : that value corresponds to the use of a NoCheckGridMeshValidator, ie
    //       no mask has been provided by the caller
    // - 1 : that value corresponds to the use of a MaskGridMeshValidator, ie
    //       a raster mask has been provided.
    // - 2 : that value corresponds to the use of a InvalidValueGridMeshValidator, ie
    //       a grid nodata value has been provided.
    let mut grid_validator_flag : u8 = 0;
    
    // Check exclusive parameters
    assert_options_exclusive!(grid_mask, grid_nodata, PyErr::new::<PyValueError, _>(
        "Only one of `grid_mask` or `grid_nodata` may be provided, not both."));
    let grid_mask_view;
    let grid_mask_array_view = match grid_mask {
        Some(a_mask_grid) => {
            grid_validator_flag += 1;
            grid_mask_view = a_mask_grid.readonly();
            let grid_mask_slice = grid_mask_view.as_slice()?;
            Some(GxArrayView::new(grid_mask_slice, 1, grid_shape.0, grid_shape.1))
        }
        None => None, 
    };
    // Get grid_nodata_value ; warning : if None a default value will be given
    let grid_nodata_value = grid_nodata.map(|val| {
        grid_validator_flag += 2;
        val.into()
    });
    
    
    match grid_validator_flag {
        0 => {
            // No validator parameter has been passed ; we set the grid_checker to the always
            // positiv NoCheckGridMeshValidator
            let grid_checker = NoCheckGridNodeValidator{};
            
            match array1_compute_resampling_grid_geometries::<W, f64, NoCheckGridNodeValidator>(
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &grid_checker, // grid_validity_checker
                    rs_grid_win, // win
               ) {
                Ok(Some(grid_metrics)) => Ok(Some(grid_metrics.into())),
                Ok(None) => Ok(None),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        1 => {
            // A grid mask parameter has been passed ; we intialize a MaskGridNodeValidator
            let mask_view = grid_mask_array_view.unwrap();
            let mask_valid_value = grid_mask_valid_value.ok_or_else(|| PyValueError::new_err(
                    "The argument `grid_mask_valid_value` is mandatory when using `grid_mask`"
                ))?;
            let grid_checker = MaskGridNodeValidator{ mask_view: &mask_view, valid_value: mask_valid_value };
            
            match array1_compute_resampling_grid_geometries::<W, f64, MaskGridNodeValidator>(
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &grid_checker, // grid_validity_checker
                    rs_grid_win, // win
               ) {
                Ok(Some(grid_metrics)) => Ok(Some(grid_metrics.into())),
                Ok(None) => Ok(None),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        2 => {
            // A grid nodata value parameter has been passed ; we intialize an InvalidValueGridNodeValidator
            let grid_checker = InvalidValueGridNodeValidator{
                invalid_value: grid_nodata_value.expect("grid_nodata was None, but a value was expected"),
                epsilon: F64_TOLERANCE
            };
            
            match array1_compute_resampling_grid_geometries::<W, f64, InvalidValueGridNodeValidator>(
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &grid_checker, // grid_validity_checker
                    rs_grid_win, // win
               ) {
                Ok(Some(grid_metrics)) => Ok(Some(grid_metrics.into())),
                Ok(None) => Ok(None),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        _ => Err(PyValueError::new_err("Grid validator mode not implemented")),
    }
}

/// This function calls the generic [`py_array1_compute_resampling_grid_geometries_f64`] with `W = f64`.
#[pyfunction]
#[pyo3(signature = (grid_row, grid_col, grid_shape, grid_resolution, grid_mask=None, grid_mask_valid_value=None, grid_nodata=None, grid_win=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_compute_resampling_grid_geometries_f64_f64(
    grid_row: &Bound<'_, PyArray1<f64>>,
    grid_col: &Bound<'_, PyArray1<f64>>,
    grid_shape: (usize, usize),
    grid_resolution: (usize, usize),
    grid_mask: Option<&Bound<'_, PyArray1<u8>>>,
    grid_mask_valid_value: Option<u8>,
    grid_nodata: Option<f64>,
    grid_win: Option<PyArrayWindow2>,
    ) -> Result<Option<PyGridGeometriesMetricsF64>, PyErr>
{
    py_array1_compute_resampling_grid_geometries_f64::<f64>(
            grid_row, //: &Bound<'_, PyArray1<f64>>,
            grid_col, //: &Bound<'_, PyArray1<f64>>,
            grid_shape, //: (usize, usize),
            grid_resolution, //: (usize, usize),
            grid_mask, //: Option<&Bound<'_, PyArray1<u8>>>,
            grid_mask_valid_value, //: Option<u8>,
            grid_nodata, //: Option<W>,
            grid_win, //: Option<PyArrayWindow2>,
            )
}