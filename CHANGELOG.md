# CHANGELOG

## [0.x.y] - YYYY-mm-dd

### Added

#### Grid Resampling

- **Performance improvement : fast path for full resolution grid** (`core.gx_resampling_grid.rs`)
  - Introduced a new structure `GridPointMesh` optimized for complete degenerated GridMesh reduced to one node only
  - Added `validate_point method` to GridMeshValidator trait for validating grid points when using a `GridPointMesh`
  - Added conditional logic to choose between the fast path using (`GridPointMesh`) and the regular interpolation path. The fast path is taken when oversampling factors equal to 1 for both dimensions.
  - Added a new test case `test_array1_grid_resampling_gridmesh_vs_gridpointmesh_idendity` to verify the correctness of the interpolation with both `GridMesh` and `GridPointMesh`.

#### Benchmarking
- **Benchmark framework**:
  - Added pytest-benchmark based framework for CPU time benchmarking
  - Automatic machine configuration information collection and storage in benchmark reports

- **New benchmarks**:
  - Added `tests/python/benchmarks/time/test_time_core_array_grid_resampling.py`:
    - Benchmarks `gridr.core.grid.grid_resampling.array_grid_resampling` method
    - Tests various parameter combinations (grid size, interpolation methods)
    - Compares with `scipy.map_coordinates` using equivalent configuration

  - Added `tests/python/benchmarks/time/test_time_chain_array_grid_resampling.py`:
    - Benchmarks `gridr.chain.grid_resampling_chain.basic_grid_resampling_chain` method
    - Tests limited parameter set (resolution, interpolation methods, multi-channel mode)
    - Compares with CNES proprietary ORION Software (requires ORION_BIN_PATH environment variable)

### Changed

#### Grid Resampling

- **Rust module `core.gx_grid_resampling.rs`**
  - Renamed `validate` method to `validate_mesh` in the `GridMeshValidator` trait and its implementations.

### Fixed

#### Makefile
- **Optimized Makefile**:
  - Limited Rust build steps to necessary cases (source changes or missing compiled library)

- **Test execution**
  - Modified `test-python` rule to exclude benchmark tests

## [0.5.2] - 2026-03-13

### Fixed

#### Grid Resampling

- **Core Grid Resampling (`grid_resampling.py`)** 
  - Fixed an UnboundLocalError in `calculate_source_extent()` when grid metrics could not be computed - the variable was referenced before assignment in that code path.
  - Added the `test_grid_resampling_standalone_no_idendity_1_1_full_invalid_grid()` in `test_grid_resampling.py` to cover the previous mentioned path.

## [0.5.1] - 2026-01-29

### Changed

#### Project
- Updated README banner logo and switched to PNG format for better mobile compatibility

### Fixed

#### Grid Resampling
- Fixed panic when resampling with degenerate windows (1-pixel in any dimension) at grid edges


## [0.5.0] - 2026-01-13

### Added

#### Interpolation

- **B-Spline Interpolation Implementation**
  - Added B-Spline interpolation support for orders 3, 5, 7, 9, and 11
  - Added Rust module `core/interp/gx_bspline_prefiltering` implementing B-Spline prefiltering methods and mask handling
  - Added Rust module `core/interp/gx_bspline_kernel` implementing `GxArrayViewInterpolator` trait for B-Spline
  - Implemented `GxBSplineInterpolatorTrait<const N: usize>` trait using monomorphic approach for compile-time B-Spline order specification
  - Created Python bindings for `BSpline<N>Interpolator` classes for all supported orders
  - Added Rust test suite for B-Spline interpolation
  - Created API documentation

- **Interpolator Margins**
  - Added `total_margins()` method to `GxArrayViewInterpolator` trait to compute required margins on each side for interpolation
  - For B-Spline interpolators, `total_margins()` includes pre-filtering domain extension for controlled precision in infinite sum approximations
  - Note: `initialize()` method must be called before `total_margins()` on B-Spline interpolator objects
  - Created Python bindings for `total_margins()` method
  - Added Python test suite for margin computation

#### Grid Operations

- **Array Arithmetic Functions**
  - Added `array1_add` Rust function in `gx_array_utils.rs` module
  - Added `array1_add_win2` Rust function in `gx_array_utils.rs` module
  - Created Python bindings for `array1_add` and `array1_add_win2` functions
  - Implemented `array_add` method in `gridr.core.array_utils` with comprehensive tests

- **Grid Coordinate Operations**
  - Added `array_shift_grid_coordinates` method to `core.grid.grid_utils` for in-place grid coordinate shifting

- **Source Boundary Computation**
  - Added `array1_compute_resampling_grid_src_boundaries` Rust function in `gx_grid_geometry.rs` module
  - Created Python bindings for `array1_compute_resampling_grid_src_boundaries` function
  - Implemented `array_compute_resampling_grid_src_boundaries` method in `gridr.core.grid.grid_utils` with comprehensive tests

#### Grid Resampling

- **Core Grid Resampling (`array_grid_resampling`)**
  - Added `check_boundaries` boolean parameter to activate safe grid boundaries computation (see Source Boundary Computation)
  - Added `boundary_condition` parameter accepting values: 'edge', 'reflect', 'symmetric', 'wrap'
  - Added `standalone` parameters to `gridr.core.grid.grid_resampling.array_grid_resampling`
  - Standalone mode performs: automatic grid metrics computation, source image required region computation (considering interpolation margins), optional padding operation, and B-Spline prefiltering
  - Updated test suite to include standalone mode validation

- **Grid Resampling Chain (`basic_grid_resampling_chain`)**
  - Added `grid_shift` parameter to apply global bias to grid coordinates
  - Added `boundary_condition` parameter accepting values: 'edge', 'reflect', 'symmetric', 'wrap'
  - Added `SAFECHECK_SOURCE_BOUNDARIES` constant (default: True) to determine read window using all valid coordinates within working grid subset (see Source Boundary Computation)
  - Method now creates and initialize interpolator objects internally

#### Boundary Handling

- **Array Padding**
  - Added `gridr.core.utils.array_pad` Python module derived from `numpy.pad`
  - Operates in-place with limited mode support
  - Created API documentation for `core.utils.array_pad`

#### Documentation

- **Theoretical Foundations**
  - Initialized "Theoretical Foundations" section in documentation
  - Added image geometry definitions
  - Added detailed grid resampling convention documentation

- **API Documentation Structure**
  - Added `core.interp` section to API documentation
  - Added `core.interp.bspline_prefiltering` documentation page
  - Added `core.interp.interpolator` documentation page
  - Added `core.utils.array_pad` documentation page

- **User Guides**
  - Expanded user documentation with standalone mode details
  - Added documentation for grid shift feature
  - Added documentation for boundary conditions
  - Added documentation for B-Spline mask handling

#### Dependencies

- **Rust**
  - Added `transpose` crate dependency (v0.2.6)
  - Upgraded `pyo3` crate dependency to version 0.27.2


### Changed

#### Interpolator Architecture

- **Interpolator Interface**
  - Added `Send` trait implementation to `GxArrayViewInterpolator`
  - Introduced `GxArrayViewInterpolatorArgs` trait to provide arguments for interpolator creation

- **Interpolator Type System**
  - Moved `PyInterpolatorType` to dedicated Rust module `pyapi/interp/py_interp.rs`
  - Added `AnyInterpolator` enum to replace `PyInterpolatorType` in `py_grid_resampling`
  - Changed interpolator passing mechanism: now passed as Python object instead of string literal
  - Implemented `FromPyObject` trait for `AnyInterpolator` to retrieve Rust object from Python object
  - `AnyInterpolator` enum variants wrap interpolator implementations using `Arc` (Atomic Reference Counting) and `RwLock` for thread-safe shared ownership
  - Note: This change requires interpolator creation on Python side before passing to Rust functions

#### Code Organization

- **Function Location**
  - Moved `read_win_from_grid_metrics()` function from `gridr.chain.grid_resampling_chain` to `gridr.core.grid.grid_utils`

#### Numerical Precision

- **Grid Coordinate Calculations**
  - Implemented controlled rounding to 12 decimal digits for internal full-resolution grid calculations in `gx_grid_resampling.rs`

#### Documentation Infrastructure

- **Sphinx Configuration**
  - Updated MathJax from version 2 to version 3
  - Changed MathJax CDN URL from `https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML` to `https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js`
  - Added `mathjax3_config` with inline math delimiters: `[['$', '$'], ['\\(', '\\)']]` and display math delimiters: `[['$$', '$$'], ['\\[', '\\]']]`
  - Added `sphinxcontrib.bibtex` extension
  - Configured `bibtex_bibfiles` to `['references.bib']`
  - Set `bibtex_default_style` to `'plain'`
  - Added MyST extensions: `dollarmath` and `amsmath`
  - Set `myst_dmath_double_inline` to `True`

- **Documentation Structure**
  - Changed page title from "API Documentation" to "API Reference" in `docs/source/api_python/modules.rst`
  - Reorganized Sphinx main menu structure
  - Removed `docs/source/developer_guide/index.rst` section heading
  - Removed `docs/source/api_python/misc/index.rst` module documentation
  - Removed `docs/source/api_python/misc/mandrill.rst` documentation page

- **Notebook References**
  - Updated notebook reference from `grid_resampling_001_work.ipynb` to `grid_resampling_001.ipynb` in `docs/source/_notebooks/index.rst`

- **Images**
  - Added new SVG file `docs/source/images/numeric_image.svg`

#### Project Metadata

- **README Badges**
  - Updated minimum pyo3 version badge from "0.26+" to "0.27.2+"
  - Changed logo image source from `./doc/images/gridr_logo.svg` to `https://github.com/CNES/gridr/tree/main/doc/images/gridr_logo.svg`

### Fixed

#### Grid Resampling

- **Out-of-Bounds Access**
  - Resolved panic when addressing index out-of-bounds of source array
  - Issue occurred for grids that do not preserve source topology
  - Fixed through implementation of safe grid source boundaries computation

#### Numerical Stability

- **Floating-Point Precision**
  - Resolved floating-point precision issues in grid coordinate calculations in `gx_grid_resampling.rs`
  - Fixed nearest neighbor interpolation bug caused by precision limitations
  - Addressed discrepancies between chain method and monolithic core method

#### Dependencies

- **Missing Dependencies**
  - Added missing Shapely dependency in `pyproject.toml`

#### Documentation

- **Styling**
  - Improved CSS styling of log extracts in `grid_resampling_chain` notebook for proper line breaking


## [0.4.3] - 2025-10-21

### [Unreleased]
- Documentation for antialiasing filter creation

### Fixed
- **Documentation**: Fixed style errors for contributing guidelines.


## [0.4.2] - 2025-10-20

### [Unreleased]
- Documentation for antialiasing filter creation

### Added
- **Documentation**: Added `.readthedocs.yaml` for automated docs builds.
- **Build**: Support for custom Sphinx HTML output directory in `Makefile`.

### Changed
- **CI/CD**: Release jobs now trigger **only** on semantic version tags (e.g., `vX.Y.Z`).
- **Dependencies**: Upgraded `pyo3` to **v0.26.0** to address security vulnerability .
- **Documentation**: Changed the contributing part in order to adhere the strict linear trunk-based branching strategy.

### Fixed
- **Build**: Generate ABI3-compatible wheels for Python 3.10+.
- **Documentation**: Fix indentations in array_utils.py docstrings.

### Security
- **Dependencies**: Patched `pyo3` to v0.26.0 (addresses GHSA-pph8-gcv7-4qj5).


## [0.4.1] - 2025-10-02

### [Unreleased]
- Documentation for antialiasing filter creation

### Added
- Added usage of pre-commit with flake8, isort and black
- Added license and opensource related files : LICENSE, NOTICE, AUTHORS.md, CONTRIBUTING.md, Clause of License Aggreements files
- Added Developer_Guide (WIP) and License sections in sphinx documentation
- scripts:
    - Added scripts directory in project tree
    - Added generate_notice.py and generate_Rust_notice.sh script
- templates:
    - Added templates directory in project tree
    - Added templates to generate main NOTICE and Python/Rust 3rd party notices sections.
- Added Rust tests for `GxNearestInterpolator` and `GxLinearInterpolator` to verify mask usage
- Added Python test for the core `grid.grid_resampling.py` module. The test currently covers only identity grid transforms at full resolution.
- Added `array_convert`, `is_clip_required` and `is_clip_to_dtype_limits_safe` Python methods in the `gridr.core.utils.array_utils.py` module as well as corresponding tests. These methods are used by the `grid_resampling_chain` to convert data type before writing to disk in order to match the output dataset type.

### Changed
- Added license related header in Python and Rust source files
- Apply pre-commit hooks (flake8, isort, black) to existing Python source files in Python/gridr and tests 
- Documentation
- Added a temporary change in `grid_resampling_chain` to adapt margin to the interpolation function.
- Tests for `grid_resampling_chain` have been enhanced to include interpolators other than `cubic` and output types different from `np.float64`. The `linear` interpolator is now tested, while the `nearest` interpolator is not due to an identified bug.
- Tests for `grid_resampling_chain` now use the newly local implemented utility method `assert_all_close_with_details`, which provides detailed information about differences when they occur.

### Fixed
- Fixed the `array1_interp2` for `GxNearestInterpolator` to properly set the nodata value for masked output pixels.
- Fixed the `grid_resampling_chain` method to work with output datasets whose types are different from float64.
- Fixed the test class name in `test_array_utils` from TestArrayWindow to TestArrayUtils.


## [0.4.0] - 2025-08-27

### [Unreleased]
- Documentation for antialiasing filter creation

### Added

- Grid Masks:
    - (Rust Core) GridNodeValidator trait and implementation supporting No Mask, Raster mask and Sentinel value options.
    - Configurable valid cell value in grid masks (replaces hardcoded value of 1)
    - Configurable sentinel value within grid coordinates to identify invalid cells
    - Using grid masks or a sentinel value are optional and mutually exclusive
    - Test coverage for new masking options
- Grid Geometric Metrics:
    - New data structure to hold grid geometric metrics
    - Computation of grid geometric metrics for in-memory grids
- Grid Resampling 
    - (Rust Core) Introduced trait strategies design concept for Input data masks, Output data masks and Bound checks
    - (Rust Core) Implemented strategies traits to cover both with and without options
    - (Rust Core) Added GxArrayViewInterpolationContextTrait trait to wrap the strategies traits
    - Nearest neighbor and linear interpolation methods
    - Configurable interpolation method
    - Configurable indexing shift for input array coordinates
    - Configurable target window for output buffer
    - Safe method `grid_resolution_window_safe` for handling edge cases in grid definitions
    - Chain module for managing I/O (input/output image, grids and input/output masks) and memory with tiling capabilities
- Antialiasing
    - Added reciprocal cell frequential filter functionality
    - Implemented core method to compute antialiasing filter from a grid using reciprocal cell frequential filter
- Documentation
    - Add documentation notebooks for the grid mask and grid resampling chains
- Branding
    - Added SVG version of GridR logo

### Changed

- Rust required version is now 1.80+
- Geometry mask definition
    - Modified signature of geometry_mask related methods (core and chain) to:
        - Allow setting both valid and invalid geometry masks
        - Remove assumptions about mask roles (previously implicit)
- Mask Default Value:
    - Changed default valid value for masks from 0 to 1
    - This change allows to use masks as factors
    - Note: This is an implicit convention change that may affect existing code
- Grid Resampling (Rust Core):
    - Fully embedded grid mesh iteration within the GridMesh implementation
    - Refactored to use GxArrayViewInterpolationContextTrait trait and related strategies
- Cubic interpolator (Rust Core)
    - Optimized weight computation through factorization of common terms for performance improvement
    - Refactored to use GxArrayViewInterpolationContextTrait trait and related strategies
- Docstrings
    - Adopted NumPy-style convention for docstrings
    - Reformulated code documentation
- Documentation
    - Updated core grid resampling notebook documentation to reflect grid resampling masks updates
    - Cleaned the `notebook_utils.py` module

### Fixed

- Resolved warning in Sphinx documentation building
- Fixed warning in Rust code compilation
