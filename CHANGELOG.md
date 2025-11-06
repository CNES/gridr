# CHANGELOG


## [0.4.4] - yyyy-mm-dd

### Fixed
- **Dependencies**: Fixed missing dependancy to shapely in pyproject.toml. 


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
    - Added generate_notice.py and generate_rust_notice.sh script
- templates:
    - Added templates directory in project tree
    - Added templates to generate main NOTICE and python/rust 3rd party notices sections.
- Added Rust tests for `GxNearestInterpolator` and `GxLinearInterpolator` to verify mask usage
- Added Python test for the core `grid.grid_resampling.py` module. The test currently covers only identity grid transforms at full resolution.
- Added `array_convert`, `is_clip_required` and `is_clip_to_dtype_limits_safe` python methods in the `gridr.core.utils.array_utils.py` module as well as corresponding tests. These methods are used by the `grid_resampling_chain` to convert data type before writing to disk in order to match the output dataset type.

### Changed
- Added license related header in python and rust source files
- Apply pre-commit hooks (flake8, isort, black) to existing python source files in python/gridr and tests 
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
