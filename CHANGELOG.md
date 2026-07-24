# CHANGELOG

## Unreleased

### Added

#### Documentation

- Added a new `Standards` section in the HTML documentation

### Changed

#### Documentation

- Updated pixel center terminology to align with OGC standard

### Build & Tooling

- **GitHub Actions** — added `release-python.yml`.
  - `release-python.yml`: builds wheels for manylinux_2_28/x86_64, macOS (Intel and Apple Silicon),
    and Windows x86_64 via `cibuildwheel`, on a `v*` tag push (auto-publish) or manual dispatch
    (build-only by default). A single `cp310-abi3` wheel per platform/arch covers Python 3.10-3.14+
    (`py_limited_api=cp310`), so no per-Python-version matrix is needed. AVX2/FMA `RUSTFLAGS` applied
    on every x86_64 target; automatically verified present in the compiled binary via `objdump` on
    Linux. Publishes to PyPI (or TestPyPI, for manual test runs) via Trusted Publishing (OIDC) — no
    stored token.

## [0.6.0] - 2026-06-10

### ⚠️ Breaking Changes

- **`GridMeshValidator` trait (Rust)** — The `validate` method has been renamed to `validate_mesh`
  to disambiguate from the newly introduced `validate_point` method. All implementors must update
  their method name accordingly.
- **`GxArrayViewInterpolatorInputMaskStrategy` trait (Rust)** — The signatures of `is_valid_window`,
  `is_valid_weighted_window` have changed:
  - The `cache` parameter has been removed from `is_valid_weighted_window`.
  - New `start_row_idx` and `start_col_idx` parameters have been added alongside the existing
    `start_idx` (flat index). Implementors may use whichever representation is most efficient for
    their implementation.
- **`array_grid_resampling` (Python)** — Two new parameters `trust_padding` and
  `array_in_mask_safe_win` have been added. `trust_padding` **must be explicitly set** when
  `boundary_condition` is not `None` ; an exception will be raised otherwise.
- **`basic_grid_resampling_array` / `basic_grid_resampling_chain` (Python)** — Same `trust_padding`
  requirement as above when a boundary condition is specified.
- **`shmutils.SharedMemoryArray` (Python)** — The `gridr.scaling.shmutils` module is now flagged as
  deprecated and replaced by `gridr.scaling.shared_array`, which exposes a unified facade over three
  interchangeable shared-memory backends (`shm`, `mmap`, `memfd`).

### Added

#### Grid Resampling — Fast Path for Full-Resolution Grids

*Module: `core::gx_resampling_grid`*

- Introduced the `GridPointMesh` structure, a degenerate single-node `GridMesh` optimized for the
  identity-resolution case (oversampling factor of 1 on both dimensions). This avoids the cost of
  full mesh interpolation when no resampling is actually needed.
- Added the `validate_point` method to the `GridMeshValidator` trait for validating individual grid
  points when operating on a `GridPointMesh`.
- Added a conditional dispatch in the resampling entry point that selects the fast `GridPointMesh`
  path when both oversampling factors equal 1, falling back to the regular `GridMesh` interpolation
  path otherwise.
- Added the `test_array1_grid_resampling_gridmesh_vs_gridpointmesh_identity` integration test, which
  verifies bit-exact equivalence between the two paths at identity resolution.

#### B-Spline Interpolators — Weight Factorization

*Module: `core::interp::gx_bspline_kernel`*

- Added `bspline{3,5,7,9,11}_all_centered` functions that compute all kernel weights for a centered
  fractional offset in a single call, factorizing intermediate computations that were previously
  repeated across per-tap evaluations.
- Eliminated bounds checks in these functions via a single `get_unchecked_mut` per call. The size
  contract (output buffer length matches kernel order) is documented in a `# Safety` section on each
  function.

#### Safe-Window Masking

*Modules:*

- *Rust:* `core::interp::gx_array_view_interp`, `core::interp::gx_bspline_prefiltering`,
  `core::interp::gx_bspline_kernels`, `pyapi::interp::py_bspline_prefiltering`,
  `pyapi::interp::py_bspline_kernel`, `pyapi::py_bindings`
- *Python:* `bspline_prefiltering.py`, `grid_resampling.py`, `grid_resampling_chain.py`

- Added `BinaryInputMaskWithSafeWindow` (`core::interp::gx_array_view_interp`), a new input mask
  strategy that allows the caller to declare a *safe window* — a rectangular sub-region of the input
  where the mask is known to be all-valid. Interpolation kernels falling entirely inside the safe
  window skip per-sample mask checks entirely.
- Added Rust
  (`core::interp::gx_bspline_prefiltering::array1_bspline_prefiltering_ext_gene_mask_safe_win`
  `core::interp::gx_bspline_kernels::array1_bspline_prefiltering_ext_mask_safe_win`,
  `pyapi::interp::py_bspline_prefiltering::py_array1_bspline_prefiltering_mask_safe_win_f64`,
  `pyapi::interp::py_bspline_kernel::PyBSpline*Interpolator::array1_bspline_prefiltering_ext_mask_safe_win`),
  and Python (`array_bspline_prefiltering_mask_safe_win`) prefiltering companion methods that update
  the safe window after B-spline prefiltering, eroding it by the kernel's influence radius.
- Wired the safe-window update into both the Python core (`grid_resampling.py`) and chain
  (`grid_resampling_chain.py`) resampling paths.
- Safe-Window is available for all convolution-based implemented kernels, i.e. all except `nearest`.

#### Exceptions

*Module: `grid_resampling.py`*

- Added the `GridMetricsError` exception, raised internally when the grid metrics required for
  source-extent computation cannot be derived from the provided grid (mainly due to invalid data).


#### Python Mask Strategy — Unified Resolution

*Module: `grid_resampling.py`*

- Added the `ResamplingMaskStrategy` named tuple, which encapsulates all mask-preparation decisions
  (padding policy, safe window, mask buffer source, etc.).
- Added `resolve_mask_strategy`, which derives the optimal `ResamplingMaskStrategy` from the input
  configuration. This function is shared between the standalone and chain code paths, guaranteeing
  that identical inputs produce identical mask decisions across both entry points.
- Added `check_mask_strategy`, a defensive validator that verifies the internal consistency of a
  resolved `ResamplingMaskStrategy` before execution.
- Added `apply_mask_strategy`, which materializes the final mask buffer from the resolved strategy.
- Added the `trust_padding` and `array_in_mask_safe_win` parameters to `array_grid_resampling` (see
  Breaking Changes).
- Extracted `standalone_preprocessing` from `array_grid_resampling`. This function now isolates
  source extent computation, data padding, mask preparation, and B-spline prefiltering as a single
  composable, standalone-testable unit.

#### Chain Mask Strategy — Unified Resolution

*Module: `grid_resampling_chain.py`*

- Added `apply_mask_strategy_chain`, the chain-specific counterpart of `apply_mask_strategy`. Unlike
  the standalone version, it operates on the pre-allocated merged buffer rather than allocating a
  new padded one, avoiding redundant memory allocations in the chain path.
- The chain path now consumes `resolve_mask_strategy` from the standalone module, guaranteeing
  behavioural parity with the standalone path for identical inputs.
- Added the `trust_padding` parameter to `basic_grid_resampling_array` and
  `basic_grid_resampling_chain` (see Breaking Changes).

#### Array Window Utilities

*Module: `array_window.py`*

- Added `complementary_window_indices`, which computes the list of index tuples covering the
  complement of a window within an array shape — i.e. the padded border zone excluding the data
  zone. Used by `apply_mask_strategy_chain` to fill the padded zone of a pre-allocated mask buffer
  without reallocation.
- Added unit tests covering nominal cases, edge windows, full-array windows, and axis-restricted
  variants.

#### Shared Memory — Pluggable Backends

*Module: `gridr.scaling.shared_array` (Python)*

- Introduced the `SharedArray` class, a unified facade for sharing `numpy.ndarray` buffers across
  processes. The class delegates to one of three interchangeable OS-level backends selected at
  runtime:
  * `shm` — based on `multiprocessing.shared_memory.SharedMemory` (named POSIX shared memory, stored
    under `/dev/shm`). Preserved as the historical default for compatibility.
  * `mmap` — anonymous `mmap(MAP_SHARED)` mapping. No `/dev/shm` footprint, inherited by forked
    workers via the kernel's address-space duplication. Targets memory-constrained Linux deployments
    such as Docker containers running with the default 64 MB `/dev/shm` cap.
  * `memfd` — Linux `memfd_create(2)` plus `mmap`. Anonymous, accessed through a file descriptor
    that can be transmitted to children via `SCM_RIGHTS` or `multiprocessing.reduction.send_handle`.
    Compatible with both `fork` and `spawn` start methods, in preparation for the `spawn` default
    change in Python 3.14.
- Added the `GRIDR_SHARED_MEMORY_BACKEND` environment variable for deployment-time backend selection
  without code changes. Accepted values: `shm`, `mmap`, `memfd`, `auto`. Takes precedence over any
  in-code call to `set_backend`.
- Added the `GRIDR_SHM_MIN_FREE` environment variable to tune the `/dev/shm` free-space threshold
  used by the auto-detection selector (default: 67108864 bytes / 64 MiB).
- Added a backend auto-detection routine that runs when neither `GRIDR_SHARED_MEMORY_BACKEND` nor
  `set_backend()` has fixed an explicit choice.
- Documented the full design and platform compatibility matrix.


#### Benchmarking Framework

- Introduced a `pytest-benchmark`-based framework for CPU time benchmarking.
- Machine configuration information is automatically collected and stored in benchmark reports for
  reproducibility tracking.

New benchmark suites:

- `tests/python/benchmarks/time/test_time_core_array_grid_resampling.py` — benchmarks
  `gridr.core.grid.grid_resampling.array_grid_resampling` across grid sizes and interpolation
  methods, with a comparison against `scipy.ndimage.map_coordinates` configured equivalently.
- `tests/python/benchmarks/time/test_time_chain_array_grid_resampling.py` — benchmarks
  `gridr.chain.grid_resampling_chain.basic_grid_resampling_chain` on a reduced parameter set
  (resolution, interpolation methods, multi-channel mode), with an optional comparison against the
  CNES proprietary ORION software (requires the `ORION_BIN_PATH` environment variable).

#### Documentation

- Added the `grid_resampling_core_standalone.ipynb` notebook, a deep-dive into the standalone
  preprocessing pipeline.
- Added the `grid_resampling_plot_utils.py` helper module supporting the notebook with adaptive
  matplotlib rendering.

#### Testing

- Added `pytest-regressions[num]` as a development dependency.
- Added regression tests for `gridr.chain.grid_resampling_chain.basic_grid_resampling_chain`
  covering: interpolation methods, multi-channel data, input mask, input grid mask, output mask
  production, resolution factors, and boundary conditions.
- Added the `test-python-regression` target to the project `Makefile`.
- Reference regression data must be generated by running `pytest tests/python/regression` on a
  reference branch. Data is stored in `tests/python/regression/_regression_data`.
- Extended `test_grid_resampling.py` with two new test classes covering the new code paths:
  - **`TestGridResamplingMonoPointGrid`** — exercises the `GridPointMesh` fast path (mono-point
    grid, identity resolution) across all four interpolators (`nearest`, `linear`, `cubic`,
    `bspline3`). This tests aim to validate the numerical results. Coverage includes:
    - Identity at the array centre and at the upper-left / bottom-right corners under all
      combinations of `boundary_condition` ∈ {`"reflect"`, `"constant"`, `None`} × `trust_padding` ∈
      {`True`, `False`}, with explicit documentation of B-spline-specific behaviour.
    - Sub-pixel column and row shifts, with the stencil position classified as *inside* vs *outside*
      the source domain.
    - Input mask invalidation patterns (invalid at `(0, 0)` and `(17, 27)`) combined with targets on
      the invalid point itself, on neighbouring valid points, and at interpolation-required
      positions.
    - Effect of the B-spline `mask_influence_threshold` parameter — comparing the default (no mask
      dilation during prefiltering) against an active-dilation configuration that propagates
      invalidity through the prefilter stencil.
    - `test_mono_point_grid__invalid_resolution` — verifies that any resolution other than `(1, 1)`
      raises a `ValueError`, with distinct error messages for the zero-resolution case
      (`"Resolution must be non zero"`) and the insufficient-grid-coverage case
      (`"InsufficientGridCoverage"`).
  - **`TestGridResamplingMultiPointGrid`** — exercises the regular `GridMesh` path with multi-point
    grids:
    - Uniformity preservation on constant input data across multiple resampling resolutions
      (including the large `(200, 300)` case).
    - Row and column sub-pixel shifts with linear interpolation.
    - Cross-product of `grid_mask` invalidation patterns (full invalid, four corners, centre) ×
      output `win` selections at resolutions `(1, 1)` and `(2, 3)`. Each parametrised case is run
      twice — once without `win`, once with — and the windowed result is cross-checked against the
      full result sliced by `window_indices(win)`.
    - **`test_multi_point_grid__safe_window__nominal_case`** — a three-way comparison verifying the
      `array_in_mask_safe_win` contract: a *correctly declared* safe window must produce results
      identical to the no-safe-window reference, while an *incorrectly declared* safe window
      (claiming validity over a region that contains invalid samples) must produce different results
      — except for `nearest`, which performs no convolution and therefore ignores the safe window.
      Parameterised across all interpolators, all four `boundary_condition` values, both
      `trust_padding` values, and both `use_standalone` values.
- Added a pytest suite of 115 tests covering the three shared memory backends with parametrized fork
  and spawn execution.

### Changed

#### Interpolation Architecture — Const-Generic Refactor

*Module: `core::interp::gx_array_view_interp`*

This is a substantial internal restructuring of the separable convolution machinery. While the
public `GxArrayViewInterpolator` API is preserved, implementors of custom interpolators or mask
strategies will be affected (see Breaking Changes).

- Refactored the module around the const generic parameters `KROWS` and `KCOLS`, which fix the
  kernel dimensions at compile time. This enables zero-cost generic implementations of the four
  separable convolution variants: masked / unmasked × bounds-checked / unchecked.
- Each variant now ships two implementations:
  - A **safe** version that uses pre-slicing and iterators to eliminate inner-loop bounds checks
    while remaining fully safe.
  - An **unsafe** version that uses `get_unchecked` throughout, under the precondition — enforced by
    the bounds classification performed in `array1_interp2_separable_core` — that the caller has
    already validated index ranges.
- Introduced a clean separation between the public API (`GxArrayViewInterpolator`) and the new
  internal computation trait (`GxArrayViewInterpolatorCore<KROWS, KCOLS>`).

**`GxArrayViewInterpolatorCore` trait**

- Provides default implementations for all four separable convolution variants and the unified
  dispatcher `array1_interp2_separable_core`.
- Concrete interpolators now only need to implement `compute_weights`; the kernel application logic
  is inherited automatically.
- B-spline, bicubic, and linear interpolators delegate their `array1_interp2` implementation to
  `array1_interp2_separable_core`, eliminating significant duplication.
- The nearest-neighbour interpolator retains its direct implementation, as its kernel is not
  separable in the same sense.
- `array1_interp2_separable_core` calls the unsafe variants internally — its bounds classification
  is the safety boundary that justifies this.

**Tests**

- Added a comprehensive unit-test module `core::interp::gx_array_view_interp_core_tests` covering
  all five trait methods across nominal, identity, multi-variable, boundary, out-of-bounds,
  masked-input, and output-mask scenarios.

#### Mask Validation — Branchless Checks and Short-Circuit Dispatch

*Module: `core::interp::gx_array_view_interp`*

- Added `is_valid_window<H, W>` to `GxArrayViewInterpolatorInputMaskStrategy`: a branchless
  AND-accumulated check returning `1` iff all samples in the `H × W` window are valid. Implemented
  for both `NoInputMask` and `BinaryInputMask`.
- Added `count_valid_window<H, W>` to the same trait: returns the number of valid samples in the
  `H × W` window. Implemented for `NoInputMask` and `BinaryInputMask`.
- Refactored `interpolate_masked_unchecked` to use a two-stage short-circuit mask validation,
  ordered by decreasing probability of the fast-exit case. This avoids the systematic call to
  `is_valid_weighted_window` and is the primary perf win on typical inputs.
- Each mask-strategy method (`is_valid`, `is_valid_window`, `count_valid_window`,
  `is_valid_weighted_window`) now provides a safe and an unsafe variant:
  - **Safe**: pre-sliced rows + iterators to eliminate inner-loop bounds checks.
  - **Unsafe**: `get_unchecked` on both mask data and weights.
- This change only meaningfully affects `BinaryInputMask`; `NoInputMask` is unchanged, as all its
  methods are compile-time constants with no data access.

#### Grid Resampling — Internal Restructuring

*Module: `core::gx_grid_resampling`*

- Renamed `validate` to `validate_mesh` in the `GridMeshValidator` trait (see Breaking Changes).
- Split the monolithic `match (ima_mask_in, ima_mask_out, check_boundaries)` block into a chain of
  three small generic helper functions, each resolving one interpolation-context component
  independently. This makes the dispatch logic substantially easier to follow and extend.

#### Grid Resampling — Handling of Unresolvable Grid Metrics

*Module: `grid_resampling.py`*

`array_grid_resampling` now degrades gracefully when grid metrics cannot be computed during
standalone preprocessing (e.g. malformed or degenerate input grid). This aligns the behaviour of the
standalone entry point with the chain entry point, which already exhibited this contract.

- **Previous behaviour:** an Exception raised by `standalone_preprocessing` propagated unhandled to
  the caller.
- **New behaviour:** the exception is caught internally and the following steps are executed in
  order:
  1. A `UserWarning` is emitted with the message
     `"Grid metrics cannot be computed. Check your input grid data"`.
  2. The interpolation core is bypassed entirely (no Rust call).
  3. The output array is filled with `nodata_out` and, if `array_out_mask` is provided, the
     corresponding region of the mask is set to `Validity.INVALID`.
  4. When `array_out_win` is provided, both substitutions are scoped to that window
     (`view[(..., *window_indices(array_out_win))] = nodata_out`); the rest of the output buffer and
     mask remain untouched.

#### Chain Subsystem — Shared Memory Plumbing

*Modules: `gridr.chain.grid_resampling_chain`, `gridr.chain.grid_mask_chain` (Python)*

- The chain subsystem now allocates its inter-process buffers through the new `SharedArray` facade
  rather than directly through `multiprocessing.shared_memory.SharedMemory`. Single-process / core
  API paths are unaffected.
- `gridr.scaling.shmutils` has been flagged as deprecated.
- `create_and_register_sma` has been renamed as `create_and_register`. It now appends `SharedArray`
  instances to the tracking list rather than name strings (see Breaking Changes).
  `SharedArray.clear_buffers` detects which form the list contains and dispatches accordingly.

#### Documentation

- Restructured the tutorial notebooks into focused, single-topic series for easier navigation and
  selective reading.
- The monolithic core resampling tutorial is now split into 8 notebooks
  (`grid_resampling_001_getting_started` through `grid_resampling_008_pipeline_mode`).
- The monolithic `grid_resampling_chain_001` notebook is now split into 8 notebooks
  (`grid_resampling_chain_000_overview` through `grid_resampling_chain_007_io_memory`), with a new
  overview page mapping every core API argument to its chain counterpart.
- The monolithic `grid_masking_001` notebook is now split into 5 notebooks
  (`core_masking_001_conventions` through `core_masking_005_combined`).
- Each series has its own Sphinx index page (`*_index.rst`) with a recommended reading order and
  per-page prerequisites.
- Deleted `doc` folder and moved logo images to `docs/source/_static/images`.

### Fixed

#### Chain Grid Resampling

*Module: `grid_resampling_chain.py`*

- `basic_grid_resampling_chain` now validates that the input mask, when provided, shares the same
  width and height as the input source array. Previously, mismatched dimensions could surface as
  opaque indexing errors deep in the Rust call stack; the check now raises a clear, early error.

### Build & Tooling

- **Makefile**
  - Rust build steps are now limited to the necessary cases (source changes or missing compiled
    library), eliminating spurious rebuilds.
  - The `test-python` rule now excludes benchmark tests, which are run separately via dedicated
    targets.
  - Added `RUSTFLAGS` environment variable with value `-C target-feature=+avx2,+fma`. This can be
    overriden by caller.

## [0.5.2] - 2026-03-13

### Fixed

#### Grid Resampling

- **Core Grid Resampling (`grid_resampling.py`)**
  - Fixed an UnboundLocalError in `calculate_source_extent()` when grid metrics could not be
    computed - the variable was referenced before assignment in that code path.
  - Added the `test_grid_resampling_standalone_no_idendity_1_1_full_invalid_grid()` in
    `test_grid_resampling.py` to cover the previous mentioned path.

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
  - Added Rust module `core/interp/gx_bspline_prefiltering` implementing B-Spline prefiltering
    methods and mask handling
  - Added Rust module `core/interp/gx_bspline_kernel` implementing `GxArrayViewInterpolator` trait
    for B-Spline
  - Implemented `GxBSplineInterpolatorTrait<const N: usize>` trait using monomorphic approach for
    compile-time B-Spline order specification
  - Created Python bindings for `BSpline<N>Interpolator` classes for all supported orders
  - Added Rust test suite for B-Spline interpolation
  - Created API documentation

- **Interpolator Margins**
  - Added `total_margins()` method to `GxArrayViewInterpolator` trait to compute required margins on
    each side for interpolation
  - For B-Spline interpolators, `total_margins()` includes pre-filtering domain extension for
    controlled precision in infinite sum approximations
  - Note: `initialize()` method must be called before `total_margins()` on B-Spline interpolator
    objects
  - Created Python bindings for `total_margins()` method
  - Added Python test suite for margin computation

#### Grid Operations

- **Array Arithmetic Functions**
  - Added `array1_add` Rust function in `gx_array_utils.rs` module
  - Added `array1_add_win2` Rust function in `gx_array_utils.rs` module
  - Created Python bindings for `array1_add` and `array1_add_win2` functions
  - Implemented `array_add` method in `gridr.core.array_utils` with comprehensive tests

- **Grid Coordinate Operations**
  - Added `array_shift_grid_coordinates` method to `core.grid.grid_utils` for in-place grid
    coordinate shifting

- **Source Boundary Computation**
  - Added `array1_compute_resampling_grid_src_boundaries` Rust function in `gx_grid_geometry.rs`
    module
  - Created Python bindings for `array1_compute_resampling_grid_src_boundaries` function
  - Implemented `array_compute_resampling_grid_src_boundaries` method in
    `gridr.core.grid.grid_utils` with comprehensive tests

#### Grid Resampling

- **Core Grid Resampling (`array_grid_resampling`)**
  - Added `check_boundaries` boolean parameter to activate safe grid boundaries computation (see
    Source Boundary Computation)
  - Added `boundary_condition` parameter accepting values: 'edge', 'reflect', 'symmetric', 'wrap'
  - Added `standalone` parameters to `gridr.core.grid.grid_resampling.array_grid_resampling`
  - Standalone mode performs: automatic grid metrics computation, source image required region
    computation (considering interpolation margins), optional padding operation, and B-Spline
    prefiltering
  - Updated test suite to include standalone mode validation

- **Grid Resampling Chain (`basic_grid_resampling_chain`)**
  - Added `grid_shift` parameter to apply global bias to grid coordinates
  - Added `boundary_condition` parameter accepting values: 'edge', 'reflect', 'symmetric', 'wrap'
  - Added `SAFECHECK_SOURCE_BOUNDARIES` constant (default: True) to determine read window using all
    valid coordinates within working grid subset (see Source Boundary Computation)
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
  - Implemented `FromPyObject` trait for `AnyInterpolator` to retrieve Rust object from Python
    object
  - `AnyInterpolator` enum variants wrap interpolator implementations using `Arc` (Atomic Reference
    Counting) and `RwLock` for thread-safe shared ownership
  - Note: This change requires interpolator creation on Python side before passing to Rust functions

#### Code Organization

- **Function Location**
  - Moved `read_win_from_grid_metrics()` function from `gridr.chain.grid_resampling_chain` to
    `gridr.core.grid.grid_utils`

#### Numerical Precision

- **Grid Coordinate Calculations**
  - Implemented controlled rounding to 12 decimal digits for internal full-resolution grid
    calculations in `gx_grid_resampling.rs`

#### Documentation Infrastructure

- **Sphinx Configuration**
  - Updated MathJax from version 2 to version 3
  - Changed MathJax CDN URL from
    `https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML` to
    `https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js`
  - Added `mathjax3_config` with inline math delimiters: `[['$', '$'], ['\\(', '\\)']]` and display
    math delimiters: `[['$$', '$$'], ['\\[', '\\]']]`
  - Added `sphinxcontrib.bibtex` extension
  - Configured `bibtex_bibfiles` to `['references.bib']`
  - Set `bibtex_default_style` to `'plain'`
  - Added MyST extensions: `dollarmath` and `amsmath`
  - Set `myst_dmath_double_inline` to `True`

- **Documentation Structure**
  - Changed page title from "API Documentation" to "API Reference" in
    `docs/source/api_python/modules.rst`
  - Reorganized Sphinx main menu structure
  - Removed `docs/source/developer_guide/index.rst` section heading
  - Removed `docs/source/api_python/misc/index.rst` module documentation
  - Removed `docs/source/api_python/misc/mandrill.rst` documentation page

- **Notebook References**
  - Updated notebook reference from `grid_resampling_001_work.ipynb` to `grid_resampling_001.ipynb`
    in `docs/source/_notebooks/index.rst`

- **Images**
  - Added new SVG file `docs/source/images/numeric_image.svg`

#### Project Metadata

- **README Badges**
  - Updated minimum pyo3 version badge from "0.26+" to "0.27.2+"
  - Changed logo image source from `./doc/images/gridr_logo.svg` to
    `https://github.com/CNES/gridr/tree/main/doc/images/gridr_logo.svg`

### Fixed

#### Grid Resampling

- **Out-of-Bounds Access**
  - Resolved panic when addressing index out-of-bounds of source array
  - Issue occurred for grids that do not preserve source topology
  - Fixed through implementation of safe grid source boundaries computation

#### Numerical Stability

- **Floating-Point Precision**
  - Resolved floating-point precision issues in grid coordinate calculations in
    `gx_grid_resampling.rs`
  - Fixed nearest neighbor interpolation bug caused by precision limitations
  - Addressed discrepancies between chain method and monolithic core method

#### Dependencies

- **Missing Dependencies**
  - Added missing Shapely dependency in `pyproject.toml`

#### Documentation

- **Styling**
  - Improved CSS styling of log extracts in `grid_resampling_chain` notebook for proper line
    breaking


## [0.4.3] - 2025-10-21

### Fixed
- **Documentation**: Fixed style errors for contributing guidelines.


## [0.4.2] - 2025-10-20

### Added
- **Documentation**: Added `.readthedocs.yaml` for automated docs builds.
- **Build**: Support for custom Sphinx HTML output directory in `Makefile`.

### Changed
- **CI/CD**: Release jobs now trigger **only** on semantic version tags (e.g., `vX.Y.Z`).
- **Dependencies**: Upgraded `pyo3` to **v0.26.0** to address security vulnerability .
- **Documentation**: Changed the contributing part in order to adhere the strict linear trunk-based
  branching strategy.

### Fixed
- **Build**: Generate ABI3-compatible wheels for Python 3.10+.
- **Documentation**: Fix indentations in array_utils.py docstrings.

### Security
- **Dependencies**: Patched `pyo3` to v0.26.0 (addresses GHSA-pph8-gcv7-4qj5).


## [0.4.1] - 2025-10-02

### Added
- Added usage of pre-commit with flake8, isort and black
- Added license and opensource related files : LICENSE, NOTICE, AUTHORS.md, CONTRIBUTING.md, Clause
  of License Aggreements files
- Added Developer_Guide (WIP) and License sections in sphinx documentation
- scripts:
    - Added scripts directory in project tree
    - Added generate_notice.py and generate_Rust_notice.sh script
- templates:
    - Added templates directory in project tree
    - Added templates to generate main NOTICE and Python/Rust 3rd party notices sections.
- Added Rust tests for `GxNearestInterpolator` and `GxLinearInterpolator` to verify mask usage
- Added Python test for the core `grid.grid_resampling.py` module. The test currently covers only
  identity grid transforms at full resolution.
- Added `array_convert`, `is_clip_required` and `is_clip_to_dtype_limits_safe` Python methods in the
  `gridr.core.utils.array_utils.py` module as well as corresponding tests. These methods are used by
  the `grid_resampling_chain` to convert data type before writing to disk in order to match the
  output dataset type.

### Changed
- Added license related header in Python and Rust source files
- Apply pre-commit hooks (flake8, isort, black) to existing Python source files in Python/gridr and
  tests
- Documentation
- Added a temporary change in `grid_resampling_chain` to adapt margin to the interpolation function.
- Tests for `grid_resampling_chain` have been enhanced to include interpolators other than `cubic`
  and output types different from `np.float64`. The `linear` interpolator is now tested, while the
  `nearest` interpolator is not due to an identified bug.
- Tests for `grid_resampling_chain` now use the newly local implemented utility method
  `assert_all_close_with_details`, which provides detailed information about differences when they
  occur.

### Fixed
- Fixed the `array1_interp2` for `GxNearestInterpolator` to properly set the nodata value for masked
  output pixels.
- Fixed the `grid_resampling_chain` method to work with output datasets whose types are different
  from float64.
- Fixed the test class name in `test_array_utils` from TestArrayWindow to TestArrayUtils.


## [0.4.0] - 2025-08-27

### Added

- Grid Masks:
    - (Rust Core) GridNodeValidator trait and implementation supporting No Mask, Raster mask and
      Sentinel value options.
    - Configurable valid cell value in grid masks (replaces hardcoded value of 1)
    - Configurable sentinel value within grid coordinates to identify invalid cells
    - Using grid masks or a sentinel value are optional and mutually exclusive
    - Test coverage for new masking options
- Grid Geometric Metrics:
    - New data structure to hold grid geometric metrics
    - Computation of grid geometric metrics for in-memory grids
- Grid Resampling
    - (Rust Core) Introduced trait strategies design concept for Input data masks, Output data masks
      and Bound checks
    - (Rust Core) Implemented strategies traits to cover both with and without options
    - (Rust Core) Added GxArrayViewInterpolationContextTrait trait to wrap the strategies traits
    - Nearest neighbor and linear interpolation methods
    - Configurable interpolation method
    - Configurable indexing shift for input array coordinates
    - Configurable target window for output buffer
    - Safe method `grid_resolution_window_safe` for handling edge cases in grid definitions
    - Chain module for managing I/O (input/output image, grids and input/output masks) and memory
      with tiling capabilities
- Antialiasing
    - Added reciprocal cell frequential filter functionality
    - Implemented core method to compute antialiasing filter from a grid using reciprocal cell
      frequential filter
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
