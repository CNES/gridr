# CHANGELOG

## GridR v0.4.0
Release Date : 2025-MM-DD

### New features

#### New Masking-Related Arguments in the `array_grid_resampling` Method

The following new arguments have been introduced to the `gridr.core.grid_resampling.array_grid_resampling` Python method:

- The value used to denote a valid grid cell in the `grid_mask` is now configurable via the `grid_mask_valid_value` argument, replacing the previous hardcoded value of 1.

- The `grid_nodata` argument now allows direct use of raster grids to determine invalid grid cells.

- Both `grid_nodata` and `grid_mask` are optional and mutually exclusive.

- **Important Note** : There has been an implicit convention change. The default value considered valid has changed from 0 to 1. This change is driven by upcoming modifications to allow masks to function as factors. It may be propagated across other codes to ensure consistency.

These updates in the Python API correspond to the equivalent changes in the Rust core code.