Using the GridR's core grid resampling API
==========================================

This guide walks you through GridR's core grid resampling API,
``array_grid_resampling``, in a series of focused tutorials. Each page is
self-contained and executable as a Jupyter notebook — you can follow them
in order or jump directly to the topic you need.

Each page declares its prerequisites and ends with a pointer to the next
logical step.

.. rubric:: Recommended reading order

The pages are ordered for first-time readers below. Use the *Previous /
Next* links at the bottom of each page to navigate sequentially.

1. **Getting started** — your first call to ``array_grid_resampling``
2. **Geometric transformations** — translation, rotation, zoom
3. **Output control** — ``nodata_out``, validity masks, output windows
4. **Masking inputs** — grid masks and array masks
5. **B-Spline masking** — cardinal B-Spline interpolation and its specifics
6. **Boundary conditions** — handling source-array edges
7. **Standalone mode** — what GridR does for you automatically
8. **Pipeline integration** — disabling standalone for chained workflows

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   generated/grid_resampling_001_getting_started
   generated/grid_resampling_002_geometric_transformations
   generated/grid_resampling_003_output_control
   generated/grid_resampling_004_masking
   generated/grid_resampling_005_bspline_masking
   generated/grid_resampling_006_boundary_conditions
   generated/grid_resampling_007_standalone_mode
   generated/grid_resampling_008_pipeline_mode
