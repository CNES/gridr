Using the GridR's chain grid resampling API
===========================================

This guide walks you through GridR's chain grid resampling API,
``basic_grid_resampling_chain``, in a series of focused tutorials. The
chain layer is the file-oriented wrapper around the core
``array_grid_resampling`` function: it reads from and writes to rasterio
datasets, and processes the output in memory-efficient strips and tiles.

Each page is self-contained and executable as a Jupyter notebook — you
can follow them in order or jump directly to the topic you need. Pages
declare their prerequisites and end with a pointer to the next logical
step.

.. rubric:: Recommended reading order

The pages are ordered for first-time readers below. Use the *Previous /
Next* links at the bottom of each page to navigate sequentially. If you
are already familiar with the core API (``array_grid_resampling``),
start with the overview to see how chain arguments map to their core
counterparts.

0. **Overview** — relationship between the chain and core layers
1. **Getting started** — your first call to ``basic_grid_resampling_chain``
2. **Output mask** — requesting an output validity mask
3. **Grid mask** — flagging grid nodes as invalid
4. **Source mask** — flagging source raster pixels as invalid
5. **Geometry masks** — using Shapely polygons as masks
6. **Shift and window** — ``grid_shift`` and computation window
7. **I/O and memory** — strips, tiles, logging

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   generated/grid_resampling_chain_000_overview
   generated/grid_resampling_chain_001_getting_started
   generated/grid_resampling_chain_002_output_mask
   generated/grid_resampling_chain_003_grid_mask
   generated/grid_resampling_chain_004_source_mask
   generated/grid_resampling_chain_005_geometry_masks
   generated/grid_resampling_chain_006_shift_and_window
   generated/grid_resampling_chain_007_io_memory
