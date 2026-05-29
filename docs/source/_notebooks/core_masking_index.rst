Using the GridR's core build mask API
=====================================

This guide demonstrates GridR's core masking feature, built around the
``grid_rasterize`` and ``grid_mask.build_mask`` utilities. It is split
into a series of focused tutorials. Each page is self-contained and
executable as a Jupyter notebook -- you can follow them in order or jump
directly to the topic you need.

Each page declares its prerequisites and ends with a pointer to the next
logical step.

.. rubric:: Recommended reading order

The pages are ordered for first-time readers below. Use the *Previous /
Next* links at the bottom of each page to navigate sequentially.

1. **Conventions** -- pixel coordinate conventions and ``geometry_origin``
2. **Rasterize** -- the ``grid_rasterize`` module
3. **Build mask: geometries** -- the ``geometry_pair`` argument
4. **Build mask: input raster** -- ``mask_in`` and related arguments
5. **Build mask: combined** -- all masking modes together

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   generated/core_masking_001_conventions
   generated/core_masking_002_rasterize
   generated/core_masking_003_build_mask_geometries
   generated/core_masking_004_build_mask_raster
   generated/core_masking_005_combined
