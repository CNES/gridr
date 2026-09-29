Using the GridR's FFT Filtering chain API
=========================================

This guide demonstrates GridR's FFT Filtering chain feature, which applies the
core filtering to rasterio datasets read from and written to disk. It is split
into a series of focused tutorials. Each page is self-contained and executable
as a Jupyter notebook -- you can follow them in order or jump directly to the
topic you need.

The chain reuses the core layer for the filtering itself, so the guide assumes
the arguments described in *Using the GridR's core FFT Filtering API* are
known. Each page declares its prerequisites and ends with a pointer to the next
logical step.

.. rubric:: Recommended reading order

The pages are ordered for first-time readers below. Use the *Previous / Next*
links at the bottom of each page to navigate sequentially.

1. **Overview** -- how the chain relates to the core layer, and which core
   argument maps to which chain argument
2. **Getting started** -- your first call to ``fft_filtering_oa_strip_chain``,
   and how to size the output dataset from the filtering parameters
3. **Production window and decimation** -- producing a region of interest,
   decimating it, and deriving the output georeferencing
4. **Strips, memory and tracing** -- cutting the window with ``strip_size``,
   following the loop with ``logger``, and bounding the memory of a run

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   generated/fft_filtering_chain_000_overview
   generated/fft_filtering_chain_001_getting_started
   generated/fft_filtering_chain_002_production_window
   generated/fft_filtering_chain_003_strips_and_memory