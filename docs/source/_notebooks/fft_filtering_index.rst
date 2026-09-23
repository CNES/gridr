Using the GridR's core FFT Filtering API
========================================

This guide demonstrates GridR's core FFT Filtering feature. It is split
into a series of focused tutorials. Each page is self-contained and
executable as a Jupyter notebook -- you can follow them in order or jump
directly to the topic you need.

Each page declares its prerequisites and ends with a pointer to the next
logical step.

.. rubric:: Recommended reading order

The pages are ordered for first-time readers below. Use the *Previous /
Next* links at the bottom of each page to navigate sequentially.

1. **Getting started** — your first call to ``fft_array_filter``
2. **Boundary conditions and output modes** -- how the input is extrapolated
   beyond its domain, and which part of the convolution is returned
3. **Limiting computation to a window** -- filtering a region of interest instead of the whole
   array
4. **Zooming out and aliasing** -- undersampling the output with ``zoom``,
   controlling the decimation phase, and sizing the kernel accordingly


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   generated/fft_filtering_001_getting_started
   generated/fft_filtering_002_boundary_and_output_mode
   generated/fft_filtering_003_production_window
   generated/fft_filtering_004_zooming