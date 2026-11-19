.. _antialiasing:

========
Aliasing
========

Resampling an image means computing its values on a new sampling grid. The destination grid may be shifted, rotated,
scaled or deformed with respect to the source grid, and its spacing may change from one place to another. Wherever the
destination cannot represent some frequencies of the source, under a decimation but also under a rotation at unit
scale, the source holds details too fine for the destination. Sampling does not discard these details: it folds them back
into coarser, false patterns, such as moiré on periodic textures, jagged edges or flicker. This is aliasing.
Interpolation does not prevent it. An interpolation kernel reconstructs the source between its samples; it does not
remove the details that the destination cannot hold.

.. toctree::
   :maxdepth: 1

   aliasing_and_reciprocal_lattice
   source_prefiltering
   experimental_options
   transition_matrix_estimation
