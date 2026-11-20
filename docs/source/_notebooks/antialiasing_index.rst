Building the GridR reciprocal cell antialiasing filter
======================================================

This guide builds the reciprocal cell antialiasing filter of
``gridr.core.filter.antialiasing.reciprocal_cell`` for a decimation by 4: how
``resolve_antialiasing_filter`` turns a ``ReciprocalCellFilter`` specification
into a kernel for a grid, how the kernel is applied, and what each parameter
changes. The page is executable as a Jupyter notebook.

The derivations and the design choices are given in the theoretical
foundations: :doc:`../theoretical_foundations/antialiasing/aliasing_and_reciprocal_lattice`
for the reciprocal lattice, :doc:`../theoretical_foundations/antialiasing/source_prefiltering`
for the filter.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   generated/antialiasing_001_filter_construction
