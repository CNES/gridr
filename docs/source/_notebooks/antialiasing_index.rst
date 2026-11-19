Building the GridR reciprocal cell antialiasing filter
======================================================

This guide demonstrates the construction of the reciprocal cell antialiasing
filter of ``gridr.core.filter.antialiasing.reciprocal_cell``: how
``resolve_antialiasing_filter`` turns a ``ReciprocalCellFilter``
specification into a kernel for a given grid, and what each option of the
specification changes. The page is executable as a Jupyter notebook.

The derivations and the design choices are given in the theoretical
foundations: :doc:`../theoretical_foundations/antialiasing/aliasing_and_reciprocal_lattice`
for the reciprocal lattice, :doc:`../theoretical_foundations/antialiasing/source_prefiltering`
for the filter.

.. note::
   Coming in a future commit !
