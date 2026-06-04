.. _environment_variables:

=====================
Environment variables
=====================

This page lists the environment variables recognized by GridR. It is a
quick reference; for the design rationale and inner workings of the
features they control, follow the links to the dedicated topic pages.

Variables are read **once at startup** of the relevant component.
Changing them at runtime has no effect on objects already created.


Reference
=========

.. list-table::
   :header-rows: 1
   :widths: 28 12 20 40

   * - Variable
     - Type
     - Default
     - Documented in

   * - :envvar:`GRIDR_SHARED_MEMORY_BACKEND`
     - string
     - auto
     - :mod:`gridr.scaling.shared_array`


   * - :envvar:`GRIDR_SHM_MIN_FREE`
     - bytes
     - 64 MiB
     - :mod:`gridr.scaling.shared_array`



Shared memory backend
=====================

These variables control how GridR's chaining subsystem shares numpy
arrays between worker processes. They affect the chain pipelines under
:mod:`gridr.chain` only; the single-process API reads no environment variable.

GridR ships three shared-memory backends — ``shm``, ``mmap`` and
``memfd`` — with different trade-offs depending on the deployment
environment (Docker, ``/dev/shm`` availability, fork vs spawn). The
default behaviour is to auto-detect the most appropriate backend; the
variables below let you override this when needed.

For the full description of the backends and the auto-detection
algorithm, see :mod:`gridr.scaling.shared_array`.

:envvar:`GRIDR_SHARED_MEMORY_BACKEND`
-------------------------------------

Forces the shared-memory backend, bypassing auto-detection. Accepts
``shm``, ``mmap`` or ``memfd``. Typical uses:

* ``shm`` — Standard environments with a properly sized ``/dev/shm``.
  This is the historical behaviour, suitable for development
  workstations.
* ``mmap`` — Memory-constrained Linux containers (e.g. Docker with the
  default 64 MB ``/dev/shm`` cap). No filesystem footprint; works only
  with the ``fork`` multiprocessing start method.
* ``memfd`` — Linux deployments planning the move to ``spawn`` workers
  (Python 3.14+). Requires kernel ≥ 3.17.

Invalid values are silently ignored and auto-detection runs instead.

See :mod:`gridr.scaling.shared_array` for the canonical reference.

:envvar:`GRIDR_SHM_MIN_FREE`
----------------------------

Threshold in bytes used by the shared-memory auto-detection routine.
When ``/dev/shm`` reports at least this many free bytes, the ``shm``
backend is selected; otherwise the chain falls back to ``mmap`` or
``memfd``. Default: 64 MiB (``67108864``).

Ignored when :envvar:`GRIDR_SHARED_BACKEND` is explicitly set.

See :mod:`gridr.scaling.shared_array` for the canonical reference.
