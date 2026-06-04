# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of GRIDR
# (see https://github.com/CNES/gridr).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
r"""
Unified cross-process numpy arrays with pluggable OS backends.

This module provides a single :class:`SharedArray` facade for sharing
:class:`numpy.ndarray` buffers between Python processes, backed by one of
three interchangeable mechanisms:

* ``shm`` — :class:`multiprocessing.shared_memory.SharedMemory` (named POSIX
  shared memory, stored under ``/dev/shm``).
* ``mmap`` — anonymous ``mmap(MAP_SHARED)`` (no filesystem footprint,
  inherited by forked children).
* ``memfd`` — Linux ``memfd_create(2)`` (anonymous, file-descriptor based,
  compatible with both ``fork`` and ``spawn``).

The active backend is chosen once at process startup, either explicitly or
by auto-detection. All :class:`SharedArray` instances created afterwards use
that backend transparently.

.. contents:: Module contents
   :local:
   :depth: 2

Why three backends?
===================

Each backend trades off three concerns:

1. **Storage location** — whether the buffer lives in ``/dev/shm`` (a
   ``tmpfs`` partition of fixed and often small size) or in the address
   space of the process.
2. **Process-start compatibility** — whether the buffer is reachable from
   children started with ``fork``, with ``spawn``, or both.
3. **Platform** — Linux, macOS, or Windows.

The pluggable design lets the same code run unchanged across:

* Tightly constrained Docker containers where ``/dev/shm`` is too small.
* Future Python releases that default to ``spawn`` on Linux (3.14+).
* Cross-platform developer workstations.

fork vs spawn — a refresher
===========================

When :mod:`multiprocessing` starts child processes, two fundamentally
different mechanisms can be used.

fork (default on Linux, available on macOS)
-------------------------------------------

``fork()`` duplicates the parent's address space (copy-on-write at the page
level). The child inherits every Python object, file descriptor, mmap
region and OS resource the parent had at the moment of the fork. No data
is serialized, and the operation is essentially free even on multi-gigabyte
processes.

For :class:`SharedArray`, a forked child simply calls
:meth:`SharedArray.load` to rebuild the numpy view on a buffer it already
has in its address space.

spawn (default on Windows, opt-in elsewhere, Linux default in 3.14+)
--------------------------------------------------------------------

``spawn()`` starts a fresh Python interpreter and re-executes the entry
point. The child knows nothing of the parent's state; whatever it needs
must be transmitted explicitly via pickle or a fd-passing mechanism.

For :class:`SharedArray`, the parent calls
:meth:`SharedArray.get_passing_payload` to obtain a serializable
description, and the child rebuilds the array with
:meth:`SharedArray.from_payload`. Whether this works depends on the chosen
backend:

* ``shm``: works (segment re-attached by name).
* ``memfd``: works, but requires the file descriptor to be transmitted via
  :func:`multiprocessing.reduction.send_handle` or ``SCM_RIGHTS``.
* ``mmap``: does **not** work (anonymous mappings have no transmissible
  handle).

Default start method by platform
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Platform
     - Python ≤ 3.13
     - Python 3.14+
     - Other modes

   * - Linux
     - ``fork``
     - ``spawn``
     - ``forkserver``

   * - macOS
     - ``spawn``
     - ``spawn``
     - ``fork``, ``forkserver``

   * - Windows
     - ``spawn``
     - ``spawn``
     - ``spawn`` only

Backend reference
=================

shm — multiprocessing.shared_memory
-----------------------------------

A named POSIX shared memory segment stored under ``/dev/shm``.

**Strengths**

* Works on every Python platform (Linux, macOS, Windows).
* Compatible with both ``fork`` and ``spawn`` — children re-attach by name.
* Inspectable from the shell with ``ls /dev/shm``.

**Weaknesses**

* Constrained by ``/dev/shm`` total size — 64 MB by default in Docker.
* Segments must be explicitly unlinked, or they leak until reboot.
* CPython bug pre-3.13: ``ResourceWarning`` may unlink the segment when
  the creator exits even if children are still attached.

**Best for**: development environments, cross-platform deployments,
spawn-based pipelines with ample ``/dev/shm``.

mmap — anonymous MAP_SHARED
---------------------------

An anonymous mapping created with ``mmap.mmap(-1, size, MAP_SHARED)``. No
name, no filesystem entry; visible only to the creator and its forked
descendants.

**Strengths**

* Zero filesystem footprint, independent of ``/dev/shm``.
* Released automatically when the last reference goes out of scope.

**Weaknesses**

* ``fork``-only. Not usable with ``spawn`` or on Windows.
* Cannot be re-attached after the creator dies.

**Best for**: Linux containers with tight ``/dev/shm`` and a ``fork`` start
method.

memfd — memfd_create + fd passing
---------------------------------

A Linux-only backend using ``memfd_create(2)`` to allocate anonymous memory
accessible through a file descriptor, then mapped with ``mmap``.

**Strengths**

* Independent of ``/dev/shm``.
* Compatible with both ``fork`` (fd inherited) and ``spawn`` (fd
  transmissible).
* Inspectable through ``/proc/<pid>/fd``.
* Cleaned up automatically when the last fd closes.

**Weaknesses**

* Linux only (kernel ≥ 3.17, glibc ≥ 2.27).
* ``spawn`` requires explicit fd-passing code in the parent.

**Best for**: Linux deployments preparing for Python 3.14's ``spawn``
default while keeping anonymous shared memory.

Backend comparison
==================

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Property
     - ``shm``
     - ``mmap``
     - ``memfd``
   * - OS resource
     - Named segment in ``/dev/shm``
     - Address-space mapping only
     - Anonymous fd + mapping
   * - Uses ``/dev/shm``
     - Yes
     - No
     - No
   * - Works with ``fork``
     - Yes
     - Yes
     - Yes
   * - Works with ``spawn``
     - Yes (by name)
     - No
     - Yes (by fd-passing)
   * - Linux
     - Yes
     - Yes
     - Yes
   * - macOS
     - Yes
     - Yes
     - No
   * - Windows
     - Yes
     - No
     - No
   * - Cleanup
     - Manual ``unlink``
     - Automatic
     - Automatic

Backend selection
=================

The backend is chosen once and reused for the lifetime of the process.
Three mechanisms drive the choice, in order of precedence:

1. The :envvar:`GRIDR_SHARED_BACKEND` environment variable.
2. An explicit call to :func:`set_backend`.
3. Auto-detection if neither of the above is set.

Auto-detection logic
--------------------

In order:

1. If the platform is Windows: ``shm``.
2. If ``/dev/shm`` reports strictly more than :envvar:`GRIDR_SHM_MIN_FREE` bytes free
   (default 64 MB): ``shm``.
3. If ``fork`` is available: ``mmap``.
4. If ``memfd_create`` is available: ``memfd``.
5. Otherwise: ``mmap`` with a warning.

Usage
=====

Basic example (single process)
------------------------------

.. code-block:: python

   import numpy as np
   from gridr.scaling.shared_array import SharedArray

   sa = SharedArray(
       shape=(1024, 1024),
       dtype=np.float32,
       name=SharedArray.build_name(prefix="buffer"),
   )
   sa.create()
   sa.array[:] = 0.0
   sa.destroy()

Sharing across forked workers
-----------------------------

With the default ``fork`` start method on Linux, no special handling is
required:

.. code-block:: python

   import multiprocessing as mp
   import numpy as np
   from gridr.scaling.shared_array import SharedArray

   def worker(sa, idx):
       sa.load()
       sa.array[idx] = idx ** 2

   if __name__ == "__main__":
       sa = SharedArray(shape=(100,), dtype=np.int64,
                        name=SharedArray.build_name("squares"))
       sa.create()
       ctx = mp.get_context("fork")
       with ctx.Pool(4) as pool:
           pool.starmap(worker, [(sa, i) for i in range(100)])
       sa.destroy()

Sharing across spawned workers
------------------------------

With ``spawn``, the child receives only what the parent transmits. The
parent calls :meth:`SharedArray.get_passing_payload`, the child rebuilds
the array with :meth:`SharedArray.from_payload`. This works for ``shm``
and ``memfd`` backends; ``mmap`` raises :exc:`RuntimeError`.

For ``memfd``, the file descriptor must additionally be transferred to the
child via :func:`multiprocessing.reduction.send_handle` or
``SCM_RIGHTS`` over a Unix-domain socket.

Cleaning up registered buffers
------------------------------

.. code-block:: python

   from gridr.scaling.shared_array import SharedArray, create_and_register

   buffers = []
   sa1 = create_and_register((512, 512), np.float32, buffers, prefix="grid")
   sa2 = create_and_register((256, 256), np.uint8,  buffers, prefix="mask")
   # ... pipeline ...
   SharedArray.clear_buffers(buffers)

Concurrent access
=================

:class:`SharedArray` does **not** provide synchronization. The caller is
responsible for consistency. Common patterns:

* **Disjoint write regions** — workers each write to a disjoint slice. No
  locking needed.
* **Phased access** — write phase, :class:`multiprocessing.Barrier`, then
  read phase.
* **Per-region atomic flags** — fine-grained progress tracking using
  atomic flags placed in a second shared buffer.

Environment variables
=====================

.. envvar:: GRIDR_SHARED_MEMORY_BACKEND

   Forces the backend used by all subsequent :class:`SharedArray`
   instances. Accepted values: ``shm``, ``mmap``, ``memfd``.

.. envvar:: GRIDR_SHM_MIN_FREE

   Threshold in bytes used by the ``auto`` selector. DThe ``shm`` backend
    is selected only when ``/dev/shm`` has **strictly more** than this many
    bytes free. Defaults to ``67108864`` (64 MiB).

Compatibility
=============

* Python 3.10+
* Linux: all three backends
* macOS: ``shm``, ``mmap``
* Windows: ``shm`` only

See also
========

* :mod:`multiprocessing.shared_memory`
* :mod:`mmap`
* :manpage:`memfd_create(2)`
* :mod:`multiprocessing.reduction`
"""
from __future__ import annotations

import abc
import ctypes
import logging
import mmap
import os
import shutil
import sys
from datetime import datetime
from functools import wraps
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# memfd_create syscall wrapper (Linux only)
# ---------------------------------------------------------------------------

_MFD_CLOEXEC = 0x0001
_memfd_create_supported: Optional[bool] = None
_libc: Optional[ctypes.CDLL] = None


def _memfd_create(name: str, size: int) -> int:
    """Call memfd_create(2) and ftruncate to `size`. Returns the fd."""
    global _libc

    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)

    if not hasattr(_libc, "memfd_create"):
        raise NotImplementedError("memfd_create symbol not available in libc (needs glibc >= 2.27)")

    _libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    _libc.memfd_create.restype = ctypes.c_int

    fd = _libc.memfd_create(name.encode("utf-8"), _MFD_CLOEXEC)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"memfd_create failed: {os.strerror(err)}")

    try:
        os.ftruncate(fd, size)
    except OSError:
        os.close(fd)
        raise
    return fd


def _has_memfd() -> bool:
    """Probe whether memfd_create is usable on this platform."""
    global _memfd_create_supported
    if _memfd_create_supported is not None:
        return _memfd_create_supported
    if not sys.platform.startswith("linux"):
        _memfd_create_supported = False
        return False
    try:
        fd = _memfd_create("gridr_probe", 4096)
        os.close(fd)
        _memfd_create_supported = True
    except (OSError, NotImplementedError, AttributeError) as e:
        logger.debug(f"memfd_create not available: {e}")
        _memfd_create_supported = False
    return _memfd_create_supported


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ("shm", "mmap", "memfd", "auto")
_ENV_BACKEND = "GRIDR_SHARED_MEMORY_BACKEND"
_ENV_SHM_MIN_FREE = "GRIDR_SHM_MIN_FREE"

_active_backend: Optional[str] = None
_requested_backend: str = "auto"


def _shm_free_bytes() -> int:
    try:
        return shutil.disk_usage("/dev/shm").free
    except OSError:
        return 0


def _fork_available() -> bool:
    try:
        import multiprocessing as mp

        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


def _resolve_backend() -> str:
    """Pick a concrete backend based on requested mode, env, and capabilities."""
    env_val = os.environ.get(_ENV_BACKEND, "").strip().lower()

    if env_val != "" and env_val not in _VALID_BACKENDS:
        logger.warning(
            f"Invalid backend {env_val!r}. Expected one of {_VALID_BACKENDS}.\n"
            f"Automatic resolving..."
        )

    if env_val in ("shm", "mmap", "memfd"):
        logger.debug(f"shared backend forced via {_ENV_BACKEND}={env_val}")
        return env_val

    if _requested_backend in ("shm", "mmap", "memfd"):
        return _requested_backend

    if sys.platform.startswith("win"):
        return "shm"

    min_free = int(os.environ.get(_ENV_SHM_MIN_FREE, 64 * 1024 * 1024))
    free = _shm_free_bytes()
    if free > min_free:
        logger.debug(f"shared backend auto: shm (/dev/shm free={free})")
        return "shm"

    if _fork_available():
        logger.debug("shared backend auto: mmap (fork available, /dev/shm tight)")
        return "mmap"

    if _has_memfd():
        logger.debug("shared backend auto: memfd (no fork, memfd available)")
        return "memfd"

    logger.warning("Falling back to mmap; spawn workers will not see the buffer.")
    return "mmap"


def set_backend(name: str) -> None:
    """Force the backend for subsequent SharedArray creations."""
    global _requested_backend, _active_backend
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Invalid backend {name!r}. Expected one of {_VALID_BACKENDS}.")
    _requested_backend = name
    _active_backend = None
    logger.info(f"shared backend requested: {name}")


def get_backend() -> str:
    """Return the currently-active concrete backend."""
    global _active_backend
    if _active_backend is None:
        _active_backend = _resolve_backend()
        logger.info(f"shared backend active: {_active_backend}")
    return _active_backend


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class _MemoryBackend(abc.ABC):
    """Internal backend handle. Owns the OS resource and exposes a numpy view."""

    array: Optional[np.ndarray]

    @abc.abstractmethod
    def create(self, shape, dtype, name: str) -> None: ...  # noqa: E704

    @abc.abstractmethod
    def attach(self, shape, dtype, name: str) -> None: ...  # noqa: E704

    @abc.abstractmethod
    def detach(self) -> None: ...  # noqa: E704

    @abc.abstractmethod
    def destroy(self) -> None: ...  # noqa: E704

    def get_passing_payload(self) -> Optional[Dict[str, Any]]:
        """Return backend-specific data needed to reattach in another process."""
        return None


# ---------------------------------------------------------------------------
# Backend: shared_memory
# ---------------------------------------------------------------------------


class _ShmBackend(_MemoryBackend):
    """multiprocessing.shared_memory: named POSIX SHM in /dev/shm."""

    def __init__(self):
        self._shm: Optional[shared_memory.SharedMemory] = None
        self.array: Optional[np.ndarray] = None

    def create(self, shape, dtype, name: str) -> None:
        dtype = np.dtype(dtype)
        size = int(dtype.itemsize * int(np.prod(shape)))
        self._shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)

    def attach(self, shape, dtype, name: str) -> None:
        dtype = np.dtype(dtype)
        self._shm = shared_memory.SharedMemory(name=name)
        self.array = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)

    def detach(self) -> None:
        self.array = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def destroy(self) -> None:
        self.array = None
        if self._shm is not None:
            try:
                self._shm.close()
            finally:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None


# ---------------------------------------------------------------------------
# Backend: anonymous mmap (fork-only)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-process registry of anonymous mmap mappings.
#
# Anonymous mmaps cannot be pickled (the mmap.mmap object has no
# transmissible identity outside the process that created it). However,
# under fork the mapping is inherited at the same virtual address in the
# child, so the child only needs a way to *re-find* the mmap object after
# unpickling.
#
# We register every anonymous mmap in this module-level dict, keyed by a
# stable identifier (the SharedArray name). After fork, the dict itself
# is duplicated into the child's address space, with the same keys
# pointing to the same mmap objects. __reduce__ on the backend then only
# transmits the key, and __setstate__ looks the object up.
# ---------------------------------------------------------------------------
_MMAP_REGISTRY: Dict[str, mmap.mmap] = {}


class _MmapBackend(_MemoryBackend):
    """Anonymous MAP_SHARED mmap. Inherited via fork()."""

    def __init__(self):
        self._mm: Optional[mmap.mmap] = None
        self._key: Optional[str] = None  # registry key, set on create()
        self.array: Optional[np.ndarray] = None

    def _build_array(self, shape, dtype) -> np.ndarray:
        flat = np.frombuffer(self._mm, dtype=np.dtype(dtype))
        flat.setflags(write=True)
        return flat.reshape(shape)

    def create(self, shape, dtype, name: str) -> None:
        size = int(np.dtype(dtype).itemsize * int(np.prod(shape)))
        self._mm = mmap.mmap(-1, size, mmap.MAP_SHARED)
        self._key = name
        _MMAP_REGISTRY[name] = self._mm
        self.array = self._build_array(shape, dtype)

    def attach(self, shape, dtype, name: str) -> None:
        # Two cases:
        # 1. Same process (or forked child): self._mm is already set, or
        #    can be recovered from the registry under self._key / name.
        # 2. Unpickled in a forked child: __setstate__ has restored self._mm
        #    from the registry.
        if self._mm is None:
            mm = _MMAP_REGISTRY.get(self._key or name)
            if mm is None:
                raise RuntimeError(
                    f"mmap backend: no mapping found for name={name!r}. "
                    "The parent must create() before fork and the child "
                    "must be forked from that parent."
                )
            self._mm = mm
        self.array = self._build_array(shape, dtype)

    def detach(self) -> None:
        self.array = None

    def destroy(self) -> None:
        # Drop the numpy view first; numpy holds a buffer export on the mmap
        # which would cause BufferError on close otherwise.
        self.array = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                # A buffer export still exists somewhere (e.g. a clone or
                # a leftover numpy slice). Let GC reclaim the mapping.
                pass
            self._mm = None
        # Remove from registry so destroy is observable from any holder.
        if self._key is not None:
            _MMAP_REGISTRY.pop(self._key, None)
            self._key = None

    # ------------------------------------------------------------------
    # Pickle support — required because mmap.mmap is not picklable.
    #
    # We pickle only the registry key. After fork, the registry has been
    # duplicated into the child's address space, so the lookup succeeds.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        return {"_key": self._key}

    def __setstate__(self, state: dict) -> None:
        self._key = state.get("_key")
        self._mm = _MMAP_REGISTRY.get(self._key) if self._key else None
        self.array = None  # rebuilt on attach()


# ---------------------------------------------------------------------------
# Backend: memfd (Linux, fork OR spawn via fd passing)
# ---------------------------------------------------------------------------


class _MemfdBackend(_MemoryBackend):
    """
    Linux memfd_create-based backend.

    Allocates anonymous memory via memfd_create(2), mmaps the fd with
    MAP_SHARED. Unlike _MmapBackend, the fd can be passed to other
    processes (even spawn) via SCM_RIGHTS or
    multiprocessing.reduction.send_handle.
    """

    def __init__(self):
        self._fd: int = -1
        self._mm: Optional[mmap.mmap] = None
        self._size: int = 0
        self._owns_fd: bool = False
        self.array: Optional[np.ndarray] = None

    def _build_array(self, shape, dtype) -> np.ndarray:
        flat = np.frombuffer(self._mm, dtype=np.dtype(dtype))
        flat.setflags(write=True)
        return flat.reshape(shape)

    def create(self, shape, dtype, name: str) -> None:
        if not _has_memfd():
            raise RuntimeError("memfd_create is not available on this platform")
        size = int(np.dtype(dtype).itemsize * int(np.prod(shape)))
        self._fd = _memfd_create(name or "gridr_buf", size)
        self._size = size
        self._owns_fd = True
        self._mm = mmap.mmap(self._fd, size, mmap.MAP_SHARED)
        self.array = self._build_array(shape, dtype)

    def attach(self, shape, dtype, name: str) -> None:
        if self._mm is None:
            if self._fd < 0:
                raise RuntimeError(
                    f"memfd backend: no fd to attach to for name={name!r}. "
                    "Use SharedArray.from_payload() in spawned workers."
                )
            self._mm = mmap.mmap(self._fd, self._size, mmap.MAP_SHARED)
        self.array = self._build_array(shape, dtype)

    def detach(self) -> None:
        self.array = None

    def destroy(self) -> None:
        self.array = None
        if self._mm is not None:
            try:
                self._mm.close()
            except BufferError:
                # A buffer export still exists; let GC reclaim it.
                pass
            self._mm = None
        if self._fd >= 0 and self._owns_fd:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1

    def get_passing_payload(self) -> Dict[str, Any]:
        return {"fd": self._fd, "size": self._size}

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], shape, dtype) -> "_MemfdBackend":
        """
        Build a backend from an fd received from the parent (spawn case).

        The fd must already be valid in the calling process (transferred
        via socket SCM_RIGHTS, multiprocessing.reduction.send_handle, or
        Popen pass_fds).
        """
        obj = cls()
        obj._fd = int(payload["fd"])
        obj._size = int(payload["size"])
        obj._owns_fd = True
        obj._mm = mmap.mmap(obj._fd, obj._size, mmap.MAP_SHARED)
        obj.array = obj._build_array(shape, dtype)
        return obj

    # ------------------------------------------------------------------
    # Pickle support — mmap.mmap is not picklable, but the fd (int) and
    # size are. After fork the fd is inherited and still valid in the
    # child, so we re-mmap on demand from these two values.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        return {"_fd": self._fd, "_size": self._size, "_owns_fd": False}

    def __setstate__(self, state: dict) -> None:
        self._fd = int(state.get("_fd", -1))
        self._size = int(state.get("_size", 0))
        # The unpickled instance never owns the fd: only the original
        # creator in the parent should close it.
        self._owns_fd = False
        self._mm = None  # rebuilt lazily on attach()
        self.array = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_backend(kind: Optional[str] = None) -> _MemoryBackend:
    kind = kind or get_backend()
    if kind == "shm":
        return _ShmBackend()
    if kind == "mmap":
        return _MmapBackend()
    if kind == "memfd":
        return _MemfdBackend()
    raise RuntimeError(f"Unknown backend {kind!r}")


# ---------------------------------------------------------------------------
# Public SharedArray facade
# ---------------------------------------------------------------------------


class SharedArray:
    """
    Process-shared numpy array with a pluggable backend.

    Drop-in replacement for the previous SharedMemoryArray. The active
    backend is determined by get_backend() and can be controlled via
    set_backend() or the GRIDR_SHARED_BACKEND env var.
    """

    COUNTER = 0

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        name: str,
        array_slice: Optional[Tuple[slice, ...]] = None,
        _backend: Optional[_MemoryBackend] = None,
    ):
        """
        Initializes a SharedArray instance.

        Parameters
        ----------
        shape : tuple of int
            The desired shape of the NumPy array.

        dtype : numpy.dtype
            The desired data type of the NumPy array.

        name : str
            A unique name for the memory segment. This name is used to create or
            connect to the shared memory.

        array_slice : tuple of slice, optional
            A tuple of slice objects (e.g., `(slice(0, 10), slice(None))`) to
            apply to the NumPy array after it is loaded from the shared memory
            buffer. This allows working with a subset of the memory seglebt.
            Defaults to `None`.

        _backend: _MemoryBackend, optional
            Defaults to `None`.
        """
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.name = name
        self.array_slice = array_slice
        self._backend: _MemoryBackend = _backend if _backend is not None else _make_backend()
        self._backend_kind: str = self._infer_kind(self._backend)

    @staticmethod
    def _infer_kind(backend: _MemoryBackend) -> str:
        if isinstance(backend, _ShmBackend):
            return "shm"
        if isinstance(backend, _MemfdBackend):
            return "memfd"
        if isinstance(backend, _MmapBackend):
            return "mmap"
        return "unknown"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def array(self) -> Optional[np.ndarray]:
        """Numpy view onto the shared buffer.

        Returns the writable :class:`numpy.ndarray` exposing the underlying
        shared memory. If an ``array_slice`` was provided at construction,
        the sliced sub-view is returned; otherwise the full-shape array is
        returned.

        Returns
        -------
        numpy.ndarray or None
            The shared array view, or ``None`` if the resource has not been
            allocated yet (no :meth:`create` or :meth:`load` call) or has
            been released via :meth:`close` or :meth:`destroy`.

        Notes
        -----
        The returned array shares memory with all other processes attached
        to the same buffer. Modifications are visible immediately to every
        attached process; no synchronization is performed by this property.
        """
        arr = self._backend.array
        if arr is not None and self.array_slice is not None:
            return arr[self.array_slice]
        return arr

    @property
    def backend(self) -> str:
        return self._backend_kind

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self) -> None:
        """
        Creates the memory buffer and associates a NumPy array view.

        This method allocates a memory segment with the specified `name`
        and `size` (derived from `shape` and `dtype`), then creates a NumPy
        array view that points to this memory segment. The `array_slice`
        attribute is not applied during creation; it's used when the array is
        loaded (e.g., by another process, or via the `load()` method).
        """
        self._backend.create(self.shape, self.dtype, self.name)

    def load(self) -> None:
        """Attach this process to the existing shared buffer.

        Re-attaches to a buffer previously allocated by :meth:`create` in
        another (or the same) process, and rebuilds the local
        :class:`numpy.ndarray` view onto the shared memory. The attach
        mechanism is backend-specific:

        * ``shm`` — reconnect to the named POSIX segment by its
          :attr:`name`.
        * ``mmap`` — rebuild the numpy view on the mapping already inherited
          via ``fork``. The mapping is located in a per-process registry
          under :attr:`name`.
        * ``memfd`` — rebuild the numpy view on the file descriptor
          inherited via ``fork`` or transmitted via SCM_RIGHTS.

        For ``mmap`` and ``memfd``, the buffer must have been created by an
        ancestor process before the current process was forked, or by the
        current process itself. For ``shm``, any process can attach by
        name.

        Raises
        ------
        RuntimeError
            If the underlying OS resource cannot be found, typically because
            :meth:`create` was not called or because the worker was started
            with ``spawn`` for a backend that requires inheritance.

        See Also
        --------
        create : allocate the shared resource (creator side).
        close : detach the local view without releasing the resource.
        """
        self._backend.attach(self.shape, self.dtype, self.name)

    def close(self) -> None:
        """Detach the local numpy view from the shared buffer.

        Releases this process's reference to the numpy view but leaves the
        underlying OS resource intact, so other processes can keep using
        it. Safe to call from worker processes after they are done with
        the buffer.

        After :meth:`close`, the :attr:`array` property returns ``None``
        until :meth:`load` is called again.

        See Also
        --------
        destroy : release the underlying OS resource (creator side).
        load : re-attach to the shared resource after a close.
        """
        self._backend.detach()

    def destroy(self) -> None:
        """Release the underlying OS resource backing this shared array.

        Performs the backend-specific cleanup:

        * ``shm`` — closes and unlinks the named POSIX segment from
          ``/dev/shm``.
        * ``mmap`` — unmaps the anonymous memory region.
        * ``memfd`` — closes the file descriptor and unmaps the region.

        Should only be called from the process that created the
        :class:`SharedArray`, after all other processes have finished
        using it. Calling :meth:`destroy` while workers are still
        attached results in undefined behaviour for the workers.

        The call is idempotent: a second :meth:`destroy` is a no-op.

        See Also
        --------
        close : detach the local view without releasing the OS resource.
        clear_buffers : release multiple shared arrays in bulk.
        """
        self._backend.destroy()

    # ------------------------------------------------------------------
    # Cross-process passing (spawn-friendly, not supported for fork)
    # ------------------------------------------------------------------

    def get_passing_payload(self) -> Dict[str, Any]:
        """
        Return a serializable dict to reconstruct this SharedArray in
        another process.

        For "memfd", the payload contains an `fd` that must be transferred
        via SCM_RIGHTS / multiprocessing.reduction.send_handle.
        For "shm", only the name is needed (already in the payload).
        For "mmap", not supported (use fork).

        .. note::
            Only neede when using ``spawn`` workers. With the default ``fork``
            start method on Linux, you do not need this - simply pass the
            :class:`SharedArray` instance to workers and call :meth:`load` in
            the worker.
        """
        payload: Dict[str, Any] = {
            "kind": self._backend_kind,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "name": self.name,
            "array_slice": self.array_slice,
        }
        backend_payload = self._backend.get_passing_payload()
        if backend_payload is not None:
            payload["backend"] = backend_payload
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SharedArray":
        """Reconstruct a SharedArray in a child process."""
        kind = payload["kind"]
        shape = tuple(payload["shape"])
        dtype = np.dtype(payload["dtype"])
        name = payload["name"]
        array_slice = payload.get("array_slice")

        if kind == "shm":
            backend: _MemoryBackend = _ShmBackend()
            backend.attach(shape, dtype, name)
        elif kind == "memfd":
            backend = _MemfdBackend.from_payload(payload["backend"], shape, dtype)
        elif kind == "mmap":
            raise RuntimeError("mmap backend cannot be reconstructed from a payload; use fork.")
        else:
            raise ValueError(f"Unknown backend kind {kind!r}")

        return cls(shape=shape, dtype=dtype, name=name, array_slice=array_slice, _backend=backend)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def clone(cls, sa: "SharedArray", **override) -> "SharedArray":
        """
        Build a new SharedArray description from an existing one.

        For mmap / memfd, the clone shares the underlying mapping.
        For shm, the clone targets the same segment; load() in the target
        process to attach.
        """
        kwargs = {
            "shape": sa.shape,
            "dtype": sa.dtype,
            "name": sa.name,
            "array_slice": sa.array_slice,
        }
        kwargs.update(override)
        obj = cls(**kwargs)

        # Specifics for mmap
        if sa._backend_kind == "mmap":
            obj._backend._mm = sa._backend._mm  # type: ignore[attr-defined]
            if obj._backend._mm is not None:  # type: ignore[attr-defined]
                obj._backend.array = obj._backend._build_array(  # type: ignore[attr-defined]
                    obj.shape, obj.dtype
                )
        # Specifics for memfd
        elif sa._backend_kind == "memfd":
            obj._backend._fd = sa._backend._fd  # type: ignore[attr-defined]
            obj._backend._mm = sa._backend._mm  # type: ignore[attr-defined]
            obj._backend._size = sa._backend._size  # type: ignore[attr-defined]
            obj._backend._owns_fd = False
            if obj._backend._mm is not None:  # type: ignore[attr-defined]
                obj._backend.array = obj._backend._build_array(  # type: ignore[attr-defined]
                    obj.shape, obj.dtype
                )
        return obj

    @classmethod
    def build_name(cls, prefix: Optional[str] = None) -> str:
        """
        Generates a supposedly unique name for a memory segment.

        The name is constructed using a class-level counter, an optional prefix,
        the current timestamp, and a UUID4 string to maximize uniqueness. The
        class counter is incremented with each call.

        Parameters
        ----------
        prefix : str, optional
            An optional string prefix to include in the generated name.
            Defaults to `None`, resulting in an empty prefix.

        Returns
        -------
        str
            A unique string suitable for use as a shared memory segment name.
            Example:
            "1-my_prefix-202310-2715-3000-abcdef12-3456-7890-abcd-ef1234567890"
        """
        if prefix is None:
            prefix = ""
        cls.COUNTER += 1
        return "-".join(
            (
                str(cls.COUNTER),
                prefix,
                datetime.now().strftime("%Y%m-%d%H-%M%S"),
                str(uuid4()),
            )
        )

    @classmethod
    def clear_buffers(cls, buffers) -> None:
        """
        Release a list of buffers.

        Accepts SharedArray instances (preferred) or legacy str names.
        Names only meaningful for the "shm" backend.

        This method iterates through a list of shared memory names and attempts
        to unlink (delete) each corresponding shared memory segment from the
        operating system. This effectively cleans up shared memory resources.

        Parameters
        ----------
        buffer_names : list of str
            A list of unique names of the shared memory buffers to be unlinked.
        """
        for item in buffers:
            if isinstance(item, SharedArray):
                try:
                    item.destroy()
                except Exception as e:
                    logger.warning(f"Error destroying SharedArray {item.name!r}: {e}")

            # Only for Shared Memory backend
            elif isinstance(item, str):
                try:
                    buf = shared_memory.SharedMemory(name=item)
                    buf.close()
                    buf.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(f"Error unlinking shm {item!r}: {e}")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def shared_array_wrap(func):
    """Auto-load and auto-close SharedArray arguments around a function call.

    This helper function simplifies working with `SharedArray` objects
    by automatically handling their `load()` and `close()` operations.
    It's intended for functions that operate on NumPy arrays but might receive
    `SharedMemoryArray` instances as inputs.

    Parameters
    ----------
    func : callable
        The function to be wrapped. Its arguments will be inspected for
        `SharedArray` instances.

    Returns
    -------
    callable
        A wrapper function that handles the loading and closing of
        `SharedArray` arguments before and after executing the original
        `func`.

    Notes
    -----
    This decorator should be used with caution as it modifies the arguments
    passed to the wrapped function by replacing `SharedArray` instances
    with their underlying NumPy arrays. It ensures `close()` is called on
    all detected `SharedArray` instances, even if the wrapped function
    raises an exception.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function created by the `shared_array_wrap` decorator.

        This function intercepts calls to the decorated function. It iterates
        through both positional and keyword arguments, identifies any
        `SharedArray` instances, calls their `load()` method to make
        their `array` attribute available, and then passes these `np.ndarray`
        views to the original function.

        It ensures that `close()` is called on all `SharedArray` instances
        that were loaded, regardless of whether the wrapped function completes
        successfully or raises an exception.

        Parameters
        ----------
        *args
            Positional arguments passed to the decorated function.
        **kwargs
            Keyword arguments passed to the decorated function.

        Returns
        -------
        any
            The return value of the wrapped function.

        Raises
        ------
        Exception
            Any exception raised by the wrapped function will be re-raised
            after ensuring all `SharedArray` instances are closed.
        """
        attached: list[SharedArray] = []

        def resolve_arg(arg):
            if isinstance(arg, SharedArray):
                attached.append(arg)
                arg.load()
                return arg.array
            return arg

        res_args = [resolve_arg(a) for a in args]
        res_kwargs = {k: resolve_arg(v) for k, v in kwargs.items()}

        try:
            return func(*res_args, **res_kwargs)
        finally:
            for sa in attached:
                sa.close()

    return wrapper


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_and_register(
    shape,
    dtype,
    register: List,
    prefix: Optional[str] = None,
) -> SharedArray:
    """Create a SharedArray and append it to a tracking list."""
    name = SharedArray.build_name(prefix)
    sa = SharedArray(shape=shape, dtype=dtype, name=name)
    sa.create()
    register.append(sa)
    return sa
