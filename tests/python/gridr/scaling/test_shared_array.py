# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
#
# Licensed under the Apache License, Version 2.0
#
"""
Pytest test suite for gridr.scaling.shared_array.

Covers:
- Backend resolution (env, set_backend, auto)
- Lifecycle: create / load / close / destroy on the 3 backends
- Multi-process IPC: fork (all backends) + spawn (shm, memfd)
- Array semantics: shape, dtype, writeability, array_slice
- Concurrent disjoint writes across workers
- clone() semantics
- share_array_wrap decorator
- create_and_register + clear_buffers
- Error paths: load before create, mmap from_payload, unknown backend
- Platform skips (memfd Linux-only, mmap fork-only)

Run with:

PYTHONPATH=${PWD}/python/:$PYTHONPATH pytest tests/python/gridr/scaling/test_shared_array.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Tuple

import numpy as np
import pytest

# Import module under test
# Adjust path if the module is named differently in your tree
from gridr.scaling.shared_array import (  # noqa: E402
    SharedArray,
    _has_memfd,
    _MemfdBackend,
    _MmapBackend,
    create_and_register,
    get_backend,
    set_backend,
    shared_array_wrap,
)

# ---------------------------------------------------------------------------
# Platform / capability flags
# ---------------------------------------------------------------------------

IS_LINUX = sys.platform.startswith("linux")
IS_WIN = sys.platform.startswith("win")
HAS_FORK = "fork" in mp.get_all_start_methods()
HAS_MEMFD = _has_memfd()

requires_fork = pytest.mark.skipif(not HAS_FORK, reason="fork start method not available")
requires_memfd = pytest.mark.skipif(not HAS_MEMFD, reason="memfd_create not available")
requires_linux = pytest.mark.skipif(not IS_LINUX, reason="Linux only")


# ---------------------------------------------------------------------------
# Worker functions (must be module-level to be picklable for spawn)
# ---------------------------------------------------------------------------


def _worker_fork_write(sa: SharedArray, value: float) -> None:
    """Fork worker: SharedArray is pickled-by-reference, attach via load()."""
    sa.load()
    sa.array[:] = value


def _worker_fork_write_region(sa: SharedArray, region: Tuple[slice, ...], value: float) -> None:
    sa.load()
    sa.array[region] = value


def _worker_spawn_write(payload: dict, value: float) -> None:
    sa = SharedArray.from_payload(payload)
    sa.array[:] = value


def _worker_fork_read_sum(sa: SharedArray, result_queue) -> None:
    sa.load()
    result_queue.put(float(sa.array.sum()))


def _worker_fork_concurrent_region(sa: SharedArray, idx: int, n_workers: int) -> None:
    """Each worker writes its own disjoint row band."""
    sa.load()
    rows_per_worker = sa.array.shape[0] // n_workers
    start = idx * rows_per_worker
    stop = start + rows_per_worker
    sa.array[start:stop] = float(idx + 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_backend_state():
    """Reset backend resolution between tests to avoid cross-test leakage."""
    import gridr.scaling.shared_array as mod

    mod._active_backend = None
    mod._requested_backend = "auto"
    # also clear env override if a previous test forced one
    os.environ.pop("GRIDR_SHARED_MEMORY_BACKEND", None)
    os.environ.pop("GRIDR_SHM_MIN_FREE", None)
    yield
    mod._active_backend = None
    mod._requested_backend = "auto"
    os.environ.pop("GRIDR_SHARED_MEMORY_BACKEND", None)
    os.environ.pop("GRIDR_SHM_MIN_FREE", None)


def _backend_params():
    """Yield available backends for parametrization."""
    backends = ["shm", "mmap"]
    if HAS_MEMFD:
        backends.append("memfd")
    return backends


@pytest.fixture(params=_backend_params())
def backend(request):
    """Parametrize tests over every available backend."""
    set_backend(request.param)
    return request.param


@pytest.fixture
def sa(backend):
    """A small float32 SharedArray ready to use, cleaned up afterwards."""
    arr = SharedArray(
        shape=(8, 8),
        dtype=np.float32,
        name=SharedArray.build_name(prefix=f"test-{backend}"),
    )
    arr.create()
    arr.array[:] = 0.0
    yield arr
    try:
        arr.destroy()
    except Exception:
        pass


# ===========================================================================
# Backend resolution
# ===========================================================================


class TestBackendResolution:

    def test_set_backend_explicit(self):
        set_backend("mmap")
        assert get_backend() == "mmap"
        set_backend("shm")
        assert get_backend() == "shm"

    def test_set_backend_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            set_backend("nonsense")

    def test_env_var_overrides_request(self):
        set_backend("shm")
        os.environ["GRIDR_SHARED_MEMORY_BACKEND"] = "mmap"
        # active backend is recomputed
        import gridr.scaling.shared_array as mod

        mod._active_backend = None
        assert get_backend() == "mmap"

    def test_env_var_invalid_ignored(self):
        set_backend("mmap")
        os.environ["GRIDR_SHARED_MEMORY_BACKEND"] = "garbage"
        import gridr.scaling.shared_array as mod

        mod._active_backend = None
        # Falls back to the requested backend
        assert get_backend() == "mmap"

    def test_auto_returns_valid_backend(self):
        set_backend("auto")
        b = get_backend()
        assert b in ("shm", "mmap", "memfd")

    @requires_memfd
    def test_memfd_capability_probe(self):
        assert _has_memfd() is True

    def test_get_backend_caches_result(self):
        set_backend("mmap")
        b1 = get_backend()
        b2 = get_backend()
        assert b1 == b2 == "mmap"


# ===========================================================================
# Lifecycle
# ===========================================================================


class TestLifecycle:

    def test_create_exposes_array(self, sa):
        assert sa.array is not None
        assert sa.array.shape == (8, 8)
        assert sa.array.dtype == np.float32

    def test_array_is_writable(self, sa):
        sa.array[:] = 42.0
        assert (sa.array == 42.0).all()

    def test_array_writeable_flag(self, sa):
        # critical for PyO3 / Rust bindings
        assert sa.array.flags.writeable is True

    def test_close_drops_view(self, sa):
        sa.close()
        assert sa._backend.array is None

    def test_destroy_releases(self, sa):
        sa.destroy()
        assert sa._backend.array is None

    def test_destroy_idempotent(self, sa):
        sa.destroy()
        sa.destroy()  # should not raise

    def test_load_without_create_raises_for_mmap(self):
        set_backend("mmap")
        arr = SharedArray(shape=(4,), dtype=np.float32, name="orphan")
        with pytest.raises(RuntimeError, match="no mapping found"):
            arr.load()

    def test_backend_property(self, sa, backend):
        assert sa.backend == backend


# ===========================================================================
# Array semantics
# ===========================================================================


class TestArraySemantics:

    def test_shape_preserved(self, backend):
        sa = SharedArray(shape=(3, 5, 7), dtype=np.float64, name=SharedArray.build_name())
        sa.create()
        assert sa.array.shape == (3, 5, 7)
        sa.destroy()

    def test_dtype_preserved(self, backend):
        for dt in (np.uint8, np.int32, np.float32, np.float64):
            sa = SharedArray(shape=(4, 4), dtype=dt, name=SharedArray.build_name())
            sa.create()
            assert sa.array.dtype == np.dtype(dt)
            sa.destroy()

    def test_array_slice_applied(self, backend):
        sa = SharedArray(
            shape=(10, 10),
            dtype=np.float32,
            name=SharedArray.build_name(),
            array_slice=(slice(2, 8), slice(2, 8)),
        )
        sa.create()
        # The exposed view follows the slice
        assert sa.array.shape == (6, 6)
        # But the underlying buffer is full-shape
        assert sa._backend.array.shape == (10, 10)
        sa.destroy()

    def test_array_slice_writes_propagate_to_full(self, backend):
        sa = SharedArray(
            shape=(10, 10),
            dtype=np.float32,
            name=SharedArray.build_name(),
            array_slice=(slice(2, 8), slice(2, 8)),
        )
        sa.create()
        sa.array[:] = 5.0
        # The full underlying buffer should have 5.0 in [2:8, 2:8] and 0 elsewhere
        full = sa._backend.array
        assert (full[2:8, 2:8] == 5.0).all()
        assert (full[:2, :] == 0.0).all()
        sa.destroy()

    def test_zero_initialized_or_explicit(self, backend):
        """Don't assume initial values; just check writability/read consistency."""
        sa = SharedArray(shape=(16,), dtype=np.int32, name=SharedArray.build_name())
        sa.create()
        sa.array[:] = np.arange(16, dtype=np.int32)
        assert (sa.array == np.arange(16, dtype=np.int32)).all()
        sa.destroy()


# ===========================================================================
# Multi-process IPC: fork
# ===========================================================================


@requires_fork
class TestForkIPC:
    """All three backends must work transparently with fork."""

    def test_child_writes_visible_to_parent(self, sa):
        ctx = mp.get_context("fork")
        p = ctx.Process(target=_worker_fork_write, args=(sa, 9.0))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0
        assert (sa.array == 9.0).all()

    def test_parent_writes_visible_to_child(self, sa):
        ctx = mp.get_context("fork")
        sa.array[:] = 3.0
        q = ctx.Queue()
        p = ctx.Process(target=_worker_fork_read_sum, args=(sa, q))
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0
        assert q.get(timeout=5) == 3.0 * 8 * 8

    def test_concurrent_disjoint_writes(self, backend):
        """Multiple workers each write their own row band; no locking needed."""
        n_workers = 4
        rows = 16
        sa = SharedArray(
            shape=(rows, 4),
            dtype=np.float32,
            name=SharedArray.build_name(prefix=f"concurrent-{backend}"),
        )
        sa.create()
        sa.array[:] = 0.0

        ctx = mp.get_context("fork")
        procs = [
            ctx.Process(target=_worker_fork_concurrent_region, args=(sa, i, n_workers))
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=10)
            assert p.exitcode == 0

        rows_per_worker = rows // n_workers
        for i in range(n_workers):
            band = sa.array[i * rows_per_worker : (i + 1) * rows_per_worker]
            assert (band == float(i + 1)).all(), f"band {i} got {band}"
        sa.destroy()


# ===========================================================================
# Multi-process IPC: spawn (shm + memfd only)
# ===========================================================================


class TestSpawnIPC:
    """shm and memfd can be used across spawn; mmap cannot."""

    def test_spawn_shm(self):
        set_backend("shm")
        sa = SharedArray(shape=(4, 4), dtype=np.float32, name=SharedArray.build_name())
        sa.create()
        sa.array[:] = 0.0
        payload = sa.get_passing_payload()
        assert payload["kind"] == "shm"

        ctx = mp.get_context("spawn")
        p = ctx.Process(target=_worker_spawn_write, args=(payload, 11.0))
        p.start()
        p.join(timeout=15)
        assert p.exitcode == 0
        assert (sa.array == 11.0).all()
        sa.destroy()

    @requires_memfd
    def test_spawn_memfd_via_fork_ctx(self):
        """
        memfd + spawn requires SCM_RIGHTS fd passing in production.
        For an in-process test we use fork ctx so the fd is inherited,
        validating that from_payload() rebuilds the mapping correctly.
        """
        set_backend("memfd")
        sa = SharedArray(shape=(4, 4), dtype=np.float32, name=SharedArray.build_name())
        sa.create()
        sa.array[:] = 0.0
        payload = sa.get_passing_payload()
        assert payload["kind"] == "memfd"
        assert "fd" in payload["backend"]
        assert payload["backend"]["fd"] > 0

        ctx = mp.get_context("fork")  # inherits fd
        p = ctx.Process(target=_worker_spawn_write, args=(payload, 13.0))
        p.start()
        p.join(timeout=15)
        assert p.exitcode == 0
        assert (sa.array == 13.0).all()
        sa.destroy()

    def test_mmap_from_payload_raises(self):
        set_backend("mmap")
        sa = SharedArray(shape=(4, 4), dtype=np.float32, name="x")
        sa.create()
        payload = sa.get_passing_payload()
        with pytest.raises(RuntimeError, match="cannot be reconstructed"):
            SharedArray.from_payload(payload)
        sa.destroy()


# ===========================================================================
# Cross-backend payload
# ===========================================================================


class TestPayload:

    def test_payload_shape_dtype(self, sa, backend):
        payload = sa.get_passing_payload()
        assert payload["kind"] == backend
        assert tuple(payload["shape"]) == (8, 8)
        assert np.dtype(payload["dtype"]) == np.float32

    def test_payload_carries_array_slice(self, backend):
        sa = SharedArray(
            shape=(10, 10),
            dtype=np.float32,
            name=SharedArray.build_name(),
            array_slice=(slice(1, 9), slice(1, 9)),
        )
        sa.create()
        payload = sa.get_passing_payload()
        assert payload["array_slice"] == (slice(1, 9), slice(1, 9))
        sa.destroy()

    def test_from_payload_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            SharedArray.from_payload(
                {
                    "kind": "nope",
                    "shape": (4,),
                    "dtype": "float32",
                    "name": "x",
                }
            )


# ===========================================================================
# clone()
# ===========================================================================


class TestClone:

    def test_clone_preserves_attributes(self, sa):
        clone = SharedArray.clone(sa)
        assert clone.shape == sa.shape
        assert clone.dtype == sa.dtype
        assert clone.name == sa.name
        assert clone.array_slice == sa.array_slice

    def test_clone_override_array_slice(self, sa):
        new_slice = (slice(0, 4), slice(0, 4))
        clone = SharedArray.clone(sa, array_slice=new_slice)
        assert clone.array_slice == new_slice

    def test_clone_shares_buffer_mmap_memfd(self, backend):
        if backend == "shm":
            pytest.skip("clone semantics for shm require load() in the new process")
        sa = SharedArray(shape=(4, 4), dtype=np.float32, name=SharedArray.build_name())
        sa.create()
        sa.array[:] = 0.0

        clone = SharedArray.clone(sa)
        # Writing through the clone should be visible through the original
        clone.array[2, 2] = 42.0
        assert sa.array[2, 2] == 42.0
        sa.destroy()


# ===========================================================================
# shared_array_wrap decorator
# ===========================================================================


class TestSharedArrayWrap:

    def test_replaces_shared_array_with_ndarray(self, sa):
        sa.array[:] = 4.0

        @shared_array_wrap
        def f(arr, scalar):
            assert isinstance(arr, np.ndarray)
            return arr.sum() + scalar

        out = f(sa, 1.0)
        expected = 4.0 * 8 * 8 + 1.0
        assert out == expected

    def test_keyword_arguments(self, sa):
        sa.array[:] = 2.0

        @shared_array_wrap
        def f(*, arr):
            return arr.mean()

        assert f(arr=sa) == 2.0

    def test_close_called_on_exception(self, sa, monkeypatch):
        close_calls = {"n": 0}
        orig_close = sa.close

        def counting_close():
            close_calls["n"] += 1
            return orig_close()

        monkeypatch.setattr(sa, "close", counting_close)

        @shared_array_wrap
        def boom(arr):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            boom(sa)
        assert close_calls["n"] == 1

    def test_passthrough_non_shared(self, sa):
        @shared_array_wrap
        def f(x, y):
            return x + y

        assert f(1, 2) == 3


# ===========================================================================
# Registry helpers
# ===========================================================================


class TestRegistry:

    def test_create_and_register_appends(self, backend):
        reg = []
        sa = create_and_register((4, 4), np.float32, reg, prefix="reg")
        assert len(reg) == 1
        assert reg[0] is sa
        sa.destroy()

    def test_clear_buffers_instances(self, backend):
        reg = []
        sa1 = create_and_register((4, 4), np.float32, reg)
        sa2 = create_and_register((2, 2), np.int8, reg)
        SharedArray.clear_buffers(reg)
        # destroy() drops the numpy view
        assert sa1._backend.array is None
        assert sa2._backend.array is None

    def test_clear_buffers_legacy_names_shm(self):
        """Legacy str-based clear_buffers only works for the shm backend."""
        set_backend("shm")
        name = SharedArray.build_name(prefix="legacy")
        sa = SharedArray(shape=(4,), dtype=np.uint8, name=name)
        sa.create()
        # Drop our handle; the OS segment still exists
        sa.close()
        # legacy path: pass the name string
        SharedArray.clear_buffers([name])
        # subsequent attach should fail (segment unlinked)
        with pytest.raises(FileNotFoundError):
            SharedArray(shape=(4,), dtype=np.uint8, name=name).load()


# ===========================================================================
# build_name uniqueness
# ===========================================================================


class TestBuildName:

    def test_unique_across_calls(self):
        names = {SharedArray.build_name("x") for _ in range(100)}
        assert len(names) == 100

    def test_prefix_present(self):
        n = SharedArray.build_name("myprefix")
        assert "myprefix" in n

    def test_no_prefix(self):
        n = SharedArray.build_name(None)
        assert isinstance(n, str) and len(n) > 0


# ===========================================================================
# Backend-specific edge cases
# ===========================================================================


class TestBackendEdgeCases:

    def test_shm_name_collision_raises(self):
        set_backend("shm")
        name = SharedArray.build_name(prefix="dup")
        sa1 = SharedArray(shape=(4,), dtype=np.uint8, name=name)
        sa1.create()
        try:
            sa2 = SharedArray(shape=(4,), dtype=np.uint8, name=name)
            with pytest.raises(FileExistsError):
                sa2.create()
        finally:
            sa1.destroy()

    @requires_memfd
    def test_memfd_fd_is_valid_after_create(self):
        set_backend("memfd")
        sa = SharedArray(shape=(4,), dtype=np.uint8, name="memfd-fd")
        sa.create()
        backend = sa._backend
        assert isinstance(backend, _MemfdBackend)
        assert backend._fd > 0
        # /proc/self/fd should list it on Linux
        if IS_LINUX:
            assert os.path.exists(f"/proc/self/fd/{backend._fd}")
        sa.destroy()
        assert backend._fd == -1

    def test_mmap_destroy_releases_memory(self):
        set_backend("mmap")
        sa = SharedArray(shape=(1024, 1024), dtype=np.float64, name="big")
        sa.create()
        backend = sa._backend
        assert isinstance(backend, _MmapBackend)
        assert backend._mm is not None
        sa.destroy()
        assert backend._mm is None


# ===========================================================================
# Pickle support (regression for "cannot pickle mmap.mmap")
# ===========================================================================


class TestPickle:
    """SharedArray must survive pickle/unpickle for all backends."""

    def test_pickle_roundtrip_in_same_process(self, sa, backend):
        import pickle

        sa.array[:] = 5.0
        data = pickle.dumps(sa)
        restored = pickle.loads(data)
        restored.load()
        assert restored.array.sum() == 5.0 * 8 * 8

    @requires_fork
    def test_pool_starmap_fork(self, backend):
        """End-to-end: Pool.starmap pickles arguments — must not raise."""
        sa = SharedArray(
            shape=(20,), dtype=np.float32, name=SharedArray.build_name(f"pool-{backend}")
        )
        sa.create()
        sa.array[:] = 0.0

        ctx = mp.get_context("fork")
        with ctx.Pool(4) as pool:
            pool.starmap(_worker_pool_starmap, [(sa, i) for i in range(20)])

        expected = np.arange(20, dtype=np.float32)
        assert np.array_equal(sa.array, expected)
        sa.destroy()

    @requires_fork
    def test_pool_map_fork(self, backend):
        """End-to-end: Pool.map (single-arg variant) under fork."""
        sa = SharedArray(
            shape=(20,), dtype=np.float32, name=SharedArray.build_name(f"map-{backend}")
        )
        sa.create()
        sa.array[:] = 0.0

        ctx = mp.get_context("fork")
        with ctx.Pool(4) as pool:
            pool.map(_worker_pool_map, [(sa, i) for i in range(20)])

        expected = np.arange(20, dtype=np.float32)
        assert np.array_equal(sa.array, expected)
        sa.destroy()


def _worker_pool_starmap(sa, idx):
    sa.load()
    sa.array[idx] = float(idx)


def _worker_pool_map(args):
    sa, idx = args
    sa.load()
    sa.array[idx] = float(idx)


# ===========================================================================
# Smoke: parametrized end-to-end
# ===========================================================================


@requires_fork
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
def test_e2e_fork_dtypes(backend, dtype):
    """End-to-end: create, write from child, read from parent, for various dtypes."""
    sa = SharedArray(
        shape=(8, 8),
        dtype=dtype,
        name=SharedArray.build_name(prefix=f"e2e-{dtype}"),
    )
    sa.create()
    sa.array[:] = 0

    ctx = mp.get_context("fork")
    p = ctx.Process(target=_worker_fork_write, args=(sa, 5))
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0
    assert (sa.array == 5).all()
    sa.destroy()
