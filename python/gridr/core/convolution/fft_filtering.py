# Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Windowed N-d convolution with explicit boundary and decimation control.

Wraps :mod:`scipy.signal` convolution and adds what tiled raster processing
needs: a production window, per-side boundary policies, and decimation fused
into the output slicing.

:func:`build_plan` computes the geometry from the input shape alone and returns
an immutable :class:`FilterPlan`. :func:`fft_array_filter` and
:func:`fft_array_filter_output_shape` both consume that plan, so a predicted
shape and a produced shape come from the same code.

Conventions
-----------
Window
    An integer array of shape ``(ndim, 2)`` holding ``(first, last)``
    inclusive indices per axis, as in :mod:`gridr.core.utils.array_window`.

Data type
    The working dtype defaults to ``np.result_type(arr, kernel)``, so an
    integer or ``float64`` kernel on a ``float32`` raster doubles the memory.
    Pass ``dtype=np.float32`` to pin it.

Non-finite values
    FFT convolution is global: one ``NaN`` in the input contaminates the whole
    output. Mask or fill nodata beforehand.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, SupportsIndex

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy import signal

from gridr.core.utils.array_pad import pad_inplace
from gridr.core.utils.array_window import window_normalize

__all__ = [
    "BOUNDARY_MODES",
    "CONVOLUTION_METHODS",
    "DECIMATION_ORIGINS",
    "OUTPUT_MODES",
    "AxisPlan",
    "BoundaryMode",
    "ConvolutionMethod",
    "DecimationOrigin",
    "FilterPlan",
    "FilterResult",
    "OutputMode",
    "Zoom",
    "align_kernel",
    "build_plan",
    "decimated_size",
    "decimation_offset",
    "fft_array_filter",
    "fft_array_filter_output_shape",
    "kernel_margin",
    "normalize_axes",
    "normalize_zoom",
    "pad_kernel_to_odd",
]


# --------------------------------------------------------------------------- #
# Option vocabularies
# --------------------------------------------------------------------------- #
# Plain strings rather than enums: these are the :func:`numpy.pad` modes and the
# :func:`scipy.signal.convolve` output modes, and
# :func:`~gridr.core.grid.grid_resampling.array_grid_resampling` already takes
# the same spellings.

#: Padding policy for one side. ``"none"``, or ``None``, synthesises nothing
#: there; the other values are :func:`numpy.pad` modes.
#:
#: A specification may name any number of sides but only one non-``"none"``
#: policy: sides say whether they are padded, not how. ``"reflect"`` on one edge
#: and ``"wrap"`` on another is rejected.
#:
#: The policy applies outside the array only. The kernel margin is always read
#: from real neighbours where they exist, so a window in the middle of a raster
#: is never padded.
#:
#: ``"wrap"`` is the only one that needs the whole axis: a production window
#: that stops short of an edge has no periodic neighbour to bring in, so it is
#: rejected rather than wrapped around the window itself.
BoundaryMode = Literal["none", "reflect", "symmetric", "edge", "wrap", "constant"]

#: Part of the convolution returned, as in :func:`scipy.signal.convolve`.
OutputMode = Literal["same", "full", "valid"]

#: Convolution backend. ``"overlap_add"`` and ``"fft"`` map to
#: :func:`scipy.signal.oaconvolve` and :func:`scipy.signal.fftconvolve`, the
#: other two to :func:`scipy.signal.convolve`. Overlap-add is the default but
#: loses to a plain FFT on small kernels.
ConvolutionMethod = Literal["overlap_add", "fft", "direct", "auto"]

#: Sample kept in each block of ``Q``: the first one, or the one at offset
#: ``(Q - 1) // 2``.
DecimationOrigin = Literal["centered", "leading"]

BOUNDARY_MODES: tuple[str, ...] = ("none", "reflect", "symmetric", "edge", "wrap", "constant")
OUTPUT_MODES: tuple[str, ...] = ("same", "full", "valid")
CONVOLUTION_METHODS: tuple[str, ...] = ("overlap_add", "fft", "direct", "auto")
DECIMATION_ORIGINS: tuple[str, ...] = ("centered", "leading")

#: ``"none"`` is the only boundary mode with no :func:`numpy.pad` equivalent;
#: every other one is spelled exactly as the mode it selects.
NO_BOUNDARY: BoundaryMode = "none"


def _normalize_choice(value: object, allowed: tuple[str, ...], name: str) -> str:
    """Validate one of the string vocabularies above.

    Wrong type raises :class:`TypeError`, wrong value :class:`ValueError`, as
    elsewhere in the module. The message lists the accepted spellings.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if value not in allowed:
        spellings = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"unknown {name} {value!r}; expected one of {spellings}")
    return value


def _normalize_boundary_mode(value: object) -> str:
    """Wrap _normalize_choice, add ``None`` as an accepted spelling of ``"none"``."""
    if value is None:
        return NO_BOUNDARY
    return _normalize_choice(value, BOUNDARY_MODES, "boundary mode")


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #
class Zoom(NamedTuple):
    """A rational zoom factor ``P / Q`` in lowest terms."""

    p: int
    q: int

    @property
    def is_supported(self) -> bool:
        """``True`` if the current implementation can honour this factor.

        Only pure decimation (``P == 1``) is implemented; interpolation
        (``P > 1``) requires a polyphase upsampling stage.
        """
        return self.p == 1 and self.q >= 1


def _is_scalar(value: object) -> bool:
    """Return ``True`` for a single value, ``False`` for anything with an axis."""
    try:
        return np.ndim(value) == 0
    except (TypeError, ValueError):
        return False


def _as_index(value: object, name: str) -> int:
    """Coerce to a Python ``int``. ``bool`` and non-integral input are refused."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a bool")
    if not isinstance(value, SupportsIndex):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    return int(value.__index__())


def normalize_zoom(zoom: int | tuple[int, int]) -> Zoom:
    """Normalize a zoom argument to a :class:`Zoom` in lowest terms.

    Parameters
    ----------
    zoom : int or tuple of two ints
        Either ``P`` (implying ``Q = 1``) or the pair ``(P, Q)``. Any object
        implementing ``__index__`` is accepted, so ``numpy`` integers work.

    Returns
    -------
    Zoom
        The simplified factor.

    Raises
    ------
    TypeError
        If ``zoom`` is neither an integer nor a pair of integers.
    ValueError
        If the pair has the wrong length, or if ``P`` or ``Q`` is not strictly
        positive.

    Examples
    --------
    >>> normalize_zoom(2)
    Zoom(p=2, q=1)
    >>> normalize_zoom((2, 6))
    Zoom(p=1, q=3)

    """
    if _is_scalar(zoom):
        p, q = _as_index(zoom, "zoom"), 1
    else:
        items = list(zoom)
        if len(items) != 2:
            raise ValueError(f"zoom pair must contain exactly two integers, got {len(items)}")
        p = _as_index(items[0], "zoom P")
        q = _as_index(items[1], "zoom Q")

    if p <= 0:
        raise ValueError(f"zoom P must be strictly positive, got {p}")
    if q <= 0:
        raise ValueError(f"zoom Q must be strictly positive, got {q}")

    gcd = math.gcd(p, q)
    return Zoom(p // gcd, q // gcd)


def decimation_offset(q: int, origin: DecimationOrigin = "centered") -> int:
    """Index of the sample kept inside each block of ``q`` samples.

    ``"centered"`` yields ``(q - 1) // 2``: the exact centre for odd ``q``, the
    lower of the two central samples for even ``q``.

    Raises
    ------
    ValueError
        If ``q`` is not strictly positive.

    """
    if q <= 0:
        raise ValueError(f"q must be strictly positive, got {q}")
    return 0 if origin == "leading" else (q - 1) // 2


def decimated_size(size: int, q: int, offset: int = 0) -> int:
    """Length of ``range(size)[offset::q]``.

    Parameters
    ----------
    size : int
        Length of the original axis, ``>= 0``.
    q : int
        Decimation step, strictly positive.
    offset : int, optional
        Index of the first sample kept, ``>= 0``. Default ``0``.

    Raises
    ------
    ValueError
        If any argument is out of domain.

    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}")
    if q <= 0:
        raise ValueError(f"q must be > 0, got {q}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if offset >= size:
        return 0
    return (size - offset - 1) // q + 1


def normalize_axes(axes: int | Iterable[int] | None, ndim: int) -> tuple[int, ...]:
    """Normalize an axis specification to distinct non-negative indices.

    Parameters
    ----------
    axes : int, iterable of int, or None
        Target axes. ``None`` means every axis. Negative indices are accepted
        with the usual ``numpy`` semantics, i.e. ``-ndim <= axis < ndim``.
    ndim : int
        Rank of the array the axes refer to.

    Returns
    -------
    tuple of int
        The distinct target axes, sorted, as non-negative indices.

    Raises
    ------
    ValueError
        If an axis is out of bounds or repeated.

    Examples
    --------
    >>> normalize_axes(-1, 3)
    (2,)
    >>> normalize_axes((2, 0), 3)
    (0, 2)
    >>> normalize_axes(None, 2)
    (0, 1)

    """
    if ndim < 0:
        raise ValueError(f"ndim must be >= 0, got {ndim}")
    if axes is None:
        return tuple(range(ndim))

    raw = (axes,) if _is_scalar(axes) else tuple(axes)

    normalized: list[int] = []
    for item in raw:
        axis = _as_index(item, "axis")
        if not -ndim <= axis < ndim:
            raise ValueError(f"axis {axis} is out of bounds for ndim={ndim}")
        axis %= ndim
        if axis in normalized:
            raise ValueError(f"axis {axis} is repeated in {axes!r}")
        normalized.append(axis)
    return tuple(sorted(normalized))


# --------------------------------------------------------------------------- #
# Kernel preparation
# --------------------------------------------------------------------------- #
def align_kernel(kernel: ArrayLike, ndim: int, axes: int | Iterable[int] | None = None) -> NDArray:
    """Broadcast a kernel so that it has exactly ``ndim`` dimensions.

    Two shapes are accepted:

    * a kernel of rank ``len(axes)``, whose ``i``-th dimension is mapped onto
      the ``i``-th target axis in increasing order, singleton dimensions being
      inserted on the remaining axes;
    * a kernel already of rank ``ndim``, which is returned unchanged provided
      its size is ``1`` on every axis outside ``axes``.

    When both readings apply (``len(axes) == ndim``) the second one wins: a
    full-rank kernel is assumed to be already expressed in array axis order.

    ``axes`` is normalized here, so the function is safe to call on its own.
    The mapping is positional against the sorted axes and does not reorder the
    kernel samples; pass a transposed kernel for another assignment.

    Raises
    ------
    ValueError
        If the rank matches neither reading, or if a full-rank kernel is not
        singleton outside ``axes``.

    """
    kernel = np.asarray(kernel)
    axes = normalize_axes(axes, ndim)

    if kernel.ndim == ndim:
        bad = [i for i in range(ndim) if i not in axes and kernel.shape[i] != 1]
        if bad:
            raise ValueError(
                f"kernel must be singleton outside the target axes {axes}; "
                f"axis/axes {bad} have sizes {[kernel.shape[i] for i in bad]}"
            )
        return kernel

    if kernel.ndim == len(axes):
        shape = [1] * ndim
        for axis, size in zip(axes, kernel.shape, strict=True):
            shape[axis] = size
        return kernel.reshape(shape)

    raise ValueError(
        f"kernel has rank {kernel.ndim}, expected {len(axes)} (one dimension per "
        f"target axis) or {ndim} (one dimension per array axis)"
    )


def pad_kernel_to_odd(kernel: NDArray, axes: int | Iterable[int] | None = None) -> NDArray:
    """Right-pad the kernel with zeros so its size is odd along every target axis.

    An odd size gives the kernel an unambiguous centre, without which the
    ``"same"`` output alignment is not well defined.

    ``axes`` is normalized here, so the function is safe to call on its own.
    """
    axes = normalize_axes(axes, kernel.ndim)
    widths = [(0, 0)] * kernel.ndim
    padded = False
    for axis in axes:
        if kernel.shape[axis] % 2 == 0:
            widths[axis] = (0, 1)
            padded = True
    if not padded:
        return kernel
    return np.pad(kernel, widths, mode="constant", constant_values=0)


def kernel_margin(kernel: NDArray, axes: int | Iterable[int] | None = None) -> tuple[int, ...]:
    """Half-width of an odd-sized kernel along every axis, ``0`` outside ``axes``.

    ``axes`` is normalized here, so the function is safe to call on its own.
    """
    axes = normalize_axes(axes, kernel.ndim)
    return tuple(kernel.shape[i] // 2 if i in axes else 0 for i in range(kernel.ndim))


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AxisPlan:
    """Geometry of a single axis, in samples."""

    #: Slice read from the input array before padding.
    source: slice
    #: Padding widths ``(before, after)`` added around :attr:`source`.
    pad_width: tuple[int, int]
    #: Length of the convolution input, i.e. :attr:`source` plus its padding.
    conv_size: int
    #: Length of the ``"full"`` convolution result along this axis.
    full_size: int
    #: Index, in the ``"full"`` result, of the first requested sample.
    origin: int
    #: Number of requested samples, before decimation.
    window_size: int
    #: Slice extracting the requested output from the ``"full"`` result.
    output: slice
    #: Length of ``output``.
    output_size: int


@dataclass(frozen=True, slots=True)
class FilterPlan:
    """Everything :func:`fft_array_filter` needs, derived from shapes alone.

    A function of the input shape, the kernel and the options, so it can be
    built and inspected without allocating the raster. Tiled schedulers use it
    to size their outputs ahead of time.
    """

    #: Kernel aligned to the array rank and padded to an odd size.
    kernel: NDArray
    #: Target axes, normalized.
    axes: tuple[int, ...]
    #: Per-axis geometry, one entry per array dimension.
    per_axis: tuple[AxisPlan, ...]
    #: The rule actually applied to the margins, ``"none"`` when this particular
    #: window needs no synthetic sample. Invariant:
    #: ``(pad_mode == "none") == (not needs_padding)``.
    pad_mode: BoundaryMode
    #: Requested output mode.
    out_mode: OutputMode
    #: Normalized zoom factor.
    zoom: Zoom
    #: Working dtype of the convolution.
    dtype: np.dtype
    #: Backend used for the convolution.
    method: ConvolutionMethod

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the array returned by :func:`fft_array_filter`."""
        return tuple(axis.output_size for axis in self.per_axis)

    @property
    def source(self) -> tuple[slice, ...]:
        """Slices reading the convolution input out of the source array."""
        return tuple(axis.source for axis in self.per_axis)

    @property
    def output(self) -> tuple[slice, ...]:
        """Slices extracting the result out of the ``"full"`` convolution."""
        return tuple(axis.output for axis in self.per_axis)

    @property
    def window(self) -> NDArray[np.int64]:
        """Production window expressed in ``"full"`` output coordinates.

        Shape ``(ndim, 2)``, inclusive bounds, following the GridR window
        convention.
        """
        return np.asarray(
            [(axis.origin, axis.origin + axis.window_size - 1) for axis in self.per_axis],
            dtype=np.int64,
        )

    @property
    def conv_shape(self) -> tuple[int, ...]:
        """Shape of the convolution input, padding included."""
        return tuple(axis.conv_size for axis in self.per_axis)

    @property
    def pad_width(self) -> tuple[tuple[int, int], ...]:
        """Padding widths per axis, in :func:`numpy.pad` order."""
        return tuple(axis.pad_width for axis in self.per_axis)

    @property
    def src_win(self) -> tuple[slice, ...]:
        """Where the real samples sit inside the padded convolution input."""
        return tuple(
            slice(axis.pad_width[0], axis.conv_size - axis.pad_width[1]) for axis in self.per_axis
        )

    @property
    def needs_padding(self) -> bool:
        """``True`` if at least one side has to be synthesised."""
        return any(any(axis.pad_width) for axis in self.per_axis)


class FilterResult(NamedTuple):
    """Result of :func:`fft_array_filter`.

    Unpacks as ``(data, window)`` for backward compatibility.
    """

    #: The filtered samples.
    data: NDArray
    #: Production window in ``"full"`` output coordinates, inclusive bounds.
    window: NDArray[np.int64]


BoundarySpec = BoundaryMode | None | Sequence[Sequence[BoundaryMode | None]]


def _normalize_boundary(
    boundary: BoundarySpec, ndim: int, axes: tuple[int, ...]
) -> tuple[tuple[tuple[bool, bool], ...], BoundaryMode]:
    """Split a boundary specification into where to pad and how.

    Returns one ``(before, after)`` pair of booleans per axis, plus the single
    declared policy (``"none"`` when no side asks to be extended).

    A scalar applies to every side. A sequence of pairs shorter than ``ndim``
    is right-aligned and the leading axes default to ``"none"``, so a 2-D
    ``((top, bottom), (left, right))`` specification still works on a stacked
    3-D array.

    Raises
    ------
    ValueError
        If two different non-``"none"`` policies appear. See :data:`BoundaryMode`.

    """
    if boundary is None or isinstance(boundary, str):
        scalar = _normalize_boundary_mode(boundary)
        pairs = [(scalar, scalar)] * ndim
    else:
        if not isinstance(boundary, Iterable):
            raise TypeError(
                "boundary must be a mode string, None, or a sequence of "
                f"(before, after) pairs, got {boundary!r}"
            )
        given = [tuple(pair) for pair in boundary]
        if len(given) > ndim:
            raise ValueError(f"boundary has {len(given)} pairs for a {ndim}-d array")
        for pair in given:
            if len(pair) != 2:
                raise ValueError(
                    f"each boundary entry must be a (before, after) pair, got {pair!r}"
                )
        pairs = [("none", "none")] * (ndim - len(given)) + given  # type: ignore[assignment]

    pairs = [tuple(_normalize_boundary_mode(side) for side in pair) for pair in pairs]

    effective = [pair if axis in axes else ("none", "none") for axis, pair in enumerate(pairs)]

    policies = {side for pair in effective for side in pair if side is not NO_BOUNDARY}
    if len(policies) > 1:
        names = ", ".join(sorted(repr(policy) for policy in policies))
        raise ValueError(
            f"a single padding policy must apply to the whole array, got {names}. "
            "Sides choose whether they are padded, not how."
        )

    where = tuple((pair[0] is not NO_BOUNDARY, pair[1] is not NO_BOUNDARY) for pair in effective)
    return where, policies.pop() if policies else NO_BOUNDARY


def _axis_output(
    out_mode: OutputMode,
    *,
    full_size: int,
    origin: int,
    window_size: int,
    kernel_size: int,
) -> slice:
    """Slice selecting the requested region inside the ``"full"`` result."""
    if out_mode == "full":
        return slice(0, full_size)
    if out_mode == "same":
        return slice(origin, origin + window_size)
    if out_mode == "valid":
        margin = kernel_size // 2
        if window_size <= 2 * margin:
            raise ValueError(
                f"the 'valid' output is empty: a kernel of size {kernel_size} does not fit "
                f"in a window of size {window_size}"
            )
        return slice(origin + margin, origin + window_size - margin)

    # Unreachable: build_plan validates the value and every mode is handled
    # above. Kept so that adding a mode fails loudly instead of silently
    # returning None.
    raise ValueError(f"unsupported output mode {out_mode!r}")  # pragma: no cover


def build_plan(
    shape: tuple[int, ...],
    kernel: ArrayLike,
    win: ArrayLike | None = None,
    *,
    boundary: BoundarySpec = "none",
    out_mode: OutputMode = "same",
    zoom: int | tuple[int, int] = 1,
    decimation: DecimationOrigin = "centered",
    axes: int | Iterable[int] | None = None,
    dtype: DTypeLike | None = None,
    method: ConvolutionMethod = "overlap_add",
) -> FilterPlan:
    """Compute the full geometry of a filtering operation without touching data.

    Shapes, slices and padding are settled here. Both :func:`fft_array_filter`
    and :func:`fft_array_filter_output_shape` consume the plan it returns.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array that will be filtered.

    kernel : array_like
        Filter taps in the spatial domain. Even-sized axes are right-padded
        with zeros so that the kernel has a well-defined centre.

    win : array_like or None, optional
        Production window as ``(ndim, 2)`` inclusive bounds. ``None`` means
        the whole array.

    boundary : str, None, or sequence of pairs, optional
        Policy applied on each side when the kernel margin falls outside the
        array. A scalar applies everywhere. Default
        ``"none"``, i.e. no margin at all.

    out_mode : str, optional
        Region of the convolution to return. Default
        ``"same"``.

    zoom : int or tuple of two ints, optional
        Rational resampling factor ``P/Q``. Only ``P == 1`` is implemented;
        ``Q > 1`` decimates the output. Default ``1``.

    decimation : str, optional
        Which sample of each block of ``Q`` is kept, on every target axis. The
        phase is counted in the output frame and not in the input one, so what
        it lands on depends on ``out_mode``. With ``"same"``, index 0 of the
        output is the first sample of ``win``, hence ``"leading"`` keeps the
        window's own first sample and ``"centered"`` keeps the one
        ``(Q - 1) // 2`` samples further in. With ``"full"``, index 0 is the first
        sample of the convolution support, which sits :attr:`AxisPlan.origin`
        samples ahead of the window's first one; with ``"valid"`` it sits
        ``kernel_size - 1`` samples into that support. In both of those, neither
        origin lands on the window's first sample. Moving ``win`` therefore
        moves the sampling grid with it: this decimates the window, it does not
        resample the array on a grid anchored at index 0. Default
        ``"centered"``.

    axes : int, iterable of int, or None, optional
        Axes along which to convolve. Other axes are passed through untouched.
        Default ``None``, i.e. every axis.

    dtype : data-type, optional
        Working dtype. Default ``np.result_type(kernel, ...)`` resolved by
        :func:`fft_array_filter` against the input array.

    method : str, optional
        Convolution backend. Default ``"overlap_add"``.

    Returns
    -------
    FilterPlan
        Immutable description of the operation.

    Raises
    ------
    ValueError
        If the options are inconsistent, if the window is not contained in the
        array, or if the requested zoom is not supported.

    """
    ndim = len(shape)
    axes = normalize_axes(axes, ndim)
    zoom_pq = normalize_zoom(zoom)
    if not zoom_pq.is_supported:
        raise ValueError(
            f"zoom P/Q = {zoom_pq.p}/{zoom_pq.q} is not supported; "
            "only pure decimation (P == 1) is implemented"
        )
    out_mode = _normalize_choice(out_mode, OUTPUT_MODES, "out_mode")
    method = _normalize_choice(method, CONVOLUTION_METHODS, "method")
    decimation = _normalize_choice(decimation, DECIMATION_ORIGINS, "decimation")

    kernel = align_kernel(kernel, ndim, axes)
    kernel = pad_kernel_to_odd(kernel, axes)
    margins = kernel_margin(kernel, axes)

    window = window_normalize(win, shape)

    pad_sides, pad_mode = _normalize_boundary(boundary, ndim, axes)
    offset = decimation_offset(zoom_pq.q, decimation)

    per_axis: list[AxisPlan] = []
    for axis in range(ndim):
        size = shape[axis]
        first, last = int(window[axis][0]), int(window[axis][1])
        window_size = last - first + 1

        if axis not in axes:
            per_axis.append(
                AxisPlan(
                    source=slice(0, size),
                    pad_width=(0, 0),
                    conv_size=size,
                    full_size=size,
                    origin=0,
                    window_size=size,
                    output=slice(0, size),
                    output_size=size,
                )
            )
            continue

        margin = margins[axis]
        synthesise_before, synthesise_after = pad_sides[axis]

        # Real neighbours are always taken when they exist, whatever the
        # policy: a boundary condition describes the edge of the array, never a
        # seam in the middle of it. The policy only decides what happens beyond
        # what the array holds.
        real_before = min(margin, first)
        real_after = min(margin, size - 1 - last)
        pad_before = margin - real_before if synthesise_before else 0
        pad_after = margin - real_after if synthesise_after else 0

        if (
            pad_mode == "wrap"
            and (pad_before or pad_after)
            and not (first == 0 and last == size - 1)
        ):
            # Periodic continuation only means something when the window covers
            # the whole axis: the samples to bring in are at the far end of the
            # array, not at the far end of the requested extent.
            raise ValueError(
                f"the 'wrap' boundary needs the window to span axis {axis}, but it "
                f"covers {first}..{last} of {size}"
            )

        source = slice(first - real_before, last + real_after + 1)
        lead = real_before + pad_before
        trail = real_after + pad_after
        conv_input_size = lead + window_size + trail
        full_size = conv_input_size + kernel.shape[axis] - 1
        origin = margin + lead

        output = _axis_output(
            out_mode,
            full_size=full_size,
            origin=origin,
            window_size=window_size,
            kernel_size=kernel.shape[axis],
        )
        output_size = output.stop - output.start

        if zoom_pq.q > 1:
            output = slice(output.start + offset, output.stop, zoom_pq.q)
            output_size = decimated_size(output_size, zoom_pq.q, offset)

        per_axis.append(
            AxisPlan(
                source=source,
                pad_width=(pad_before, pad_after),
                conv_size=conv_input_size,
                full_size=full_size,
                origin=origin,
                window_size=window_size,
                output=output,
                output_size=output_size,
            )
        )

    # A declared policy that ends up padding nothing (an interior tile reading
    # real neighbours on every side) is not the rule executed for this plan.
    if not any(any(axis.pad_width) for axis in per_axis):
        pad_mode = NO_BOUNDARY

    return FilterPlan(
        kernel=kernel,
        axes=axes,
        per_axis=tuple(per_axis),
        pad_mode=pad_mode,
        out_mode=out_mode,
        zoom=zoom_pq,
        dtype=np.dtype(dtype) if dtype is not None else np.dtype(kernel.dtype),
        method=method,
    )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _make_convolution_input(arr: NDArray, plan: FilterPlan) -> NDArray:
    """Build the padded, dtype-converted input of the convolution.

    One allocation and one copy whatever the number of padded sides. The buffer
    is created at its final size, the real samples are written into it , and the
    margins are filled in place by :func:`gridr.core.utils.array_pad.pad_inplace`.
    """
    source = arr[plan.source]
    if plan.pad_mode == NO_BOUNDARY:
        return np.asarray(source, dtype=plan.dtype)

    buffer = np.empty(plan.conv_shape, dtype=plan.dtype)
    src_win = plan.src_win
    buffer[src_win] = source
    pad_inplace(buffer, src_win, plan.pad_width, mode=plan.pad_mode)
    return buffer


_CONVOLVERS = {
    "overlap_add": lambda a, k, axes: signal.oaconvolve(a, k, mode="full", axes=axes),
    "fft": lambda a, k, axes: signal.fftconvolve(a, k, mode="full", axes=axes),
    "direct": lambda a, k, axes: signal.convolve(a, k, mode="full", method="direct"),
    "auto": lambda a, k, axes: signal.convolve(a, k, mode="full", method="auto"),
}


def fft_array_filter(
    arr: NDArray,
    kernel: ArrayLike,
    win: ArrayLike | None = None,
    *,
    boundary: BoundarySpec = "none",
    out_mode: OutputMode = "same",
    zoom: int | tuple[int, int] = 1,
    decimation: DecimationOrigin = "centered",
    axes: int | Iterable[int] | None = None,
    dtype: DTypeLike | None = None,
    method: ConvolutionMethod = "overlap_add",
    plan: FilterPlan | None = None,
) -> FilterResult:
    """Convolve ``arr`` with ``kernel`` over a production window.

    See :func:`build_plan` for the options; this function executes the plan.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.

    kernel : array_like
        Filter taps in the spatial domain. Even-sized axes are right-padded
        with zeros so that the kernel has a well-defined centre.

    win : array_like or None, optional
        Production window as ``(ndim, 2)`` inclusive bounds. ``None`` means
        the whole array.

    boundary : str, None, or sequence of pairs, optional
        Policy applied on each side when the kernel margin falls outside the
        array. A scalar applies everywhere. Default
        ``"none"``, i.e. no margin at all.

    out_mode : str, optional
        Region of the convolution to return. Default
        ``"same"``.

    zoom : int or tuple of two ints, optional
        Rational resampling factor ``P/Q``. Only ``P == 1`` is implemented;
        ``Q > 1`` decimates the output. Default ``1``.

    decimation : str, optional
        Which sample of each block of ``Q`` is kept, on every target axis. The
        phase is counted in the output frame and not in the input one, so what
        it lands on depends on ``out_mode``. With ``"same"``, index 0 of the
        output is the first sample of ``win``, hence ``"leading"`` keeps the
        window's own first sample and ``"centered"`` keeps the one
        ``(Q - 1) // 2`` samples further in. With ``"full"``, index 0 is the first
        sample of the convolution support, which sits :attr:`AxisPlan.origin`
        samples ahead of the window's first one; with ``"valid"`` it sits
        ``kernel_size - 1`` samples into that support. In both of those, neither
        origin lands on the window's first sample. Moving ``win`` therefore
        moves the sampling grid with it: this decimates the window, it does not
        resample the array on a grid anchored at index 0. Default
        ``"centered"``.

    axes : int, iterable of int, or None, optional
        Axes along which to convolve. Other axes are passed through untouched.
        Default ``None``, i.e. every axis.

    dtype : data-type, optional
        Working dtype. Default ``np.result_type(kernel, ...)`` resolved by
        :func:`fft_array_filter` against the input array.

    method : str, optional
        Convolution backend. Default ``"overlap_add"``.

    plan : FilterPlan, optional
        A plan built beforehand by :func:`build_plan`. When given, every other
        option is ignored. Reuse one across the tiles of a raster to skip
        recomputing the geometry each time.

    Returns
    -------
    FilterResult
        Named tuple ``(data, window)``. ``window`` locates the production
        window inside the ``"full"`` convolution frame, with inclusive bounds.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    >>> kernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    >>> out, window = fft_array_filter(arr, kernel)
    >>> out.shape
    (5, 5)
    >>> window.tolist()  # inclusive bounds, in the "full" convolution frame
    [[1, 5], [1, 5]]

    """
    arr = np.asarray(arr)
    if plan is None:
        plan = build_plan(
            arr.shape,
            kernel,
            win,
            boundary=boundary,
            out_mode=out_mode,
            zoom=zoom,
            decimation=decimation,
            axes=axes,
            dtype=dtype if dtype is not None else np.result_type(arr, np.asarray(kernel)),
            method=method,
        )
    elif plan.per_axis and len(plan.per_axis) != arr.ndim:
        raise ValueError(f"plan was built for a {len(plan.per_axis)}-d array, got {arr.ndim}-d")

    conv_arr = _make_convolution_input(arr, plan)
    conv_kernel = np.asarray(plan.kernel, dtype=plan.dtype)

    full = _CONVOLVERS[plan.method](conv_arr, conv_kernel, plan.axes)
    return FilterResult(data=full[plan.output], window=plan.window)


def fft_array_filter_output_shape(
    arr_or_shape: NDArray | tuple[int, ...],
    kernel: ArrayLike,
    win: ArrayLike | None = None,
    *,
    boundary: BoundarySpec = "none",
    out_mode: OutputMode = "same",
    zoom: int | tuple[int, int] = 1,
    decimation: DecimationOrigin = "centered",
    axes: int | Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Shape :func:`fft_array_filter` would return, without doing the work.

    Takes an array or a plain shape, so outputs can be sized before any pixel
    is read.

    Examples
    --------
    >>> fft_array_filter_output_shape((50, 60), np.ones((3, 3)), zoom=(1, 5))
    (10, 12)

    """
    shape = (
        tuple(arr_or_shape.shape)
        if isinstance(arr_or_shape, np.ndarray)
        else tuple(int(size) for size in arr_or_shape)
    )
    return build_plan(
        shape,
        kernel,
        win,
        boundary=boundary,
        out_mode=out_mode,
        zoom=zoom,
        decimation=decimation,
        axes=axes,
    ).output_shape
