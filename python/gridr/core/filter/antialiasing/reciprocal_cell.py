#
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
"""
Reciprocal cell antialiasing filter.

Conventions: the transition matrix is ``W = [w1 w2]``, where ``w1`` and
``w2`` are the source displacements ``(dx, dy)`` for one destination column
and one destination row.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import spence

from gridr.core.filter.antialiasing import KernelFilter
from gridr.core.filter.lattice import voronoi_cell_facets

#: Tolerance on ``|a|_1 - 1`` when telling whether the band of a facet cuts
#: the source Nyquist square.
EDGE_TOL = 1e-9

#: Largest jump of the template across the Nyquist border accepted without a
#: Nyquist gate (experimental option).
JUMP_TOL = 1e-3


# --------------------------------------------------------------------------- #
# Cell of the reciprocal lattice and decision to filter
# --------------------------------------------------------------------------- #
def reciprocal_basis(w: np.ndarray) -> np.ndarray:
    """
    Basis of the reciprocal lattice ``W^{-T} Z^2``.

    Parameters
    ----------
    w : np.ndarray
        Transition matrix ``[w1 w2]``.

    Returns
    -------
    np.ndarray
        Matrix ``B = [b1 b2]`` with ``b_i . w_j = delta_ij``.
    """
    return np.linalg.inv(np.asarray(w, dtype=np.float64)).T


def metric_matrix(w: np.ndarray, isotropic_frame: str | np.ndarray) -> np.ndarray:
    """
    Metric ``G`` on source frequencies selected by ``isotropic_frame``.

    Parameters
    ----------
    w : np.ndarray
        Transition matrix ``[w1 w2]``.

    isotropic_frame : {"destination", "source"} or np.ndarray
        ``"destination"`` gives ``W W^T``, so that the norm is that of the
        destination frequency ``g = W^T f``; ``"source"`` gives the identity;
        a 2x2 symmetric positive definite matrix is returned as is.

    Returns
    -------
    np.ndarray
        The 2x2 metric.

    Raises
    ------
    ValueError
        If ``isotropic_frame`` is neither a known name nor a 2x2 symmetric
        positive definite matrix.
    """
    if isinstance(isotropic_frame, str):
        if isotropic_frame == "destination":
            return w @ w.T
        if isotropic_frame == "source":
            return np.eye(2)
        raise ValueError(
            f"Unknown isotropic_frame {isotropic_frame!r}, expected 'destination', "
            "'source' or a 2x2 matrix"
        )
    g = np.asarray(isotropic_frame, dtype=np.float64)
    if (
        g.shape != (2, 2)
        or not np.allclose(g, g.T)
        or np.any(np.linalg.eigvalsh(g) <= 0)
    ):
        raise ValueError(
            "An explicit isotropic_frame must be symmetric positive definite"
        )
    return g


def cell_facets(
    w: np.ndarray, isotropic_frame: str | np.ndarray = "destination"
) -> list[np.ndarray]:
    """
    Facet vectors of the Voronoi cell of the reciprocal lattice.

    The cell is the set of source frequencies closer to the origin than to any
    other point of ``W^{-T} Z^2``, in the metric selected by
    ``isotropic_frame``. In the destination metric it is the parallelogram
    ``{f : ||W^T f||_inf <= 1/2}``, whose facets are ``w1`` and ``w2``.

    Parameters
    ----------
    w : np.ndarray
        Transition matrix ``[w1 w2]``.

    isotropic_frame : {"destination", "source"} or np.ndarray, optional
        Metric of the Voronoi construction (see :func:`metric_matrix`).

    Returns
    -------
    list[np.ndarray]
        Two or three facet vectors; the cell is ``{f : |f . a| <= 1/2}``.
    """
    w = np.asarray(w, dtype=np.float64)
    return voronoi_cell_facets(reciprocal_basis(w), metric_matrix(w, isotropic_frame))


def filter_required(facets: list[np.ndarray]) -> bool:
    """
    Tell whether the Nyquist square of the source leaves the cell.

    The spectrum of the source is confined to the Nyquist square. The cell is
    convex, so the square lies inside it if and only if its four corners do.
    At the corners, ``|f . a|`` reaches ``|a|_1 / 2``: the square leaves the
    cell when a facet has ``|a|_1 > 1``. The test is sufficient for the
    absence of aliasing, not necessary.

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors of the cell.

    Returns
    -------
    bool
        True if a corner of the Nyquist square lies outside the cell.
    """
    return any(abs(a[0]) + abs(a[1]) > 1.0 + EDGE_TOL for a in facets)


# --------------------------------------------------------------------------- #
# Frequency template
# --------------------------------------------------------------------------- #
# TODO: decide the fate of both experimental options (keep, make default or
# drop).
@dataclass(frozen=True)
class TemplateOptions:
    """
    Experimental options of the frequency template, off by default.

    Without option, the template is the product of the gates of all the
    facets of the cell. The options are described in
    ``theoretical_foundations/antialiasing/experimental_options`` and in
    :func:`template_gates`.

    Attributes
    ----------
    nyquist_gates : bool
        Add a gate on a facet of the source Nyquist square, on each axis where
        the template jumps across the border of the square. For oblique grids
        near unit scale.

    progressive_gates : bool
        Give each facet a tolerance that grows continuously from 1, no gate,
        as its band starts to cut the source Nyquist square (see
        :func:`facet_tolerance`). For grids that keep or zoom an axis, or are
        close to unit scale.
    """

    nyquist_gates: bool = False
    progressive_gates: bool = False


#: Template options used when none is given.
DEFAULT_TEMPLATE_OPTIONS = TemplateOptions()


def tanh_gate(g: np.ndarray, slope: float, t: float) -> np.ndarray:
    """
    Smooth gate on a normalized facet coordinate, equal to 1 at the origin.

    ``phi(g) = (tanh(d (g + t)) - tanh(d (g - t))) / (2 tanh(d t))`` with
    ``d = 2 slope``.

    Parameters
    ----------
    g : np.ndarray
        Normalized coordinate ``f . a``; the cell boundary is at ``+-1/2``.

    slope : float
        Steepness of the transition.

    t : float
        Half-width of the gate.

    Returns
    -------
    np.ndarray
        Gate values in ``(0, 1]``.
    """
    d = 2.0 * slope
    return (np.tanh(d * (g + t)) - np.tanh(d * (g - t))) / (2.0 * np.tanh(d * t))


def cutoff_half_width(slope: float, tolerance: float) -> float:
    """
    Half-width ``t`` of the gate such that its value on the cell boundary is
    the tolerance.

    Writing the tanh gate at 1/2 leads to solving
    ``cosh(2 s t) = sinh(s) sqrt(tolerance / (1 - tolerance))`` with ``s = slope``.
    A tolerance of 0.5 gives ``t`` close to 1/2, a transition centered on the
    boundary.

    Parameters
    ----------
    slope : float
        Steepness of the transition.

    tolerance : float
        Gate value on the cell boundary, in ``(0, 1)``.

    Returns
    -------
    float
        Half-width ``t``.

    Raises
    ------
    ValueError
        If the tolerance is out of ``(0, 1)``, or if the slope is too small for
        the gate to fall to the tolerance on the boundary.
    """
    if not 0 < tolerance < 1:
        raise ValueError("The gate tolerance must lie in (0, 1)")
    c = np.sinh(slope) * np.sqrt(tolerance / (1.0 - tolerance))
    if c < 1.0:
        raise ValueError(
            f"The slope {slope:g} is too small for the gate tolerance {tolerance:g}"
        )
    return float(np.arccosh(c) / (2.0 * slope))


def cell_response(
    facets: list[np.ndarray],
    fx: np.ndarray,
    fy: np.ndarray,
    slope: float,
    t: float | list[float],
) -> np.ndarray:
    """
    Product of the tanh gates of a list of facets.

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors.

    fx, fy : np.ndarray
        Source frequencies, in cycles per source pixel.

    slope : float
        Steepness of the transition.

    t : float or list of float
        Half-width of each gate, one value for all or one per facet.

    Returns
    -------
    np.ndarray
        Gate product, of the broadcast shape of ``fx`` and ``fy``.
    """
    ts = [t] * len(facets) if np.ndim(t) == 0 else t
    response = np.ones(np.broadcast(fx, fy).shape)
    for a, ta in zip(facets, ts, strict=True):
        response = response * tanh_gate(fx * a[0] + fy * a[1], slope, ta)
    return response


def border_mismatch(
    facets: list[np.ndarray], axis: int, slope: float, t: float | list[float]
) -> float:
    """
    Largest difference of a gate product between two opposite edges of the
    Nyquist square.

    On the source grid the edges ``f_axis = -1/2`` and ``f_axis = +1/2`` are
    the same frequencies. A non-zero difference is a jump of the response
    that a discrete kernel would have to follow.

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors.

    axis : int
        0 for the edges ``fx = +-1/2``, 1 for the edges ``fy = +-1/2``.

    slope : float
        Steepness of the gates.

    t : float or list of float
        Half-width of the gates, one value for all or one per facet.

    Returns
    -------
    float
        ``max |H(edge+) - H(edge-)|``.
    """
    s = np.linspace(-0.5, 0.5, 1001)
    half = np.full_like(s, 0.5)
    if axis == 0:
        upper = cell_response(facets, half, s, slope, t)
        lower = cell_response(facets, -half, s, slope, t)
    else:
        upper = cell_response(facets, s, half, slope, t)
        lower = cell_response(facets, s, -half, slope, t)
    return float(np.abs(upper - lower).max())


def ramp_width(a: np.ndarray, slope: float, tolerance: float) -> float:
    """
    Width of the ramp of the progressive gates for a facet.

    With ``d = 2 slope``, ``k = (1 - tolerance) / tolerance`` and
    ``c = ln(1 / (tolerance (1 - tolerance))) - 1``, the width is::

        min(c / d + min(|a_x|, |a_y|), (c + sqrt(c^2 + 2 m)) / d)

    where ``m = -Li2(-k) - ln(1 + k)`` is the first moment of the blur of a
    gate.
    A facet aligned with an axis gets ``c / d``.

    Parameters
    ----------
    a : np.ndarray
        Facet vector.

    slope : float
        Steepness of the gates.

    tolerance : float
        Aliasing tolerance, in ``(0, 1)``.

    Returns
    -------
    float
        Width of the ramp, in units of ``|a|_1``.
    """
    # TODO: cite the derivation of the ramp width once it is published.
    d = 2.0 * slope
    k = (1.0 - tolerance) / tolerance
    c = np.log(1.0 / (tolerance * (1.0 - tolerance))) - 1.0
    # first moment of the blur, -Li2(-k) - ln(1 + k), with Li2(z) = spence(1 - z)
    m = -spence(1.0 + k) - np.log(1.0 + k)
    axis = c / d + min(abs(a[0]), abs(a[1]))
    oblique = (c + np.sqrt(c * c + 2.0 * m)) / d
    return float(min(axis, oblique))


def facet_tolerance(a: np.ndarray, slope: float, tolerance: float) -> float:
    """
    Gate value on the cell boundary for a facet, with the progressive gates.

    The band ``|f . a| <= 1/2`` of the facet reaches furthest over the source
    Nyquist square at a corner, where ``|f . a| = |a|_1 / 2``: it cuts the
    square when ``|a|_1 > 1``. The tolerance is 1, no gate, up to
    ``|a|_1 = 1``; it then falls linearly to ``tolerance`` at
    ``|a|_1 = 1 + ramp_width(a, slope, tolerance)``, and stays there beyond.

    Parameters
    ----------
    a : np.ndarray
        Facet vector.

    slope : float
        Steepness of the gates.

    tolerance : float
        Aliasing tolerance, the gate value on the boundary past the ramp.

    Returns
    -------
    float
        Gate value on the cell boundary, in ``[tolerance, 1]``.
    """
    excess = abs(a[0]) + abs(a[1]) - 1.0
    if excess <= EDGE_TOL:
        return 1.0
    ramp = ramp_width(a, slope, tolerance)
    if excess >= ramp:
        return tolerance
    return 1.0 - (1.0 - tolerance) * excess / ramp


def template_gates(
    facets: list[np.ndarray],
    slope: float,
    t: float,
    options: TemplateOptions = DEFAULT_TEMPLATE_OPTIONS,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Facets and half-widths of the gates that make the template.

    By default, these are all the facets of the cell, with the half-width
    ``t``. Two experimental options change this list.

    With ``options.progressive_gates``, each facet gets the half-width of its
    own tolerance (:func:`facet_tolerance`), and the facets whose band
    contains the whole Nyquist square are dropped.

    With ``options.nyquist_gates``, the facet of the Nyquist square on an axis
    is added when the template jumps by more than :data:`JUMP_TOL` across the
    border on that axis. On the source grid, ``f`` and ``f + m`` (``m`` in
    ``Z^2``) are the same frequency, so such a jump makes the kernel oscillate
    and grow to its size bound. The added gate has the half-width ``t``: it
    equals the tolerance on both edges, which divides the jump by at least
    ``1 / tolerance``. Only a jump of value adds a gate; a change of slope
    only affects the response near the border, where the accuracy is not
    measured (:data:`ERROR_DOMAIN`).

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors of the cell.

    slope : float
        Steepness of the gates.

    t : float
        Half-width of the gates, set by the aliasing tolerance.

    options : TemplateOptions, optional
        Experimental options.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        Facets of the template and the half-width of each gate.
    """
    gates = [np.asarray(a, dtype=np.float64) for a in facets]
    ts = [t] * len(gates)
    if options.progressive_gates:
        tolerance = float(tanh_gate(np.array(0.5), slope, t))
        kept = []
        for a in gates:
            tau = facet_tolerance(a, slope, tolerance)
            if tau < 1.0:
                ta = t if tau == tolerance else cutoff_half_width(slope, tau)
                kept.append((a, ta))
        gates = [a for a, _ in kept]
        ts = [ta for _, ta in kept]
    if options.nyquist_gates:
        for axis, e in enumerate(np.eye(2)):
            if border_mismatch(gates, axis, slope, ts) > JUMP_TOL:
                gates.append(e)
                ts.append(t)
    return gates, ts


def template(
    facets: list[np.ndarray],
    fx: np.ndarray,
    fy: np.ndarray,
    slope: float,
    t: float,
    options: TemplateOptions = DEFAULT_TEMPLATE_OPTIONS,
) -> np.ndarray:
    """
    Frequency template of the filter for a cell.

    The template is the product of the gates of :func:`template_gates`. Without
    experimental option, these are the gates of every facet of the cell, with
    the same slope and the same half-width.

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors of the cell.

    fx, fy : np.ndarray
        Source frequencies, in cycles per source pixel.

    slope : float
        Steepness of the transition.

    t : float
        Half-width of the gates, set by the aliasing tolerance.

    options : TemplateOptions, optional
        Experimental options.

    Returns
    -------
    np.ndarray
        Template values.
    """
    gates, ts = template_gates(facets, slope, t, options)
    return cell_response(gates, fx, fy, slope, ts)


# --------------------------------------------------------------------------- #
# Kernel synthesis
# --------------------------------------------------------------------------- #
#: Half-side of the frequency square on which the response error is measured.
#: On the border of the Nyquist square the template may have a corner (or a
#: jump) that no finite kernel follows; measuring the error there would only
#: make the size saturate.
ERROR_DOMAIN = 0.45

#: The size bound is this multiple of the size estimate, plus one; the search
#: for the accurate size goes at least as far.
#: The value is empirical.
#: Grids near unit scale that need more get a warning
#: (see ``ReciprocalCellFilter.design``).
SIZE_BOUND_MARGIN = 2

#: Smallest size of the fine grid, for very small kernels.
MIN_FINE_SIZE = 129


def estimated_kernel_size(
    gates: list[np.ndarray], slope: float, accuracy: float
) -> int:
    """
    Odd kernel size predicted by the decay of the tanh gates.

    The kernel of the template decays exponentially in every direction, at
    least as fast as ``exp(-r / L)`` with ``r`` the distance to the center,
    ``L = 2 slope |a| / pi^2`` and ``a`` the longest facet; for the
    parallelogram, this rate is reached along ``a``. Cutting the kernel
    where it has fallen to ``accuracy`` on both sides of the center gives
    ``n ~ (4 / pi^2) ln(1 / accuracy) slope max|a|``. This is a calibrated
    estimate; it only initializes the search. It only holds when the template
    is negligible outside the Nyquist square: a jump of the template on the
    border of the square leaves an error that no size removes.

    Parameters
    ----------
    gates : list[np.ndarray]
        Facets of the template (see :func:`template_gates`).

    slope : float
        Steepness of the gates.

    accuracy : float
        Target relative error.

    Returns
    -------
    int
        Odd size estimate.
    """
    rho = max((float(np.linalg.norm(a)) for a in gates), default=1e-3)
    n = 4.0 / np.pi**2 * np.log(1.0 / accuracy) * slope * rho
    return int(np.ceil(n)) | 1


def fine_kernel(
    gates: list[np.ndarray], slope: float, t: float | list[float], size: int
) -> np.ndarray:
    """
    Sample a gate product on a fine odd grid and return its inverse DFT.

    The gate product is sampled at ``f = m / size`` over the source Nyquist
    square. With an odd ``size`` the samples are symmetric about the origin, so
    the kernel is real, even and centered.

    Parameters
    ----------
    gates : list[np.ndarray]
        Facets of the template (see :func:`template_gates`).

    slope : float
        Steepness of the gates.

    t : float or list of float
        Half-width of the gates, one value for all or one per facet.

    size : int
        Odd size of the fine grid.

    Returns
    -------
    np.ndarray
        Fine kernel, shape ``(size, size)``, centered at ``(size // 2, size // 2)``.
    """
    f = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(f, f)
    h = cell_response(gates, fx, fy, slope, t)
    return np.real(np.fft.fftshift(np.fft.ifft2(h)))


def kernel_response(
    kernel: np.ndarray, n_fft: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    True frequency response of a centered odd kernel on a fine grid.

    Parameters
    ----------
    kernel : np.ndarray
        Odd-sized kernel, centered.

    n_fft : int, optional
        Minimum size of the zero-padded DFT. It is raised to the next power of
        two above twice the kernel size when needed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(f, response)``: frequencies, zero at the center, and the real
        response ``response[fy, fx]``.
    """
    hr, hc = kernel.shape[0] // 2, kernel.shape[1] // 2
    n_fft = max(n_fft, 1 << int(np.ceil(np.log2(2 * max(kernel.shape)))))
    buffer = np.zeros((n_fft, n_fft))
    buffer[: kernel.shape[0], : kernel.shape[1]] = kernel
    buffer = np.roll(buffer, (-hr, -hc), axis=(0, 1))
    f = np.fft.fftshift(np.fft.fftfreq(n_fft))
    return f, np.real(np.fft.fftshift(np.fft.fft2(buffer)))


def response_error(
    kernel: np.ndarray,
    gates: list[np.ndarray],
    slope: float,
    t: float | list[float],
) -> float:
    """
    Sup-norm error between the response of a kernel and a gate product.

    The error is measured on ``||f||_inf <= ERROR_DOMAIN`` (see
    :data:`ERROR_DOMAIN`).

    Parameters
    ----------
    kernel : np.ndarray
        Odd-sized kernel.

    gates : list[np.ndarray]
        Facets of the template (see :func:`template_gates`).

    slope : float
        Steepness of the gates.

    t : float or list of float
        Half-width of the gates, one value for all or one per facet.

    Returns
    -------
    float
        ``max |H_kernel - H_template|`` over ``||f||_inf <= ERROR_DOMAIN``.
    """
    # TODO: perf, the template on the DFT does not depend on the kernel;
    # evaluate it once per synthesis.
    f, response = kernel_response(kernel)
    fx, fy = np.meshgrid(f, f)
    inside = (np.abs(fx) <= ERROR_DOMAIN) & (np.abs(fy) <= ERROR_DOMAIN)
    error = np.abs(response - cell_response(gates, fx, fy, slope, t))
    return float(error[inside].max())


def crop_kernel(fine: np.ndarray, n: int) -> np.ndarray:
    """
    Central ``n x n`` crop of a fine kernel, normalized to a unit sum.

    Parameters
    ----------
    fine : np.ndarray
        Fine kernel, odd size, centered.

    n : int
        Odd crop size, at most the size of the fine kernel.

    Returns
    -------
    np.ndarray
        Cropped kernel.
    """
    c, h = fine.shape[0] // 2, n // 2
    kernel = fine[c - h : c + h + 1, c - h : c + h + 1]
    return kernel / kernel.sum()


def smallest_accurate_size(accurate, start: int, n_max: int) -> int | None:
    """
    Smallest odd size accepted by ``accurate``, searched up to ``n_max``.

    The search starts from ``start``, doubles the size (``n -> 2 n + 1``)
    until ``accurate`` holds, then bisects on odd sizes. The bisection ends on
    an accurate size ``n`` whose neighbour ``n - 2`` failed. The response
    error is not strictly monotone in the size, so the two odd sizes below
    that failure, ``n - 4`` and ``n - 6``, are also tried, and the smallest
    accurate one is kept.

    Parameters
    ----------
    accurate : callable
        ``accurate(n)`` tells whether the odd size ``n`` meets the accuracy.

    start : int
        First size tried, usually :func:`estimated_kernel_size`.

    n_max : int
        Odd upper end of the search.

    Returns
    -------
    int or None
        The size, or None if no size up to ``n_max`` is accurate.
    """
    hi = min(max(3, start), n_max)
    lo = 1
    failed = set()
    while not accurate(hi):
        if hi >= n_max:
            return None
        failed.add(hi)
        lo, hi = hi, min(2 * hi + 1, n_max)
    while hi - lo > 2:
        mid = ((lo + hi) // 2) | 1
        if accurate(mid):
            hi = mid
        else:
            failed.add(mid)
            lo = mid
    # the error may rise and fall again by a little, so try the two sizes
    # below the failure lo, skipping those the bisection already rejected
    for n in (lo - 4, lo - 2):
        if n >= 1 and n not in failed and accurate(n):
            return n
    return hi


def _check_odd(name: str, value: int | None) -> None:
    """
    Check that an optional size is a positive odd integer.

    Parameters
    ----------
    name : str
        Name of the size, for the message.

    value : int or None
        Size to check; None is accepted.

    Raises
    ------
    ValueError
        If ``value`` is not None and not a positive odd integer.
    """
    if value is not None and (value < 1 or value % 2 == 0):
        raise ValueError(f"{name} must be a positive odd integer or None")


def synthesize_kernel(
    facets: list[np.ndarray],
    slope: float,
    tolerance: float,
    accuracy: float = 0.01,
    max_size: int | None = None,
    size: int | None = None,
    options: TemplateOptions = DEFAULT_TEMPLATE_OPTIONS,
) -> tuple[np.ndarray, dict]:
    """
    Synthesize an odd kernel whose response follows the template.

    The template is sampled on a fine odd grid, about four times the size
    estimate; its inverse DFT is a faithful sample of the continuous kernel.
    The kernel is the central crop of the smallest odd size meeting
    ``accuracy``, searched up to the size bound ``max_size``. Beyond the
    bound, the accurate kernel is truncated to the bound and the template is
    not changed. A forced ``size`` is used as is.

    Parameters
    ----------
    facets : list[np.ndarray]
        Facet vectors of the cell.

    slope : float
        Steepness of the gates.

    tolerance : float
        Aliasing tolerance, gate value on the cell boundary.

    accuracy : float, optional
        Target sup-norm error of the response.

    max_size : int or None, optional
        Odd upper bound on the size. None sets it to
        ``SIZE_BOUND_MARGIN * estimate + 1``.

    size : int or None, optional
        Forced odd size.

    options : TemplateOptions, optional
        Experimental options.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(kernel, info)``. ``info`` holds the half-width ``t``, the gates of
        the template ``gates`` and their half-widths ``ts``, the size estimate
        ``estimate``, the bound used ``size_bound``, the size of the fine grid
        ``n_fine``, the accurate size ``accurate_size`` (None if forced or
        beyond the search range), ``accuracy_met`` and ``response_error``.

    Raises
    ------
    ValueError
        If ``max_size`` or ``size`` is not a positive odd integer.
    """
    # TODO: perf, the template of a diagonal W in the destination metric is
    # separable; return a separable kernel.
    _check_odd("max_size", max_size)
    _check_odd("size", size)
    t = cutoff_half_width(slope, tolerance)
    gates, ts = template_gates(facets, slope, t, options)
    estimate = estimated_kernel_size(gates, slope, accuracy)
    bound = SIZE_BOUND_MARGIN * estimate + 1 if max_size is None else max_size
    # the search may go past an explicit bound, so that the diagnostics can tell
    # how far the accurate size is; the fine grid holds the largest candidate twice
    n_search = max(SIZE_BOUND_MARGIN * estimate + 1, bound, size or 1)
    n_fine = max(MIN_FINE_SIZE, 2 * n_search + 1)
    fine = fine_kernel(gates, slope, ts, n_fine)
    if size is not None:
        n_accurate, n = None, size
    else:

        def accurate(m: int) -> bool:
            """Tell whether the ``m x m`` crop meets the accuracy."""
            return response_error(crop_kernel(fine, m), gates, slope, ts) <= accuracy

        n_accurate = smallest_accurate_size(accurate, estimate, n_search)
        n = bound if n_accurate is None else min(n_accurate, bound)
    kernel = crop_kernel(fine, n)
    error = response_error(kernel, gates, slope, ts)
    info = {
        "t": t,
        "gates": gates,
        "ts": ts,
        "estimate": estimate,
        "size_bound": bound,
        "n_fine": n_fine,
        "accurate_size": n_accurate,
        "accuracy_met": error <= accuracy,
        "response_error": error,
    }
    return kernel, info


# --------------------------------------------------------------------------- #
# Specification, diagnostics and design
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReciprocalCellDiagnostics:
    """
    Quantities computed while designing a :class:`ReciprocalCellFilter`.

    Attributes
    ----------
    facets : list[np.ndarray]
        Facet vectors of the cell.

    template_facets : list[np.ndarray]
        Facets whose gates make the template (see :func:`template_gates`).

    filter_required : bool
        Whether a filter is needed.

    half_width : float
        Half-width ``t`` of the gates, set by the aliasing tolerance.

    kernel_size : int or None
        Size of the kernel, None when no filter is needed.

    size_bound : int or None
        Bound on the size actually used: ``max_kernel_size``, or the bound
        derived from the size estimate when it is None.

    accurate_size : int or None
        Smallest size meeting the accuracy, None if forced or out of range.

    accuracy_met : bool or None
        Whether the kernel meets the accuracy.

    response_error : float or None
        Measured sup-norm error of the kernel response.

    template_tolerances : list[float] or None
        Value of each gate of ``template_facets`` on its own boundary
        ``|f . a| = 1/2``, None when no filter is needed. It is the aliasing
        tolerance by default; with ``experimental_progressive_gates``, a facet
        inside the ramp gets a larger value, between the tolerance and 1 (see
        :func:`facet_tolerance`).

    border_jump : float or None
        Largest jump of the template between two opposite edges of the
        source Nyquist square, None when no filter is needed. A jump above
        about the accuracy leaves an error floor that no kernel size
        removes; the experimental Nyquist gates reduce it.
    """

    facets: list[np.ndarray]
    template_facets: list[np.ndarray]
    filter_required: bool
    half_width: float
    kernel_size: int | None = None
    size_bound: int | None = None
    accurate_size: int | None = None
    accuracy_met: bool | None = None
    response_error: float | None = None
    template_tolerances: list[float] | None = None
    border_jump: float | None = None


@dataclass(frozen=True)
class ReciprocalCellFilter:
    """
    Specification of the reciprocal cell antialiasing filter.

    The filter keeps the frequencies of a cell of the reciprocal lattice of
    the destination geometry. :meth:`design` builds it from the transition
    matrix of the grid.

    The class implements
    :class:`~gridr.core.filter.antialiasing.AntialiasingFilter`.

    Attributes
    ----------
    cutoff_tanh_slope : float
        Steepness of the tanh transition across each facet of the cell.

    aliasing_tolerance : float
        Gate value on the cell boundary, in ``(0, 0.5]``: the largest
        amplitude that a frequency just beyond the boundary keeps before it
        folds. ``0.5`` centers the transition on the boundary (relaxed
        specification, sharper result); smaller values move it inside the
        cell (strict specification, less aliasing, softer result).
        The default ``0.1`` strongly reduces the aliased amplitude compared
        with ``0.5`` for a small loss of gain in the upper passband.

    isotropic_frame : {"destination", "source"} or np.ndarray
        Metric of the Voronoi cell: the destination pixel frame, the source
        pixel frame, or an explicit 2x2 symmetric positive definite matrix
        acting on source frequencies.

    kernel_accuracy : float
        Target sup-norm error between the response of the kernel and the
        template, used to size the kernel.

    max_kernel_size : int or None
        Odd upper bound on the kernel size. None, the default, sets it to twice
        the size estimate, so that it is mostly reached when the template jumps
        on the border of the source Nyquist square. An explicit bound caps the
        margins of the tiles. When the accuracy needs a larger kernel, the
        accurate kernel is truncated to the bound, the diagnostics report the
        error reached and ``design`` issues a ``RuntimeWarning``.

    kernel_size : int or None
        Forced odd kernel size, used without accuracy guarantee. None selects
        the smallest size meeting ``kernel_accuracy``.

    experimental_nyquist_gates : bool
        Experimental, off by default. See :class:`TemplateOptions`.

    experimental_progressive_gates : bool
        Experimental, off by default. See :class:`TemplateOptions`.
    """

    cutoff_tanh_slope: float = 10.0
    aliasing_tolerance: float = 0.1
    isotropic_frame: str | np.ndarray = "destination"
    kernel_accuracy: float = 0.01
    max_kernel_size: int | None = None
    kernel_size: int | None = None
    experimental_nyquist_gates: bool = False
    experimental_progressive_gates: bool = False

    def __post_init__(self) -> None:
        """
        Validate the specification.

        Raises
        ------
        ValueError
            If a parameter is out of its domain, or if the slope is too small
            for the tolerance: the half-width of the gates would fall to 1/4 or
            below, leaving less than half of the cell in the passband.
        """
        if not self.cutoff_tanh_slope > 0:
            raise ValueError("cutoff_tanh_slope must be positive")
        if not 0 < self.aliasing_tolerance <= 0.5:
            raise ValueError("aliasing_tolerance must lie in (0, 0.5]")
        if not 0 < self.kernel_accuracy < 1:
            raise ValueError("kernel_accuracy must lie in (0, 1)")
        _check_odd("max_kernel_size", self.max_kernel_size)
        _check_odd("kernel_size", self.kernel_size)
        # validates isotropic_frame
        metric_matrix(np.eye(2), self.isotropic_frame)
        if cutoff_half_width(self.cutoff_tanh_slope, self.aliasing_tolerance) <= 0.25:
            raise ValueError(
                "cutoff_tanh_slope too small for aliasing_tolerance: the passband "
                "would shrink below half of the cell"
            )

    def _accuracy_message(self, info: dict, jump: float, size: int) -> str:
        """
        Explain why the kernel misses the accuracy, and what to do.

        Parameters
        ----------
        info : dict
            Information returned by :func:`synthesize_kernel`.

        jump : float
            Largest jump of the template on the border of the Nyquist square.

        size : int
            Size of the kernel.

        Returns
        -------
        str
            Message of the warning.
        """
        msg = (
            f"antialiasing kernel of size {size} misses the "
            f"accuracy {self.kernel_accuracy:g} (response error "
            f"{info['response_error']:.3g})"
        )
        if info["accurate_size"] is not None:
            msg += f"; size {info['accurate_size']} would meet it"
        if jump > self.kernel_accuracy and not self.experimental_nyquist_gates:
            return msg + (
                f": the template jumps by {jump:.2g} on the border of the source "
                "Nyquist square; experimental_nyquist_gates=True reduces the jump"
            )
        if self.max_kernel_size is not None:
            return msg + ": raise max_kernel_size or lower cutoff_tanh_slope"
        return msg + ": set max_kernel_size or lower cutoff_tanh_slope"

    def design(
        self, transition_matrix: np.ndarray
    ) -> tuple[KernelFilter | None, ReciprocalCellDiagnostics]:
        """
        Build the filter for a transition matrix.

        Parameters
        ----------
        transition_matrix : np.ndarray
            Transition matrix ``W = [w1 w2]`` of the grid.

        Returns
        -------
        tuple[KernelFilter or None, ReciprocalCellDiagnostics]
            The filter, None when the source Nyquist square lies inside the
            cell, and the diagnostics.

        Warns
        -----
        RuntimeWarning
            If the kernel misses ``kernel_accuracy``. The message gives the
            accurate size when the search found it, and the likely cause: the
            jump of the template on the border of the Nyquist square, with a
            pointer to ``experimental_nyquist_gates`` when it is off, or else
            ``max_kernel_size`` and ``cutoff_tanh_slope``. No warning is issued
            for a size forced with ``kernel_size``.
        """
        slope = self.cutoff_tanh_slope
        facets = cell_facets(transition_matrix, self.isotropic_frame)
        t = cutoff_half_width(slope, self.aliasing_tolerance)
        if not filter_required(facets):
            return None, ReciprocalCellDiagnostics(facets, list(facets), False, t)
        options = TemplateOptions(
            nyquist_gates=self.experimental_nyquist_gates,
            progressive_gates=self.experimental_progressive_gates,
        )
        kernel, info = synthesize_kernel(
            facets,
            slope,
            self.aliasing_tolerance,
            self.kernel_accuracy,
            self.max_kernel_size,
            self.kernel_size,
            options,
        )
        gates, ts = info["gates"], info["ts"]
        jump = max(border_mismatch(gates, axis, slope, ts) for axis in (0, 1))
        # a size forced by the caller is a choice: the diagnostics report it
        if not info["accuracy_met"] and self.kernel_size is None:
            warnings.warn(
                self._accuracy_message(info, jump, kernel.shape[0]),
                RuntimeWarning,
                stacklevel=2,
            )
        # the gate value on the boundary, from the half-width of each gate
        tolerances = [float(tanh_gate(np.array(0.5), slope, ti)) for ti in ts]
        return KernelFilter(kernel), ReciprocalCellDiagnostics(
            facets,
            gates,
            True,
            t,
            kernel_size=kernel.shape[0],
            size_bound=info["size_bound"],
            accurate_size=info["accurate_size"],
            accuracy_met=bool(info["accuracy_met"]),
            response_error=info["response_error"],
            template_tolerances=tolerances,
            border_jump=jump,
        )
