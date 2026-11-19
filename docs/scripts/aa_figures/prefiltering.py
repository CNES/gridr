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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Ellipse, Polygon, Rectangle

from gridr.core.filter.antialiasing import reciprocal_cell as rc
from gridr.core.filter.lattice import (
    gauss_reduce,
    relevant_vectors,
    voronoi_cell_vertices,
)

from .aliasing import draw_tiling
from .common import (
    C_ALT,
    C_BAD,
    C_DST,
    C_GREY,
    C_LAT,
    C_SRC,
    CELL_FRAMES,
    N_OUT,
    SLOPE,
    T_DEFAULT,
    TOL,
    W_EX,
    centered_grid,
    circular_convolve,
    exact_resample,
    filter_by_cell,
    filter_by_template,
    inside,
    lattice_points,
    resample,
    resolve,
    rot,
    w_example,
    zone_plate,
)
from .plot import (
    CB_GAP,
    CB_ROOM,
    CB_SPACE,
    CB_W,
    HALO,
    PX_GAP,
    PX_LEFT,
    PX_OUT,
    PX_TICKS,
    PX_TITLE,
    RESULTS,
    cell_patch,
    colour_bar,
    freq_ticks,
    nyquist_square,
    pixel_axes,
    pixel_figure,
    save,
    show_output,
    show_pixels,
    show_source,
    show_spectrum,
    step,
)

SCALES = (1.5, 2.0, 2.5, 3.0, 4.0)
SLOPES = (5.0, 10.0, 20.0)


# Size estimate of the page, for the largest facet norm of the cell.
def estimated_size(slope, max_norm, accuracy=0.01):
    return 4 / np.pi**2 * np.log(1 / accuracy) * slope * max_norm


@step
def fig_tilings():
    W = W_EX
    B = rc.reciprocal_basis(W)
    # parallelogram of the basis (b1 + b2, b2)
    tmp = np.linalg.inv(B @ np.array([[1.0, 1.0], [0.0, 1.0]]))
    kinds = [
        (rc.cell_facets(W, "destination"), "Voronoi, destination metric", C_DST),
        (rc.cell_facets(W, "source"), "Voronoi, source metric", C_SRC),
        ([tmp[0], tmp[1]], "parallelogram of another basis", C_ALT),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5))
    L, T, Bt = PX_LEFT, PX_TITLE, PX_TICKS
    figp = pixel_figure(L + 512 + 8, T + 512 + Bt)
    axp = pixel_axes(figp, L, T, 512, 512)
    show_source(
        axp,
        zone_plate(),
        "the three domains on the zone plate: (a) green, (b) orange, (c) purple",
        annotate=False,
    )
    rp = lattice_points(B, 6)
    f = np.linspace(-0.5, 0.5, 801)
    fx, fy = np.meshgrid(f, f)
    # mean spectrum of natural scenes, finite at the origin
    S = 1.0 / (fx**2 + fy**2 + 1e-4)
    res = {}
    for i, (facets, title, col) in enumerate(kinds):
        P = voronoi_cell_vertices(facets)
        ax = axs[i]
        draw_tiling(ax, P, B, ["#e6e6e6", "#f4f4f4", "#d6d6d6"], 6, 0.5)
        ax.add_patch(Polygon(P, closed=True, fc=col, alpha=0.35, ec=col, lw=1.2))
        ax.scatter(rp[:, 0], rp[:, 1], s=5, c=C_LAT, zorder=3)
        ax.add_patch(nyquist_square(ec=C_GREY, lw=0.7))
        # the facet a is the line nu . a = 1/2, at 1 / (2 |a|) from the origin
        r_in = min(1 / (2 * np.linalg.norm(a)) for a in facets)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r_in * np.cos(th), r_in * np.sin(th), "k:", lw=0.6)
        ax.set(xlim=(-0.6, 0.6), ylim=(0.6, -0.6), aspect="equal")
        ax.set_title(f"({'abc'[i]}) {title}", fontsize=7.5)
        freq_ticks(ax)
        axp.add_patch(
            Polygon(P, closed=True, fill=False, ec=col, lw=1.6, path_effects=HALO)
        )
        area = 0.5 * abs(
            np.dot(P[:, 0], np.roll(P[:, 1], 1)) - np.dot(P[:, 1], np.roll(P[:, 0], 1))
        )
        res[title] = {
            "area": float(area),
            "inscribed_radius": float(r_in),
            "farthest_vertex": float(np.linalg.norm(P, axis=1).max()),
            "loss_1_f2_pct": float(
                100 * (1 - (S * inside(fx, fy, facets)).sum() / S.sum())
            ),
        }
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_tilings")
    save(figp, "fig_tilings_source")
    RESULTS["tilings"] = res


@step
def fig_metric():
    W = W_EX
    rp = lattice_points(rc.reciprocal_basis(W), 5)
    # a made-up pixel to ground Jacobian
    D = np.array([[1.0, 0.0], [0.35, 1.3]])
    frames = [
        ("destination", "destination metric\n$G = WW^{\\top}$", C_DST),
        ("source", "source metric\n$G = I$", C_SRC),
        (
            np.linalg.inv(D) @ np.linalg.inv(D).T,
            "explicit metric\n$G = D^{-1}D^{-\\top}$",
            C_ALT,
        ),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5))
    for ax, (frame, title, col) in zip(axs, frames, strict=True):
        G = rc.metric_matrix(W, frame)
        facets = rc.cell_facets(W, frame)
        ax.add_patch(cell_patch(facets, fc=col, alpha=0.3, ec=col, lw=1.2))
        # largest ball of the metric inside the cell, G normalised to compare
        Gn = G / np.sqrt(np.linalg.det(G))
        w, V = np.linalg.eigh(Gn)
        r = min(1 / (2 * np.sqrt(a @ np.linalg.solve(Gn, a))) for a in facets)
        ax.add_patch(
            Ellipse(
                (0, 0),
                2 * r / np.sqrt(w[0]),
                2 * r / np.sqrt(w[1]),
                angle=np.degrees(np.arctan2(V[1, 0], V[0, 0])),
                fill=False,
                ec="k",
                ls=":",
                lw=0.8,
            )
        )
        ax.scatter(rp[:, 0], rp[:, 1], s=6, c=C_LAT, zorder=3)
        ax.add_patch(nyquist_square(ec=C_GREY, lw=0.7))
        ax.set(xlim=(-0.6, 0.6), ylim=(0.6, -0.6), aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("$\\nu_x$ (cycles/px)")
        freq_ticks(ax)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_metric")


def arrow(ax, v, lw, col):
    ax.annotate(
        "", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=col)
    )


@step
def fig_reduction():
    B = rc.reciprocal_basis(W_EX)
    # a unimodular change of basis that makes the basis long and oblique
    Bu = B @ (np.array([[1, 1], [0, 1]]) @ np.array([[1, 0], [1, 1]]))
    u1, u2 = gauss_reduce(Bu[:, 0], Bu[:, 1])
    if u1 @ u2 > 0:
        u2 = -u2
    pts = lattice_points(B, 6)
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.5))

    ax = axs[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c=C_LAT, zorder=3)
    for v, label, col in (
        (Bu[:, 0], r"$\mathbf{b}_1'$", C_GREY),
        (Bu[:, 1], r"$\mathbf{b}_2'$", C_GREY),
        (u1, r"$\mathbf{e}_1$", C_BAD),
        (u2, r"$\mathbf{e}_2$", C_BAD),
    ):
        arrow(ax, v, 1.0, col)
        e = v / np.linalg.norm(v)
        ax.text(
            *(v + 0.16 * e + 0.07 * np.array([-e[1], e[0]])),
            label,
            color=col,
            fontsize=7,
            ha="center",
            va="center",
        )
    lim = 1.15 * max(np.abs(Bu).max(), 0.6)
    ax.set(xlim=(-lim, lim), ylim=(lim, -lim), aspect="equal")
    ax.set_title("(a) any basis (grey) and\nreduced obtuse basis (red)")

    ax = axs[1]
    vecs = [u1, u2, u1 + u2]
    ax.add_patch(
        cell_patch([v / (v @ v) for v in vecs], fc=C_SRC, alpha=0.25, ec=C_SRC, lw=1.2)
    )
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c=C_LAT, zorder=3)
    labels = [r"$\mathbf{e}_1$", r"$\mathbf{e}_2$", r"$\mathbf{e}_1{+}\mathbf{e}_2$"]
    for v, label in zip(
        [*vecs, *(-v for v in vecs)], [*labels, "", "", ""], strict=True
    ):
        arrow(ax, v, 0.8, C_BAD)
        if label:
            ax.text(*(v * 1.15), label, color=C_BAD, fontsize=7, ha="center")
    ax.set(xlim=(-0.65, 0.65), ylim=(0.65, -0.65), aspect="equal")
    ax.set_title("(b) relevant vectors and\nVoronoi cell (perpendicular bisectors)")

    # unit e1 and e2 at angles from 60 (regular hexagon) to 90 degrees
    ax = axs[2]
    angles = np.linspace(60, 90, 200)
    edge = []
    for a in angles:
        v1 = np.array([1.0, 0.0])
        v2 = np.array([np.cos(np.radians(180 - a)), np.sin(np.radians(180 - a))])
        P = voronoi_cell_vertices([v / (v @ v) for v in relevant_vectors(v1, v2)])
        e = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        edge.append(e.min() if len(P) == 6 else 0.0)
    ax.plot(angles, edge, color=C_SRC, lw=1.2)
    ax.set_xlabel(r"angle between $\mathbf{e}_1$ and $\mathbf{e}_2$ (deg)")
    ax.set_ylabel("shortest hexagon edge")
    ax.set_title("(c) the hexagon degenerates\ncontinuously into a rectangle")
    ax.set_xlim(60, 90)
    ax.set_ylim(0, None)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_reduction")


@step
def fig_decision():
    cases = [
        ("zoom x2: no filter", 0.5 * np.eye(2)),
        ("decimation x2: filter", 2.0 * np.eye(2)),
        ("30° rotation, unit scale: filter", rot(30)),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35))
    for ax, (title, W) in zip(axs, cases, strict=True):
        facets = rc.cell_facets(W)
        ax.add_patch(cell_patch(facets, fc=C_DST, alpha=0.25, ec=C_DST, lw=1.1))
        ax.add_patch(nyquist_square(ec="k", lw=0.9))
        for c in ((0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)):
            out = np.any(np.abs(np.array(c) @ np.array(facets).T) > 0.5 + 1e-9)
            ax.plot(*c, "o", ms=4, color=C_BAD if out else C_DST)
        ax.set(xlim=(-1.1, 1.1), ylim=(1.1, -1.1), aspect="equal")
        ax.set_title(title, fontsize=7.5)
        freq_ticks(ax, 1.0)
        ax.set_xlabel("$\\nu_x$ (cycles/px)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_decision")


@step
def fig_gate_shape():
    gam = np.linspace(-1, 1, 1001)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    panels = (
        (
            axs[0],
            f"(a) slope {SLOPE:g}, three half-widths",
            [(SLOPE, t, f"$t$ = {t:g}") for t in (0.3, 0.4, 0.5)],
        ),
        (
            axs[1],
            "(b) $t$ = 0.4, four slopes",
            [(sl, 0.4, f"slope {sl:g}") for sl in (3, 5, 10, 20)],
        ),
    )
    for ax, title, cases in panels:
        ax.plot(gam, (np.abs(gam) <= 0.5).astype(float), color="k", lw=0.6)
        for (sl, t, label), col in zip(
            cases, (C_BAD, C_SRC, C_DST, C_LAT), strict=False
        ):
            ax.plot(gam, rc.tanh_gate(gam, sl, t), color=col, lw=1.1, label=label)
        for x in (-0.5, 0.5):
            ax.axvline(x, color="k", ls=":", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel(r"$\gamma = \boldsymbol{\nu}\cdot\mathbf{a}$")
        ax.legend(fontsize=6, frameon=False, loc="center right")
    axs[0].set_ylabel(r"$\varphi(\gamma)$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_gate_shape")


@step
def fig_gate():
    tols = ((0.5, C_BAD), (0.1, C_SRC), (0.01, C_DST))
    gam = np.linspace(0, 1, 600)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axs[0]
    ax.plot(gam, (np.abs(gam) <= 0.5).astype(float), color="k", lw=0.7, label="ideal")
    rows = []
    for tol, col in tols:
        t = rc.cutoff_half_width(SLOPE, tol)
        ax.plot(
            gam,
            rc.tanh_gate(gam, SLOPE, t),
            color=col,
            lw=1.1,
            label=f"$\\varepsilon$ = {tol:g} ($t$ = {t:.3f})",
        )
        rows.append({"tolerance": tol, "t": t})
    ax.axvline(0.5, color="k", ls=":", lw=0.7)
    ax.set_xlabel(
        r"normalised facet coordinate $\gamma = \boldsymbol{\nu}\cdot\mathbf{a}$"
    )
    ax.set_ylabel(r"$\varphi(\gamma)$")
    ax.set_title(f"(a) gate, slope {SLOPE:g}")
    ax.legend(fontsize=6, frameon=False)

    ax = axs[1]
    delta = np.linspace(0, 0.5, 400)
    for tol, col in tols:
        t = rc.cutoff_half_width(SLOPE, tol)
        ax.semilogy(
            0.5 - delta,
            np.maximum(rc.tanh_gate(0.5 + delta, SLOPE, t), 1e-8),
            color=col,
            lw=1.1,
            label=f"$\\varepsilon$ = {tol:g}",
        )
    ax.set_xlabel(r"facet coordinate where the alias lands, $|\gamma| = 1/2 - \delta$")
    ax.set_ylabel("aliased amplitude")
    ax.set_ylim(1e-6, 1)
    ax.set_title(r"(b) a component at $1/2 + \delta$ folds to $1/2 - \delta$")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_gate")
    RESULTS["gate"] = rows

    # the zone plate filtered by the two templates, central part only
    half = 176
    L, T, Bt, G = PX_LEFT, PX_TITLE, PX_TICKS, PX_GAP
    P = 2 * half
    fig = pixel_figure(L + P + G + P + 8, T + P + Bt)
    mire = zone_plate()
    for i, (frame, title, col) in enumerate(CELL_FRAMES):
        facets = rc.cell_facets(W_EX, frame)
        ax = pixel_axes(fig, L + i * (P + G), T, P, P)
        show_source(
            ax,
            filter_by_template(mire, facets),
            f"({'ab'[i]}) {title}, $\\varepsilon$ = {TOL:g}",
            annotate=False,
            half=half,
        )
        if i:
            ax.set_yticklabels([])
        ax.add_patch(
            cell_patch(facets, fill=False, ec=col, lw=1.4, ls="--", path_effects=HALO)
        )
    save(fig, "fig_template_plate")


@step
def fig_synthesis_steps():
    facets = rc.cell_facets(W_EX)
    t = T_DEFAULT
    # the fine grid of GridR for this template
    _, info = rc.synthesize_kernel(facets, SLOPE, TOL, 0.01)
    nf = info["n_fine"]
    f = np.fft.fftshift(np.fft.fftfreq(nf))
    fx, fy = np.meshgrid(f, f)
    H = rc.template(facets, fx, fy, SLOPE, t)
    fine = rc.fine_kernel(facets, SLOPE, t, nf)
    kf, _ = resolve(W_EX)
    k = kf.kernel
    n = k.shape[0]
    h = n // 2
    # true response on the frequencies of the template
    tmp = np.zeros((nf, nf))
    tmp[:n, :n] = k
    tmp = np.roll(tmp, (-h, -h), axis=(0, 1))
    err = np.abs(np.fft.fftshift(np.real(np.fft.fft2(tmp))) - H)
    # the value quoted comes from the finer grid GridR uses
    fr, Hr = rc.kernel_response(k)
    gx, gy = np.meshgrid(fr, fr)
    err_max = float(np.abs(Hr - rc.template(facets, gx, gy, SLOPE, t)).max())

    P = nf
    zoom = 4
    L, T, Bt = PX_LEFT, PX_TITLE, PX_TICKS
    col = L + P + CB_GAP + CB_W + CB_ROOM
    y2 = T + P + Bt + T
    fig = pixel_figure(2 * col, 2 * (T + P + Bt))
    lo, hi = f[0] - 0.5 / nf, f[-1] + 0.5 / nf

    ax = pixel_axes(fig, L, T, P, P)
    show_pixels(ax, H, extent=(lo, hi, hi, lo), cmap="viridis", vmin=0, vmax=1)
    freq_ticks(ax)
    ax.set(xlim=(lo, hi), ylim=(hi, lo))
    ax.set_title(f"1. template $H$ sampled\non {nf} x {nf} frequencies", fontsize=7.5)
    colour_bar(fig, ax, L + P, T, P)

    # the whole fine kernel, one sample per pixel
    e = nf // 2 + 0.5
    ax = pixel_axes(fig, col + L, T, P, P)
    show_pixels(
        ax,
        np.log10(np.abs(fine) + 1e-12),
        extent=(-e, e, e, -e),
        cmap="magma",
        vmin=-8,
        vmax=-1,
    )
    ax.add_patch(
        Rectangle((-h - 0.5, -h - 0.5), n, n, fill=False, ec="w", ls="--", lw=0.8)
    )
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set(xlim=(-e, e), ylim=(e, -e))
    ax.set_title(
        f"2. inverse DFT: fine kernel {nf} x {nf}\n$\\log_{{10}}|h|$, crop {n} x {n}",
        fontsize=7.5,
    )
    colour_bar(fig, ax, col + L + P, T, P)

    # symmetric log scale: in linear the lobes vanish next to the center
    vmax = float(np.abs(k).max())
    off = (P - zoom * n) // 2
    e = h + 0.5
    ax = pixel_axes(fig, L + off, y2 + off, zoom * n, zoom * n)
    show_pixels(
        ax,
        np.kron(k, np.ones((zoom, zoom))),
        extent=(-e, e, e, -e),
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1e-4, linscale=0.5, vmin=-vmax, vmax=vmax),
    )
    ax.set_xticks([-30, -15, 0, 15, 30])
    ax.set_yticks([-30, -15, 0, 15, 30])
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_title(
        f"3. central crop $h$, {n} x {n}\nunit sum, symmetric log scale", fontsize=7.5
    )
    colour_bar(fig, ax, L + P, y2 + off, zoom * n, ticks=[-1e-2, -1e-3, 0, 1e-3, 1e-2])

    ax = pixel_axes(fig, col + L, y2, P, P)
    show_pixels(ax, err, extent=(lo, hi, hi, lo), cmap="viridis", vmin=0, vmax=0.01)
    ax.add_patch(cell_patch(facets, fill=False, ec="w", lw=0.6))
    freq_ticks(ax)
    ax.set(xlim=(lo, hi), ylim=(hi, lo))
    ax.set_title(f"4. |response - template|\nmax {err_max:.4f} < 0.01", fontsize=7.5)
    colour_bar(fig, ax, col + L + P, y2, P)
    save(fig, "fig_synthesis_steps")
    RESULTS["synthesis_steps"] = {"n": int(n), "max_error": err_max}


@step
def fig_synthesis():
    t = T_DEFAULT
    sizes = np.arange(9, 82, 2)
    # 523 is odd and well above eight times the largest size tried
    curves = []
    for frame in ("destination", "source"):
        facets = rc.cell_facets(W_EX, frame)
        fine = rc.fine_kernel(facets, SLOPE, t, 523)
        curves.append(
            [
                rc.response_error(rc.crop_kernel(fine, int(n)), facets, SLOPE, t)
                for n in sizes
            ]
        )
    facets = rc.cell_facets(W_EX)
    direct = []
    for n in sizes:
        k = rc.fine_kernel(facets, SLOPE, t, int(n))
        direct.append(rc.response_error(k / k.sum(), facets, SLOPE, t))
    kf, diag = resolve(W_EX)
    kf_src, diag_src = resolve(W_EX, isotropic_frame="source")
    n_dst, n_src = kf.kernel.shape[0], kf_src.kernel.shape[0]

    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    ax.semilogy(
        sizes,
        curves[0],
        "o-",
        ms=2.5,
        color=C_DST,
        lw=1.0,
        label="crop of the fine kernel",
    )
    ax.semilogy(
        sizes,
        direct,
        "s-",
        ms=2.5,
        color=C_GREY,
        lw=1.0,
        label=r"inverse DFT of $n \times n$ samples",
    )
    ax.semilogy(
        sizes,
        curves[1],
        "^-",
        ms=2.5,
        color=C_SRC,
        lw=1.0,
        label="crop, hexagon (source metric)",
    )
    ax.axhline(0.01, color="k", ls=":", lw=0.7)
    ax.axvline(n_dst, color=C_DST, ls="--", lw=0.7)
    ax.axvline(n_src, color=C_SRC, ls="--", lw=0.7)
    ax.set_xlabel("kernel size $n$")
    ax.set_ylabel("max |response - template|")
    ax.set_title("response error, running example")
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_synthesis")

    # for the parallelogram the facet vectors are w1 and w2
    norm_src = max(np.linalg.norm(a) for a in rc.cell_facets(W_EX, "source"))
    RESULTS["synthesis"] = {
        "selected_size": n_dst,
        "response_error": diag.response_error,
        "source_metric": {
            "selected_size": n_src,
            "response_error": diag_src.response_error,
            "max_facet_norm": float(norm_src),
            "estimated_size": float(estimated_size(SLOPE, norm_src)),
        },
        "estimated_sizes": {
            f"slope_{slope:g}": {
                f"{s:g}": float(
                    estimated_size(slope, np.linalg.norm(w_example(s), axis=0).max())
                )
                for s in SCALES
            }
            for slope in SLOPES
        },
    }


@step
def fig_moire():
    mire = zone_plate()
    grid = centered_grid(W_EX, N_OUT)
    facets = rc.cell_facets(W_EX)
    facets_src = rc.cell_facets(W_EX, "source")
    # (key, title, kernel tolerance, metric, facets); the first one is not drawn
    cases = [
        ("none", None, None, None, facets),
        ("0.5", "parallelogram, $\\varepsilon$ = 0.5", 0.5, "destination", facets),
        ("0.1", "parallelogram, $\\varepsilon$ = 0.1", 0.1, "destination", facets),
        ("source 0.1", f"hexagon, $\\varepsilon$ = {TOL:g}", TOL, "source", facets_src),
        ("0.01", None, 0.01, "destination", facets),
    ]
    L, T, B, G, P = PX_LEFT, PX_TITLE, PX_TICKS, PX_GAP, PX_OUT
    fig = pixel_figure(8 + 3 * P + 2 * G + 8, 2 * (T + P) + 8)
    figs = pixel_figure(L + 512 + CB_SPACE, 3 * (T + 256 + B))
    res = {}
    i = 0
    for key, title, tol, frame, fa in cases:
        k = None
        if tol is not None:
            kwargs = (
                {"isotropic_frame": "source"}
                if frame == "source"
                else {"aliasing_tolerance": tol}
            )
            k = resolve(W_EX, **kwargs)[0].kernel
        src = mire if k is None else circular_convolve(mire, k)
        ref = exact_resample(src, grid)
        alias = ref - exact_resample(filter_by_cell(src, fa), grid)
        res[key] = {
            "rms_alias": float(alias.std()),
            "max_alias": float(np.abs(alias).max()),
        }
        if k is not None:
            res[key]["kernel_size"] = int(k.shape[0])
        if key == "0.01":
            continue
        out = resample(src, grid)
        interp = out - ref
        res[key].update(
            {
                "rms_signal": float((ref - 0.5).std()),
                "rms_interp": float(interp.std()),
                "max_interp": float(np.abs(interp).max()),
            }
        )
        if title is None:
            continue
        x = 8 + i * (P + G)
        show_output(pixel_axes(fig, x, T, P, P), out, f"({'abc'[i]}) {title}")
        show_output(
            pixel_axes(fig, x, 2 * T + P, P, P),
            alias,
            f"({'def'[i]}) aliased part x10, rms {alias.std():.4f}",
            gain=10.0,
        )
        show_spectrum(
            figs,
            L,
            T + i * (T + 256 + B),
            src,
            fa,
            f"({'abc'[i]}) {title}: spectrum of the prefiltered source"
            " (upper half, dB)",
            upper_half=True,
        )
        i += 1
    save(fig, "fig_moire")
    save(figs, "fig_moire_spectra")
    RESULTS["moire"] = res


@step
def size_table():
    data = {}
    for slope in SLOPES:
        data[f"slope_{slope:g}"] = {
            f"{s:g}": rc.synthesize_kernel(
                rc.cell_facets(w_example(s)), slope, TOL, 0.01, max_size=301
            )[1]["accurate_size"]
            for s in SCALES
        }
    data["slope_10_source"] = {
        f"{s:g}": rc.synthesize_kernel(
            rc.cell_facets(w_example(s), "source"), SLOPE, TOL, 0.01, max_size=301
        )[1]["accurate_size"]
        for s in SCALES
    }
    RESULTS["size_table"] = {"running_example": data}
