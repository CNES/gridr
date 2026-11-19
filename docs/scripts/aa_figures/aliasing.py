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
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from gridr.core.filter.antialiasing import reciprocal_cell as rc
from gridr.core.filter.lattice import voronoi_cell_vertices

from .common import (
    C_BAD,
    C_DST,
    C_GREY,
    C_LAT,
    C_SRC,
    N_OUT,
    W_EX,
    K,
    centered_grid,
    exact_resample,
    filter_by_cell,
    lattice_points,
    resample,
    zone_plate,
)
from .plot import (
    CB_SPACE,
    HALO,
    PX_GAP,
    PX_LEFT,
    PX_OUT,
    PX_TICKS,
    PX_TITLE,
    RESULTS,
    cell_patch,
    nyquist_square,
    pixel_axes,
    pixel_figure,
    save,
    show_output,
    show_source,
    show_spectrum,
    step,
)


# Zone plate resampled on the running example, its reference without aliasing
# and its aliased part.
@cache
def problem_case():
    mire = zone_plate()
    grid = centered_grid(W_EX, N_OUT)
    facets = rc.cell_facets(W_EX)
    ideal = exact_resample(filter_by_cell(mire, facets), grid)
    return {
        "mire": mire,
        "facets": facets,
        "out": resample(mire, grid),
        "ideal": ideal,
        "alias": exact_resample(mire, grid) - ideal,
    }


@step
def fig_problem():
    data = problem_case()
    err = data["out"] - data["ideal"]
    RESULTS["problem"] = {
        "rms_signal": float((data["ideal"] - 0.5).std()),
        "rms_error": float(err.std()),
        "max_error": float(np.abs(err).max()),
        "rms_alias": float(data["alias"].std()),
        "rms_interp": float((err - data["alias"]).std()),
    }
    L, T, B, G, P = PX_LEFT, PX_TITLE, PX_TICKS, PX_GAP, PX_OUT

    fig = pixel_figure(L + 512 + 8, T + 512 + B)
    show_source(
        pixel_axes(fig, L, T, 512, 512), data["mire"], "The source Fresnel Zone Plate"
    )
    save(fig, "fig_zp_source")

    fig = pixel_figure(L + 512 + CB_SPACE, T + 512 + B)
    show_spectrum(
        fig, L, T, data["mire"], None, "Spectrum of the Zone Plate (dB), (cycles/px)"
    )
    save(fig, "fig_zp_spectrum")

    fig = pixel_figure(8 + 3 * P + 2 * G + 8, T + P + 8)
    show_output(
        pixel_axes(fig, 8, T, P, P),
        data["out"],
        "(a) resampled on the grid\nof the running example",
    )
    show_output(
        pixel_axes(fig, 8 + P + G, T, P, P),
        data["ideal"],
        "(b) reference: expected\nwithout aliasing",
    )
    show_output(
        pixel_axes(fig, 8 + 2 * (P + G), T, P, P),
        err,
        f"(c) error (a) - (b) x3,\nrms {err.std():.3f}",
        gain=3.0,
    )
    save(fig, "fig_problem")


@step
def fig_lattice():
    W = W_EX
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.8))
    ax = axs[0]
    for i in range(-1, 5):
        ax.axhline(i - 0.5, color="#e2e2e2", lw=0.5, zorder=0)
        ax.axvline(i - 0.5, color="#e2e2e2", lw=0.5, zorder=0)
    jj, ii = np.meshgrid(np.arange(4), np.arange(4))
    ax.scatter(jj, ii, s=12, c=C_DST, zorder=3)
    ax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0))
    ax.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.0))
    ax.text(0.5, -0.25, "$j'$", ha="center")
    ax.text(-0.3, 0.55, "$i'$", va="center")
    ax.set(
        xlim=(-0.8, 3.8),
        ylim=(3.8, -0.8),
        aspect="equal",
        xlabel="$x' = j'$",
        ylabel="$y' = i'$",
    )
    ax.set_title(r"(a) destination grid $\mathscr{G}_d$")

    ax = axs[1]
    for i in range(-2, 12):
        ax.axhline(i - 0.5, color="#e2e2e2", lw=0.5, zorder=0)
        ax.axvline(i - 0.5, color="#e2e2e2", lw=0.5, zorder=0)
    p0 = np.array([2.0, 1.0])
    pts = np.array([p0 + W @ np.array([j, i]) for i in range(4) for j in range(4)])
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c=C_DST, zorder=3)
    ax.add_patch(
        Polygon(
            [p0, p0 + W[:, 0], p0 + W[:, 0] + W[:, 1], p0 + W[:, 1]],
            closed=True,
            fc=C_DST,
            alpha=0.15,
        )
    )
    for v, label, off in (
        (W[:, 0], r"$\mathbf{w}_1$", (0.0, -0.45)),
        (W[:, 1], r"$\mathbf{w}_2$", (-0.55, 0.0)),
    ):
        ax.annotate("", xy=p0 + v, xytext=p0, arrowprops=dict(arrowstyle="-|>", lw=1.0))
        ax.text(*(p0 + 0.55 * v + np.array(off)), label, ha="center", va="center")
    ax.set(
        xlim=(-1, 11),
        ylim=(11, -1),
        aspect="equal",
        xlabel="$x$ (source)",
        ylabel="$y$ (source)",
    )
    ax.set_title(r"(b) $\Phi(\mathscr{G}_d)$ on the source grid $\mathscr{G}_s$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_lattice")


# Tile the plane with copies of a cell, three shades to tell neighbours apart.
def draw_tiling(ax, poly, B, shades, r, lw):
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            ax.add_patch(
                Polygon(
                    poly + B @ np.array([a, b]),
                    closed=True,
                    fc=shades[(a - b) % 3],
                    ec="white",
                    lw=lw,
                )
            )


@step
def fig_periodisation():
    W = W_EX
    B = rc.reciprocal_basis(W)
    rp = lattice_points(B, 4)
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.55))
    ax = axs[0]
    for p in rp:
        ax.add_patch(Circle(p, 0.33, fc=C_BAD, alpha=0.22, ec="none"))
    ax.add_patch(Circle((0, 0), 0.33, fc="none", ec=C_BAD, lw=1.0))
    ax.scatter(rp[:, 0], rp[:, 1], s=6, c=C_LAT, zorder=3)
    for v, label in ((B[:, 0], r"$\mathbf{b}_1$"), (B[:, 1], r"$\mathbf{b}_2$")):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=0.9))
        ax.text(*(1.25 * v), label, ha="center", va="center", fontsize=7)
    ax.add_patch(nyquist_square(ec=C_GREY, lw=0.8))
    ax.set(xlim=(-0.62, 0.62), ylim=(0.62, -0.62), aspect="equal")
    ax.set_title("(a) spectrum copied at every\npoint of $\\Lambda^*$: overlaps")
    ax.set_xlabel("$\\nu_x$ (cycles/px)")
    ax.set_ylabel("$\\nu_y$")

    for ax, frame, col, shades, title in (
        (
            axs[1],
            "destination",
            C_DST,
            ["#d9ead3", "#eef5ea", "#c4dfba"],
            "(b) spectrum limited to a cell:\ncopies tile the plane",
        ),
        (
            axs[2],
            "source",
            C_SRC,
            ["#f2c48d", "#f7dfbf", "#eaa75c"],
            "(c) another cell of the same\nlattice: another tiling",
        ),
    ):
        poly = voronoi_cell_vertices(rc.cell_facets(W, frame))
        draw_tiling(ax, poly, B, shades, 5, 0.6)
        ax.add_patch(Polygon(poly, closed=True, fill=False, ec=col, lw=1.3))
        ax.scatter(rp[:, 0], rp[:, 1], s=6, c=C_LAT, zorder=3)
        ax.add_patch(nyquist_square(ec=C_GREY, lw=0.8))
        ax.set(xlim=(-0.62, 0.62), ylim=(0.62, -0.62), aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("$\\nu_x$ (cycles/px)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_periodisation")


@step
def fig_aliasing_cell():
    data = problem_case()
    facets = data["facets"]
    B = rc.reciprocal_basis(W_EX)
    rp = lattice_points(B, 4)
    err = data["out"] - data["ideal"]
    n = data["out"].shape[0]
    L, T, Bt, G, P = PX_LEFT, PX_TITLE, PX_TICKS, PX_GAP, PX_OUT

    fig = pixel_figure(L + 512 + CB_SPACE, T + 512 + Bt)
    ax = show_spectrum(
        fig,
        L,
        T,
        data["mire"],
        facets,
        "spectrum of the zone plate (dB), points of $\\Lambda^*$ (blue)"
        " and a cell (white)",
    )
    ax.scatter(rp[:, 0], rp[:, 1], s=14, c="#7fc8ff", zorder=3)
    save(fig, "fig_aliasing_spectrum")

    # the cell on the zone plate, where a position codes a frequency
    fig = pixel_figure(L + 512 + 8, T + 512 + Bt)
    ax = pixel_axes(fig, L, T, 512, 512)
    show_source(
        ax,
        data["mire"],
        "zone plate, cell (red), points of $\\Lambda^*$ (blue),\n"
        "source area seen by the output (dashed)",
        annotate=False,
    )
    ax.add_patch(cell_patch(facets, fill=False, ec=C_BAD, lw=1.6, path_effects=HALO))
    # an output pixel at offset u reads the source at W u, of local frequency K W u
    a0, a1 = -n / 2 - 0.5, n / 2 - 0.5
    corners = np.array([[a0, a0], [a1, a0], [a1, a1], [a0, a1]])
    ax.add_patch(
        Polygon(
            (K * W_EX @ corners.T).T,
            closed=True,
            fill=False,
            ec=C_DST,
            lw=1.4,
            ls="--",
            path_effects=HALO,
        )
    )
    near = rp[np.abs(rp).max(1) < 0.45]
    ax.scatter(near[:, 0], near[:, 1], s=22, c=C_LAT, ec="white", lw=0.8, zorder=3)
    bb = B @ np.array([1.0, 1.0])
    box = dict(fc="white", ec="none", pad=0.6, alpha=0.85)
    for sgn in (1, -1):
        ax.text(
            sgn * bb[0] + 0.02,
            sgn * bb[1] - 0.025,
            ("" if sgn > 0 else "$-$") + "$(\\mathbf{b}_1 + \\mathbf{b}_2)$",
            color=C_LAT,
            fontsize=8,
            bbox=box,
        )
    save(fig, "fig_aliasing_source")

    # backwards: the output pixels whose local frequency is on the cell boundary
    cell = voronoi_cell_vertices(facets)
    boundary = (np.linalg.inv(K * W_EX) @ cell.T).T + n / 2
    fig = pixel_figure(8 + P + G + P + 8, T + P + 8)
    axa = pixel_axes(fig, 8, T, P, P)
    show_output(axa, data["out"], "(a) output: local frequency\ninside the cell (red)")
    axb = pixel_axes(fig, 8 + P + G, T, P, P)
    show_output(axb, err, "(b) its error (x3): the false\nrings lie outside", gain=3.0)
    for ax in (axa, axb):
        ax.add_patch(Polygon(boundary, closed=True, fill=False, ec=C_BAD, lw=1.0))
    save(fig, "fig_aliasing_cell")

    # a pixel is inside the cell when |mu|_inf <= 1/2, mu = W^T nu
    off = np.arange(n) - n / 2
    U = np.stack(np.meshgrid(off, off), -1)
    mu = np.abs(U @ (K * W_EX.T @ W_EX).T).max(-1)
    RESULTS["aliasing_cell"] = {
        "rms_error_inside": float(err[mu <= 0.5].std()),
        "rms_error_center": float(err[mu <= 0.25].std()),
        "rms_error_outside": float(err[mu > 0.5].std()),
        "fraction_outside": float((mu > 0.5).mean()),
    }
