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
from scipy.optimize import brentq

from gridr.core.filter.antialiasing import reciprocal_cell as rc
from gridr.core.filter.lattice import voronoi_cell_vertices

from .common import (
    C_BAD,
    C_DST,
    C_GREY,
    C_LAT,
    CELL_FRAMES,
    NYQ,
    PROGRESSIVE,
    SLOPE,
    T_DEFAULT,
    TOL,
    W_EX,
    inside,
    line_response,
    rot,
    w_example,
)
from .plot import RESULTS, cell_patch, freq_ticks, nyquist_square, save, step

# Nyquist gates: rotation by 30 degrees at unit scale, cut along nu_y = 0.122
# where the jump on the border is large. Kernels use the default size bound.
ANGLE = 30
FY_CUT = 0.122
SMALL = 21


# Energy kept inside the cell for a flat spectrum, energy that folds, largest
# response outside the cell.
def kernel_shares(kernel, facets):
    f, R = rc.kernel_response(kernel, n_fft=1024)
    fx, fy = np.meshgrid(f, f)
    m = inside(fx, fy, facets)
    return {
        "kept_pct": float(100 * (R[m] ** 2).mean()),
        "aliased_pct": float(100 * (R[~m] ** 2).sum() / (R[m] ** 2).sum()),
        "leak": float(np.abs(R[~m]).max()),
    }


# Cell and Nyquist square, the part of the cell outside the square in red.
def draw_cell_in_square(ax, facets, title, lim=0.75):
    square = nyquist_square(ec=C_GREY, lw=0.9, zorder=3)
    ax.add_patch(square)
    ax.add_patch(cell_patch(facets, fc="#f2b4ad", ec="none", zorder=1))
    tmp = cell_patch(facets, fc="#c4e3cd", ec="none", zorder=2)
    ax.add_patch(tmp)
    tmp.set_clip_path(square)
    ax.add_patch(cell_patch(facets, fill=False, ec=C_DST, lw=1.0, zorder=4))
    ax.set(xlim=(-lim, lim), ylim=(lim, -lim), aspect="equal")
    ax.set_xlabel("$\\nu_x$ (cycles/px)")
    ax.set_title(title, fontsize=7.5)
    freq_ticks(ax, 0.5)


# Template along the cut, on the plane and repeated with period 1, and the
# response of the kernel GridR synthesises from it.
def draw_cut(ax, facets, options, label):
    fx = np.linspace(-1.0, 1.0, 2001)
    fy = np.full_like(fx, FY_CUT)
    kernel, info = rc.synthesize_kernel(facets, SLOPE, TOL, 0.01, options=options)
    ax.axvspan(-0.5, 0.5, color="#eef3fb", zorder=0)
    ax.plot(
        fx,
        rc.template(facets, fx, fy, SLOPE, T_DEFAULT, options),
        ":",
        color=C_GREY,
        lw=1.0,
        label="template on the plane",
    )
    rep = rc.template(facets, (fx + 0.5) % 1.0 - 0.5, fy, SLOPE, T_DEFAULT, options)
    # split at the jumps, so that they are not drawn as vertical strokes
    cuts = np.flatnonzero(np.abs(np.diff(rep)) > 0.05) + 1
    for i, part in enumerate(np.split(np.arange(fx.size), cuts)):
        ax.plot(
            fx[part],
            rep[part],
            color=C_LAT,
            lw=1.4,
            label="template seen by a discrete kernel" if i == 0 else None,
        )
    n = kernel.shape[0]
    ax.plot(
        fx,
        line_response(kernel, fx, FY_CUT),
        color=C_BAD,
        lw=0.8,
        label="response of the kernel",
    )
    edge = rc.template(
        facets, np.array([-0.5, 0.5]), np.full(2, FY_CUT), SLOPE, T_DEFAULT, options
    )
    bound = "" if info["accuracy_met"] else ", size bound"
    ax.set_title(f"{label}: {n} x {n} kernel{bound}", fontsize=7.5)
    ax.set_xlabel(f"$\\nu_x$ (cycles/px), at $\\nu_y$ = {FY_CUT}")
    ax.set_ylabel("response")
    ax.set_ylim(-0.08, 1.1)
    return {
        "jump": float(abs(edge[1] - edge[0])),
        "size": n,
        "accurate_size": info["accurate_size"],
        "accuracy_met": bool(info["accuracy_met"]),
        "response_error": float(info["response_error"]),
    }


@step
def fig_periodic():
    facets = rc.cell_facets(rot(ANGLE))
    res = {"fy": FY_CUT, "tolerance": TOL}
    fig, axs = plt.subplots(
        1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1, 1.75]}
    )
    draw_cell_in_square(
        axs[0], facets, f"(a) cell of a {ANGLE}° rotation at unit scale"
    )
    axs[0].set_ylabel("$\\nu_y$ (cycles/px)")
    axs[0].plot([-0.75, 0.75], [FY_CUT, FY_CUT], color=C_LAT, lw=0.8, ls="-.")
    axs[0].annotate(
        "cut", (0.52, FY_CUT), (0.52, FY_CUT - 0.04), fontsize=6, color=C_LAT
    )
    res["cell"] = draw_cut(axs[1], facets, rc.TemplateOptions(), "(b) default template")
    axs[1].legend(
        fontsize=6.2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
    )
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_periodic")

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    res["template"] = draw_cut(ax, facets, NYQ, "with the Nyquist gates")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles, labels, fontsize=6.5, frameon=False, loc="lower center", ncol=2)
    save(fig, "fig_periodic_nyquist")
    RESULTS["periodic"] = res


@step
def fig_cells_nyquist():
    # a decimation where the hexagon fits in the Nyquist square and the
    # parallelogram does not
    s = 1.25
    W = w_example(s)
    fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.75))
    res = {"scale": s}
    for ax, (frame, name, _), letter in zip(axs, CELL_FRAMES, "ab", strict=True):
        facets = rc.cell_facets(W, frame)
        draw_cell_in_square(ax, facets, f"({letter}) {name}, {frame} metric", 0.65)
        res[name] = {
            "max_abs_vertex": float(
                np.abs(np.array(voronoi_cell_vertices(facets))).max()
            ),
            "border_jump": float(
                max(
                    rc.border_mismatch(facets, axis, SLOPE, T_DEFAULT)
                    for axis in (0, 1)
                )
            ),
        }
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_cells_nyquist")
    RESULTS["cells_nyquist"] = res


@step
def fig_nyquist():
    facets = rc.cell_facets(rot(ANGLE))
    t = T_DEFAULT
    nf = 513
    f = np.fft.fftfreq(nf)
    fx, fy = np.meshgrid(f, f)
    fs = np.fft.fftshift(f)
    fig, axs = plt.subplots(2, 2, figsize=(4.8, 4.6))
    res = {}
    for i, (name, options) in enumerate(
        (("cell gates only", rc.TemplateOptions()), ("cell and Nyquist gates", NYQ))
    ):
        H = rc.template(facets, fx, fy, SLOPE, t, options)
        axs[0, i].imshow(
            np.fft.fftshift(H),
            extent=(fs[0], fs[-1], fs[-1], fs[0]),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axs[0, i].set_title(f"({'ab'[i]}) template, {name}")
        freq_ticks(axs[0, i])
        # central crop of the fine kernel, normalised, as the synthesis does
        fine = np.real(np.fft.fftshift(np.fft.ifft2(H)))
        c, h = nf // 2, SMALL // 2
        k = fine[c - h : c + h + 1, c - h : c + h + 1]
        k = k / k.sum()
        fr, R = rc.kernel_response(k)
        gx, gy = np.meshgrid(fr, fr)
        dev = np.abs(R - rc.template(facets, gx, gy, SLOPE, t, options))
        err = float(dev[np.maximum(np.abs(gx), np.abs(gy)) <= rc.ERROR_DOMAIN].max())
        axs[1, i].imshow(
            R, extent=(fr[0], fr[-1], fr[-1], fr[0]), cmap="RdBu_r", vmin=-0.3, vmax=1.3
        )
        axs[1, i].add_patch(cell_patch(facets, fill=False, ec="k", lw=0.6))
        axs[1, i].set_title(
            f"({'cd'[i]}) response of the {SMALL} x {SMALL} kernel\nmax error {err:.3f}"
        )
        freq_ticks(axs[1, i])
        kernel, info = rc.synthesize_kernel(facets, SLOPE, TOL, 0.01, options=options)
        res[name] = {
            f"error_{SMALL}": err,
            "size": int(kernel.shape[0]),
            "accuracy_met": bool(info["accuracy_met"]),
            **kernel_shares(kernel, facets),
        }
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save(fig, "fig_nyquist")

    # the gates equal the tolerance on the border: the jump left is the
    # tolerance times the jump without them
    jumps = {}
    for tol in (0.01, 0.1, 0.5):
        tt = rc.cutoff_half_width(SLOPE, tol)
        kernel, info = rc.synthesize_kernel(facets, SLOPE, tol, 0.01, options=NYQ)
        jumps[f"{tol:g}"] = {
            "cell": rc.border_mismatch(facets, 0, SLOPE, tt),
            "with_gates": rc.border_mismatch(
                rc.template_gates(facets, SLOPE, tt, NYQ)[0], 0, SLOPE, tt
            ),
            "size_with_gates": int(kernel.shape[0]),
            "accurate_with_gates": bool(info["accuracy_met"]),
            "error_with_gates": float(info["response_error"]),
            "leak_with_gates": kernel_shares(kernel, facets)["leak"],
        }
    res["border_jump_by_tolerance"] = jumps
    # at 45 degrees the template takes the same values on opposite edges
    f45 = rc.cell_facets(rot(45))
    res["rot45_gates_added"] = len(rc.template_gates(f45, SLOPE, t, NYQ)[0]) - len(f45)
    RESULTS["nyquist"] = res


@step
def fig_blur():
    facets = rc.cell_facets(np.diag([1.0, 2.0]))
    f = np.fft.fftshift(np.fft.fftfreq(257))
    fx, fy = np.meshgrid(f, f)
    fig, axs = plt.subplots(
        1, 3, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.25]}
    )
    for ax, options, title in (
        (axs[0], rc.TemplateOptions(), "(a) default template"),
        (axs[1], PROGRESSIVE, "(b) progressive gates"),
    ):
        H = rc.template(facets, fx, fy, SLOPE, T_DEFAULT, options)
        ax.imshow(H, extent=(f[0], f[-1], f[-1], f[0]), cmap="viridis", vmin=0, vmax=1)
        # the vertical edges of the cell lie on the border of the square
        for y in (-0.25, 0.25):
            ax.axhline(y, color="w", lw=0.7, ls="--")
        for x in (-0.5, 0.5):
            ax.axvline(x, color="w", lw=1.4, ls="--")
        ax.set(xlim=(-0.5, 0.5), ylim=(0.5, -0.5))
        ax.set_title(title, fontsize=7.5)
        freq_ticks(ax)
        ax.set_xlabel("$\\nu_x$ (cycles/px)")
    axs[1].set_yticklabels([])
    axs[0].set_ylabel("$\\nu_y$ (cycles/px)")

    # along the kept axis, for scales of that axis across the ramp
    ax = axs[2]
    fx1 = np.linspace(0, 0.5, 501)
    ramp = rc.ramp_width(np.array([1.0, 0.0]), SLOPE, TOL)
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate((0, 0.25, 0.5, 0.75, 1)):
        s = 1 + k * ramp
        tmp = rc.cell_facets(np.diag([s, 2.0]))
        ax.plot(
            fx1,
            rc.template(tmp, fx1, 0 * fx1, SLOPE, T_DEFAULT, PROGRESSIVE),
            color=cmap(0.85 * i / 4),
            lw=1.2,
            label=f"$s$ = {s:.3f}",
        )
    tmp = rc.cell_facets(np.diag([1.0, 2.0]))
    ax.plot(
        fx1,
        rc.template(tmp, fx1, 0 * fx1, SLOPE, T_DEFAULT),
        color=C_BAD,
        ls="--",
        lw=1.0,
        label="default, $s$ = 1",
    )
    ax.axhline(TOL, color=C_GREY, lw=0.6, ls=":")
    ax.set(
        xlim=(0.2, 0.5),
        ylim=(-0.03, 1.05),
        xlabel="$\\nu_x$ (cycles/px), at $\\nu_y = 0$",
    )
    ax.set_title("(c) progressive gates, kept axis\nat scale $s$", fontsize=7.5)
    ax.legend(fontsize=5.6, frameon=False, loc="lower left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_blur")
    RESULTS["blur"] = {
        name: {
            "cell_facets": len(rc.cell_facets(W)),
            "template_facets": len(
                rc.template_gates(rc.cell_facets(W), SLOPE, T_DEFAULT, PROGRESSIVE)[0]
            ),
        }
        for name, W in (
            ("rows decimated by 2", np.diag([1.0, 2.0])),
            ("decimation x2", 2.0 * np.eye(2)),
            ("running example", W_EX),
        )
    }


# Loosest gate of a grid s R(theta) with the progressive gates: the largest
# tolerance among the facets that cut the square, nan without filter.
def loosest_gate(s, theta):
    facets = rc.cell_facets(s * rot(theta))
    if not rc.filter_required(facets):
        return np.nan
    data = [rc.facet_tolerance(a, SLOPE, TOL) for a in facets]
    return max(tmp for tmp in data if tmp < 1.0)


# Scale where the ramp of an axis-like facet starts and ends, for a rotation.
def ramp_scales(theta):
    c, sn = abs(np.cos(np.radians(theta))), abs(np.sin(np.radians(theta)))
    s0 = 1.0 / (c + sn)
    s = s0
    for _ in range(100):
        s = (1.0 + rc.ramp_width(s * np.array([c, sn]), SLOPE, TOL)) / (c + sn)
    return s0, s


@step
def fig_ramp_region():
    angles = np.linspace(0, 45, 181)
    scales = np.linspace(0.6, 1.3, 281)
    data = np.array([[loosest_gate(s, a) for a in angles] for s in scales])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.set_facecolor("#e8e8e8")
    mesh = ax.pcolormesh(
        angles, scales, data, cmap="magma", vmin=TOL, vmax=1.0, shading="nearest"
    )
    ax.contour(
        angles,
        scales,
        np.nan_to_num(data, nan=1.0) > TOL + 1e-9,
        levels=[0.5],
        colors="w",
        linewidths=0.6,
    )
    ax.axhline(1.0, color=C_GREY, lw=0.7, ls="--")
    ax.text(30, 0.66, "no filter", fontsize=7.5, ha="center")
    ax.text(12, 1.2, "default gates", fontsize=7.5, ha="center", color="w")
    ax.text(27, 0.815, "ramp", fontsize=7.5, ha="center", color="w", rotation=-12)
    ax.set(
        xlabel="rotation $\\theta$ (degrees)",
        ylabel="scale $s$ of $W = s\\,R(\\theta)$",
        xlim=(0, 45),
        ylim=(0.6, 1.3),
    )
    bar = fig.colorbar(mesh, ax=ax, pad=0.02)
    bar.set_label("loosest gate on the cell boundary", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig_ramp_region")
    res = {f"{a:g} deg": [round(v, 3) for v in ramp_scales(a)] for a in (0, 10, 45)}
    # at unit scale, the largest rotation still inside the ramp
    angle = brentq(
        lambda t: (
            np.cos(t)
            + np.sin(t)
            - 1.0
            - rc.ramp_width(np.array([np.cos(t), np.sin(t)]), SLOPE, TOL)
        ),
        1e-4,
        0.7,
    )
    res["unit scale, last angle in the ramp (deg)"] = round(float(np.degrees(angle)), 1)
    RESULTS["ramp_region"] = res
