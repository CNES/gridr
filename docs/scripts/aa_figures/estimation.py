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

from gridr.core.filter.antialiasing import reciprocal_cell as rc
from gridr.core.grid.grid_antialiasing import (
    estimate_transition_matrix,
    fit_affine_least_squares,
)

from .common import C_BAD, C_DST, C_GREY, inside
from .plot import RESULTS, cell_patch, freq_ticks, save, step


def matrix_stats(W):
    return {
        "rotation_deg": float(np.degrees(np.arctan2(W[1, 0], W[0, 0]))),
        "norm_w1": float(np.hypot(*W[:, 0])),
        "norm_w2": float(np.hypot(*W[:, 1])),
    }


@step
def fig_estimation():
    # curved grid z -> 2 z + a z^2, with z = c + i r the node index
    a = 4e-3
    r, c = np.mgrid[0:48, 0:48].astype(float)
    z = 2.0 * (c + 1j * r) + a * (c + 1j * r) ** 2
    gr, gc = np.ascontiguousarray(z.imag), np.ascontiguousarray(z.real)
    # what a perfect estimator would return: the mean local Jacobian
    dydr, dydc = np.gradient(gr)
    dxdr, dxdc = np.gradient(gc)
    W_mean = np.array([[dxdc.mean(), dxdr.mean()], [dydc.mean(), dydr.mean()]])
    modes = (("least-square-fit", C_DST, "-"), ("longest-distance", C_BAD, "--"))

    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.6))
    ax = axs[0]
    ax.scatter(gc[::3, ::3], gr[::3, ::3], s=1.5, c=C_GREY)
    ax.plot(gc[0, :], gr[0, :], color=C_BAD, lw=1.4)
    ax.plot(gc[:, 0], gr[:, 0], color=C_BAD, lw=1.4)
    ax.plot([gc[0, 0], gc[0, -1]], [gr[0, 0], gr[0, -1]], color=C_BAD, ls=":", lw=0.9)
    ax.plot([gc[0, 0], gc[-1, 0]], [gr[0, 0], gr[-1, 0]], color=C_BAD, ls=":", lw=0.9)
    ax.set(aspect="equal", xlabel="$x$ (source)", ylabel="$y$ (source)")
    ax.invert_yaxis()
    ax.set_title(
        "(a) curved grid; longest row and\ncolumn used by longest-distance",
        fontsize=7.5,
    )
    ax.tick_params(labelsize=6)

    ax = axs[1]
    facets = rc.cell_facets(W_mean)
    ax.add_patch(
        cell_patch(facets, fc=C_GREY, alpha=0.25, ec="none", label="mean Jacobian")
    )
    f = np.linspace(-0.5, 0.5, 801)
    fx, fy = np.meshgrid(f, f)
    ref = inside(fx, fy, facets)
    res = {}
    for mode, col, ls in modes:
        W = estimate_transition_matrix(gr, gc, mode=mode)
        facets = rc.cell_facets(W)
        ax.add_patch(cell_patch(facets, fill=False, ec=col, lw=1.1, ls=ls, label=mode))
        res[mode] = {
            **matrix_stats(W),
            "cell_mismatch_pct": float(
                100 * np.mean(inside(fx, fy, facets) != ref) / np.mean(ref)
            ),
        }
    ax.set(
        xlim=(-0.48, 0.48),
        ylim=(0.36, -0.60),
        aspect="equal",
        xlabel="$\\nu_x$ (cycles/px)",
    )
    ax.set_title("(b) cells from the two estimates", fontsize=7.5)
    ax.legend(
        fontsize=5.3, frameon=False, loc="upper center", ncol=1, borderaxespad=0.1
    )
    freq_ticks(ax, 0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_estimation")
    RESULTS["estimation"] = {
        "mean_jacobian": matrix_stats(W_mean),
        "modes": res,
        "affine_residual": fit_affine_least_squares(gr, gc)[2],
    }
