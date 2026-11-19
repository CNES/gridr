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
from matplotlib import patheffects
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

from gridr.core.filter.lattice import voronoi_cell_vertices

from .common import C_BAD, N_OUT, REPO_ROOT, K, N

OUTDIR = REPO_ROOT / "docs" / "source" / "images" / "antialiasing"

# Figures full of fine rings
PALETTE = {
    "fig_moire",
    "fig_nyquist",
    "fig_problem",
    "fig_aliasing_cell",
    "fig_zp_source",
    "fig_zp_spectrum",
    "fig_aliasing_spectrum",
    "fig_moire_spectra",
}

# The pages show DISPLAY_DPI CSS pixels per figure inch. Pixel figures are
# saved at PX_RATIO PNG pixels per CSS pixel, so that no image is resampled.
PNG_DPI = 200
DISPLAY_DPI = 120
PX_RATIO = 2

# Margins of the pixel figures: tick labels, two-line title, ticks, gap
PX_LEFT, PX_TITLE, PX_TICKS, PX_GAP = 40, 36, 24, 16
OUT_ZOOM = 2
PX_OUT = OUT_ZOOM * N_OUT
# Colour bar right of a panel: gap, bar, room for its labels
CB_GAP, CB_W, CB_ROOM = 6, 8, 40
CB_SPACE = CB_GAP + CB_W + CB_ROOM

HALO = [patheffects.withStroke(linewidth=3.2, foreground="white")]

STEPS = []
RESULTS = {}
WIDTHS = {}
CONF = {"outdir": OUTDIR, "pdf": False}


def step(fn):
    STEPS.append(fn)
    return fn


# Write a figure and record the width the page must give it.
def save(fig, name):
    if CONF["pdf"]:
        fig.savefig(CONF["outdir"] / f"{name}.pdf")
    path = CONF["outdir"] / f"{name}.png"
    css = getattr(fig, "css_size", None)
    if css is None:
        dpi = PNG_DPI
        fig.savefig(path, dpi=dpi, metadata={"Software": None})
    else:
        dpi = DISPLAY_DPI * PX_RATIO
        # a tight box would crop a few pixels and break the integer scale
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(path, dpi=dpi, metadata={"Software": None})
    with Image.open(path) as im:
        width = im.size[0]
        if name in PALETTE:
            dither = Image.Dither.NONE if css else Image.Dither.FLOYDSTEINBERG
            tmp = im.convert("RGB").quantize(colors=256, dither=dither)
    if name in PALETTE:
        tmp.save(path, optimize=True)
    WIDTHS[name] = round(width * DISPLAY_DPI / dpi)
    plt.close(fig)


def cell_patch(facets, **kwargs):
    return Polygon(voronoi_cell_vertices(facets), closed=True, **kwargs)


def nyquist_square(**kwargs):
    return Rectangle((-0.5, -0.5), 1, 1, fill=False, ls="--", **kwargs)


def freq_ticks(ax, lim=0.4):
    t = [-lim, -lim / 2, 0, lim / 2, lim]
    ax.set_xticks(t)
    ax.set_yticks(t)
    ax.tick_params(labelsize=6, length=2, pad=1)


# Figure laid out in CSS pixels.
def pixel_figure(width, height):
    fig = plt.figure(
        figsize=(width / DISPLAY_DPI, height / DISPLAY_DPI), dpi=DISPLAY_DPI * PX_RATIO
    )
    fig.css_size = (width, height)
    return fig


def pixel_axes(fig, x, y, w, h, **kwargs):
    W, H = fig.css_size
    return fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H), **kwargs)


def show_pixels(ax, img, extent=None, **kwargs):
    if extent is None:
        n, m = img.shape[:2]
        extent = (-0.5, m - 0.5, n - 0.5, -0.5)
    ax.imshow(img, interpolation="nearest", aspect="auto", extent=extent, **kwargs)


def colour_bar(fig, ax, x, y, height, label=None, **kwargs):
    cax = pixel_axes(fig, x + CB_GAP, y, CB_W, height)
    bar = fig.colorbar(ax.images[-1], cax=cax, **kwargs)
    bar.ax.tick_params(labelsize=6, length=2, pad=1)
    if label:
        bar.set_label(label, fontsize=6.5, labelpad=2)


# Spectrum in dB, with its colour bar on the right of the panel.
def show_spectrum(fig, x, y, img, facets, title, upper_half=False):
    n = img.shape[0]
    h = 256 if upper_half else 512
    ax = pixel_axes(fig, x, y, 512, h)
    F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean())))
    S = 20 * np.log10(F / F.max() + 1e-12)
    lo, hi = -0.5 - 0.5 / n, 0.5 - 0.5 / n
    if upper_half:
        S, extent = S[: n // 2], (lo, hi, -0.5 / n, lo)
    else:
        extent = (lo, hi, hi, lo)
    show_pixels(ax, S, extent=extent, cmap="magma", vmin=-60, vmax=0)
    if facets is not None:
        ax.add_patch(cell_patch(facets, fill=False, ec="w", lw=0.9))
    freq_ticks(ax)
    if upper_half:
        ax.set_yticks([-0.4, -0.2, 0])
    # limits after the ticks, which would widen the view
    ax.set(xlim=extent[:2], ylim=extent[2:])
    ax.set_title(title, fontsize=7.5)
    colour_bar(fig, ax, x + 512, y, h, "amplitude (dB)", ticks=[-60, -40, -20, 0])
    return ax


# Output image, each pixel shown as OUT_ZOOM x OUT_ZOOM CSS pixels.
def show_output(ax, img, title, gain=1.0):
    n = img.shape[0]
    m = round(ax.get_position().width * ax.figure.css_size[0])
    if m != OUT_ZOOM * n:
        raise ValueError(f"axes of {m} CSS pixels for an image of {n} pixels")
    tmp = img if gain == 1.0 else 0.5 + gain * img
    tmp = np.kron(np.clip(tmp, 0, 1), np.ones((OUT_ZOOM, OUT_ZOOM)))
    show_pixels(
        ax, tmp, extent=(-0.5, n - 0.5, n - 0.5, -0.5), cmap="gray", vmin=0, vmax=1
    )
    ax.set_title(title, fontsize=7.5)
    ax.axis("off")


# Central part of the zone plate, axes in local frequency.
def show_source(ax, img, title, annotate=True, half=N // 2):
    c = N // 2
    lo, hi = -(half + 0.5) * K, (half - 0.5) * K
    show_pixels(
        ax,
        img[c - half : c + half, c - half : c + half],
        extent=(lo, hi, hi, lo),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    freq_ticks(ax, 0.4 if half * K > 0.41 else 0.2)
    ax.set(xlim=(lo, hi), ylim=(hi, lo))
    ax.set_title(title, fontsize=7.5)
    if not annotate:
        return
    box = dict(fc="white", ec="none", pad=0.6, alpha=0.85)
    pf = 0.3 * np.array([np.cos(0.55), np.sin(0.55)])
    ax.annotate(
        "", xy=pf, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=1.2)
    )
    ax.plot([0], [0], "o", ms=3.5, color=C_BAD)
    ax.plot([pf[0]], [pf[1]], "o", ms=3.5, color=C_BAD)
    ax.text(
        -0.012, 0.035, r"$\mathbf{c}$", color=C_BAD, fontsize=9, ha="right", bbox=box
    )
    ax.text(
        pf[0] + 0.015, pf[1] + 0.03, r"$\mathbf{p}$", color=C_BAD, fontsize=9, bbox=box
    )
    ax.text(
        0.13,
        0.215,
        r"$\boldsymbol{\nu} = k(\mathbf{p} - \mathbf{c})$",
        color=C_BAD,
        fontsize=8,
        bbox=box,
    )
