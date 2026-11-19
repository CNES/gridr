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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gridr.core.filter.antialiasing import reciprocal_cell as rc
from gridr.core.filter.antialiasing.reciprocal_cell import ReciprocalCellFilter
from gridr.core.grid.grid_antialiasing import resolve_antialiasing_filter
from gridr.core.grid.grid_resampling import array_grid_resampling

REPO_ROOT = Path(__file__).resolve().parents[3]

# Same colour, same meaning on every page
C_DST = "#1b7f3b"  # destination metric, the default cell
C_SRC = "#d17a00"  # source metric
C_ALT = "#7b4fa0"  # any other choice
C_LAT = "#2c5d9e"  # lattice points
C_GREY = "#7f7f7f"
C_BAD = "#c0392b"  # what goes wrong

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

# Zone plate: the local frequency at p is K (p - c), FMAX on the edge of the
# disk. The rings fade to grey over the last TAPER pixels.
N = 512
FMAX = 0.45
K = FMAX / (N / 2)
TAPER = 16

SLOPE = 10.0
TOL = 0.1
T_DEFAULT = rc.cutoff_half_width(SLOPE, TOL)
NYQ = rc.TemplateOptions(nyquist_gates=True)
PROGRESSIVE = rc.TemplateOptions(progressive_gates=True)

INTERPOLATOR = "bspline5"
BSPLINE_KWARGS = {"epsilon": 1e-6, "mask_influence_threshold": 1e-3}

# Output size of the zone plate resamplings: keeps the cell and the two false
# zone plates, and stays inside the source.
N_OUT = 116

CELL_FRAMES = (
    ("destination", "parallelogram", C_DST),
    ("source", "hexagon", C_SRC),
)


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# Running example: decimation by s, rotation by 20 degrees, shear of 0.5.
def w_example(s=3.0):
    return s * rot(20) @ np.array([[1.0, 0.5], [0.0, 1.0]])


W_EX = w_example()


# True where (fx, fy) lies inside the cell.
def inside(fx, fy, facets):
    return np.all(np.abs(np.stack([fx, fy], -1) @ np.array(facets).T) <= 0.5, -1)


def zone_plate(n=N, fmax=FMAX):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    r2 = (x - n / 2) ** 2 + (y - n / 2) ** 2
    k = fmax / (n / 2)
    t = np.clip((n / 2 - np.sqrt(r2)) / TAPER, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * t)
    return 0.5 + 0.5 * w * np.cos(np.pi * k * r2)


def lattice_points(B, r=5):
    return np.array(
        [B @ np.array([a, b]) for a in range(-r, r + 1) for b in range(-r, r + 1)]
    )


# Backward grid of W, centered on the source.
def centered_grid(W, n, n_src=N):
    r, c = np.mgrid[0:n, 0:n].astype(float)
    cc, rr = c - n / 2, r - n / 2
    gx = W[0, 0] * cc + W[0, 1] * rr + n_src / 2
    gy = W[1, 0] * cc + W[1, 1] * rr + n_src / 2
    return np.ascontiguousarray(gy), np.ascontiguousarray(gx)


# Kernel and diagnostics of ReciprocalCellFilter for W, as GridR resolves them.
def resolve(W, **kwargs):
    gr, gc = centered_grid(W, 32)
    kf, diag = resolve_antialiasing_filter(ReciprocalCellFilter(**kwargs), gr, gc)
    return kf, diag.filter


def circular_convolve(img, kernel):
    m, n = img.shape
    kh, kw = kernel.shape
    tmp = np.zeros((m, n))
    tmp[
        m // 2 - kh // 2 : m // 2 - kh // 2 + kh,
        n // 2 - kw // 2 : n // 2 - kw // 2 + kw,
    ] = kernel
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(np.fft.ifftshift(tmp))))


# Exact filtering by the template, or by a mask of the frequencies.
def filter_by(img, response):
    return np.real(np.fft.ifft2(np.fft.fft2(img - 0.5) * response)) + 0.5


def filter_by_template(img, facets):
    f = np.fft.fftfreq(img.shape[0])
    fx, fy = np.meshgrid(f, f)
    return filter_by(img, rc.template(facets, fx, fy, SLOPE, T_DEFAULT))


def filter_by_cell(img, facets):
    f = np.fft.fftfreq(img.shape[0])
    fx, fy = np.meshgrid(f, f)
    return filter_by(img, inside(fx, fy, facets))


def resample(img, grid, interpolator=INTERPOLATOR):
    kwargs = BSPLINE_KWARGS if interpolator.startswith("bspline") else None
    # the B-spline prefilter runs in place when no padding is needed: pass a copy
    out, _ = array_grid_resampling(
        interpolator,
        np.array(img, dtype=np.float64, order="C", copy=True),
        grid[0],
        grid[1],
        (1, 1),
        None,
        nodata_out=0.0,
        interp_kwargs=kwargs,
        boundary_condition="edge",
        trust_padding=True,
    )
    return out


# Near exact sampling: zero padded FFT upsampling, then cubic interpolation.
def exact_resample(img, grid, up=4):
    n = img.shape[0]
    tmp = np.zeros((n * up, n * up), complex)
    o = (n * up - n) // 2
    tmp[o : o + n, o : o + n] = np.fft.fftshift(np.fft.fft2(img))
    img_up = np.real(np.fft.ifft2(np.fft.ifftshift(tmp))) * up * up
    return resample(img_up, (grid[0] * up, grid[1] * up), "cubic")


# Response of a kernel along the line nu_y = fy.
def line_response(kernel, fx, fy):
    h = kernel.shape[0] // 2
    q = np.arange(-h, h + 1)
    rows = kernel.T @ np.exp(-2j * np.pi * fy * q)
    return np.real(np.exp(-2j * np.pi * np.outer(fx, q)) @ rows)
