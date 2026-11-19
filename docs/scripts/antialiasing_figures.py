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
# Figures and numbers of the antialiasing pages, with results.json for
# build_figures.py: python docs/scripts/antialiasing_figures.py [--only STEP ...]
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from aa_figures import aliasing, estimation, options, plot, prefiltering
from aa_figures.common import REPO_ROOT

# Bump when the layout of results.json changes, an older file is then ignored
RESULTS_FORMAT = 3

PAGES = (aliasing, prefiltering, options, estimation)

# Files every step depends on, besides its own module
SHARED = (
    "docs/scripts/aa_figures/__init__.py",
    "docs/scripts/aa_figures/common.py",
    "docs/scripts/aa_figures/plot.py",
    "python/gridr/core/filter/lattice.py",
    "python/gridr/core/filter/antialiasing/__init__.py",
    "python/gridr/core/filter/antialiasing/reciprocal_cell.py",
    "python/gridr/core/grid/grid_antialiasing.py",
)


# SHA-256 with LF line endings, the same on a Windows checkout.
def sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def provenance(fn):
    module = Path(sys.modules[fn.__module__].__file__).resolve()
    files = [module.relative_to(REPO_ROOT).as_posix(), *SHARED]
    # the prefiltering page draws its tilings with the aliasing module
    if fn.__module__.endswith("prefiltering"):
        files.append("docs/scripts/aa_figures/aliasing.py")
    return {f: sha256(REPO_ROOT / f) for f in files}


def main():
    steps = [
        fn for page in PAGES for fn in plot.STEPS if fn.__module__ == page.__name__
    ]
    names = [fn.__name__ for fn in steps]
    parser = argparse.ArgumentParser(description="Figures of the antialiasing pages.")
    parser.add_argument("--outdir", type=Path, default=plot.OUTDIR)
    parser.add_argument("--only", nargs="+", metavar="STEP", choices=names)
    parser.add_argument("--pdf", action="store_true", help="also write PDF files")
    parser.add_argument("--list", action="store_true", help="print the steps")
    args = parser.parse_args()
    if args.list:
        print("\n".join(names))
        return 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot.CONF.update(outdir=args.outdir, pdf=args.pdf)

    path = args.outdir / "results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if data.get("format") != RESULTS_FORMAT:
        data = {}
    done = data.get("steps", {})
    results = data.get("results", {})
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for fn in steps:
        if args.only and fn.__name__ not in args.only:
            continue
        keys = set(plot.RESULTS)
        plot.WIDTHS.clear()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        keys = sorted(set(plot.RESULTS) - keys)
        done[fn.__name__] = {
            "outputs": [f"{name}.png" for name in plot.WIDTHS],
            "display_widths": dict(plot.WIDTHS),
            "results": keys,
            "date": time.strftime("%Y-%m-%d"),
            "elapsed_s": round(elapsed, 1),
            "provenance": provenance(fn),
            "environment": env,
        }
        results.update({key: plot.RESULTS[key] for key in keys})
        print(f"{fn.__name__} done ({elapsed:.1f} s)", flush=True)

    data = {
        "format": RESULTS_FORMAT,
        "all_steps": names,
        "steps": {name: done[name] for name in names if name in done},
        "results": results,
    }
    path.write_text(json.dumps(data, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
