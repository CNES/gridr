# coding: utf8
#
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
benchmarks/scripts/bench.py

CLI to run benchmarks with version control and save.

Example :
   PYTHONPATH=$PWD/python:$PYTHONPATH python3 benchmarks/scripts/bench.py --version v0.5.1
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

BENCHMARK_TESTS_ROOT = "tests/python/benchmarks/"
BENCHMARK_RESULTS_ROOT = f"{PROJECT_ROOT}/benchmarks/results/pytest"

# ---------------------------------------------------------------------------
# GIT Detection
# ---------------------------------------------------------------------------


def _run_git(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def detect_version() -> str | None:
    # 1. Exact TAG on HEAD
    v = _run_git("git describe --tags --exact-match")
    if v:
        return v
    # 2. git describe (v1.2.0-3-gabc123-dirty)
    v = _run_git("git describe --tags --dirty --always")
    if v:
        return v
    # 3. Installed version
    try:
        from importlib.metadata import version

        return version("gridr")
    except Exception:
        return None


def detect_commit() -> str:
    return _run_git("git rev-parse HEAD") or "unknown"


def is_dirty() -> bool:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True)
        return bool(result.stdout.strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Run pytest-benchmark
# ---------------------------------------------------------------------------


def run_benchmarks(args: argparse.Namespace) -> int:
    runner_slug = args.runner.replace(" ", "_").replace("/", "-")
    bench_name = f"{args.version or 'untagged'}_{runner_slug}"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        BENCHMARK_TESTS_ROOT,
        f"--benchmark-save={bench_name}",
        f"--benchmark-storage={BENCHMARK_RESULTS_ROOT}",
        "--benchmark-group-by=group",
        "--benchmark-columns=min,median,mean,stddev,iqr,outliers,ops,rounds",
        "-v",
    ]

    if args.filter:
        cmd += ["-k", args.filter]

    if args.group:
        cmd += ["--benchmark-group-by=group", "-m", args.group]

    # Environment variables read by conftest.py
    env = os.environ.copy()
    env["BENCH_VERSION"] = args.version or ""
    env["BENCH_RUNNER"] = args.runner
    env["BENCH_COMMIT"] = detect_commit()
    env["BENCH_DIRTY"] = "1" if is_dirty() else "0"

    print(f"  $ {' '.join(cmd)}\n")
    return subprocess.run(cmd, env=env).returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch benchmarks with version control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        "-v",
        metavar="VERSION",
        default=None,
        help=(
            "Version to map to the test (e.g.: v0.5.1). "
            "Auto-detected with git describe if missing."
        ),
    )
    parser.add_argument(
        "--runner",
        metavar="NOM",
        default=platform.node(),
        help=f"Human-readable machine name (Default: {platform.node()}).",
    )
    parser.add_argument(
        "--filter",
        "-k",
        metavar="EXPR",
        default=None,
        help="Filter to pass to pytest -k (e.g.: 'interpolation and not memory').",
    )
    parser.add_argument(
        "--group",
        "-g",
        metavar="GROUP",
        default=None,
        help="Benchmark group (e.g.: interpolation_time).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.version is None:
        args.version = detect_version()

    dirty_warn = " WARNING : non commited files in working tree !" if is_dirty() else ""
    print(
        f"""
┌─ Benchmark ──────────────────────────────────────────────────
│  Version    : {args.version or 'not specified'}{dirty_warn}
│  Commit     : {detect_commit()[:8]}
│  Runner     : {args.runner}
│  Filter     : {args.filter or 'tous'}
│  Group      : {args.group or 'tous'}
└──────────────────────────────────────────────────────────────
"""
    )

    # 1. Run pytest-benchmark
    rc = run_benchmarks(args)
    if rc != 0:
        print(f"\n !!! pytest failed (code {rc})")
        sys.exit(rc)


if __name__ == "__main__":
    main()
