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
benchmarks/parse_results.py

Parse pytest-benchmark JSON files into pandas DataFrames.

Library usage:
    from benchmarks.parse_results import load_benchmark_json, load_benchmark_dir

    # Single file
    df = load_benchmark_json("benchmark_results/0001_abc123.json")

    # All files in a directory — merged into one DataFrame
    df = load_benchmark_dir("benchmark_results/", merge=True)

    # All files in a directory — one DataFrame per file
    dfs = load_benchmark_dir("benchmark_results/", merge=False)

CLI usage:
    # Single file → CSV
    python parse_results.py path/to/run.json

    # Directory, merged output → one CSV
    python parse_results.py path/to/results/ --merge

    # Directory, separate outputs → one CSV per JSON file
    python parse_results.py path/to/results/

    # Specify output directory
    python parse_results.py path/to/results/ --output-dir ./csv_exports/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# UNSERIALIZABLE adapter name extraction
# ---------------------------------------------------------------------------

_UNSERIALIZABLE_RE = re.compile(
    r"UNSERIALIZABLE\[<[\w.]+\.(\w+)\s+object\s+at\s+0x[0-9a-f]+>\]",
    re.IGNORECASE,
)


def _parse_adapter_name(raw: str | None) -> str | None:
    """
    Extract the adapter class name from a pytest-benchmark UNSERIALIZABLE string.

    Example:
        'UNSERIALIZABLE[<test_foo.OrionAdapter object at 0x14c7...>]'
        → 'OrionAdapter'

    Returns the raw string unchanged if the pattern does not match.
    """
    if raw is None:
        return None
    m = _UNSERIALIZABLE_RE.match(str(raw))
    return m.group(1) if m else str(raw)


def _parse_outliers(outliers_str: str) -> tuple[int, int]:
    """
    Parse the pytest-benchmark outliers string 'near;far' into two integers.

    pytest-benchmark reports two outlier counts separated by a semicolon:
      - near outliers: values between 1.5× and 3× the IQR
      - far  outliers: values beyond 3× the IQR

    Returns (0, 0) if parsing fails.
    """
    try:
        near_s, far_s = str(outliers_str).split(";")
        return int(near_s), int(far_s)
    except (ValueError, AttributeError):
        return 0, 0


# ---------------------------------------------------------------------------
# Single-file parser
# ---------------------------------------------------------------------------

def load_benchmark_json(path: str | Path) -> pd.DataFrame:
    """
    Load a single pytest-benchmark JSON file and return a flat DataFrame.

    Each row corresponds to one benchmark entry (one combination of adapter,
    method, and parameters).

    Columns produced
    ----------------
    Context (one value per file, repeated on every row):
        datetime        : run timestamp (datetime64, UTC)
        lib_version     : library version from machine_info
        git_commit      : short git commit hash (8 chars)
        git_branch      : git branch name from commit_info
        git_dirty       : True if the working tree had uncommitted changes
        bench_runner    : hostname / runner identifier from machine_info
        cpu             : CPU brand string from machine_info
        python_version  : Python version string (e.g. '3.11.10')

    Benchmark identity:
        group           : benchmark group label
        name            : full pytest node id (useful for investigation)
        adapter         : adapter class name parsed from the UNSERIALIZABLE param
        <param_keys>    : all other params keys (method, n, resolution, …)
                          list-valued params (e.g. resolution) are stored as tuples

    Timing metrics (seconds):
        time_min        : minimum observed time
        time_median     : median time (recommended primary metric)
        time_mean       : mean time
        time_stddev     : standard deviation
        time_max        : maximum observed time
        time_iqr        : interquartile range (Q3 - Q1)
        time_q1         : first quartile
        time_q3         : third quartile
        ops_per_sec     : throughput (1 / mean)
        rounds          : number of measured rounds
        outliers_near   : count of near outliers (IQR × 1.5)
        outliers_far    : count of far  outliers (IQR × 3.0)
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    machine      = data.get("machine_info", {})
    commit_info  = data.get("commit_info", {})
    datetime_str = data.get("datetime", "")

    # Resolve git commit: prefer the enriched machine_info field, fall back
    # to the standard commit_info block written by pytest-benchmark.
    raw_commit = machine.get("git_commit") or commit_info.get("id", "")
    short_commit = str(raw_commit)[:8] if raw_commit else ""

    # Resolve CPU brand: machine_info.cpu may be a nested dict (py-cpuinfo)
    # or a plain string depending on the conftest enrichment.
    cpu_info = machine.get("cpu", "")
    cpu_brand = (
        cpu_info.get("brand_raw", "")
        if isinstance(cpu_info, dict)
        else str(cpu_info)
    )

    # Metadata shared by all benchmarks in this file
    meta = {
        "datetime":      pd.to_datetime(datetime_str, utc=True),
        "lib_version":   machine.get("lib_version", ""),
        "git_commit":    short_commit,
        "git_branch":    commit_info.get("branch", ""),
        "git_dirty":     bool(
            machine.get("git_dirty", commit_info.get("dirty", False))
        ),
        "bench_runner":  machine.get("bench_runner", machine.get("node", "")),
        "cpu":           cpu_brand,
        "python_version": machine.get("python_version", ""),
    }

    rows = []
    for bm in data.get("benchmarks", []):
        stats  = bm.get("stats", {})
        params = bm.get("params") or {}

        outliers_near, outliers_far = _parse_outliers(stats.get("outliers", "0;0"))

        # Parse adapter class name and collect remaining params
        adapter_raw  = params.get("adapter")
        adapter_name = _parse_adapter_name(adapter_raw)

        # All params except "adapter"; list values (e.g. resolution) become
        # tuples so they are hashable and display cleanly in DataFrames.
        extra_params: dict = {}
        for k, v in params.items():
            if k == "adapter":
                continue
            extra_params[k] = tuple(v) if isinstance(v, list) else v

        row = {
            **meta,
            # Benchmark identity
            "group":         bm.get("group", ""),
            "name":          bm.get("name", ""),
            "adapter":       adapter_name,
            **extra_params,
            # Timing metrics
            "time_min":      stats.get("min",    None),
            "time_median":   stats.get("median", None),
            "time_mean":     stats.get("mean",   None),
            "time_stddev":   stats.get("stddev", None),
            "time_max":      stats.get("max",    None),
            "time_iqr":      stats.get("iqr",    None),
            "time_q1":       stats.get("q1",     None),
            "time_q3":       stats.get("q3",     None),
            "ops_per_sec":   stats.get("ops",    None),
            "rounds":        stats.get("rounds", None),
            "outliers_near": outliers_near,
            "outliers_far":  outliers_far,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure numeric types for all timing columns
    time_cols = [
        "time_min", "time_median", "time_mean", "time_stddev",
        "time_max", "time_iqr",    "time_q1",   "time_q3",
        "ops_per_sec",
    ]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------

def load_benchmark_dir(
    directory: str | Path,
    pattern: str = "**/*.json",
    merge: bool = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Load all pytest-benchmark JSON files found under *directory*.

    Parameters
    ----------
    directory : path to search recursively for JSON files
    pattern   : glob pattern used to locate files (default: '**/*.json')
    merge     : if True concatenate all files into one DataFrame and return it
                if False return a dict mapping each filename to its DataFrame

    The 'source_file' column is always added to each DataFrame so that rows
    can be traced back to their origin file even after merging.
    """
    directory = Path(directory)
    results: dict[str, pd.DataFrame] = {}

    for path in sorted(directory.glob(pattern)):
        try:
            df = load_benchmark_json(path)
            if not df.empty:
                df["source_file"] = path.name
                results[path.name] = df
                print(f"  {path.name}  ({len(df)} benchmarks)")
        except Exception as exc:
            print(f"  Cannot load {path.name}: {exc}", file=sys.stderr)

    if not results:
        print("  No benchmark files found.", file=sys.stderr)
        return pd.DataFrame() if merge else {}

    if merge:
        return pd.concat(results.values(), ignore_index=True)

    return results


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def _current_timestamp() -> str:
    """Return the current UTC datetime as a compact sortable string."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Write *df* to a semicolon-separated CSV file.

    Using ';' as separator avoids conflicts with the decimal comma used in
    some locales and makes the file directly openable in Excel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep=";", index=False, encoding="utf-8")
    print(f"  → {output_path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse pytest-benchmark JSON files into semicolon-separated CSV files.\n\n"
            "Single-file mode  : one CSV named <timestamp>_<stem>.csv\n"
            "Directory mode    : one CSV per file, or one merged CSV (--merge)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Path to a single JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help=(
            "Directory mode only: concatenate all JSON files into a single CSV. "
            "Without this flag each JSON file produces its own CSV."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=".",
        help="Directory where CSV files are written (default: current directory).",
    )
    parser.add_argument(
        "--pattern",
        metavar="GLOB",
        default="**/*.json",
        help="Glob pattern for JSON discovery in directory mode (default: '**/*.json').",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)
    timestamp   = _current_timestamp()

    # ── Single-file mode ────────────────────────────────────────────────────
    if input_path.is_file():
        print(f"Loading {input_path.name} …")
        df = load_benchmark_json(input_path)
        if df.empty:
            print("No benchmarks found in file.", file=sys.stderr)
            sys.exit(1)
        # CSV name: <timestamp>_<original_stem>.csv
        csv_name = f"{timestamp}_{input_path.stem}.csv"
        save_csv(df, output_dir / csv_name)

    # ── Directory mode ───────────────────────────────────────────────────────
    elif input_path.is_dir():
        print(f"Scanning {input_path} (pattern='{args.pattern}') …")

        if args.merge:
            # One merged CSV for the whole directory
            df = load_benchmark_dir(input_path, pattern=args.pattern, merge=True)
            if isinstance(df, pd.DataFrame) and df.empty:
                print("No benchmarks found.", file=sys.stderr)
                sys.exit(1)
            csv_name = f"{timestamp}_merged.csv"
            save_csv(df, output_dir / csv_name)

        else:
            # One CSV per JSON file
            dfs = load_benchmark_dir(input_path, pattern=args.pattern, merge=False)
            if not dfs:
                print("No benchmarks found.", file=sys.stderr)
                sys.exit(1)
            print(f"\nWriting {len(dfs)} CSV file(s) …")
            for filename, df in dfs.items():
                stem     = Path(filename).stem
                csv_name = f"{timestamp}_{stem}.csv"
                save_csv(df, output_dir / csv_name)

    else:
        print(f"Error: '{input_path}' is neither a file nor a directory.", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()

