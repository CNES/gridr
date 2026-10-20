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
benchmarks/scripts/bench_rust.py

CLI to run the Rust (criterion) benchmarks with version control and save.

The results are stored as one JSON file per run, in the same layout and the
same schema as the pytest-benchmark results saved by ``bench.py``:

    benchmarks/results/criterion/<System>-rustc-<X.Y>-64bit/NNNN_<version>_<runner>.json

Timings are expressed in seconds per benchmark iteration; one iteration of
``interp/<method>/<scenario>`` interpolates ``elements_per_iteration`` points
(see ``extra_info``). The raw criterion samples are kept in ``extra_info``.

By default the benchmarks run on the current working tree.
With ``--ref``, they run on another commit, checked out in a temporary git worktree; the
results are still saved in the current tree.
With ``--inject-bench``, the benchmark sources and their Cargo declarations are copied
from the current tree into that worktree, which allows measuring a commit that predates
the benchmarks, with the same benchmark code as the current tree.

Examples :
RUSTFLAGS="-C target-feature=+avx2,+fma" \
python3 benchmarks/scripts/bench_rust.py --version v0.8.0

RUSTFLAGS="-C target-feature=+avx2,+fma" \
python3 benchmarks/scripts/bench_rust.py --ref v0.7.0 --inject-bench
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Version, commit and working tree detection are shared with bench.py.
sys.path.insert(0, str(Path(__file__).parent))
from bench import detect_commit, detect_version, is_dirty

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUST_WORKSPACE = PROJECT_ROOT / "rust"
BENCH_CRATE = "gridr"
BENCHMARK_RESULTS_ROOT = PROJECT_ROOT / "benchmarks" / "results" / "criterion"
BASELINE_NAME = "bench_rust"

# ---------------------------------------------------------------------------
# Measured source
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """Source tree on which the benchmarks run.

    Attributes
    ----------
    rust_workspace : Path
        Rust workspace directory (``rust/``) of the measured tree.
    commit : str
        Full hash of the measured commit.
    version : str
        Version label of the measured commit.
    branch : str
        Branch name, empty for a detached commit.
    dirty : bool
        True if the measured tree has uncommitted changes.
    bench_injected : bool
        True if the benchmark sources were copied from the current tree.
    """

    rust_workspace: Path
    commit: str
    version: str
    branch: str
    dirty: bool
    bench_injected: bool = False


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command and return its stripped standard output.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    cwd : Path, optional
        Working directory.

    Returns
    -------
    str
        Standard output, or an empty string if the command fails.
    """
    try:
        return (
            subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def describe_ref(sha: str) -> str:
    """Return a version label for a commit, as ``detect_version`` does for HEAD.

    Parameters
    ----------
    sha : str
        Commit hash.

    Returns
    -------
    str
        The exact tag of the commit, otherwise its ``git describe`` label,
        otherwise the short hash.
    """
    return (
        _run(["git", "describe", "--tags", "--exact-match", sha], cwd=PROJECT_ROOT)
        or _run(["git", "describe", "--tags", "--always", sha], cwd=PROJECT_ROOT)
        or sha[:8]
    )


def current_source(version: str | None) -> Source:
    """Describe the current working tree.

    Parameters
    ----------
    version : str, optional
        Version label given on the command line.

    Returns
    -------
    Source
        The current tree.
    """
    return Source(
        rust_workspace=RUST_WORKSPACE,
        commit=detect_commit(),
        version=version or detect_version() or "",
        branch=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT),
        dirty=is_dirty(),
    )


def checkout_ref(ref: str, version: str | None, worktree: Path) -> Source:
    """Check out a commit in a temporary git worktree.

    Parameters
    ----------
    ref : str
        Any git reference to a commit (tag, branch, hash).
    version : str, optional
        Version label given on the command line.
    worktree : Path
        Directory of the worktree, which must not exist.

    Returns
    -------
    Source
        The checked-out tree.

    Raises
    ------
    SystemExit
        If the reference does not resolve to a commit or the checkout fails.
    """
    sha = _run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=PROJECT_ROOT)
    if not sha:
        sys.exit(f" !!! unknown git reference: {ref}")
    rc = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), sha],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        sys.exit(f" !!! git worktree add failed for {ref}")
    return Source(
        rust_workspace=worktree / "rust",
        commit=sha,
        version=version or describe_ref(sha),
        branch="",
        dirty=False,
    )


def remove_worktree(worktree: Path) -> None:
    """Remove a temporary git worktree.

    Parameters
    ----------
    worktree : Path
        Directory of the worktree.
    """
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Benchmark injection
# ---------------------------------------------------------------------------


def _bench_declarations(manifest: str) -> tuple[str | None, list[str]]:
    """Extract the criterion dependency and the ``[[bench]]`` blocks.

    Parameters
    ----------
    manifest : str
        Content of a ``Cargo.toml``.

    Returns
    -------
    tuple
        The ``criterion = ...`` line (or None) and the list of ``[[bench]]``
        blocks.
    """
    dep = re.search(r"^criterion\s*=.*$", manifest, re.MULTILINE)
    benches = re.findall(r"^\[\[bench\]\]\n(?:(?!\[).*\n?)*", manifest, re.MULTILINE)
    return (dep.group(0) if dep else None), [b.strip() for b in benches]


def inject_benches(target_workspace: Path) -> None:
    """Copy the benchmark sources and declarations of the current tree.

    The ``benches`` directory of the benchmarked crate is replaced by the
    current one. The criterion dev-dependency and the ``[[bench]]`` blocks
    are added to the target ``Cargo.toml`` when missing.

    Parameters
    ----------
    target_workspace : Path
        Rust workspace directory of the measured tree.

    Raises
    ------
    SystemExit
        If the current tree has no benchmark to inject.
    """
    src_crate = RUST_WORKSPACE / BENCH_CRATE
    dst_crate = target_workspace / BENCH_CRATE
    if not (src_crate / "benches").is_dir():
        sys.exit(" !!! no benches directory in the current tree")
    shutil.rmtree(dst_crate / "benches", ignore_errors=True)
    shutil.copytree(src_crate / "benches", dst_crate / "benches")

    src_manifest = (src_crate / "Cargo.toml").read_text(encoding="utf-8")
    dst_path = dst_crate / "Cargo.toml"
    dst_manifest = dst_path.read_text(encoding="utf-8")
    dep, benches = _bench_declarations(src_manifest)
    if dep and not re.search(r"^criterion\s*=", dst_manifest, re.MULTILINE):
        if re.search(r"^\[dev-dependencies\]", dst_manifest, re.MULTILINE):
            dst_manifest = re.sub(
                r"^\[dev-dependencies\]\n",
                f"[dev-dependencies]\n{dep}\n",
                dst_manifest,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            dst_manifest += f"\n[dev-dependencies]\n{dep}\n"
    for block in benches:
        name = re.search(r'name\s*=\s*"([^"]+)"', block)
        if name and f'name = "{name.group(1)}"' not in dst_manifest:
            dst_manifest += f"\n{block}\n"
    dst_path.write_text(dst_manifest, encoding="utf-8")


# ---------------------------------------------------------------------------
# Environment description
# ---------------------------------------------------------------------------


def rustc_version(workspace: Path) -> str:
    """Return the version of the rustc used in a workspace (e.g. ``1.96.0``).

    Parameters
    ----------
    workspace : Path
        Rust workspace directory; its toolchain file, if any, applies.

    Returns
    -------
    str
        The release number, or ``"unknown"``.
    """
    out = _run(["rustc", "-vV"], cwd=workspace)
    match = re.search(r"^release:\s*(\S+)", out, re.MULTILINE)
    return match.group(1) if match else "unknown"


def criterion_version(workspace: Path) -> str:
    """Return the criterion version locked in a workspace.

    Parameters
    ----------
    workspace : Path
        Rust workspace directory.

    Returns
    -------
    str
        The version, or ``"unknown"`` if criterion is not in ``Cargo.lock``.
    """
    try:
        text = (workspace / "Cargo.lock").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r'name = "criterion"\nversion = "([^"]+)"', text)
    return match.group(1) if match else "unknown"


def machine_info(args: argparse.Namespace, source: Source, rustc: str) -> dict:
    """Describe the machine, the build and the measured source.

    The keys used for the Python benchmarks are kept; the Rust specific ones
    are added.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    source : Source
        Measured source tree.
    rustc : str
        Version of the rustc used for the build.

    Returns
    -------
    dict
        The ``machine_info`` section of the result file.
    """
    info: dict = {
        "node": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
    }
    try:
        from cpuinfo import get_cpu_info

        cpu = get_cpu_info()
        relevant = {"avx", "avx2", "avx512f", "sse4_1", "sse4_2", "fma", "aes"}
        info["cpu"] = cpu
        info["cpu_flags"] = [f for f in cpu.get("flags", []) if f in relevant]
        info["cpu_hz_actual"] = cpu.get("hz_actual_friendly", "?")
        info["cpu_l2_cache"] = cpu.get("l2_cache_size")
        info["cpu_l3_cache"] = cpu.get("l3_cache_size")
    except ImportError:
        info["cpu"] = platform.processor()
        info["cpu_flags"] = []
        info["cpu_hz_actual"] = "? (pip install py-cpuinfo)"
    try:
        import psutil

        info["memory_available_gb"] = round(psutil.virtual_memory().available / 1e9, 2)
        info["swap_total_gb"] = round(psutil.swap_memory().total / 1e9, 2)
    except ImportError:
        pass

    info["bench_runner"] = args.runner
    info["bench_runner_id"] = os.environ.get("CI_RUNNER_ID", "local")
    info["lib_version"] = source.version
    info["git_commit"] = source.commit
    info["git_dirty"] = source.dirty

    info["rustc_version"] = rustc
    info["rustflags"] = os.environ.get("RUSTFLAGS", "")
    info["criterion_version"] = criterion_version(source.rust_workspace)
    info["cargo_profile"] = "bench"
    info["bench_injected"] = source.bench_injected
    if source.bench_injected:
        info["bench_source_commit"] = detect_commit()
    return info


# ---------------------------------------------------------------------------
# Run criterion
# ---------------------------------------------------------------------------


def run_benchmarks(
    args: argparse.Namespace, source: Source, criterion_home: Path
) -> int:
    """Run ``cargo bench`` and save the results under a named baseline.

    The cargo target directory of the current tree is shared, so that the
    dependencies are not rebuilt for another commit.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    source : Source
        Measured source tree.
    criterion_home : Path
        Directory where criterion writes its results for this run.

    Returns
    -------
    int
        Return code of ``cargo bench``.
    """
    cmd = ["cargo", "bench", "-p", BENCH_CRATE, "--bench", args.bench, "--"]
    cmd += ["--save-baseline", BASELINE_NAME]
    if args.criterion_args:
        cmd += shlex.split(args.criterion_args)
    if args.filter:
        cmd.append(args.filter)

    env = os.environ.copy()
    env["CRITERION_HOME"] = str(criterion_home)
    env.setdefault("CARGO_TARGET_DIR", str(RUST_WORKSPACE / "target"))
    print(f"  $ RUSTFLAGS={env.get('RUSTFLAGS', '')!r} {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=source.rust_workspace, env=env).returncode


# ---------------------------------------------------------------------------
# Conversion to the pytest-benchmark schema
# ---------------------------------------------------------------------------


def _quartiles(values: list[float]) -> tuple[float, float]:
    """Return the first and third quartiles of a sample.

    Parameters
    ----------
    values : list of float
        Sample, at least two values.

    Returns
    -------
    tuple of float
        ``(q1, q3)``.
    """
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], q[2]


def sample_stats(iters: list[float], times_ns: list[float]) -> dict:
    """Compute pytest-benchmark statistics from criterion samples.

    Each criterion sample measures ``iters[i]`` iterations in ``times_ns[i]``
    nanoseconds; the per-iteration time ``times_ns[i] / iters[i]`` is one
    round in the pytest-benchmark sense.

    Parameters
    ----------
    iters : list of float
        Number of iterations of each sample.
    times_ns : list of float
        Total time of each sample, in nanoseconds.

    Returns
    -------
    dict
        The ``stats`` section of a benchmark entry, times in seconds.
    """
    per_iter = [t / n * 1e-9 for t, n in zip(times_ns, iters, strict=True)]
    mean = statistics.fmean(per_iter)
    stddev = statistics.stdev(per_iter) if len(per_iter) > 1 else 0.0
    q1, q3 = _quartiles(per_iter)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = [v for v in per_iter if low <= v <= high]
    stddev_outliers = sum(1 for v in per_iter if abs(v - mean) > stddev)
    iqr_outliers = sum(1 for v in per_iter if v < low or v > high)
    return {
        "min": min(per_iter),
        "max": max(per_iter),
        "mean": mean,
        "stddev": stddev,
        "rounds": len(per_iter),
        "median": statistics.median(per_iter),
        "iqr": iqr,
        "q1": q1,
        "q3": q3,
        "iqr_outliers": iqr_outliers,
        "stddev_outliers": stddev_outliers,
        "outliers": f"{stddev_outliers};{iqr_outliers}",
        "ld15iqr": min(inside) if inside else min(per_iter),
        "hd15iqr": max(inside) if inside else max(per_iter),
        "ops": 1.0 / mean if mean > 0 else 0.0,
        "total": sum(times_ns) * 1e-9,
        "iterations": round(statistics.fmean(iters)),
    }


def collect(criterion_home: Path) -> list[dict]:
    """Read the saved baseline and build the benchmark entries.

    Parameters
    ----------
    criterion_home : Path
        Directory where criterion wrote its results.

    Returns
    -------
    list of dict
        One entry per benchmark, in the pytest-benchmark schema.
    """
    entries = []
    for meta_path in sorted(criterion_home.glob(f"**/{BASELINE_NAME}/benchmark.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sample = json.loads((run_dir / "sample.json").read_text(encoding="utf-8"))
        estimates = json.loads((run_dir / "estimates.json").read_text(encoding="utf-8"))

        group_id = meta.get("group_id") or ""
        scenario = meta.get("value_str") or meta.get("function_id") or ""
        domain, _, method = group_id.partition("/")
        throughput = meta.get("throughput") or {}

        entries.append(
            {
                "group": group_id,
                "name": meta["full_id"],
                "fullname": f"rust/{BENCH_CRATE}/benches::{meta['full_id']}",
                "params": {"bench": domain, "method": method, "scenario": scenario},
                "param": scenario,
                "extra_info": {
                    "elements_per_iteration": throughput.get("Elements"),
                    "criterion_estimates_ns": {
                        key: {
                            "point_estimate": value["point_estimate"],
                            "lower_bound": value["confidence_interval"]["lower_bound"],
                            "upper_bound": value["confidence_interval"]["upper_bound"],
                        }
                        for key, value in estimates.items()
                        if value is not None
                    },
                    "criterion_samples": {
                        "sampling_mode": sample.get("sampling_mode"),
                        "iters": sample["iters"],
                        "times_ns": sample["times"],
                    },
                },
                "options": {"timer": "criterion", "warmup": True},
                "stats": sample_stats(sample["iters"], sample["times"]),
            }
        )
    return entries


def output_path(args: argparse.Namespace, source: Source, rustc: str) -> Path:
    """Build the result file path, numbered after the existing files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments (``runner``).
    source : Source
        Measured source tree (``version``).
    rustc : str
        Version of the rustc used for the build.

    Returns
    -------
    Path
        ``<results>/<System>-rustc-<X.Y>-64bit/NNNN_<version>_<runner>.json``.
    """
    major_minor = ".".join(rustc.split(".")[:2])
    bits = platform.architecture()[0].replace("bit", "")
    directory = (
        BENCHMARK_RESULTS_ROOT / f"{platform.system()}-rustc-{major_minor}-{bits}bit"
    )
    directory.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name[:4]) for p in directory.glob("[0-9][0-9][0-9][0-9]_*.json")]
    runner_slug = args.runner.replace(" ", "_").replace("/", "-")
    version_slug = (source.version or "untagged").replace("/", "-")
    number = max(numbers, default=0) + 1
    return directory / f"{number:04d}_{version_slug}_{runner_slug}.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Launch the Rust benchmarks with version control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version",
        "-v",
        metavar="VERSION",
        default=None,
        help=(
            "Version to map to the run (e.g.: v0.8.0). "
            "Auto-detected with git describe if missing."
        ),
    )
    parser.add_argument(
        "--ref",
        metavar="REF",
        default=None,
        help=(
            "Git reference (tag, branch, hash) to benchmark instead of the "
            "current working tree. It is checked out in a temporary worktree."
        ),
    )
    parser.add_argument(
        "--inject-bench",
        action="store_true",
        help=(
            "With --ref: use the benchmark sources and Cargo declarations of "
            "the current tree."
        ),
    )
    parser.add_argument(
        "--runner",
        metavar="NAME",
        default=platform.node(),
        help=f"Human-readable machine name (Default: {platform.node()}).",
    )
    parser.add_argument(
        "--bench",
        metavar="TARGET",
        default="*",
        help="Cargo bench target (Default: all bench targets).",
    )
    parser.add_argument(
        "--filter",
        "-k",
        metavar="REGEX",
        default=None,
        help="Criterion filter on benchmark names (e.g.: 'interp/(linear|cubic)/').",
    )
    parser.add_argument(
        "--criterion-args",
        metavar="ARGS",
        default="",
        help=(
            "Extra criterion arguments "
            "(e.g.: '--measurement-time 5 --sample-size 100')."
        ),
    )
    args = parser.parse_args()
    if args.inject_bench and args.ref is None:
        parser.error("--inject-bench requires --ref")
    return args


def benchmark(args: argparse.Namespace, source: Source) -> None:
    """Run the benchmarks on a source tree and save the result file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    source : Source
        Measured source tree.
    """
    rustc = rustc_version(source.rust_workspace)
    dirty_warn = (
        " WARNING : non commited files in working tree !" if source.dirty else ""
    )
    injected = " (benchmarks from the current tree)" if source.bench_injected else ""
    print(
        f"""
┌─ Rust benchmark ─────────────────────────────────────────────
│  Version    : {source.version or "not specified"}{dirty_warn}
│  Commit     : {source.commit[:8]}{injected}
│  Runner     : {args.runner}
│  rustc      : {rustc}
│  RUSTFLAGS  : {os.environ.get("RUSTFLAGS", "")}
│  Bench      : {args.bench}
│  Filter     : {args.filter or "all"}
└──────────────────────────────────────────────────────────────
"""
    )

    with tempfile.TemporaryDirectory(prefix="gridr_criterion_") as tmp:
        criterion_home = Path(tmp)
        started = datetime.now(timezone.utc)
        rc = run_benchmarks(args, source, criterion_home)
        if rc != 0:
            sys.exit(f"\n !!! cargo bench failed (code {rc})")
        benchmarks = collect(criterion_home)
    if not benchmarks:
        sys.exit("\n !!! no criterion result found")

    result = {
        "machine_info": machine_info(args, source, rustc),
        "commit_info": {
            "id": source.commit,
            "dirty": source.dirty,
            "project": "gridr",
            "branch": source.branch,
        },
        "benchmarks": benchmarks,
        "datetime": started.isoformat(),
        "version": f"criterion {criterion_version(source.rust_workspace)}",
    }
    path = output_path(args, source, rustc)
    path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    print(f"\n  Saved {len(benchmarks)} benchmarks to {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    """Run the benchmarks on the requested tree and save the results."""
    args = parse_args()
    if args.ref is None:
        benchmark(args, current_source(args.version))
        return

    with tempfile.TemporaryDirectory(prefix="gridr_worktree_") as tmp:
        worktree = Path(tmp) / "src"
        source = checkout_ref(args.ref, args.version, worktree)
        try:
            if args.inject_bench:
                inject_benches(source.rust_workspace)
                source.bench_injected = True
            elif not (source.rust_workspace / BENCH_CRATE / "benches").is_dir():
                sys.exit(
                    f" !!! {args.ref} has no benchmark; "
                    "use --inject-bench to run the current ones"
                )
            benchmark(args, source)
        finally:
            remove_worktree(worktree)


if __name__ == "__main__":
    main()
