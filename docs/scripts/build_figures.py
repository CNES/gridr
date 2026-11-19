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
Check, and regenerate on request, the figures of the documentation.

Some pages show figures made by a script. The figures are committed with the
pages and are not regenerated at every build, which would take minutes.
``docs/source/figures_list.json`` lists the scripts::

    [
        {
            "name": "antialiasing",
            "script": "docs/scripts/antialiasing_figures.py",
            "outdir": "docs/source/images/antialiasing",
        }
    ]

Paths are relative to the repository root. A script accepts ``--outdir DIR``,
``--only STEP ...`` and ``--list``, which prints its steps. It writes
``DIR/results.json`` with, for each step, the PNG files it produced
(``outputs``) and the SHA-256 of the files it depends on (``provenance``).

Three modes:

- ``check`` (default for the documentation build): nothing runs; a warning
  names the steps whose figures are missing or older than their sources;
- ``stale``: runs those steps only, plus the steps the script now provides
  and ``results.json`` does not know yet;
- ``all``: runs every script in full.

``conf.py`` calls :func:`build`. The mode comes from the configuration value
``gridr_build_figures`` (``sphinx-build -D gridr_build_figures=stale``) or, if
it is not set, from the environment variable ``GRIDR_DOC_BUILD_FIGURES``;
``1`` means ``all`` and ``0`` means ``check``.

Usage::

    python docs/scripts/build_figures.py [--mode check|stale|all] [--name NAME ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# this file is docs/scripts/build_figures.py
REPO_ROOT = Path(__file__).resolve().parents[2]

FIGURES_LIST = REPO_ROOT / "docs" / "source" / "figures_list.json"

MODES = ("check", "stale", "all")

# what people tend to type in an environment variable
_MODE_ALIASES = {
    "": "check",
    "0": "check",
    "false": "check",
    "no": "check",
    "off": "check",
    "1": "all",
    "true": "all",
    "yes": "all",
    "on": "all",
}


def resolve_mode(config_value: str | bool | None, env_value: str | None) -> str:
    """Return the mode from the Sphinx configuration value or the environment.

    Parameters
    ----------
    config_value : str, bool or None
        Value of ``gridr_build_figures``; an empty string or None means unset.
    env_value : str or None
        Value of ``GRIDR_DOC_BUILD_FIGURES``, used when the configuration
        value is unset.

    Returns
    -------
    str
        One of :data:`MODES`.

    Raises
    ------
    ValueError
        If the value is neither a mode nor one of the aliases.
    """
    if isinstance(config_value, bool):
        config_value = "1" if config_value else "0"
    value = config_value if config_value not in (None, "") else env_value
    value = (value or "").strip().lower()
    mode = _MODE_ALIASES.get(value, value)
    if mode not in MODES:
        raise ValueError(
            f"invalid figure build mode {value!r}: expected one of {MODES}, or 0 / 1"
        )
    return mode


def load_figures_list(path: Path = FIGURES_LIST) -> list[dict]:
    """Read the list of figure scripts.

    Parameters
    ----------
    path : pathlib.Path
        The JSON file.

    Returns
    -------
    list of dict
        One entry per script, with ``name``, ``script`` and ``outdir``.
    """
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def file_sha256(path: Path) -> str | None:
    """Return the SHA-256 of a text file, line endings normalised to LF.

    Same computation as the figure scripts, so that a Windows checkout gives
    the same digests.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str or None
        Hexadecimal digest, or None if the file does not exist (a module that
        was removed or renamed).
    """
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def stale_steps(
    entry: dict,
    repo_root: Path = REPO_ROOT,
    current_steps: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """Find the steps of a script whose figures are out of date or missing.

    Parameters
    ----------
    entry : dict
        Entry of the figure list.
    repo_root : pathlib.Path
        Repository root.
    current_steps : list of str or None
        Steps the script provides now, from ``--list``. When None, the list
        recorded in ``results.json`` by the last run is used instead.

    Returns
    -------
    tuple
        ``(stale, missing)``. ``stale`` maps each out of date step to the
        files that changed since it last ran. ``missing`` lists the steps
        that never ran or whose figures are gone; it is ``["*"]`` when there
        is no ``results.json`` at all.
    """
    outdir = repo_root / entry["outdir"]
    path = outdir / "results.json"
    if not path.is_file():
        return {}, ["*"]
    with open(path, encoding="utf-8") as fp:
        results = json.load(fp)
    steps = results.get("steps", {})
    known = current_steps if current_steps is not None else results.get("all_steps", [])
    missing = [s for s in known if s not in steps]
    stale = {}
    digests: dict[str, str | None] = {}  # each file hashed once
    for name, step in steps.items():
        if current_steps is not None and name not in current_steps:
            continue  # a step that no longer exists
        if any(not (outdir / f).is_file() for f in step.get("outputs", [])):
            missing.append(name)
            continue
        changed = []
        for f, digest in step.get("provenance", {}).items():
            if f not in digests:
                digests[f] = file_sha256(repo_root / f)
            if digests[f] != digest:
                changed.append(f)
        if changed:
            stale[name] = changed
    return stale, missing


def _script_env(repo_root: Path) -> dict:
    """Return the environment for a figure script: GridR from the tree, no display."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(repo_root / "python"), env.get("PYTHONPATH")) if p
    )
    env["MPLBACKEND"] = "Agg"
    return env


def list_steps(entry: dict, repo_root: Path = REPO_ROOT) -> list[str]:
    """Ask a figure script for the steps it provides now.

    Parameters
    ----------
    entry : dict
        Entry of the figure list.
    repo_root : pathlib.Path
        Repository root.

    Returns
    -------
    list of str
        Step names, in the order the script runs them.
    """
    done = subprocess.run(
        [sys.executable, str(repo_root / entry["script"]), "--list"],
        check=True,
        env=_script_env(repo_root),
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return done.stdout.split()


def run_script(
    entry: dict, only: list[str] | None = None, repo_root: Path = REPO_ROOT
) -> None:
    """Run a figure script on the repository tree.

    The script imports the GridR of the tree, not an installed one, as the
    documentation build does.

    Parameters
    ----------
    entry : dict
        Entry of the figure list.
    only : list of str or None
        Steps to run; None runs them all.
    repo_root : pathlib.Path
        Repository root.

    Raises
    ------
    subprocess.CalledProcessError
        If the script fails; its output is printed first, since it is
        captured.
    """
    cmd = [
        sys.executable,
        str(repo_root / entry["script"]),
        "--outdir",
        str(repo_root / entry["outdir"]),
    ]
    if only:
        cmd += ["--only", *only]
    try:
        subprocess.run(
            cmd,
            check=True,
            env=_script_env(repo_root),
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        raise


def _describe(steps: list[str]) -> str:
    """Name a few steps, or just count them when there are many."""
    return ", ".join(steps) if len(steps) <= 4 else f"{len(steps)} steps"


def build(
    mode: str = "check",
    names: list[str] | None = None,
    figures_list: Path = FIGURES_LIST,
    repo_root: Path = REPO_ROOT,
    log: Callable[[str], None] = print,
    warn: Callable[[str], None] = print,
) -> int:
    """Check or regenerate the figures of the documentation.

    Parameters
    ----------
    mode : str
        One of :data:`MODES`.
    names : list of str or None
        Entries of the figure list to consider; None takes them all.
    figures_list : pathlib.Path
        List of figure scripts.
    repo_root : pathlib.Path
        Repository root.
    log, warn : callable
        Where to send information and warnings; ``conf.py`` passes the Sphinx
        logger.

    Returns
    -------
    int
        Number of scripts whose figures are still out of date; always 0
        outside the ``check`` mode.
    """
    if mode not in MODES:
        raise ValueError(f"invalid figure build mode {mode!r}")
    outdated = 0
    for entry in load_figures_list(figures_list):
        if names and entry["name"] not in names:
            continue
        if mode == "all":
            log(f"[figures] {entry['name']}: running {entry['script']}")
            run_script(entry, repo_root=repo_root)
            continue
        if mode == "stale":
            # asking the script catches the steps added since the last run
            stale, missing = stale_steps(entry, repo_root, list_steps(entry, repo_root))
            if not stale and not missing:
                continue
            only = None if "*" in missing else sorted({*stale, *missing})
            what = "all steps" if only is None else _describe(only)
            log(f"[figures] {entry['name']}: running {entry['script']} ({what})")
            run_script(entry, only, repo_root)
            continue
        stale, missing = stale_steps(entry, repo_root)
        if not stale and not missing:
            continue
        outdated += 1
        if "*" in missing:
            detail = f"no results.json in {entry['outdir']}"
        else:
            parts = []
            if stale:
                changed = sorted({f for files in stale.values() for f in files})
                parts.append(
                    f"out of date: {_describe(sorted(stale))} "
                    f"(changed: {', '.join(changed)})"
                )
            if missing:
                parts.append(f"missing: {', '.join(sorted(missing))}")
            detail = "; ".join(parts)
        warn(
            f"[figures] {entry['name']}: {detail}. Regenerate them with "
            "'make build-doc-figures', or build the documentation with "
            "GRIDR_DOC_BUILD_FIGURES=stale."
        )
    return outdated


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.

    Parameters
    ----------
    argv : list of str or None
        Arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Exit status: 1 in ``check`` mode when figures are out of date, so
        that CI can fail on it.
    """
    parser = argparse.ArgumentParser(
        description="Check or regenerate the figures of the documentation."
    )
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument("--name", nargs="+", help="entries of figures_list.json")
    parser.add_argument("--figures-list", type=Path, default=FIGURES_LIST)
    args = parser.parse_args(argv)
    outdated = build(args.mode, args.name, args.figures_list)
    return 1 if outdated else 0


if __name__ == "__main__":
    sys.exit(main())
