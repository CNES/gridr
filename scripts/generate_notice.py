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
This python module can be used to generate the NOTICE file.

The python 3rd party dependancies informations are retrieved using the 
piplicenses_lib. The module have a simple workaround mechanism in case of 
missing license information.

The rust 3rd party dependancies informations are retrieved using the cargo-about
rust utility. This utility is wrapped in the generate_rust_notice.sh script.

This modules uses templates located in the project `templates` directory.
"""
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Tuple, Optional

import piplicenses_lib

# Templates root path
TEMPLATES_PATH = Path(__file__).parent.parent / "templates"
# The python 3rd party template
PYTHON_TEMPLATE_PATH = TEMPLATES_PATH / "NOTICE_python_3rdparty.template"
# The notice template
NOTICE_TEMPLATE_PATH = TEMPLATES_PATH / "NOTICE.template"
# Path to the script to generate the rust 3rd party section
GENERATE_RUST_NOTICE_SCRIPT_PATH = Path(__file__).parent / "generate_rust_notice.sh"
# Python packages to ignore
IGNORE = ['gridr', 'pip', 'pip-licenses', 'pip-licenses-lib']
# Workarounds if License are not automatically found
PY_WORKAROUNDS = {
    'pyparsing': 'MIT License',
    'attrs': 'MIT License',
    'click': 'BSD-3-Clause License',
    }

def execute_shell_script(
        script_path: str,
        args: Optional[list] = None,
        shell: bool = False,
        timeout: Optional[int] = None
    ) -> Tuple[str, str, int]:
    """
    Executes a shell script and returns its output, error, and return code.

    Args:
        script_path: Path to the shell script
        args: List of arguments for the script
        shell: If True, executes via a shell (less secure)
        timeout: Maximum execution time in seconds

    Returns:
        Tuple containing (stdout, stderr, returncode)

    Raises:
        subprocess.TimeoutExpired: If the script exceeds timeout
        subprocess.CalledProcessError: If the script returns a non-zero error code
    """
    # Prepare arguments
    cmd = [script_path]
    if args:
        cmd.extend(args)

    # Execute the script
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode

    except subprocess.TimeoutExpired as e:
        print(f"Script timed out after {timeout} seconds")
        raise

    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise

    except FileNotFoundError:
        print(f"Script {script_path} not found or not executable")
        raise

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

def py_workarounds(name, info):
    if not info["license"]:
        try:
            return PY_WORKAROUNDS[name]
        except KeyError:
            if len(info["license_classifiers"]) == 0:
                return info["license"]
            else:
                return ' '.join(info["license_classifiers"])
    else:
        if len(info["license_classifiers"]) == 0:
            return info["license"]
        else:
            return ' '.join(info["license_classifiers"])

def notice_python_pckg_info(template: str, pckg_info: dict) -> str:
    """Create the notice part for the input pckg_info using template
    """
    values = {"name": pckg_info["name"],
            "version": pckg_info["version"],
            "author": pckg_info["author"],
            "repository": pckg_info["homepage"],
            "license": py_workarounds(pckg_info["name"], pckg_info),}
    ret = template.format(**values)
    return ret
    
def notice_python_3rdparties(template_file: str) -> str:
    """Create the python 3rd parties section using template.
    """
    info_str = []
    with open(template_file, "r") as fp:
        template = fp.read()
        for pckg_info in piplicenses_lib.get_packages(from_source="mixed"):
            if not pckg_info.__dict__["name"] in IGNORE:
                info_str.append(notice_python_pckg_info(template, pckg_info.__dict__))
    return '\n'.join(info_str)

def notice_rust_3rdparties() -> str:
    stdout, _, _ = execute_shell_script(GENERATE_RUST_NOTICE_SCRIPT_PATH)
    stdout = stdout.replace('&lt;', '<').replace('&gt;','>').strip()
    _, _, info_str = stdout.partition('[rust-deps-licenses]')
    return info_str
    
def generate_notice():
    with open(NOTICE_TEMPLATE_PATH, 'r') as fp:
        notice_template = fp.read()
        repl = {'rust_3rdparty_licenses': notice_rust_3rdparties(),
            'python_3rdparty_licenses': notice_python_3rdparties(PYTHON_TEMPLATE_PATH)}
        notice = notice_template.format(**repl)
        print(notice)

if __name__ == '__main__':
    generate_notice()