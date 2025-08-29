# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os
from pathlib import Path
import json
import subprocess
import inspect
import shutil
from sphinx.util import logging

logger = logging.getLogger(__name__)

project = 'GridR'
copyright = '2025, Cnes'
author = 'Arnaud Kelbert'
release = '0.4.0'

sphinx_source_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(sphinx_source_path)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../python"))


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Pour les docstrings Google/NumPy
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    #"sphinxcontrib.apidoc",
    #"sphinxcontrib.rustdoc",  # Pour la doc Rust
]

mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

napoleon_google_docstring = True  # Pour activer le support Google Style
napoleon_numpy_docstring = True # Pour désactiver le support NumPy Style (si vous ne l'utilisez pas)

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
        }



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Use read the doc theme
html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "vcs_pageview_mode": "blob",
    "style_nav_header_background": "#2980B9",  # Header color
}

html_static_path = ['_static']
html_css_files = [
    'css/style.css',
]

# Automatic generation of api_python
#apidoc_module_dir = "../../python"
#apidoc_output_dir = "source/api_python"
#apidoc_excluded_paths = []
#apidoc_separate_modules = True

# Notebooks management
#-------------------------------------------------------------------------------
notebooks_in_path = os.path.join(sphinx_source_path, "..", "..", "notebooks")
notebooks_out_path = os.path.join(sphinx_source_path, "_notebooks", "generated")
nbconvert_config_path = os.path.join(sphinx_source_path, "nbconvert_config.py")

notebooks_list = json.load(open(Path(sphinx_source_path)/"notebooks_list.json"))
notebooks = [
        {
            "source": f"{notebooks_in_path}/{nb}",
            "md_output": f"{notebooks_out_path}/{nb.replace('.ipynb','.md')}",
            "img_dir": f"{notebooks_out_path}/{nb.replace('.ipynb','_files')}",
        } for nb in notebooks_list ]

def run_notebook_builds():
    for nb in notebooks:
        try:
            os.makedirs(os.path.dirname(nb["md_output"]), exist_ok=True)
            os.makedirs(nb["img_dir"], exist_ok=True)
            env = os.environ.copy()
            env["DOC_BUILD"] = "1"
            env["DOC_BUILD_FILES_OUTPUT_DIR_PATH"] = nb["img_dir"]
            env["DOC_BUILD_NOTEBOOK_OUTPUT_PATH"] = nb["md_output"]
            subprocess.run([
                "jupyter", "nbconvert",
                "--to", "markdown",
                "--execute", nb["source"],
                "--output", nb["md_output"],
                "--config", nbconvert_config_path,
                "--ExecutePreprocessor.kernel_name=python3",
            ], check=True, env=env, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("stdout:\n", e.stdout)
            print("stderr:\n", e.stderr)
            raise


def copy_notebook_images(app, exception):
    if exception is None:
        for nb in notebooks:
            src = nb["img_dir"]
            dst = Path(app.outdir) / "_notebooks" / "generated" / Path(src).name
        
            try:
                shutil.copytree(src, str(dst), dirs_exist_ok=True)
                logger.info("Copied notebook images to build folder.")
            except Exception as e:
                logger.warning(f"Could not copy notebook images: {e}")

def setup(app):
    app.connect("builder-inited", lambda app: run_notebook_builds())
    app.connect("build-finished", copy_notebook_images)
