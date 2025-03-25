# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

project = 'GridR'
copyright = '2025, Arnaud Kelbert'
author = 'Arnaud Kelbert'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../python"))


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Pour les docstrings Google/NumPy
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    #"sphinxcontrib.apidoc",
    #"sphinxcontrib.rustdoc",  # Pour la doc Rust
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # Utilisation du thème Read the Docs
html_theme_options = {
    "canonical_url": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "vcs_pageview_mode": "blob",
    "style_nav_header_background": "#2980B9",  # Couleur du header
}

html_static_path = ['_static']
html_css_files = [
    'css/style.css',  # Remplace par le nom de ton fichier CSS si différent
]

# Génération automatique de la doc Python
#apidoc_module_dir = "../../python"
#apidoc_output_dir = "source/api_python"
#apidoc_excluded_paths = []
#apidoc_separate_modules = True
