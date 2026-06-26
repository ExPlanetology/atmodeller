"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import sphinx.builders.latex.transforms

# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY: Path = Path(__file__).resolve().parent

# -- Project information -----------------------------------------------------

project = "Atmodeller"
copyright = "2024, Dan J. Bower"
author = "Dan J. Bower"

# The full version, including alpha/beta/rc tags
release = "1.0.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "jaxmod": ("https://jaxmod.readthedocs.io/en/latest/", None),
}

bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_logo = "logo200x200.png"
html_context = {
    "display_github": True,
    "github_user": "ExPlanetology",
    "github_repo": "atmodeller",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

numfig = True

bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

pygments_style = "sphinx"

autodoc_member_order = "bysource"

autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
    "inherited-members": True,
    "undoc-members": False,
    "private-members": True,
}

latex_logo = "logo.png"
# Disable the automatic inclusion of the Python module index in the LaTeX PDF output
latex_domain_indices = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#known-issues-and-workarounds


class DummyTransform(sphinx.builders.latex.transforms.BibliographyTransform):
    def run(self, **kwargs):
        pass


sphinx.builders.latex.transforms.BibliographyTransform = DummyTransform


# https://stackoverflow.com/questions/62398231/building-docs-fails-due-to-missing-pandoc
def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on the PATH, the installed
    # version will be used. Otherwise, we will download a copy of pandoc into docs/bin/ and add
    # that to our PATH.
    pandoc_dir: Path = DOCS_DIRECTORY / "bin"

    # Add dir containing pandoc binary to the PATH environment variable, if not already included
    if str(pandoc_dir) not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + str(pandoc_dir)
    pypandoc.ensure_pandoc_installed(
        targetfolder=str(pandoc_dir),
        delete_installer=True,
    )


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)


# This avoids a warning about unable to copy the examples notebook
# Path to the existing file in the build output
destination_path = Path("_build/html/examples.ipynb")

# Remove the file if it already exists
if destination_path.exists():
    destination_path.unlink()
