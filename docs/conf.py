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
import importlib.metadata
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("."))

import pypandoc
import sphinx.builders.latex.transforms

# -- Project information -----------------------------------------------------

project = "Atmodeller"
copyright = "2024-2026, Dan J. Bower and Paolo A. Sossi"
author = "Dan J. Bower"

# The full version, including alpha/beta/rc tags, from the installed package (pyproject.toml)
release = importlib.metadata.version("atmodeller")


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
    "myst_parser",
]

intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "optimistix": ("https://docs.kidger.site/optimistix/", None),
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
html_logo = "logo/logo200x200.png"
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
    "private-members": False,
}

latex_logo = "logo/logo800x800.png"
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

# For development where the notebooks may (temporarily) break, set to True
nbsphinx_allow_errors = False

# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#known-issues-and-workarounds


class DummyTransform(sphinx.builders.latex.transforms.BibliographyTransform):
    def run(self, **kwargs):
        pass


sphinx.builders.latex.transforms.BibliographyTransform = DummyTransform

# nbsphinx requires pandoc on the PATH. Prefer a system pandoc if one is available, otherwise use
# the copy bundled with pypandoc-binary (docs extra). Note that pypandoc-binary's macOS arm64 wheels
# currently bundle an x86_64 pandoc, which only runs on Apple Silicon if Rosetta is installed.
if shutil.which("pandoc") is None:
    try:
        pandoc_path: str = pypandoc.get_pandoc_path()
    except OSError as exc:
        raise RuntimeError(
            "No working pandoc found. Install pandoc (e.g. `brew install pandoc` on macOS), or on "
            "Apple Silicon install Rosetta (`softwareupdate --install-rosetta`) to run the copy "
            "bundled with pypandoc-binary."
        ) from exc
    os.environ["PATH"] = os.path.dirname(pandoc_path) + os.pathsep + os.environ["PATH"]


def _skip_equinox_enumeration_members(app, what, name, obj, skip, options):
    """Skips the members that enumerations inherit from Equinox, such as ``RESULTS.promote``,
    whose docstrings are Markdown rather than reStructuredText"""
    if getattr(obj, "__module__", None) == "equinox._enum":
        return True
    return None


def _generate_model_catalogue(app):
    """Writes the tables of the model catalogue (models.rst) from the model registries"""
    from pathlib import Path

    import model_catalogue

    model_catalogue.generate(Path(app.srcdir) / "_generated")


def setup(app):
    app.connect("autodoc-skip-member", _skip_equinox_enumeration_members)
    app.connect("builder-inited", _generate_model_catalogue)
