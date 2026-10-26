.. _DevelopersGuideFile:

Developer's Guide
=================

Introduction
------------

Community development of *Atmodeller* is strongly encouraged, and bug fixes and new features are welcome. *Atmodeller* uses JAX so familiarise yourself with `How to think in JAX <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_ and `JAX - The Sharp Bits <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_, as well as other resources offered on the JAX site and the web in general. You are welcome to enquire about the reasoning behind the structure and design of the code with the development team.

Contributing
------------

Contributions follow the usual GitHub workflow:

1. For larger changes, open an `issue <https://github.com/ExPlanetology/atmodeller/issues>`__ first so the approach can be discussed.
2. Fork the repository to your own account (personal or organisational).
3. Make your changes on a branch, including tests and documentation where required (see below).
4. Run the tests and pre-commit, then open a pull request against the upstream repository and complete the pull request template.

A pull request can be merged once the automated checks pass and a maintainer has reviewed it. The checks are:

- **Tests**: all tests pass on Ubuntu (Python 3.11 and 3.13) and macOS (Python 3.13).
- **Code style and types**: the pre-commit checks pass (ruff linting and formatting, and pyright type checking).
- **Test coverage**: at least 80% of the lines that the pull request adds or changes in ``atmodeller/`` are covered by tests (see :ref:`Tests <DevelopmentTests>` below for running coverage locally).
- **Documentation**: if the pull request changes the documentation, the package, the notebooks or the dependencies, the documentation builds without warnings.

Don't let these checks put you off contributing. You don't need to get everything right before opening a pull request: open it anyway, even as a draft, and the development team is happy to help you add tests, fix failing checks and get your changes merged.

To report a bug or request a feature, please open an `issue <https://github.com/ExPlanetology/atmodeller/issues>`__. For questions, ideas, or to share how you are using *Atmodeller*, start a thread in `Discussions <https://github.com/ExPlanetology/atmodeller/discussions>`__. For bugs, include a minimal example that reproduces the problem, the versions of *Atmodeller* and JAX you are using, and any error message.

Under the Apache License 2.0 you are not required to make your changes public, but we encourage you to contribute them back so the whole community benefits. Contributions submitted by pull request are licensed under the same terms as the project.

Installation
------------

See :ref:`developer_install` for how to install *Atmodeller* and the extra dependencies required for the developer tools and documentation.
 
Pre-commit
----------

Run pre-commit before issuing a pull request to the main repository::

    pre-commit run --all-files

Documentation
-------------

.. note::
    Documentation should be consistent with the source code, so any changes to the source code should be reflected in changes to the documentation, if required.

The doc dependencies include a bundled copy of pandoc (via ``pypandoc-binary``), which is used if pandoc is not already installed. On Apple Silicon Macs the bundled copy is currently an x86_64 binary that requires Rosetta, so either install pandoc natively (``brew install pandoc``) or install Rosetta (``softwareupdate --install-rosetta``). When the doc dependencies have been installed the documentation can be compiled::

    cd docs
    sphinx-apidoc -f -o source ../atmodeller

To generate HTML documentation use the appropriate command for your operating system:

Mac/Linux::

    make html SPHINXOPTS="-W --keep-going"

Windows::

    .\make html

The ``-W --keep-going`` options turn warnings into errors, as in the documentation check for pull requests (on Windows, set ``SPHINXOPTS`` in the environment instead).

To generate PDF documentation, noting that a LaTeX distribution must be available on your system::

    make latexpdf

Documentation is built in the appropriately named subdirectory in ``_build``.

.. _DevelopmentTests:

Tests
-----

.. note::
    Tests should be consistent with the source code, so any changes to the source code should be reflected in changes to the test suite, if required.

You can confirm that all tests pass by navigating to the root directory of *Atmodeller* and running::
    
    pytest

The performance tests, which time the solver for batches of models, are slow and are skipped by default. To run them::

    pytest -m performance
    
Similarly, to test coverage::

    pytest --cov=atmodeller --cov-report=xml

To check the coverage of the lines you changed, as the pull request check does, compare with the ``main`` branch of the upstream repository (here the remote ``upstream``)::

    diff-cover coverage.xml --compare-branch=upstream/main --fail-under=80

The overall coverage is reported by Codecov, and shown by the badge in ``README.md``.

Add a corresponding test for new features that are developed.