# Contributing to Atmodeller

Community development of Atmodeller is strongly encouraged, and bug fixes and new features are welcome.

## Reporting bugs and asking for help

Please open an [issue](https://github.com/ExPlanetology/atmodeller/issues) to report a bug or request a feature. For questions, ideas, or to share how you are using Atmodeller, start a thread in [Discussions](https://github.com/ExPlanetology/atmodeller/discussions). For bugs, include a minimal example that reproduces the problem, the versions of Atmodeller and JAX you are using, and any error message.

## Submitting changes

1. For larger changes, open an [issue](https://github.com/ExPlanetology/atmodeller/issues) first so the approach can be discussed.
2. Fork the repository to your own account (personal or organisational).
3. Make your changes on a branch, including tests and documentation where required.
4. Run the tests and pre-commit, then open a pull request against the upstream repository and complete the pull request template.

The [Developer's Guide](https://atmodeller.readthedocs.io/en/latest/development.html) explains how to set up a development install and run the tests, pre-commit checks and documentation build.

## Checks for pull requests

A pull request can be merged once the automated checks pass and a maintainer has reviewed it. The checks are:

1. **Tests**: all tests pass on Ubuntu (Python 3.11 and 3.13) and macOS (Python 3.13). Run `pytest` locally.
2. **Code style and types**: the pre-commit checks pass (ruff linting and formatting, and pyright type checking). Run `pre-commit run --all-files` locally.
3. **Test coverage**: at least 80% of the lines that the pull request adds or changes in `atmodeller/` are covered by tests. Run `pytest --cov=atmodeller --cov-report=xml` and then `diff-cover coverage.xml --compare-branch=upstream/main --fail-under=80` locally, where `upstream` is the remote for this repository.
4. **Documentation**: if the pull request changes the documentation, the package, the notebooks or the dependencies, the documentation builds without warnings. Run `sphinx-apidoc -f -o docs/source atmodeller` and then `sphinx-build -W --keep-going -b html docs docs/_build/html` locally (the first command generates the API pages, which are not stored in the repository).

Codecov also comments on the pull request with the coverage of the changed lines and any change in overall coverage.

Don't let these checks put you off contributing. You don't need to get everything right before opening a pull request: open it anyway, even as a draft, and the development team is happy to help you add tests, fix failing checks and get your changes merged.

## Licence

Under the Apache License 2.0 you are not required to make your changes public, but we encourage you to contribute them back so the whole community benefits. Contributions submitted by pull request are licensed under the same terms as the project.
