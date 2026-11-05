# Changelog

All notable changes to *Atmodeller* are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each pull request adds its changes
under **Unreleased**, which becomes the next version when it is released.

## [Unreleased]

## [2.0.0] - 2026-11-05

Version 2 is a substantial rewrite of version 1, which was a prototype. The interface has changed
throughout, so code written for version 1 needs updating; see the documentation and the example
notebooks.

### Added

- Explicit phases (gas, silicate melt, silicate solid, metal and pure condensates), with mole
  fractions, activities and optional activity models, and background masses.
- A thermodynamic database that users can extend, with end-members selected by name and
  stitched together in order of temperature, and controlled extrapolation outside the range of
  the data.
- Real-gas equations of state that depend on the gas composition, and pressure-dependent
  condensed phases.
- An initializer that solves an ideal-gas equilibrium in the element potentials, and warm starts
  from a previous solution.
- A result code, residuals and the best attempt for each model, and checks that the constraints
  of a model match its components.
- Updating every input of a model, in a loop or within `jax.lax.scan`, without compiling again.

### Changed

- Released under the Apache 2.0 licence, instead of GPL-3.0-or-later.

### Deprecated

- `HeuristicInitializer`, to be removed in version 3. Use `ElementPotentialInitializer`, the
  default.

[Unreleased]: https://github.com/ExPlanetology/atmodeller/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/ExPlanetology/atmodeller/compare/v1.0.1...v2.0.0
