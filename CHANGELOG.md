# Changelog

All notable changes to *Atmodeller* are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each pull request adds its changes
under **Unreleased**, which becomes the next version when it is released.

## [Unreleased]

### Added

- A check before solving that every mass constraint is positive: a mass constraint of zero or
  less cannot be solved, and now raises a `ValueError` that explains how to leave an element out.
- `CONDENSATE_ALLOCATION_FRACTION` in `atmodeller.initializer.core`, the fraction of its limiting
  element's budget that a condensate predicted or switched on as stable starts with (0.5),
  shared by both initializers. It is the default of
  `HeuristicInitializer.condensate_allocation_fraction`, which is unchanged.

### Changed

- `ElementPotentialInitializer` starts each gas and dissolved species at its share of the budget
  of its limiting element, instead of an even split of the total over the gas species, and starts
  an element without a mass constraint from its imposed fugacity. Its element sweep counts the
  dissolved species, which follow the element potentials through their solubility laws, so it
  places each element between the gas and the melt. This solves systems with a large melt
  reservoir and a skewed inventory (e.g. a primordial H/He atmosphere with trace C, N, O and S),
  where the cold guess failed, and halves the solver steps on such systems. The initial guess
  can differ slightly from version 2.0.0.
- `ElementPotentialInitializer` defaults to looser sweep tolerances, `sweep_step_tolerance` of
  `1e-4` (was `1e-10`) and `sweep_tolerance` of `1e-2` (was `1e-8`): the sweep only sets the
  scale of every species before the Newton step, which converges the system. A system without
  condensates is solved once, without a switching pass. The sweep keeps the log moles of the
  dissolved species within the solver's bounds (`log_number_moles_lower` and
  `log_number_moles_upper` of `SolverParameters`).

### Fixed

- A solubility law that depends on the oxygen fugacity (the N2 laws, and the S2 laws of Boulliung
  and Wood) now requires O2 as a gas species, which is checked when the model is built and raises
  a `ValueError`. Before, a missing O2 was caught by a runtime check that was off by default, so
  the model failed without explanation, and with `EQX_ON_ERROR` set to `nan` or `raise` the check
  also fired for a diverged model in a batch and acted on every model of the batch: models that
  solve on their own were reported as not solved. The fO2 can still be imposed in some models of
  a batch and solved for in others. Laws flag the dependence with the class attribute
  `Solubility.requires_fO2`.

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
