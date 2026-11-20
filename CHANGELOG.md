# Changelog

All notable changes to *Atmodeller* are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each pull request adds its changes
under **Unreleased**, which becomes the next version when it is released.

## [Unreleased]

### Added

- A check before solving that every mass constraint is positive: a mass constraint of zero or
  less cannot be solved, and now raises a `ValueError` that explains how to leave an element out.
  Inside a JIT-compiled function of your own, where the constraints are traced, such a model is
  not well posed (`is_valid_model`): its result is `invalid_model` and it is not retried.
- `CONDENSATE_ALLOCATION_FRACTION` in `atmodeller.initializer.core`, the fraction of its limiting
  element's budget that a condensate predicted or switched on as stable starts with (0.5),
  shared by both initializers. It is the default of
  `HeuristicInitializer.condensate_allocation_fraction`, which is unchanged.
- A temperature range for the heat capacity fits of Vassiliev and Taldrik (2021)
  (`VassilievHeatCapacity.temperature_range`, fields `temperature_min` and `temperature_max`):
  0.1 K to T_m, as in their Table 6 with T_m from their Table 9, i.e. 4600 K for carbon (the
  graphite triple point) and 1688 K for silicon. The fits now meet `HeatCapacityProtocol`, so they
  can be used with `IntegratedThermodynamicProperties`. Each fit's docstring notes where its data
  end.
- Registries that pair each library model with its catalogue information:
  `atmodeller.eos.get_eos_catalogue` and `atmodeller.solubility.get_solubility_catalogue` return
  a `CatalogueEntry` (the model and its `ModelInfo`: species, phase and the citation keys of its
  sources) per key. `get_eos_models` and `get_solubility_models` are derived from them and are
  unchanged. The model catalogue in the documentation is generated from this information, and a
  test checks that every library model has it, with sources in the bibliography.

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
- The extrapolation of thermodynamic data now works independently on each side of the
  temperature range. When one side was `"constant_cp"`, an `"off"` side was also extended with
  constant heat capacity instead of being held at its value at the bound (the penalty was still
  added). The library's phases set both sides the same, so this only affected a side set
  explicitly, e.g. `extrapolate_below="constant_cp"` on a pure liquid.
- The documentation of the N2 solubility law `N2_basalt_bernadou21`: its Equation 18 gives the
  mass fraction of N, and its N3- term (reduced conditions, below about IW-1.5) as published
  overestimates the authors' reduced experiments, by about 8 times at 0.9 kbar. The N2 term
  (oxidised conditions) reproduces their data to within a factor of 3. Its test is now anchored
  to their oxidised experiment AR1FeN at a realistic fO2.
- The model catalogue in the documentation: the calibrated pressure range of an EOS made of several
  calibrated segments now spans all of them (it showed only the first, e.g. 1-1000 bar for
  `H2_shi92`, whose calibration reaches 100 kbar); the H2-He mixtures of Chabrier and Debras are
  listed as such; the sources no longer include papers cited only for comparison; the melt of
  seven solubility laws is correct; and `NO_SOLUBILITY` is listed. `models.rst` now says that only
  pressure is bounded and that above the calibration the compressibility factor, not the volume,
  is held.
- The `info` sheet of the Excel output gives units for the initial-guess quantities
  (`initial_guess`, `log_error`, `log_error_mae` and `log_error_rmse`).
- Documentation and docstrings that did not match the code, among them: with one model solved,
  the failed models restart from its solution (`solver.rst`); the tau sweep runs only if a model
  fails at the final tau, and only for systems whose condensate stability is solved for
  (`troubleshooting.rst`); the solver statistics are logged by `output.solver_stats_to_logger()`;
  `update(core_mass_fraction=...)` can add a background core to a planet built without one; the
  planet mass does not include pure condensates; `species_number_moles` counts only the species
  with `include_in_phase_mass=True`; the units of the noble-gas solubility laws; and the EOS
  docstrings (e.g. the Chabrier tables are interpolated linearly, and the bounded models hold the
  compressibility factor above their calibration).

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
