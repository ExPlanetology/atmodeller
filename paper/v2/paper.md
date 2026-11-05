---
# Use plain text in the title; do not use backticks for code formatting.
title: 'Atmodeller: A JAX-based Python package for thermochemical equilibrium of planetary interiors and atmospheres'
tags:
  - Python
  - JAX
  - planetary science
  - exoplanets
  - geochemistry
  - thermodynamics
  - chemical equilibrium
  - atmospheres
  - magma oceans
authors:
  - name: Dan J. Bower
    orcid: 0000-0002-0673-4860
    affiliation: "1"
  - name: Paolo A. Sossi
    orcid: 0000-0002-1462-1882
    affiliation: "1"
affiliations:
  - name: Institute of Geochemistry and Petrology, Department of Earth and Planetary Sciences, ETH Zurich, Clausiusstrasse 25, 8092 Zurich, Switzerland
    index: 1
    ror: 057rvn534
date: 25 October 2026
bibliography: paper.bib
---

# Summary

Rocky planets exchange volatiles between reservoirs such as the atmosphere and
the mantle, notably through volcanic outgassing. Gas species react, dissolve
into molten silicates, and may condense. At high pressures they behave
non-ideally. Modelling these processes requires a self-consistent treatment of
chemical equilibrium and mass balance. `Atmodeller` is a Python package built on
JAX [@jax2018github] that performs these calculations. It thus combines the
convenience of a high-level language with the modern features of a
high-performance computational framework. The package is modular: its
solubility laws, real-gas equations of state, and thermodynamic data can each be
used in isolation. Applications include the magma ocean stage of young planets,
volcanic outgassing on Earth and other Solar System bodies, and
interior--atmosphere coupling on exoplanets such as super-Earths and
sub-Neptunes [@Bower2025]. Because its engine is a general thermodynamic solver,
`Atmodeller` can also be used for laboratory geochemistry, such as the
equilibrium between gas and pure condensed phases in a gas-mixing furnace.

# Statement of need

Calculating how volatiles partition between a planet's interior and atmosphere
requires solving gas-phase speciation, condensate stability, and volatile
solubility in silicate melt together, often at pressures where gases are far
from ideal. Existing equilibrium codes typically address only parts of this
problem (see State of the field). `Atmodeller` solves them within one framework
and is designed for researchers in planetary science, exoplanet science, and
geochemistry who need such calculations, either standalone or as one component
of a larger model. A model can describe either a planet, specified by bulk
properties such as its mass and radius, or a normalised system at a given
pressure and temperature.

Thermodynamic data, solubility laws, and equations of state for planetary
conditions are often uncertain, calibrated over limited
pressure--temperature--composition ranges, and revised as new experiments become
available. Exploring how the choice of model affects the results therefore
requires a transparent and flexible framework. In `Atmodeller`, users assemble a
model from the species, solubility laws, and equations of state they require,
and the reaction network is derived automatically from the species, so models of
increasing complexity can be built with the same code. Each model is documented
with its original source and calibration range.

Propagating these uncertainties into the results calls for parameter sweeps,
Monte Carlo sampling, and Bayesian parameter estimation, which require code that
is fast, handles many conditions at once, and ideally provides derivatives for
gradient-based samplers such as Hamiltonian Monte Carlo. `Atmodeller` is
therefore built on JAX. Just-in-time compilation makes it fast enough to embed
within other frameworks, and vectorisation solves a batch of conditions in a
single call, avoiding the outer Python loops that often bottleneck even a fast
solver. Automatic differentiation provides derivatives of the equilibrium
solution with respect to its inputs.

# State of the field

Codes developed for stellar and planetary atmospheres, such as `TEA`
[@Blecic2016], `GGchem` [@Woitke2018], and `FastChem Cond` [@Kitzmann2024],
assume ideal gases; those that include condensates treat them as pure phases,
and none includes volatile dissolution in melt. `GGchem` and `FastChem Cond`
include hundreds of molecules and condensates. The general-purpose code `CEA`
[@Gordon1994], which makes the same assumptions, has been used for lava-world
atmospheres, where gases are approximately ideal [@Charnoz2023].

Vaporisation codes compute the vapour above a volatile-free silicate melt, also
assuming ideal gases. `MAGMA` [@Fegley1987; @Schaefer2004], long the reference
code, has never been released as open source. `VapoRock` [@Wolf2023] prescribes
the oxygen fugacity, so there is no mass balance between melt and gas, whereas
`LavAtmos` [@vanBuchem2023] solves for it by assuming congruent vaporisation.
`LavAtmos` 2.0 [@vanBuchem2025] adds C, H, N, S, and P through `FastChem`, but
includes neither volatile dissolution nor condensation.

Codes that couple volatiles to the melt are closer in scope. `EVo`
[@Liggins2022] models volcanic degassing in the C--O--H--S--N system along
decompression paths, with real-gas behaviour from pure-species fugacity
coefficients combined by ideal mixing. However, its species are hard-coded and
graphite is its only condensed phase. The global chemical equilibrium model
[@Schlichting2022; @Young2023; @Werlen2025] couples the atmosphere, silicate
melt, and metal of super-Earths and sub-Neptunes, but separate versions define
the species and reactions of each chemical system, gases and silicate melt are
ideal, and solutions are found from an ensemble of initial guesses [@Grimm2026].
Melt solubility models such as `VESIcal` [@Iacovino2021; @Wieser2022] compute
volatile saturation and degassing but not gas-phase speciation.

General-purpose frameworks such as `Reaktoro` [@Leal2015], `Cantera` [@Cantera],
and `GEMS` [@Kulik2013] provide comprehensive multiphase solvers with extensive
non-ideal models, including real-gas equations of state, but are not built
around the empirical solubility laws used for volatiles in planetary melts, nor
around planetary constraints such as a surface pressure set by the atmospheric
mass. Commercial packages such as `FactSage` and `Thermo-Calc` offer extensive
databases but are closed-source and require a paid licence.

No existing open-source code therefore combines real-gas equations of state,
volatile solubility in silicate melt, coupling to solid and metal reservoirs,
and automatic determination of species stability with batched, differentiable
solves. Extending an existing code was considered but is not practical:
differentiable, batched solves require every component, from thermodynamic data
to the nonlinear solver, to be expressed in a single array-programming
framework, so retrofitting codes written in C++ or Fortran, or built around
`scipy.optimize`, would amount to rewriting their core. `Atmodeller` was
therefore built natively in JAX. Its thermodynamic library is currently smaller
than those of some other codes but can be extended from standard thermochemical
tables.

# New features in version 2.0.0

`Atmodeller` v2.0.0 extends the capabilities of v1.0.0 [@Bower2025]:

1. **Thermodynamic phases.** The gas, silicate melt, silicate solid, metal, and
   pure condensed phases are represented explicitly. This enables mole fractions
   and activities to be computed self-consistently, with optional
   activity-coefficient models. A phase can include a background mass, such as a
   silicate mantle or an iron core, into which species dissolve.
2. **Solubility laws.** New laws include the empirical model of @MVC98 for
   H$_2$O in silicate melts, calibrated to 3 kbar.
3. **Non-ideal gas mixing.** Real-gas equations of state can depend on the gas
   composition, for example the mixture model of @ZD09 with Lorentz--Berthelot
   mixing rules, rather than treating each species as a pure gas.
4. **Pressure-dependent condensed phases.** The Gibbs energy of a condensed
   species can include a pressure correction [@HP98; @HP11]. This matters at the
   GPa pressures beneath thick envelopes, for example for the stability of
   graphite relative to diamond.
5. **A modular thermodynamic database.** Thermodynamic data are accessed through
   a database interface that users can extend with their own data. New species
   include graphite and diamond [@Gustafson1986].
6. **Planetary structure.** The surface pressure can be calculated from a
   scaling law for planets with thick envelopes [@Schlichting2022], the
   hydrostatic balance for a thin atmosphere, or set directly for laboratory
   conditions. Other relationships can be easily added.
7. **End-to-end compilation and differentiation.** The whole workflow, from
   initial guess to solution, can be JIT-compiled, with the root finding
   performed by `Optimistix` [@optimistix2024]. Batches of conditions are solved
   in a single vectorised call, and an automatic retry strategy improves
   robustness across wide parameter ranges. The equilibrium solution is
   differentiable with respect to the model inputs, such as temperature, element
   budgets, and imposed fugacities.
8. **Initial guess.** A new initialiser solves an ideal-gas equilibrium for the
   element potentials [@Gordon1994], including stable condensates and
   dissolution in the melt, with each element first solved separately as in
   `FastChem` [@Stock2018]. Fewer solver iterations and retries are then needed,
   which lowers the total run time. A previous solution can also be supplied as
   a warm start, for example when coupling to a time-stepping code.
9. **Performance.** Species-specific models are evaluated directly for each
   species, so only the models that are needed are evaluated, even under
   vectorisation. State updates are applied in a single pass, which reduces
   overhead when `Atmodeller` is called repeatedly from another code.
10. **Outputs.** All outputs can be computed within JIT-compiled code, so
    `Atmodeller` can be embedded in other frameworks without loss of
    performance. New outputs include quantities of astronomical interest, such
    as elemental abundances relative to solar and the metallicity of the gas.
11. **Licence.** `Atmodeller` is now released under the permissive Apache 2.0
    licence, replacing GPL-3.0-or-later, so that it can be incorporated into
    other software without imposing copyleft obligations.

# Software design

`Atmodeller` separates the physical and chemical components from the numerical
engine that solves for equilibrium. Components interact with the engine only
through small protocols, for example a solubility law returns a dissolved
concentration and an equation of state returns an activity, so they can be
added, replaced, or evaluated on their own without changing the solver. They are
represented with `Equinox` [@kidger2021equinox] as immutable classes that JAX
can transform directly, which keeps the physics readable. Each equation of
state, solubility law, activity model, and set of thermodynamic data is tested
in isolation against values from its original publication, such as tabulated
experimental data, published figures, and reference phase boundaries, so that a
result can be traced to the models and data that produced it.

General-purpose thermodynamic packages require the pressure, temperature, and
bulk composition to be specified directly. `Atmodeller` adds a lightweight
physics layer that accepts inputs in the form most natural to planetary
problems, converts them into a scaled chemical system for the solver, and
converts the solution back into physical quantities. This makes `Atmodeller`
accessible to researchers in planetary science and astronomy for whom detailed
chemistry is often not routine, since a model can be set up without first
learning the options of a general-purpose package.

Because equations of state, solubility laws, and activity models are assembled
from different sources and do not derive from a single self-consistent Gibbs
energy, a unique solution is not guaranteed in all cases. Although most systems
have a single solution, a solver may, as in any nonlinear root-finding problem,
occasionally converge to a second solution or find none. `Atmodeller` mitigates
this: amounts are solved in logarithmic form, convergence is judged from the
residual of the governing equations rather than from the solver's internal
status, failed solves are retried from perturbed initial guesses, and systems
with stability variables are solved by continuation.

JAX was chosen because models can be written in Python yet run as fast compiled
code that can also be vectorised and differentiated. Compilation takes time,
however, and can dominate the total run time of a small model that is solved
only once. A further cost is a discipline that shapes the design: functions must
be pure, static configuration must be kept apart from traced quantities,
branching that breaks or obscures gradients must be avoided, and array shapes
must be fixed at compile time. The last determines how phase stability is
treated. Many equilibrium codes add and remove condensates during the solve,
which changes the size of the system. `Atmodeller` instead includes every
candidate species throughout and assigns each a stability variable [@Leal2016],
so the system has a fixed size. It is larger, but it is compiled once, can be
vectorised over conditions with different stable phase assemblages, and can be
differentiated through. Because JAX handles compilation, `Atmodeller` itself is
pure Python, installs with `pip`, and runs on Linux, macOS, and Windows.

# Research impact statement

`Atmodeller` has already been used to study the atmospheres of rocky planets
[@Bower2025; @Seidler2026] and sub-Neptunes [@Bower2025; @Hakim2026], to
post-process the outcomes of planet formation models [@Polman2026], and has been
coupled to time-dependent models to explore the consequences of atmospheric loss
on rocky planets [@Cherubim2025]. Several of these studies were led by groups
outside the development team [@Cherubim2025; @Polman2026], and `Atmodeller` has
been adopted independently as an optional chemical-equilibrium backend in the
JAX-based retrieval code `Exo Skryer` [@Lee2026]. It has also been adopted as a
chemistry module in `PROTEUS` [@Lichtenberg2021; @Nicholls2024; @Nicholls2025],
a community-developed framework for modelling the coupled evolution of planetary
interiors and atmospheres. `Atmodeller` grew out of `CALLIOPE`, a prototype
volatile module that originated in @Bower2022 and @Sossi2023 and remains
available in `PROTEUS`.

Beyond these applications, `Atmodeller` brings to planetary science the
vectorised and differentiable modelling that JAX has made practical in other
fields. Most existing planetary models are built on established codes that lack
these capabilities, so coupling chemistry to other processes within a single
differentiable model has not been practical. `Atmodeller` provides a chemistry
component that such models can adopt directly, and its design offers a template
for new codes, supporting the move towards a new generation of differentiable
and high-performance models in Earth and planetary sciences.

# AI usage disclosure

Version 1 of `Atmodeller` was developed with limited, targeted use of ChatGPT
(OpenAI; GPT-4 and GPT-5 models) for specific parts of the codebase. With that
foundation established, AI coding agents, principally Claude Code (Anthropic),
have become an integral part of developing version 2, used to implement new
features, refactor existing code, and review the codebase, both to probe the
robustness of the solver and to check model coefficients against the original
publications. Throughout, the authors directed this work, setting the design,
scope, and physical assumptions of each change and steering the AI output to
keep the codebase as accessible and understandable as possible to its users, who
are primarily scientists and researchers. The authors reviewed every line of
AI-generated code and verified all changes against the test suite before
accepting them. Claude Code was also used to edit the text of this paper, which
the authors reviewed in full.

# Acknowledgements

`Atmodeller` was created as part of a SERI-funded ERC Starting grant '2ATMO'
granted to P. Sossi (Contract no. MB22.00033), with additional funding provided
through a Swiss National Science Foundation (SNSF) Eccellenza Professorship
(#203668).

`Atmodeller` has benefitted from the work of many in the chemical equilibrium
community, including the authors of the studies and codes cited above, both for
elucidating methodology and numerical techniques and for making their codes open
source and available for benchmarking and code comparison. We also thank the JAX
community, and Patrick Kidger and Johanna Haffner in particular, for their
sustained efforts in developing and supporting the JAX ecosystem on which
`Atmodeller` is built.

# References