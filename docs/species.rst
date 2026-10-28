Species
=======

This page explains how species are named and how to select them from the thermodynamic data.

Terminology
-----------

*Atmodeller* follows the terminology of :cite:t:`HP11`:

- A **phase component** is an end-member with its own thermodynamic data, which fixes its formula: the ``gas``, the ``liquid``, or a mineral such as ``quartz-alpha`` or ``graphite``. Each species in *Atmodeller* is a phase component, or several phase components of one formula stitched together.
- A **phase** is the gas, the silicate melt, the silicate solid, the metal, or a pure phase such as graphite. A phase can be mixed, with several species and an activity model, or pure, with a single species of unity activity.
- A **component**, in the sense of the phase rule, is an element. The element balances are computed from the formulas of the species.

Names
-----

A species is named from the Hill formula of its formula and a suffix, ``<Hill formula>_<suffix>``. The Hill formula orders carbon first and hydrogen second, followed by the other elements alphabetically (all elements alphabetically if there is no carbon), so that every formula has one name. Names are built from the formula and suffix, never parsed.

.. list-table:: Suffixes
   :widths: 25 35 40
   :header-rows: 1

   * - Suffix
     - Species
     - Example
   * - ``g``
     - Gas
     - ``ChemicalSpecies.create_gas("SO2")`` is ``O2S_g``
   * - ``l``
     - Liquid
     - ``ChemicalSpecies.from_component("liquid", formula="H2O")`` is ``H2O_l``
   * - ``<mineral>``
     - One mineral
     - ``ChemicalSpecies.from_component("graphite")`` is ``C_graphite``
   * - ``cd``
     - All the condensed phase components of a formula, stitched together
     - ``PurePhase.from_formula("SiO2")`` is ``O2Si_cd``
   * - ``<name>``
     - A chosen set of phase components, stitched together
     - ``PurePhase.from_components(("quartz-alpha", "quartz-beta"), name="quartz")`` is ``O2Si_quartz``
   * - ``d``
     - Dissolved in the silicate melt
     - ``ReservoirSpecies.create_dissolved("H2O", ...)`` is ``H2O_d``

The same names are used for constraints and in the output, so ``"O2S_g"`` (not ``"SO2_g"``) is the key of an activity constraint on SO2. ``species_name`` gives the name of a species from any formula, e.g. ``species_name("SO2")`` is ``O2S_g`` and ``species_name("SiO2", "l")`` is ``O2Si_l``. Common formulas in Hill order are:

.. list-table:: Hill formulas
   :widths: 25 25 25 25
   :header-rows: 1

   * - Formula
     - Hill formula
     - Formula
     - Hill formula
   * - SO2
     - O2S
     - SiO2
     - O2Si
   * - NH3
     - H3N
     - SiC
     - CSi
   * - HCl
     - ClH
     - Si3N4
     - N4Si3
   * - H2SO4
     - H2O4S
     - NH4Cl
     - ClH4N

Selecting species
-----------------

Species are created from the default database, ``get_default_database()``, unless another is given with ``database=`` (see :ref:`Add thermodynamic data <CommonTasksThermodynamicData>`).

- **Gases**: ``ChemicalSpecies.create_gas("H2O")``. The activity defaults to an ideal gas, and can be a real gas equation of state.
- **One condensed phase component**: ``ChemicalSpecies.from_component(component, formula=...)``. A mineral fixes its formula, e.g. ``from_component("cristobalite-beta")``, whereas ``liquid`` is a phase component of many formulas, so the formula is given, e.g. ``from_component("liquid", formula="SiO2")``. The species can be an end-member of a mixed phase, such as liquid SiO2 in the silicate melt or liquid Si in the metal with an activity model, or the species of a pure phase.
- **A pure phase**: the methods of ``PurePhase`` create a phase with a single species, named as the species:

  - ``PurePhase.from_formula("SiO2")`` stitches together all the condensed phase components of the formula from the data file, here quartz to cristobalite to liquid, in order of temperature (``O2Si_cd``).
  - ``PurePhase.from_components(("quartz-alpha", "quartz-beta"), name="quartz")`` stitches together a chosen set (``O2Si_quartz``).
  - ``PurePhase.from_component("cristobalite-beta")`` selects one phase component.

To see what can be selected, with ``db = get_default_database()``, ``db.components_of("SiO2")`` gives the condensed phase components of a formula from the data file, in order of temperature, and ``db.phase_components(("C", "H", "O"))`` lists the phase components of every formula made of the given elements.

Stitching phase components together
-----------------------------------

The thermodynamic data are at 1 bar, and the temperature ranges of the condensed phase components of a formula give their stable sequence at 1 bar, so stitching them together gives the stable phase component at each temperature. Stitching requires that:

- the phase components are condensed, since a gas differs from a condensate in phase and standard state. To allow evaporation and condensation, include the gas and the condensate as separate species;
- the phase components are of one formula;
- their temperature ranges do not overlap or leave a gap;
- several phase components are all ones that the database was created with, such as the packaged data, and have no volume model, since a pressure dependence would move the transitions between them. A single phase component can be from anywhere and keeps any volume model, so ``PurePhase.from_formula("C")`` is graphite with its volume, whereas diamond, which is added to the default database with its own volume, is only selected on its own with ``from_component("diamond")``.

To let different phase components compete at a given temperature and pressure instead, such as graphite and diamond, include each as a separate pure phase.

Temperature range
-----------------

Outside the temperature range of their data, gases are extrapolated, whereas condensates are not stable by default, because the range of a condensed phase is often bounded by a phase boundary, such as a melting point. The same condensates can therefore be included at any temperature. To extrapolate the data of a condensate, pass ``extrapolate_below=True`` or ``extrapolate_above=True`` when creating it, e.g. ``PurePhase.from_component("ice", extrapolate_below=True)`` for ice below 200 K.

Output
------

Each pure phase has its own output, keyed by the name of its species, e.g. ``out["C_graphite"]`` (see :doc:`output`).
