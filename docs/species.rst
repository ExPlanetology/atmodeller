Species
=======

This page explains how species are named and how to select them from the thermodynamic data.

Terminology
-----------

*Atmodeller* follows the terminology of :cite:t:`HP11`:

- An **end-member** has its own thermodynamic data, which fixes its formula, and is named as the ``gas``, the ``liquid``, or a mineral such as ``quartz-alpha`` or ``graphite``. Each species created from the thermodynamic data is an end-member, or several end-members of one formula stitched together, whereas a dissolved species follows a solubility law.
- A **phase** is the gas, the silicate melt, the silicate solid, the metal, or a pure phase such as graphite. A phase can be mixed, with several species and an activity model, or pure, with a single species, of unity activity by default.

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
     - ``ChemicalSpecies.from_end_member("liquid", formula="H2O")`` is ``H2O_l``
   * - ``<mineral>``
     - One mineral
     - ``ChemicalSpecies.from_end_member("graphite")`` is ``C_graphite``
   * - ``cd``
     - All the condensed end-members of a formula, stitched together
     - ``PurePhase.from_formula("SiO2")`` is ``O2Si_cd``
   * - ``s``
     - All the solid end-members of a formula, stitched together without the liquid
     - ``PurePhase.from_formula("SiO2", liquid=False)`` is ``O2Si_s``
   * - ``<name>``
     - A chosen set of end-members, stitched together
     - ``PurePhase.from_end_members(("quartz-alpha", "quartz-beta"), name="quartz")`` is ``O2Si_quartz``
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

Species are created from the default database, ``DataBase.default()``, unless another is given with ``database=`` (see :ref:`Add thermodynamic data <CommonTasksThermodynamicData>`).

- **Gases**: ``ChemicalSpecies.create_gas("H2O")``. The activity defaults to an ideal gas, and can be a real gas equation of state.
- **One condensed end-member**: ``ChemicalSpecies.from_end_member(end_member, formula=...)``. A mineral fixes its formula, e.g. ``from_end_member("cristobalite-beta")``, whereas ``liquid`` is an end-member of many formulas, so the formula is given, e.g. ``from_end_member("liquid", formula="SiO2")``. The species can be an end-member of a mixed phase, such as liquid SiO2 in the silicate melt or liquid Si in the metal with an activity model, or the species of a pure phase.
- **A pure phase**: the methods of ``PurePhase`` create a phase with a single species, named as the species:

  - ``PurePhase.from_formula("SiO2")`` stitches together the condensed end-members of the formula given by ``stitchable_end_members``, here quartz to cristobalite to liquid, in order of temperature (``O2Si_cd``). With ``liquid=False`` only the solids are stitched together (``O2Si_s``), so the condensate is not stable above the melting point, for example when liquid SiO2 is an end-member of the silicate melt instead.
  - ``PurePhase.from_end_members(("quartz-alpha", "quartz-beta"), name="quartz")`` stitches together a chosen set (``O2Si_quartz``).
  - ``PurePhase.from_end_member("cristobalite-beta")`` selects one end-member.

- **Dissolved species**: ``ReservoirSpecies.create_dissolved("H2O", solubility=...)``, which is ``H2O_d``. Its amount follows from a solubility law and the fugacity of its gas, not from thermodynamic data, so it is not created from a database.

To see what can be selected, with ``db = DataBase.default()``, ``db.stitchable_end_members("SiO2")`` gives the condensed end-members of a formula that can be stitched together, in order of temperature, and ``db.end_members(("C", "H", "O"))`` lists the end-members of every formula made of the given elements.

Placing species in phases
-------------------------

``ThermodynamicState.from_species`` and ``Planet.from_species`` take the species of each phase:

- ``gas_species``: gases only, from ``ChemicalSpecies.create_gas``;
- ``silicate_melt_species``: condensed species, from ``ChemicalSpecies.from_end_member``, and dissolved species, since the silicate melt is the solvent of the solubility laws;
- ``silicate_solid_species`` and ``metal_species``: condensed species only, with an activity model if needed;
- ``condensates``: pure phases, from the methods of ``PurePhase``.

A species given for the wrong phase raises an error that says how to create the right one, as do two species, or two pure phases, with the same name.

Stitching end-members together
-----------------------------------

The thermodynamic data are at 1 bar, and the temperature ranges of the condensed end-members of a formula give their stable sequence at 1 bar, so stitching them together gives the stable end-member at each temperature. Stitching requires that:

- the end-members are condensed, since a gas differs from a condensate in phase and standard state. To allow evaporation and condensation, include the gas and the condensate as separate species;
- the end-members are of one formula;
- their temperature ranges do not overlap or leave a gap;
- several end-members are all ones that the database was created with, such as the packaged data, and have no volume model, since a pressure dependence would move the transitions between them. A single end-member can be from anywhere and keeps any volume model, so ``PurePhase.from_formula("C")`` is graphite with its volume, whereas diamond, which is added to the default database with its own volume, is only selected on its own with ``from_end_member("diamond")``.

To let different end-members compete at a given temperature and pressure instead, such as graphite and diamond, include each as a separate pure phase.

.. _species-extrapolation:

Temperature range and extrapolation
-----------------------------------

The thermodynamic data of each end-member cover a temperature range (``temperature_range()``), which for a condensate is often bounded by a phase boundary, such as a melting point. Outside the range, each side of the data has its own extrapolation, set with ``extrapolate_below`` and ``extrapolate_above``:

- ``"off"``: the species is not stable there. Its Gibbs energy is held at its value at the bound and a large penalty (``NO_DATA_GIBBS_OVER_RT`` in ``atmodeller.constants``) is added.
- ``"data"``: the data are evaluated in their own form, e.g. the nearest NASA Glenn polynomial.
- ``"constant_cp"``: the heat capacity is held at its value at the bound :math:`T_b`, so that the enthalpy and entropy are continuous there:

  .. math::

     H(T) = H(T_b) + C_p(T_b)\,(T - T_b), \qquad S(T) = S(T_b) + C_p(T_b) \ln\frac{T}{T_b}

The data files are unchanged: the extension is computed from the data at the bound. Only the Gibbs energy at 1 bar is extended, and the pressure dependence of a volume model, :math:`\int V\,dP`, is evaluated at the temperature, as within the range.

The defaults follow the role of the species:

- **Gases** use ``"data"``.
- **Pure phases** use ``"off"``, because the range of a pure condensate is its stable range at 1 bar. SiO2 is liquid above its melting point and solid below it, so the same condensates can be included at any temperature. To extend a pure phase where the bound is only the end of the data, set the extrapolation on that side, e.g. ``PurePhase.from_end_member("ice", extrapolate_below="data")`` for ice below 200 K.
- **End-members of a mixed phase**, such as the silicate melt, the metal, or any ``CondensedPhase`` with several species, use ``"constant_cp"``. Unlike a pure phase, a mixture needs its end-members where they are metastable as pure phases: liquid SiO2 is an end-member of a basaltic melt at 1500 K, far below its melting point of 1996 K. Both sides are extended, so a solid end-member of a solid solution is likewise available above its melting point, and a liquid above the end of its data (usually 6000 K). The constant heat capacity keeps the extension well behaved for any form of the data.

The default is decided by the phase, so a species created with ``ChemicalSpecies.from_end_member`` and no extrapolation is ``"off"`` in a ``PurePhase`` and ``"constant_cp"`` in a ``CondensedPhase``. An extrapolation that is set explicitly is kept.

This allows mixtures to be modelled from end-members at any temperature, for example a melt of oxide liquids:

.. code-block:: python

    from atmodeller import ChemicalSpecies, Planet

    # Liquid end-members, whose data start at their melting points (1996, 3100 and 1652 K)
    melt_species = tuple(
        ChemicalSpecies.from_end_member("liquid", formula=formula)
        for formula in ("SiO2", "MgO", "FeO")
    )
    # In the silicate melt, they are extended to lower temperatures with a constant heat capacity
    planet = Planet.from_species(gas_species, silicate_melt_species=melt_species, temperature=1500)

The extension is not calibrated, so its accuracy decreases with the distance from the bound. An error :math:`\delta S` in the entropy of fusion gives an error of about :math:`\delta S\,(T_b - T)` in the Gibbs energy, which is significant for MgO and CaO at the temperatures of basaltic melts. Below the Kauzmann temperature of a liquid (about 1250 K for SiO2 and MgO), its extended entropy falls below that of the solid.

Activities in a mixed phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, a mixed phase is ideal: the activity of each species is its mole fraction, :math:`a_i = x_i`. A non-ideal model is set for each end-member with ``activity``, and receives the mole fractions of the phase, e.g. ``Si_iron_badro15`` for Si in liquid Fe. The interaction parameters of the end-members should be consistent with each other. Ideal mixing of oxide liquids is a poor approximation for silicate melts, whose mixing is strongly non-ideal, so a realistic melt needs an activity model.

The mole fractions, and hence the activities, are taken over the species that are included in the phase total, together with any background moles. Species that are not included in the phase total (``include_in_phase_mass=False``), e.g. dissolved volatiles in the dilute limit, do not dilute the end-members, so they do not change the mole fractions or activities of the end-members.

Mixed phases and solids
^^^^^^^^^^^^^^^^^^^^^^^

The silicate melt, silicate solid and metal are prescribed: they exist with the background mass that is given, and *Atmodeller* does not test whether a phase as a whole is stable, e.g. whether a melt should crystallise or not exist at all. The solution is the equilibrium given that these phases exist.

A solid can compete with the end-members of a mixed phase if it is included as a pure phase. For example, with liquid SiO2 in the melt and cristobalite as a pure phase, the reaction :math:`\mathrm{SiO_2(melt)} \rightleftharpoons \mathrm{SiO_2(cristobalite)}` divides SiO2 between them:

- If the melt is undersaturated, all the SiO2 is in the melt and cristobalite is absent.
- If the melt is saturated, cristobalite takes the excess and the activity of SiO2 in the melt is held at saturation, :math:`\ln a_\mathrm{SiO_2} = -\Delta G_\mathrm{fus}(T)/RT`, where :math:`\Delta G_\mathrm{fus}` is the Gibbs energy of the liquid end-member relative to the solid. For ideal mixing, this gives the mole fraction of SiO2 at the liquidus, :math:`x_\mathrm{SiO_2} = \exp(-\Delta G_\mathrm{fus}/RT)`. Below the melting point, the liquid is extended with a constant heat capacity, so the extension sets where the solid saturates.

The stability :math:`\sigma` of an absent solid measures how undersaturated the melt is in it, :math:`\sigma = (\mu^\circ_\mathrm{solid} - \mu_\mathrm{melt})/RT`, so a small value means that the solid is close to saturating. Its natural logarithm is ``<species>_stability`` in the solution output (see :doc:`output`). Only the solids that are included can saturate, and the existence and mass of the melt remain inputs.

Output
------

Each pure phase has its own output, keyed by the name of its species, e.g. ``out["C_graphite"]`` (see :doc:`output`).
