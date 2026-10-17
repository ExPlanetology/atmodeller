Output
======

The `Output` class processes the solution to provide output, which can be in the form of a dictionary of arrays, Pandas dataframes, or an Excel file. The dictionary keys (or sheet names in the case of Excel output) provide a complete output of quantities.

Gas species
-----------

Species output have a dictionary key associated with the species name and its state of aggregation (e.g., CO2_g, H2_g).

All gas species
~~~~~~~~~~~~~~~

.. list-table:: Outputs for gas species
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - gas_mass
     - kg
     - Mass in the gas
   * - gas_number
     - mol
     - Number of moles in the gas
   * - gas_number_density
     - mol m\ :math:`^{-3}`
     - Number density in the gas
   * - dissolved_mass
     - kg
     - Mass dissolved in the melt
   * - dissolved_number
     - mol
     - Number of moles in the melt
   * - dissolved_ppmw
     - kg kg\ :math:`^{-1}` (ppm by weight)
     - Dissolved mass relative to melt mass
   * - fugacity
     - bar
     - Fugacity
   * - fugacity_coefficient
     - dimensionless
     - Fugacity relative to (partial) pressure
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - pressure
     - bar
     - Partial pressure
   * - total_mass
     - kg
     - Mass in all reservoirs
   * - total_number
     - mol
     - Number of moles in all reservoirs
   * - volume_mixing_ratio
     - mol mol\ :math:`^{-1}`
     - Volume mixing ratio in the gas
   * - gas_mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction in the gas

O2_g additional outputs
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Additional outputs for O2_g
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - log10dIW_1_bar
     - dimensionless
     - Log10 shift relative to the IW buffer at 1 bar
   * - log10dIW_P
     - dimensionless
     - Log10 shift relative to the IW buffer at the total pressure

Condensed species
-----------------

Species output have a dictionary key associated with the species name and its state of aggregation (e.g., H2O_l, S_cr).

.. list-table:: Outputs for condensed species
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - activity
     - dimensionless
     - Activity
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - total_mass
     - kg
     - Mass
   * - total_number
     - mol
     - Number of moles

Elements
--------

Element outputs have a dictionary key associated with the element name with an `element_` prefix (e.g., element_H, element_S).

.. list-table:: Outputs for elements
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - gas_mass
     - kg
     - Mass in the gas
   * - gas_number
     - mol
     - Number of moles in the gas
   * - gas_number_density
     - mol m\ :math:`^{-3}`
     - Number density in the gas
   * - condensed_mass
     - kg
     - Mass in condensed species
   * - condensed_number
     - mol
     - Number of moles in condensed species
   * - degree_of_condensation
     - dimensionless
     - Degree of condensation
   * - dissolved_mass
     - kg
     - Mass dissolved in the melt
   * - dissolved_number
     - mol
     - Number of moles in the melt
   * - logarithmic_abundance
     - dimensionless
     - Logarithmic abundance
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - total_mass
     - kg
     - Mass in all reservoirs
   * - total_number
     - mol
     - Number of moles in all reservoirs
   * - volume_mixing_ratio
     - mol mol\ :math:`^{-1}`
     - Volume mixing ratio
   * - gas_mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction in the gas

State
-----

The thermodynamic state output has a dictionary key of `state`. The exact set of outputs depends on the type of thermodynamic state being considered:

.. list-table:: Outputs for all thermodynamic states
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - temperature
     - K
     - Temperature
   * - pressure
     - bar
     - Pressure

For a planet, the thermodynamic state provides the following additional outputs:

.. list-table:: Planet-specific outputs
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - core_mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction of iron core relative to total planet mass
   * - mantle_mass
     - kg
     - Mass of the silicate mantle
   * - mantle_melt_fraction
     - kg kg\ :math:`^{-1}`
     - Fraction of silicate mantle that is molten
   * - mantle_melt_mass
     - kg
     - Mass of molten silicate
   * - mantle_solid_mass
     - kg
     - Mass of solid silicate
   * - planet_mass
     - kg
     - Total mass of the planet
   * - surface_area
     - m\ :math:`^2`
     - Surface area at the surface radius
   * - surface_gravity
     - m s\ :math:`^{-2}`
     - Gravitational acceleration at the surface radius
   * - surface_radius
     - m
     - Radius of the planetary surface

Instead, a generic thermodynamic state has the following additional outputs:

.. list-table:: Thermodynamic state-specific outputs
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - mass
     - kg
     - Mass of the condensed phase
   * - melt_fraction
     - kg kg\ :math:`^{-1}`
     - Fraction of the condensed phase that is molten
   * - melt_mass
     - kg
     - Mass of the molten condensed phase
   * - solid_mass
     - kg
     - Mass of the solid condensed phase

Gas phase (totals)
------------------

The gas phase output has a dictionary key of `gas`.

.. list-table:: Outputs for gas
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - species_number
     - mol
     - Number of moles of species
   * - species_number_density
     - mol m\ :math:`^{-3}`
     - Number density of species
   * - mass
     - kg
     - Mass
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass
   * - element_number
     - mol
     - Number of moles of elements
   * - element_number_density
     - mol m\ :math:`^{-3}`
     - Number density of elements
   * - volume
     - m\ :math:`^3`
     - Volume from the ideal gas law

Constraints
-----------

The constraints have a dictionary key of `constraints`.

.. list-table:: Outputs for constraints
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - [element]\_number
     - mol
     - Number of moles of "element"
   * - [element]\_mass
     - kg
     - Mass of "element"
   * - [species]\_fugacity
     - bar
     - Fugacity of species

Solver
------

The solver has a dictionary key of `solver`.

.. list-table:: Outputs for solver
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - solver_success
     - Boolean
     - Indicates whether the solver terminated successfully according to its internal convergence criteria
   * - steps
     - Integer
     - Number of iterations taken during the successful attempt
   * - attempts
     - Integer
     - Attempt on which the solution succeeded in a multistart procedure (``0`` if no attempt succeeded)
   * - converged
     - Boolean
     - Indicates whether the solution meets the objective-based convergence criteria (independent of solver status).
   * - success
     - Boolean
     - Indicates whether the solution is both converged and reported as successful by the solver. Rows where this is ``False`` are dropped by ``drop_unsuccessful_solves`` and highlighted in Excel output.

Other output
------------

- raw: Raw solution from the solver, i.e. number of moles and active stabilities
- residual: Residuals of the reaction network and mass balance