Output
======

Solving a model returns an :class:`~atmodeller.output.Output` object, which provides the solution as
a nested dictionary of arrays (:meth:`~atmodeller.output.Output.to_dict`), a dictionary of Pandas
dataframes (:meth:`~atmodeller.output.Output.to_dataframes`), an Excel file
(:meth:`~atmodeller.output.Output.to_excel`), or a pickle file
(:meth:`~atmodeller.output.Output.to_pickle`). For a quick look at the output, use
:meth:`~atmodeller.output.Output.quick_look`.

All quantities are arrays with the batch dimension first, so a batch of models is returned in a
single output. For dataframes and Excel, each top-level dictionary key becomes a dataframe (or
sheet) with one row per model, and the nested keys are joined by ``.`` to form the column names
(e.g. ``species.partial_pressure.H2O_g`` in the ``gas`` dataframe). In Excel, rows of models that
were not solved successfully are highlighted.

Output formats
--------------

The same quantities are available in three formats, selected with ``output_format``:

.. list-table:: Output formats
   :widths: 20 40 40
   :header-rows: 1

   * - Format
     - Layout
     - Example: partial pressure of H2O_g
   * - ``named_arrays`` (default)
     - Grouped by phase, then by ``phase``, ``elements``, or ``species``, then by quantity, then by
       element or species name
     - ``out["gas"]["species"]["partial_pressure"]["H2O_g"]``
   * - ``elements_species``
     - Grouped by element or species name, then by phase, then by quantity. Phase-level quantities
       remain under ``out[<phase>]["phase"]``
     - ``out["H2O_g"]["gas"]["partial_pressure"]``
   * - ``natural``
     - As ``named_arrays``, but element and species quantities are 2-D arrays (batch, element or
       species) with the column order given by a ``names`` entry. Not available for dataframes
       or Excel
     - ``out["gas"]["species"]["partial_pressure"][:, i]`` where
       ``out["gas"]["species"]["names"][i] == "H2O_g"``

Phases
------

Each phase that contains at least one species has its own output. The phase keys are:

- ``gas``: the gas phase (atmosphere)
- ``silicate_melt``: species dissolved in the silicate melt
- ``silicate_solid``: species in the silicate solid
- ``metal``: species in the metal phase
- condensates: each condensed (pure) phase, keyed by its species name (e.g. ``C_graphite``, ``H2O_l``)

Phase-level outputs
~~~~~~~~~~~~~~~~~~~

Phase-level outputs are under ``out[<phase>]["phase"]`` in all formats.

.. list-table:: Outputs for each phase
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - background_mass
     - kg
     - Mass of the background (untracked, chemically inert) component of the phase
   * - background_molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass of the background component
   * - background_number_moles
     - mol
     - Number of moles of the background component
   * - elements_number_moles
     - mol
     - Total number of moles of elements (atoms) in the tracked species, excluding the background
   * - mass
     - kg
     - Total mass of the phase: the background mass plus the mass of species with
       ``include_in_phase_mass=True``
   * - molar_mass
     - kg mol\ :math:`^{-1}`
     - Molar mass of the phase
   * - species_number_moles
     - mol
     - Total number of moles of species, including the background
   * - species_to_phase_mass_ratio
     - kg kg\ :math:`^{-1}`
     - Mass of all tracked species relative to ``mass``. Less than 1 when there is background
       mass; greater than 1 if any species has ``include_in_phase_mass=False``

The gas phase has the following additional phase-level outputs:

.. list-table:: Additional outputs for the gas phase
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - pressure
     - bar
     - Total pressure
   * - volume
     - m\ :math:`^3`
     - Volume of the gas phase
   * - log10dIW_1_bar
     - dimensionless
     - :math:`\log_{10}` shift of the O\ :math:`_2` fugacity relative to the iron-wüstite buffer
       evaluated at 1 bar (``NaN`` if O\ :math:`_2` is not a gas species)
   * - log10dIW_P
     - dimensionless
     - :math:`\log_{10}` shift of the O\ :math:`_2` fugacity relative to the iron-wüstite buffer
       evaluated at the total pressure (``NaN`` if O\ :math:`_2` is not a gas species)
   * - metallicity_by_mass
     - kg kg\ :math:`^{-1}`
     - Mass fraction of metals (*Z*), where metals are all elements other than H and He
   * - metals_to_hydrogen
     - mol mol\ :math:`^{-1}`
     - Number of moles of metals (atoms) relative to H,
       :math:`N_\mathrm{M}/N_\mathrm{H} = \sum_\mathrm{X} N_\mathrm{X}/N_\mathrm{H}`. Only
       present if H is in the gas phase
   * - metals_to_hydrogen_by_mass
     - kg kg\ :math:`^{-1}`
     - Mass of metals relative to the mass of H (*Z/X*). For comparison, the present-day solar
       photospheric value is 0.0187 :cite:p:`AAG21`. Only present if H is in the gas phase
   * - M_H
     - dex
     - Metallicity relative to solar,
       :math:`[\mathrm{M}/\mathrm{H}] = \log_{10}\left[\left(\sum_\mathrm{X}
       N_\mathrm{X}/N_\mathrm{H}\right) / \left(\sum_\mathrm{X}
       N_\mathrm{X}/N_\mathrm{H}\right)_\odot\right]`, using the solar abundances of
       :cite:t:`AAG21`. Both sums are over the same metals: those in the gas phase that have a
       solar abundance, so metals that are not modelled do not contribute to the solar sum. For
       example, :math:`[\mathrm{M}/\mathrm{H}] = 2` is 100 times solar. ``NaN`` if there are no
       such metals. Only present if H is in the gas phase

Elements
--------

Element outputs are under ``out[<phase>]["elements"][<quantity>][<element>]`` (``named_arrays``) or
``out[<element>][<phase>][<quantity>]`` (``elements_species``). An element appears in a phase only
if at least one species in that phase contains it.

.. list-table:: Outputs for elements in each phase
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - mass
     - kg
     - Mass of the element in the phase
   * - number_moles
     - mol
     - Number of moles of the element (atoms) in the phase

The gas phase has the following additional element outputs, which are only present if H is in the
gas phase:

.. list-table:: Additional outputs for elements in the gas phase
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - log_epsilon
     - dex
     - Logarithmic abundance relative to H,
       :math:`\log \epsilon(\mathrm{X}) = \log_{10}(N_\mathrm{X}/N_\mathrm{H}) + 12`, where
       :math:`N` is the number of moles of each element (atoms) in the gas phase. By definition,
       :math:`\log \epsilon(\mathrm{H}) = 12`. An element with no moles is ``-inf``
   * - X_H
     - dex
     - Abundance relative to H normalised to solar,
       :math:`[\mathrm{X}/\mathrm{H}] =
       \log \epsilon(\mathrm{X}) - \log \epsilon(\mathrm{X})_\odot`,
       using the present-day solar photospheric abundances of :cite:t:`AAG21`
       (:data:`~atmodeller.sci_utils.solar`). For example, :math:`[\mathrm{C}/\mathrm{H}] = 2`
       is 100 times the solar C/H ratio. By definition, :math:`[\mathrm{H}/\mathrm{H}] = 0`.
       ``NaN`` for elements without a solar photospheric abundance

Species
-------

Species outputs are under ``out[<phase>]["species"][<quantity>][<species>]`` (``named_arrays``) or
``out[<species>][<phase>][<quantity>]`` (``elements_species``). Species names include a suffix for
the phase (e.g. ``H2O_g`` for gas, ``H2O_d`` for dissolved in the melt).

.. list-table:: Outputs for species in each phase
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - activity
     - dimensionless
     - Activity. For gas species this is the fugacity in bar, i.e. relative to a 1 bar standard
       state
   * - include_in_phase_mass
     - boolean
     - Whether the species mass is included in the phase ``mass``
   * - mass
     - kg
     - Mass
   * - mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction in the phase
   * - mole_fraction
     - mol mol\ :math:`^{-1}`
     - Mole fraction in the phase
   * - number_moles
     - mol
     - Number of moles

Gas species have the following additional output:

.. list-table:: Additional outputs for gas species
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - partial_pressure
     - bar
     - Partial pressure

Constraints
-----------

The imposed constraints are under the ``constraints`` key (``named_arrays`` and ``natural``), or
under ``out[<element or species>]["constraints"]`` (``elements_species``). Every element and
species is listed, and those without the constraint are ``NaN``.

.. list-table:: Outputs for constraints
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - elements.mass
     - kg
     - Imposed total mass of each element (mass constraints)
   * - elements.number_moles
     - mol
     - Imposed total number of moles of each element (mass constraints)
   * - species.activity
     - dimensionless
     - Imposed activity of each species, evaluated at the solution temperature and pressure
       (activity constraints). For gas species this is the fugacity in bar

State
-----

The thermodynamic state is under the ``state`` key. The outputs depend on the type of state. All
states include:

.. list-table:: Outputs for all thermodynamic states
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - temperature
     - K
     - Temperature
   * - pressure
     - bar
     - Total pressure

For a planet, the state has the following additional outputs:

.. list-table:: Planet-specific outputs
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - mantle_mass
     - kg
     - Mass of the silicate mantle (melt and solid)
   * - mantle_melt_fraction
     - kg kg\ :math:`^{-1}`
     - Fraction of the silicate mantle that is molten
   * - mantle_melt_mass
     - kg
     - Mass of the molten silicate mantle
   * - mantle_solid_mass
     - kg
     - Mass of the solid silicate mantle
   * - metallic_core_mass
     - kg
     - Mass of the metallic core
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

For a generic thermodynamic state (not a planet), the state has the following additional outputs:

.. list-table:: Generic thermodynamic state outputs
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - silicate_mass
     - kg
     - Mass of the silicate (melt and solid)
   * - silicate_melt_fraction
     - kg kg\ :math:`^{-1}`
     - Fraction of the silicate that is molten
   * - silicate_melt_mass
     - kg
     - Mass of the molten silicate
   * - silicate_solid_mass
     - kg
     - Mass of the solid silicate

.. _OutputSolver:

Solver
------

The solver statistics are under the ``solver`` key. A model is solved if ``success`` is true. This
is the flag used to drop unsuccessful solves (``drop_unsuccessful_solves``), to highlight rows in
Excel, and to count the successful models in the solver log.

.. list-table:: Outputs for solver
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - success
     - boolean
     - Whether the model is solved, i.e. an attempt passed both the objective-based convergence
       check and the solver status
   * - solver_success
     - boolean
     - Whether the solver terminated successfully according to its internal criteria. For a model
       that was not solved, this is the status of an attempt that failed the objective-based check
       or the solver status, so it can be true even if ``success`` is false
   * - steps
     - integer
     - Number of iterations taken during the successful attempt. Zero if the model was not solved
   * - attempts
     - integer
     - Attempt on which the model was solved (e.g. in a multistart procedure). Zero if no attempt
       succeeded

Solution and initial guess
--------------------------

The ``solution`` key contains the raw solution from the solver for each species: the natural
logarithm of the number of moles (``<species>``) and the natural logarithm of the stability
(``<species>_stability``).

The ``initial_guess`` key compares the solution with the initial guess the solver started from,
which is useful for diagnosing and improving the initial guess. This is the guess of the model's
initializer for models solved from a cold start, and the given initial guess otherwise (e.g. a
warm start from a previous solution):

- ``<species>_solution`` and ``<species>_stability_solution``: the solution, as in ``solution``
- ``<species>_guess`` and ``<species>_stability_guess``: the initial guess
- ``<species>_log_error`` and ``<species>_stability_log_error``: absolute difference between the
  guess and the solution (``NaN`` for inactive stabilities)
- ``log_error_mae`` and ``log_error_rmse``: mean absolute error and root-mean-square error of the
  guess across all solution quantities
