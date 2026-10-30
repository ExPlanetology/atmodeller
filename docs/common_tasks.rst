Common Tasks
============

If you extend the capabilities of *Atmodeller*, you are encouraged to contribute your modifications to the main repository. For more information, see the :ref:`Developer's Guide <DevelopersGuideFile>`.

.. note::
    *Atmodeller* is built on `JAX <https://docs.jax.dev/en/latest/>`_, so most code must be compatible with JAX throughout the codebase. The main exceptions are components related to pre-processing (e.g., assembly of matrices and vectors) and output generation. However, any data structures or containers that are passed into the JAX-based numerical engine must be JAX-compliant.

    While JAX code closely resembles NumPy, there are important differences---particularly around functional programming, immutability, and tracing. The easiest way to get started is to explore the relevant sub-package, review its structure and examples, and adapt the existing code to your needs. Numerous tutorials and resources explaining JAX are also available online.

    *Atmodeller* is designed to be highly modular, enabling each sub-package to be developed and tested independently. This modular architecture makes it easier to focus on specific functionality without needing to modify or execute the entire codebase.

.. _CommonTasksThermodynamicData:

Add thermodynamic data
----------------------

Thermodynamic data are stored in the file ``nasa_glenn_coefficients.txt``, located within the ``thermodata`` sub-package under the ``data`` directory. These data follow the formulation of :cite:t:`MZG02` (https://ntrs.nasa.gov/citations/20020085330), in which thermodynamic properties---heat capacity, enthalpy, and entropy---are expressed using polynomial coefficients. You can extend the range of available species in *Atmodeller* by adding data from :cite:t:`MZG02`. For consistency, *Atmodeller* adopts Hill notation when representing chemical formulae.

In some cases, :cite:t:`MZG02` may not include the species of interest, or it may lack coefficients for the required temperature range. In such situations, you must fit the thermodynamic data yourself for the species and temperature range in question, ensuring consistency with the standard state (1 bar), the reference temperature (298.15 K) and the definition of reference enthalpy, as discussed in the introduction of :cite:t:`MZG02`. Additional guidance is provided through the methods and docstrings in the ``thermodata.core`` and ``thermodata.glenn`` modules, including how to relate these data to JANAF tables :cite:p:`Cha98`.

Each row of the file gives the coefficients of one temperature range of an end-member, with the columns ``end_member,hill_formula,T_min,T_max,a1,...,a7,b1,b2``. The end-member is ``gas``, ``liquid`` or the name of a mineral, such as ``quartz-alpha``, which must refer to one formula only. Rather than editing the packaged file, you can also load your own file in the same format and create species from it by passing the database:

.. code-block:: python

    from atmodeller import ChemicalSpecies, DataBase, PurePhase

    db = DataBase.from_glenn_file("my_glenn_coefficients.txt")
    h2o_g = ChemicalSpecies.create_gas("H2O", database=db)
    h2o_l = ChemicalSpecies.from_end_member("liquid", formula="H2O", database=db)
    sio2 = PurePhase.from_formula("SiO2", database=db)

Without a database, species are created from the default database, ``DataBase.default()``, which is the packaged data with graphite given the volume of :cite:t:`Gustafson1986` and diamond added from the same assessment. The volumes make the Gibbs energies of graphite and diamond, and hence the equilibrium constant of any reaction involving them, depend on pressure; their activities are unchanged, as are gases and other condensates. To provide data from another source, create a ``DataBase`` (in the ``database`` module) with the data keyed by Hill formula and then end-member, e.g. ``DataBase({"H2O": {"gas": ..., "liquid": ...}})``. The packaged data without these changes are ``DataBase(glenn_properties)``, with ``glenn_properties`` from ``atmodeller.thermodata``.

You can add an end-member to a database, or replace an existing one, with ``set_end_member``. For example, diamond from the assessment of :cite:t:`Gustafson1986` can be added alongside graphite in a database of the packaged data. Diamond has a volume model, so graphite must be given the volume model of the same assessment, replacing the packaged graphite, which has none. Otherwise only diamond depends on pressure and the graphite-diamond transition is wrong above about 1 bar:

.. code-block:: python

    from atmodeller import ChemicalSpecies, DataBase
    from atmodeller.thermodata import glenn_properties
    from atmodeller.thermodata.gustafson import GRAPHITE_VOLUME_GUSTAFSON, diamond_gustafson

    db = DataBase(glenn_properties)
    graphite = db.get_end_member("C", "graphite")
    db.set_end_member("C", "graphite", graphite.with_volume(GRAPHITE_VOLUME_GUSTAFSON))
    db.set_end_member("C", "diamond", diamond_gustafson)
    c_graphite = ChemicalSpecies.from_end_member("graphite", database=db)  # named "C_graphite"
    c_diamond = ChemicalSpecies.from_end_member("diamond", database=db)  # named "C_diamond"

This is the default database, so you can start from ``DataBase.default()`` and add further end-members. Diamond (``diamond_gustafson`` in ``atmodeller.thermodata.gustafson``) has the Gibbs energy of graphite plus the Gibbs energy of diamond relative to graphite from the assessment of :cite:t:`Gustafson1986`, which is valid to 6000 K, and graphite and diamond take the volumes of the same assessment. An end-member that is replaced, such as graphite, is still stitched together with the other end-members of its formula by ``PurePhase.from_formula``, whereas an end-member that is added, such as diamond, can only be selected on its own.

Data with only a heat capacity model can be created with ``IntegratedThermodynamicProperties`` in ``atmodeller.thermodata``, which integrates the heat capacity from the enthalpy and entropy at the reference temperature, and added to a database with ``set_end_member``.

The temperature range of the data you add matters for condensates. By default, a pure condensate is not stable outside the temperature range of its data (``temperature_range()``): its Gibbs energy is held at its value at the bound, and a large penalty (``NO_DATA_GIBBS_OVER_RT`` in ``atmodeller.constants``) is added. This is because the range of a pure phase is often bounded by a phase boundary, such as a melting point. A species of a solution, such as the silicate melt, is instead extended with a constant heat capacity, and gas data are extrapolated in their own form. To change the extrapolation below or above the range, pass ``extrapolate_below`` or ``extrapolate_above`` (``"off"``, ``"data"`` or ``"constant_cp"``) to ``ChemicalSpecies.from_end_member`` or the ``PurePhase`` methods (see :ref:`species-extrapolation`).

Add solubility laws
-------------------

Solubility laws are encapsulated in the ``solubility`` sub-package, with the base class ``Solubility`` located in ``solubility.core``. To define a custom solubility model, inherit from the base class and implement the ``concentration`` method. Private modules (indicated by a leading underscore) separate the solubility laws based on speciation, which is determined by the elemental composition. The ``core`` module also includes concrete classes for commonly used solubility laws, such as the power law, which can be used directly without the need to implement a custom class. Once a new solubility law is added, it should be imported into ``solubility.library`` and added to the dictionary returned by ``get_solubility_models``, with a key that starts with the formula of the species (e.g. ``H2O_basalt_dixon95``). After that, the solubility law can be accessed and used.


Each solubility module in the main codebase is mirrored by a corresponding module in the *tests* suite, ensuring that implemented solubility laws produce the expected results. When you add a new solubility law, you should also add a corresponding test. Ideally, the test should be anchored to values reported in the original source, such as those found in a table or figure. If such reference values are unavailable, a well-defined anchor value may be used instead.

Add real gas EOS
----------------

Real gas equations of state (EOS) are organised within the ``eos`` sub-package. The base class, ``RealGas``, is defined in ``eos.core``. To implement a custom EOS, create a subclass and override the ``volume`` and ``volume_integral`` methods. Private modules (indicated by a leading underscore) group EOS formulations by source or study. The ``core`` module also includes utility classes that can serve as components or base classes when building new EOS implementations.

Some EOS formulations require solving a root-finding problem to determine the molar volume---for example, see ``eos._zhang_duan_base`` and the :ref:`nested solvers <NestedSolvers>`. When defining a new EOS class, you may optionally combine it with an ``ExperimentalCalibration`` (in ``atmodeller.sci_utils``) using ``CombinedRealGas.create`` to produce a bounded variant, which extrapolates smoothly beyond the calibrated range; see ``eos._holley`` for examples. Once finalised, the EOS should be added to the dictionary returned by the ``get_*_eos_models`` function of its module (e.g. ``get_holley_eos_models``), with a key formed from the Hill formula of the species and a suffix for the source (e.g. ``H3N_beattie_holley58``). These dictionaries are merged in ``eos.library`` into the dictionary returned by ``get_eos_models``.

Each EOS module in the main codebase is mirrored by a corresponding module in the *tests* suite, ensuring that implemented EOS produce the expected results. When you add a new EOS, you should also add a corresponding test. Ideally, the test should be anchored to values reported in the original source, such as those found in a table or figure. If such reference values are unavailable, a well-defined anchor value may be used instead.