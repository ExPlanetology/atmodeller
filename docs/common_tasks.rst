Common Tasks
============

If you extend the capabilities of *Atmodeller*, you are encouraged to contribute your modifications to the main repository. For more information, see the :ref:`Developer's Guide <DevelopersGuideFile>`.

.. note::
    *Atmodeller* is built on `JAX <https://docs.jax.dev/en/latest/>`_, so most code must be compatible with JAX throughout the codebase. The main exceptions are components related to pre-processing (e.g., assembly of matrices and vectors) and output generation. However, any data structures or containers that are passed into the JAX-based numerical engine must be JAX-compliant.
    
    While JAX code closely resembles NumPy, there are important differences---particularly around functional programming, immutability, and tracing. The easiest way to get started is to explore the relevant sub-package, review its structure and examples, and adapt the existing code to your needs. Numerous tutorials and resources explaining JAX are also available online.
    
    *Atmodeller* is designed to be highly modular, enabling each sub-package to be developed and tested independently. This modular architecture makes it easier to focus on specific functionality without needing to modify or execute the entire codebase.

Add thermodynamic data
----------------------

Thermodynamic data are stored in the file `nasa_glenn_coefficients.txt`, located within the `thermodata` sub-package under the `data` directory. These data follow the formulation of :cite:t:`MZG02` (https://ntrs.nasa.gov/citations/20020085330), in which thermodynamic properties---heat capacity, enthalpy, and entropy---are expressed using polynomial coefficients. You can extend the range of available species in *Atmodeller* by adding data from :cite:t:`MZG02`. For consistency, *Atmodeller* adopts Hill notation when representing chemical formulae.

In some cases, :cite:t:`MZG02` may not include the species of interest, or it may lack coefficients for the required temperature range. In such situations, you must fit the thermodynamic data yourself for the species and temperature range in question, ensuring consistency with the standard state (1 bar), the reference temperature (273.15 K) and the definition of reference enthalpy, as discussed in the introduction of :cite:t:`MZG02`. Additional guidance is provided through the methods and docstrings in the ``thermodata.core`` and ``thermodata.janaf`` modules, including how to relate these data to JANAF tables :cite:p:`Cha98`.

Add solubility laws
-------------------

Solubility laws are encapsulated in the `solubility` sub-package, with the base class ``Solubility`` located in ``solubility.core``. To define a custom solubility model, inherit from the base class and implement the ``concentration`` method. Private modules (indicated by a leading underscore) separate the solubility laws based on speciation, which is determined by the elemental composition. The ``core`` module also includes concrete classes for commonly used solubility laws, such as the power law, which can be used directly without the need to implement a custom class. Once a new solubility law is added, it should be imported into the ``library`` and referenced in the dictionary returned by ``get_solubility_models``. After that, the solubility law can be accessed and used.


Each solubility module in the main codebase is mirrored by a corresponding module in the *tests* suite, ensuring that implemented solubility laws produce the expected results. When you add a new solubility law, you should also add a corresponding test. Ideally, the test should be anchored to values reported in the original source, such as those found in a table or figure. If such reference values are unavailable, a well-defined anchor value may be used instead.

Add real gas EOS
----------------

Real gas equations of state (EOS) are organised within the `eos` sub-package. The base class, ``RealGas``, is defined in ``eos.core``. To implement a custom EOS, create a subclass and override the ``volume`` and ``volume_integral`` methods. Private modules (indicated by a leading underscore) group EOS formulations by source or study. The ``core`` module also includes utility classes that can serve as components or base classes when building new EOS implementations.

Some EOS formulations require solving a root-finding problem to determine the molar volume---for example, see ``eos._zhang_duan``. When defining a new EOS class, you may optionally combine it with an ``ExperimentalCalibration`` to produce a bounded variant. Once finalised, the EOS should be registered in the local EOS dictionary within its module. This dictionary is then imported into ``library`` and merged into a master EOS dictionary, which can be accessed using ``get_eos_models``.

Each EOS module in the main codebase is mirrored by a corresponding module in the *tests* suite, ensuring that implemented EOS produce the expected results. When you add a new EOS, you should also add a corresponding test. Ideally, the test should be anchored to values reported in the original source, such as those found in a table or figure. If such reference values are unavailable, a well-defined anchor value may be used instead.