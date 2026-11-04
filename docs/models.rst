.. _models:

Model catalogue
===============

These tables list the real-gas equations of state and the solubility laws that *Atmodeller*
provides, by the key with which they are selected. They are generated from the model registries
when the documentation is built, so they always match the installed version.

.. important::

   Real-gas equations of state and solubility laws are often calibrated against experiments, so
   each has a range of validity in temperature, pressure and, for solubility laws, melt
   composition. Conditions far outside that range force a model to be evaluated where it was never
   tested, and the results may then be meaningless.

   The solver may also step outside the range of validity on its way to a solution, even when the
   solution itself lies inside it, and this can affect whether a solution is found. The initial
   guess of the initializer mitigates this by starting close to the solution.

   Nevertheless, it is up to you to decide whether the equations of state and solubility laws you
   choose are valid for the problem at hand.

Real-gas equations of state
---------------------------

A real-gas equation of state (EOS) is selected by its key in :func:`~atmodeller.eos.get_eos_models`
and given to a gas species, e.g.

.. code-block:: python

    from atmodeller import ChemicalSpecies, get_eos_models

    eos_models = get_eos_models()
    H2O_g = ChemicalSpecies.create_gas("H2O", activity=eos_models["H2O_cork_holland98"])

Most models are bounded: they are used within their calibrated temperature and pressure range,
and outside it they are extended (an ideal gas at low pressure and a bounded volume at high
pressure). The calibrated range is given below; a model that is ``not bounded`` is used as it is
at all conditions. Models with ``_unbounded`` in their key are the same model without the bounds.

.. include:: _generated/models_eos.rst

Solubility laws
---------------

A solubility law is selected by its key in :func:`~atmodeller.solubility.get_solubility_models`
and given to a dissolved species, e.g.

.. code-block:: python

    from atmodeller import ReservoirSpecies, get_solubility_models

    solubility_models = get_solubility_models()
    H2O_d = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_basalt_dixon95"]
    )

Each law is calibrated for a melt composition and a range of conditions, which are given in its
source; outside them it is extrapolated.

.. include:: _generated/models_solubility.rst
