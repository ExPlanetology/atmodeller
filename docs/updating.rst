.. _UpdatingFile:

Updating a model
================

A model is compiled the first time it is solved, which takes seconds, and later solves of a model with the same structure take milliseconds. To solve the same model for other inputs, e.g. in a loop over the steps of an evolution, change it with its update methods rather than building it again. The example :doc:`examples/updating_models` shows this in practice, with timings.

What can be updated
-------------------

Each update method takes keyword arguments and returns a new model with the same structure, so the compiled solver is reused. The updates also work with traced values, inside ``jax.jit`` or ``jax.lax.scan``.

- ``model.update_state(...)`` changes the thermodynamic state:

  - for both states: ``temperature``, ``pressure``, ``background_silicate_molar_mass`` (for the silicate melt and solid) and ``background_metal_molar_mass``;
  - for a ``Planet``: ``planet_mass``, ``core_mass_fraction``, ``mantle_melt_fraction`` and ``surface_radius``, with the background masses of the core, melt and solid recomputed together;
  - for a ``ThermodynamicState``: ``mass``, ``melt_fraction`` and ``background_metal_mass``, with the background melt and solid masses recomputed together.

- ``model.update_constraints(...)`` changes the constraints:

  - ``mass_constraints``, keyed by element (e.g. ``{"H": mass}``), in the units of ``mass_units`` (``"mass"``, the default, or ``"moles"``) of that call;
  - the values of ``activity_constraints``, e.g. ``{"O2_g": IronWustiteBuffer(log10_shift=1.0)}``.

  A constraint is switched off with ``NaN`` and on with a finite value. A constraint cannot change its type, e.g. from a fixed fugacity to a buffer, which raises an error.

Updated values must broadcast to the shapes of the values they replace, e.g. a single value or one value per model of the batch; a value of another shape raises an error. A different number of models in the batch therefore needs a new model. Anything else, such as the species, their equations of state or solubility laws, or the solver settings, needs a new model, which is compiled again.

Performance in loops
--------------------

- **Warm starts.** When consecutive solves are close, pass the previous ``output.solution`` to ``model.solve`` as the initial guess. This skips the initializer and usually needs fewer solver steps.
- **Compile what is called at every step.** Outside ``jit``, :meth:`~atmodeller.output.Output.to_dict` computes every output quantity one operation at a time, which can take a large fraction of a second. Taking the quantities you need inside a function compiled with ``equinox.filter_jit`` computes only those, typically in well under a millisecond.
- **Compile the whole loop.** Writing the loop with ``jax.lax.scan`` and compiling it with ``equinox.filter_jit`` runs every step without returning to Python. Values that change between calls of the compiled function, e.g. a rate, must be passed as JAX arrays: ``filter_jit`` treats Python numbers as static, so each new number would compile the function again. A model that fails to solve never raises an error, inside or outside ``jit``, so check ``output.checked_solution.success``, e.g. by collecting it at each step of the scan.
- **Batch.** Solve many models together as a batch rather than one at a time.
- **Cache compilation.** Compilation can be cached across Python sessions with JAX's `persistent compilation cache <https://docs.jax.dev/en/latest/501/compilation-cache.html>`__, e.g. ``jax.config.update("jax_compilation_cache_dir", "jax_cache")``.

``equinox.filter_jit`` is used rather than ``jax.jit`` because the model and its output are `Equinox <https://docs.kidger.site/equinox/>`__ modules, which hold arrays together with other Python objects, such as the names of the species. ``jax.jit`` traces every argument as an array and fails on these objects, whereas ``filter_jit`` traces only the arrays.
