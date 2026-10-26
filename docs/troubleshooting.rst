.. _TroubleshootingFile:

Troubleshooting
===============

Pipeline integration
--------------------
:ref:`Installation <InstallationFile>` of *Atmodeller* is straightforward, but some package dependencies may conflict with others in a larger Python pipeline. In general, *Atmodeller* has relatively flexible requirements, though the JAX-related dependencies are typically pinned to ensure stability on older hardware whilst still taking advantage of the latest JAX features and performance improvements where possible (see :ref:`Dependency notes <DependencyNotes>`).

If you encounter compatibility issues when integrating *Atmodeller* into your pipeline, please ask in the `Discussions <https://github.com/ExPlanetology/atmodeller/discussions>`__ or open an `issue <https://github.com/ExPlanetology/atmodeller/issues>`__.

The solver fails
----------------

*Atmodeller* assembles and solves systems of non-linear equations, and is therefore subject to all the usual considerations that arise when dealing with such systems. Under the hood, it uses the JAX-based solver library `Optimistix <https://github.com/patrick-kidger/optimistix>`_, whose documentation is available `here <https://docs.kidger.site/optimistix>`_. Pay particular attention to the FAQ section. How *Atmodeller* uses it is described in :doc:`solver`.

Each model in a batch is solved independently, and ``success`` under the ``solver`` key of the output records which models were solved (see :ref:`Solver output <OutputSolver>`). Unsuccessful models are highlighted in the Excel output and can be removed from the exported output with ``drop_unsuccessful_solves=True``.

Because the solution process is compiled and optimised by JAX, it can be challenging to inspect the solver's internal workings without using `JAX debugging tools <https://docs.jax.dev/en/latest/debugging.html>`_. However, the nature of the failure can often provide insight into the underlying issue:

- A solution cannot be found
- The solver detects a NaN (not a number) or infinity in a function

A solution cannot be found
~~~~~~~~~~~~~~~~~~~~~~~~~~

- The initial guess may not be sufficient to reach convergence. Try modifying the initial guess---especially if you're iterating systematically over a range of parameters. In such cases, reusing the previous solution as the initial guess for the next step can improve convergence: pass ``output.solution`` from the previous solve to ``model.solve``. Rows of the initial guess that contain NaN are replaced by the initializer's guess.

- Allow more attempts. Models that fail are retried from a new initial guess up to ``max_starts`` times (default 10), and systems with condensates are first solved at a larger :math:`\tau`, controlled by ``tau_max`` and ``tau_num``. All of these are fields of ``SolverParameters``, which is passed to the model as ``solver_parameters`` (see :doc:`solver`).

- Ensure that a solution physically exists for the system you've constructed. While *Atmodeller* allows arbitrary user-defined constraints (e.g., fugacity, mass balance), this does not guarantee a valid solution. If no solution is found, it may be due to over-constraining or conflicting conditions. To diagnose this, simplify the system by reducing the number of species and/or constraints. If a solution can be found for a simpler case, gradually reintroduce complexity to identify when the system becomes unsolvable.

- *Atmodeller* uses a bounded solver, with upper and lower limits on the log number of moles and log stability set by ``SolverParameters`` (``log_number_moles_lower``, ``log_number_moles_upper``, ``log_stability_lower``, ``log_stability_upper``), whose defaults are in ``atmodeller.constants``. These bounds are generally generous, but it is still possible that a solution lies outside them---particularly when modelling large atmospheres in the absence of solubility, or trace species in strongly oxidised or reduced systems. If a species sits at a bound in an unconverged solution, widen that bound, e.g. ``SolverParameters(log_number_moles_lower=-600.0)``, and pass it to the model as ``solver_parameters``.

The solver detects a NaN
~~~~~~~~~~~~~~~~~~~~~~~~

- During solution, *Atmodeller* evaluates various functions involving thermodynamic data, solubility laws, and real gas equations of state. These functions are not guaranteed to return finite values across all inputs. Although the bounded solver helps mitigate this, NaNs may still arise if any function behaves poorly for the given inputs. In such cases, iteratively simplifying the system---removing or isolating specific solubility laws or EOS---can help identify the problematic function.

- Thermodynamic data in *Atmodeller* are typically valid between 200 K and 6000 K, though in some cases extend to 20,000 K. Outside these ranges the data are extrapolated, so the results become less reliable. Equations of state and solubility laws also have calibrated ranges of temperature and pressure, given in their docstrings.

Additional information
~~~~~~~~~~~~~~~~~~~~~~

- To see the solver log, including how many models were solved and how many attempts they needed, call ``atmodeller.debug_logger()`` before solving.

- To find where a NaN first appears, enable JAX's NaN checking before solving, with ``jax.config.update("jax_debug_nans", True)``. This slows down the calculation, so use it only for debugging.
