.. _SolverFile:

How the solver works
====================
Most of the time you can treat *Atmodeller*'s solver as a black box: you describe the planet and the chemistry, and it returns the equilibrium state. It is still worth knowing roughly what it does, because when a calculation fails to converge, this is the quickest way to see why and what to change.

*Atmodeller* keeps the physics and chemistry (thermodynamic data, equations of state, solubility laws) separate from the numerical engine that solves the resulting equations. The same engine therefore handles any combination of these components. Like most non-linear problems, thermochemical equilibrium cannot be solved in one step. The solver starts from an initial guess and refines it iteratively until the equations are satisfied, and it converges reliably only if that guess is reasonably close to the answer.

This page follows the solution process in order: how the equations are built, how the initial guess is made, and how the solver refines it. For practical advice when a solve fails, see :doc:`troubleshooting`.

Equations
---------
*Atmodeller* turns the question "what is the equilibrium composition?" into a set of non-linear equations, built from two principles: mass balance (atoms are neither created nor destroyed) and chemical equilibrium (each reaction satisfies its equilibrium constant).

The unknowns are the natural logarithms of the number of moles of each species. Working in logs keeps every amount positive, handles species whose abundances differ by hundreds of orders of magnitude, and turns the equilibrium-constant equations into simple sums. Each condensate (e.g. graphite or liquid water) also has a *stability* variable: it stays near zero when the condensate is present and grows large when it is not, which switches the condensate off. This lets condensates appear and disappear as conditions change, without the user deciding in advance which phases are present.

What makes the problem harder is that the parts of the model are coupled. A real-gas equation of state can depend on the gas composition as well as on temperature and pressure, and the amount of a gas dissolved in magma depends on its partial pressure. Equations that are simple for a single ideal gas therefore become interdependent once these effects are included, and must be solved together.

Initializer
-----------
*Atmodeller* makes the initial guess with an *initializer*: a simplified version of the problem (pure condensates, and an ideal gas speciation so that the Gibbs energies of the gases are held fixed) that can be solved quickly and reliably.

Composing an initializer is something of a dark art. It must be simplified enough to solve robustly from almost no information, yet honour enough of the necessary complexity to produce a meaningful initial guess for the main solver. For example, nonideality has a strong control on dissolution, because solubility depends on fugacity rather than partial pressure. *Atmodeller*'s initializer therefore treats the gas speciation as ideal, but evaluates the solubility laws with the real-gas fugacities, so that the amount dissolved in the melt, which can be a large share of the volatile budget, is already close to the answer.

The initializer uses element potentials. The composition of a system is specified by its element abundances (e.g. total H, C and O), not by its species, and there are usually far fewer elements than species. It is therefore attractive to solve for one variable per element instead of one per species. At equilibrium each element has an *element potential*, and these are the building blocks of the species potentials: the potential of a species is the sum of the potentials of the elements it contains, weighted by its stoichiometry. Once the element potentials are known, the amount of every gas species follows directly. This element potential method is standard in chemical equilibrium codes :cite:p:`Gordon1994`.

1. **Start.** Every condensate starts absent and dissolution in the melt is not yet considered, so all of the elements are in the gas phase. The initializer spreads the total element moles evenly over the gas species as a first rough guess, and fits element potentials to that guess.

2. **Condensate selection.** A loop then brings in dissolution and decides which condensates are stable, switching condensates on one at a time and off as their amounts vanish:

   1. **Sweep.** Visit each mass-constrained element in turn, from most to least abundant, and adjust its element potential so that its own mass balance is satisfied exactly, holding the others fixed, as in FastChem :cite:p:`Stock2018`. Proceeding element by element reduces each step to a one-variable problem with a guaranteed single solution, and keeps minor species that Newton alone can lose when another species dominates their element.
   2. **Newton.** Refine all the unknowns together with damped Newton steps until the whole simplified system is satisfied: mass balance, pressure, any imposed fugacities, the present condensates and dissolution in the melt.
   3. **Update condensates.** Re-evaluate the Gibbs energies of the condensates at the new pressure, since which condensate is stable can depend on pressure (e.g. graphite versus diamond).
   4. **Switch.** Switch on the most supersaturated condensate, starting it at half the budget of its limiting element, and switch off any present condensate whose amount has dropped to trace level.
   5. **Repeat** for a fixed number of passes, enough for each condensate to be switched on and off once, then solve one final time.

3. **Hand-off.** Absent condensates are given a stability from how undersaturated they are, so the result is in the form the full solver expects.

Main solver
-----------
The main solver starts from the initializer's result and drops its simplifications: it brings back the real-gas equations of state in the gas speciation and the coupling between the parts of the model described in `Equations`_, and solves the complete system together. It uses Newton's method from Optimistix :cite:p:`optimistix2024`, with the Jacobian from automatic differentiation and full Newton steps clipped to fixed bounds on every unknown (see :doc:`troubleshooting`). A solve counts as converged only if Newton reports success *and* the largest residual of the equations is below the tolerance (``atol``, default :math:`10^{-6}`).

1. **Solve and retry.** All conditions in the batch are solved together from their initial guesses. Conditions that fail are given a new guess and solved again, up to ``max_starts`` attempts in total (default 10), while converged conditions are kept as they are. The new guess is drawn at random around the conditions that did converge: each log number of moles is sampled uniformly over the range spanned by the converged solutions, centred on their median, and the stabilities are then recomputed from the new guess.

2. **Condensates.** The stability of a condensate is scaled by a small factor :math:`\tau` (default :math:`10^{-25}`), which sets how many moles remain of a condensate that is switched off :cite:p:`LKK16`. A very small :math:`\tau` makes the switch between present and absent sharp, so the equations change steeply near it and Newton can fail from a poor guess. If any condition fails at the default :math:`\tau`, every condition is first solved at a much larger :math:`\tau` (:math:`10^{-3}`), where the problem is easier, and that solution becomes the starting point for a second solve at the default :math:`\tau` :cite:p:`Leal2014`. The final :math:`\tau`, the starting :math:`\tau` and the number of :math:`\tau` values in the sweep are set by ``tau``, ``tau_max`` and ``tau_num`` of ``SolverParameters``.

3. **Derivatives.** A final solve from the converged root attaches derivatives with respect to the inputs, using the implicit function theorem. The results can therefore be differentiated, however many retries were needed to find the root.

This describes the default solver, which is one of the more extensive solver options. Other options bypass the retries and the :math:`\tau` sweep to reduce compilation and execution time, at the cost of potentially less robust convergence.

.. _NestedSolvers:

Nested solvers
--------------
Some real-gas equations of state give the pressure as a function of volume and temperature, :math:`P(V, T)`, and cannot be rearranged to give the volume directly. These include the van der Waals, Beattie–Bridgeman :cite:p:`HWZ58`, implicit Redlich–Kwong :cite:p:`HP91` and Zhang and Duan :cite:p:`ZD09` equations of state. For these, *Atmodeller* finds the molar volume at the given temperature and pressure with its own Newton solve, and then computes the fugacity coefficient :math:`\phi` from the volume by integrating over pressure, :math:`RT \ln \phi = \int_0^P (V - RT/P')\,dP'`.

These volume solves are nested inside the main solver: every evaluation of the fugacity, and hence of the equations, solves for the volume of each such species. The derivatives of the volume, which the main solver needs for its Jacobian, follow from the implicit function theorem, so they do not depend on how many iterations the volume solve took.