# SPDX-FileCopyrightText: 2024-2026 Dan J. Bower <dbower@eaps.ethz.ch>
# SPDX-FileCopyrightText: 2024-2026 Paolo A. Sossi
#
# SPDX-License-Identifier: Apache-2.0

"""Heuristic initializer for the solver's initial guess

Provides :class:`HeuristicInitializer`, which allocates element budgets to species by the
limiting-reagent principle and predicts the stable condensates from the reaction affinities. See
its docstring for the method.

The initializer acts on a single system and is vmapped over the batch by the solver.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float

from atmodeller.engine import compute_implied_log_stability
from atmodeller.initializer.core import LOG_TRACE_VALUE, max_moles_by_limiting_element
from atmodeller.jax_utils import FloatArray
from atmodeller.parameters import Parameters


class HeuristicInitializer(eqx.Module):
    r"""Initializer that allocates element budgets heuristically

    The cold guess shares the budget of each element among the species that contain it, by the
    limiting-reagent principle, without solving for equilibrium. The thermodynamics is used only to
    decide which condensates are present: a condensate is predicted stable if a reaction that forms
    it has a positive affinity at the allocated moles. The guess is cheap, but the moles of the
    gas species can be far from equilibrium, which
    :class:`~atmodeller.initializer.element_potential.ElementPotentialInitializer` (the default)
    solves for.

    Notation: :math:`b_e` is the budget of element :math:`e` (moles; undefined if :math:`e` is not
    mass constrained), :math:`a_{es}` the formula matrix (moles of element :math:`e` in species
    :math:`s`), :math:`n_s` the moles of species :math:`s`, :math:`\phi` the
    ``condensate_allocation_fraction``, :math:`\epsilon = \exp(\texttt{log\_trace\_value})` the
    trace moles (``log_trace_value``), :math:`S` the set of
    condensates predicted stable, :math:`\nu_{rs}` the stoichiometry of species :math:`s` in
    reaction :math:`r` of the solver's reaction network, and :math:`K_r` and :math:`Q_r` the
    equilibrium constant and the reaction quotient of reaction :math:`r`.

    .. rubric:: Limiting-element estimate

    For a set :math:`M` of species,
    :func:`~atmodeller.initializer.core.max_moles_by_limiting_element` gives

    .. math::

        \hat{n}_s(b, M) = \min_{e:\ a_{es} > 0,\ b_e\ \text{defined}}
            \frac{b_e}{\sum_{s' \in M} a_{es'}}, \qquad s \in M,

    the moles of :math:`s` if all species of :math:`M` that contain element :math:`e` had the same
    moles and together used all of :math:`e`. The tightest element sets the estimate, which is
    undefined for a species that contains no mass-constrained element.

    .. rubric:: Allocation for a given set of stable condensates

    1. *Condensates*: :math:`n_c = \phi\, \hat{n}_c(b, S)` for :math:`c \in S`, and
       :math:`n_c = \epsilon` otherwise. With :math:`\phi < 1` part of each budget is left to the
       other species, which would otherwise start unrealistically small.
    2. *Other species except* O\ :sub:`2`: :math:`n_s = \hat{n}_s(b', M)`, with the remaining
       budget :math:`b'_e = \max(b_e - \sum_{c \in S} a_{ec} n_c,\ 0)` and :math:`M` the gas and
       dissolved species other than O\ :sub:`2`, and :math:`n_s = 0` where the estimate is
       undefined.
    3. O\ :sub:`2` (*if it is a gas species*): the oxygen not used in step 2,
       :math:`n_{\mathrm{O_2}} = b''_\mathrm{O} / a_{\mathrm{O},\mathrm{O_2}}` with
       :math:`b''_e = \max(b'_e - \sum_{s \in M} a_{es} n_s,\ 0)`. O\ :sub:`2` is left out of
       step 2 because an equal share of the oxygen would give it as much as the major O-bearing
       species, which is usually far too much.
    4. *Fallbacks*: a species with zero moles (its elements are used up, or it contains no
       mass-constrained element) gets :math:`\epsilon`. An undefined estimate, which arises only
       for O\ :sub:`2` when oxygen is not mass constrained, is replaced by the geometric mean of
       the moles of the other species of step 2 that have nonzero moles.

    .. rubric:: Stability of condensates

    From an allocation, the activities are estimated as ideal: for a gas species
    :math:`\ln a_s = \ln (n_s / N_\mathrm{gas}) + \ln P`, with :math:`N_\mathrm{gas}` the total gas
    moles and :math:`P = P(n)` the pressure of the system, and :math:`\ln a_s = 0` for condensates
    and dissolved species (dilution and non-ideality are ignored, since they need the solution).
    Condensate :math:`c` is predicted stable if, for some reaction :math:`r`,

    .. math::

        \nu_{rc} \left( \ln K_r(T, P) - \sum_s \nu_{rs} \ln a_s \right) > 0,

    that is, if the reaction, written in the direction that forms :math:`c`, has a positive
    affinity. Only species with a stability unknown in the solver enter as :math:`c`.

    The set is found by fixed-point iteration: :math:`S_0 = \emptyset`, and :math:`S_{k+1}` is
    :math:`S_k` together with the condensates predicted stable from the allocation for
    :math:`S_k`. The first pass is therefore a gas-only pre-screen. The set never shrinks, so the
    iteration stops, when it no longer grows, after at most :math:`n_\mathrm{condensates} + 1`
    passes. The solver makes the final decision on which condensates are present.

    .. rubric:: Guess

    - The moles are the allocation for the final set :math:`S`: steps 1 and 4 for condensates,
      steps 2 to 4 for the other species.
    - A gas species :math:`i` with an imposed fugacity :math:`f_i` gets the moles that give it the
      partial pressure :math:`f_i` relative to the other gas species:

      .. math::

          \ln n_i = \ln f_i(T, P) - \ln P
              + \ln \sum_{s \in \text{gas, no imposed fugacity}} n_s,

      with :math:`P` the pressure of the allocation.
    - The log stabilities :math:`\ln \zeta_s` follow from the implied-stability relation
      (:func:`~atmodeller.engine.compute_implied_log_stability`), with :math:`\tau` the solver's
      stability parameter (``tau``),

      .. math::

          \ln \zeta_s = \min_{e:\ a_{es} > 0} \ln b_e + \ln \tau - \ln n_s,

      so a condensate in :math:`S` starts with a small stability and a condensate outside
      :math:`S` (at :math:`\epsilon`) with a large one.
    - An error is raised if the guess contains ``NaN``.

    .. rubric:: JAX

    The guess is computed per system and vmapped over the batch by the solver. The stability
    iteration is a ``lax.while_loop``, which under vmap runs until the set has stopped growing for
    every system in the batch. Whether O\ :sub:`2` is a species is known from the species formulae
    outside JIT, so the O\ :sub:`2` steps are plain Python branches. Gradients are stopped through
    the guess, since the equilibrium solve is differentiated implicitly (and ``while_loop`` is not
    reverse-differentiable).

    Args:
        condensate_allocation_fraction: Fraction :math:`\phi` of the element budget to allocate to
            predicted-stable condensates. Defaults to ``0.5``.
        log_trace_value: Log number of moles given to species with a negligible element budget,
            and to condensates predicted absent. Defaults to
            :data:`~atmodeller.initializer.core.LOG_TRACE_VALUE`.
    """

    condensate_allocation_fraction: float = 0.5
    r"""Fraction :math:`\phi` of the element budget to allocate to predicted-stable condensates

    This value, although arbitrary to some extent, prevents condensates from being allocated an
    excessively large portion of the element budget, which can drive gas species to unrealistically
    low initial abundances.
    """
    log_trace_value: float = LOG_TRACE_VALUE
    """Log number of moles given to species with a negligible element budget"""

    def start(self, parameters: Parameters) -> Float[Array, " twice_species"]:
        """Cold initial guess from the heuristic.

        See the class docstring for the method.

        Args:
            parameters: Parameters of a single system

        Returns:
            Concatenated array of [log_number_moles, log_stability]
        """
        # The guess needs no gradients (the solve is differentiated implicitly), and reverse-mode
        # differentiation cannot pass through the while loop
        parameters = jax.tree_util.tree_map(
            lambda x: lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, parameters
        )

        # -----------------------------------------------------------------------------------
        # STEP 1. Setup: element budgets, species masks, imposed fugacities.
        #
        # Unconstrained elements have a NaN budget.
        # -----------------------------------------------------------------------------------
        temperature: FloatArray = parameters.state.temperature

        element_abundance: Float[Array, " n_elements"] = parameters.mass_constraints.abundance()
        # jax.debug.print("element_abundance = {out}", out=element_abundance)

        gas_mask: Bool[Array, " n_species"] = jnp.asarray(
            parameters.reaction_system.phase_system.gas_species_mask
        )
        condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
            parameters.reaction_system.phase_system.condensates_species_mask
        )
        active_activity_constraints: Bool[Array, " n_species"] = (
            parameters.activity_constraints.active()
        )
        # jax.debug.print("active_activity_constraints = {out}", out=active_activity_constraints)

        gas_no_imposed_fugacity: Bool[Array, " n_species"] = (
            gas_mask & ~active_activity_constraints
        )
        # jax.debug.print("gas_no_imposed_fugacity = {out}", out=gas_no_imposed_fugacity)

        # -----------------------------------------------------------------------------------
        # STEP 2. Stability of condensates: fixed-point iteration on the predicted-stable set.
        #
        # Each pass allocates the budget for the set so far (ALLOCATION 1 to 4 in
        # _allocate_element_budget), estimates ideal activities, and adds the condensates with a
        # positive stability signal. See the class docstring.
        # -----------------------------------------------------------------------------------
        def _one_stability_pass(
            condensate_stable_known: Bool[Array, " n_species"],
        ) -> Bool[Array, " n_species"]:
            """One stability-prediction pass

            Given the condensates predicted stable so far, allocates the element budget (to them
            first), evaluates the stability signal with ideal activities, and returns the union of
            the new predictions with the input mask.

            Args:
                condensate_stable_known: Boolean mask of condensates predicted stable so far

            Returns:
                Updated mask - superset of ``condensate_stable_known``.
            """
            _, n_other = self._allocate_element_budget(
                parameters, element_abundance, condensate_stable_known
            )
            # jax.debug.print("n_other in stability pass = {out}", out=n_other)
            stability_signal: Float[Array, "n_reactions n_species"] = self._get_stability_signal(
                parameters, n_other
            )
            # jax.debug.print("stability_signal in stability pass = {out}", out=stability_signal)
            new_predictions: Bool[Array, " n_species"] = (
                jnp.any(stability_signal > 0, axis=0) & condensate_mask
            )
            # jax.debug.print("new_predictions in stability pass = {out}", out=new_predictions)
            # jax.debug.print("new_predictions = {out}", out=new_predictions)

            # Monotone union: condensates are never retracted once predicted stable, so the set
            # either grows or is unchanged, and the iteration terminates without a cap. The solver
            # makes the definitive stable/absent decision.
            return condensate_stable_known | new_predictions

        # Iterate until the predicted-stable set stops growing. Start with no condensates known
        # stable, so the first pass is the gas-only pre-screen. The monotone union in
        # _one_stability_pass guarantees termination.
        init_stable: Bool[Array, " n_species"] = jnp.zeros_like(condensate_mask)
        # jax.debug.print("init_stable = {out}", out=init_stable)
        first_stable: Bool[Array, " n_species"] = _one_stability_pass(init_stable)
        # jax.debug.print("first_stable = {out}", out=first_stable)

        def _cond_fn(carry: tuple) -> Bool[Array, ""]:
            prev, curr = carry
            return jnp.any(prev != curr)

        def _body_fn(carry: tuple) -> tuple:
            _, curr = carry
            return curr, _one_stability_pass(curr)

        _, condensate_stable_predicted = lax.while_loop(
            _cond_fn, _body_fn, (init_stable, first_stable)
        )

        # -----------------------------------------------------------------------------------
        # STEP 3. Allocation for the final predicted-stable set.
        # -----------------------------------------------------------------------------------
        n_condensate, n_other = self._allocate_element_budget(
            parameters, element_abundance, condensate_stable_predicted
        )

        # Condensates take their allocation (trace if not predicted stable) and the other species
        # the allocation of the remaining budget
        n_estimate: Float[Array, " n_species"] = jnp.where(condensate_mask, n_condensate, n_other)
        # jax.debug.print("n_estimate after merge = {out}", out=n_estimate)

        log_number_moles: Float[Array, " n_species"] = jnp.log(n_estimate)
        # jax.debug.print("log_number_moles = {out}", out=log_number_moles)

        # -----------------------------------------------------------------------------------
        # STEP 4. Gas species with an imposed fugacity: moles that give the imposed partial
        # pressure relative to the total moles of the other gas species, at the pressure of the
        # allocation.
        # -----------------------------------------------------------------------------------
        log_n_gas_known_total: Float[Array, ""] = logsumexp(
            jnp.where(gas_no_imposed_fugacity, log_number_moles, -jnp.inf)
        )
        # jax.debug.print("log_n_gas_known_total = {out}", out=log_n_gas_known_total)

        pressure: Float[Array, ""] = parameters.state.get_pressure(log_number_moles)
        # jax.debug.print("pressure = {out}", out=pressure)

        log_fug: Float[Array, " n_species"] = parameters.activity_constraints.log_activity(
            temperature, pressure
        )
        # jax.debug.print("log_fug = {out}", out=log_fug)

        log_n_fug: Float[Array, " n_species"] = log_fug + log_n_gas_known_total - jnp.log(pressure)
        # jax.debug.print("log_n_fug = {out}", out=log_n_fug)

        log_number_moles = jnp.where(
            gas_mask & active_activity_constraints, log_n_fug, log_number_moles
        )
        # jax.debug.print("log_number_moles (fugacity) = {out}", out=log_number_moles)

        # -----------------------------------------------------------------------------------
        # STEP 5. Output: log stabilities from the implied-stability relation, then
        # [log moles, log stability]. An error is raised if the guess contains NaN.
        # -----------------------------------------------------------------------------------
        log_stability = compute_implied_log_stability(parameters, log_number_moles)
        # jax.debug.print("log_stability = {out}", out=log_stability)

        result: Float[Array, " twice_species"] = jnp.concatenate(
            (log_number_moles, log_stability), axis=-1
        )
        # jax.debug.print("Initial guess (log_number_moles, log_stability) = {out}", out=result)

        result = eqx.error_if(
            result,
            jnp.any(jnp.isnan(result)),
            "Initial guess contains NaNs, which will break the solver.",
        )

        return result

    def _allocate_element_budget(
        self,
        parameters: Parameters,
        element_abundance: Float[Array, " n_elements"],
        condensate_stable_mask: Bool[Array, " n_species"],
    ) -> tuple[Float[Array, " n_species"], Float[Array, " n_species"]]:
        """Allocates the element budget between predicted-stable condensates and other species.

        Steps 1 to 4 of the allocation in the class docstring (ALLOCATION 1 to 4 in the code): a
        fraction of the budget to the predicted-stable condensates by the limiting-element
        estimate, the remainder to the other species except O2, the oxygen left to O2, and
        fallbacks for species with zero or undefined moles.

        Args:
            parameters: Parameters
            element_abundance: Available abundance of each element
            condensate_stable_mask: Boolean mask indicating which species are predicted-stable
                condensates

        Returns:
            Tuple of two arrays:
                - n_condensate: Moles allocated to each condensate species
                - n_other: Moles allocated to each non-condensate species
        """
        formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
            parameters.reaction_system.formula_matrix, dtype=float
        )
        condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
            parameters.reaction_system.phase_system.condensates_species_mask
        )

        # -----------------------------------------------------------------------------------
        # ALLOCATION 1. Condensates: a fraction of the limiting-element estimate for the
        # predicted-stable condensates, trace for the others.
        # -----------------------------------------------------------------------------------
        n_condensate: Float[Array, " n_species"] = max_moles_by_limiting_element(
            formula_matrix, element_abundance, condensate_stable_mask
        )
        # jax.debug.print("n_condensate = {out}", out=n_condensate)

        n_condensate = n_condensate * self.condensate_allocation_fraction

        # Species outside the predicted-stable set (NaN) get the trace value exp(log_trace_value).
        # They do not consume element budget, because only predicted-stable condensates enter
        # element_used below, and the entries of non-condensate species are not used.
        n_condensate_fallback: Float[Array, " n_species"] = jnp.where(
            jnp.isnan(n_condensate), jnp.exp(self.log_trace_value), n_condensate
        )
        # jax.debug.print("n_condensate_fallback = {out}", out=n_condensate_fallback)

        # Total moles of each element consumed by the allocated condensate budget. Note here that
        # the mask ensures that only predicted-stable condensates contribute to the element
        # consumption.
        element_used: Float[Array, " n_elements"] = formula_matrix @ (
            condensate_stable_mask * n_condensate_fallback
        )
        # jax.debug.print("element_used = {out}", out=element_used)

        # Remaining element budget after stable-condensate allocation, which must be positive or
        # zero.
        remaining_b: Float[Array, " n_elements"] = jnp.maximum(
            element_abundance - element_used, 0.0
        )
        # jax.debug.print("remaining_b = {out}", out=remaining_b)

        # -----------------------------------------------------------------------------------
        # ALLOCATION 2. Other species except O2: the limiting-element estimate for the
        # remaining budget.
        # -----------------------------------------------------------------------------------
        # If present, exclude O2 from the shared allocation, because an equal share of oxygen would
        # give it as much as the major O-bearing species. O2 instead takes the oxygen left over.
        # Found from the gas formulae (as the gas phase's O2_index), which are static under JIT
        gas_formulae: list[str] = [
            species.data.hill_formula
            for species in parameters.reaction_system.phase_system.gas.species
        ]
        O2_index: int | None = gas_formulae.index("O2") if "O2" in gas_formulae else None
        non_condensate_no_O2_mask: Bool[Array, " n_species"] = ~condensate_mask
        if O2_index is not None:
            non_condensate_no_O2_mask = non_condensate_no_O2_mask.at[O2_index].set(False)

        # Allocate remaining budget to non-condensate species excluding O2
        n_other: Float[Array, " n_species"] = max_moles_by_limiting_element(
            formula_matrix, remaining_b, non_condensate_no_O2_mask
        )
        # NaN for condensates, O2 (if present) and species without a mass-constrained element, all
        # zeroed for calculating the budget
        n_other = jnp.where(jnp.isnan(n_other), 0.0, n_other)
        # jax.debug.print("n_other = {out}", out=n_other)

        # Recalculate element_used after n_other allocation
        element_used_no_O2: Float[Array, " n_elements"] = formula_matrix @ n_other
        # jax.debug.print("element_abundance = {out}", out=element_abundance)
        # jax.debug.print("element_used_no_O2 = {out}", out=element_used_no_O2)

        # Update element budget after allocation excluding O2, which must be positive or zero.
        remaining_b = jnp.maximum(element_abundance - element_used - element_used_no_O2, 0.0)
        # jax.debug.print("remaining_b (after n_other) = {out}", out=remaining_b)

        # -----------------------------------------------------------------------------------
        # ALLOCATION 3. O2 (if a gas species): the oxygen left over, which is NaN if O is not
        # mass constrained.
        # -----------------------------------------------------------------------------------
        if O2_index is not None:
            O_index: int = parameters.element_names.index("O")
            n_O2: Float[Array, ""] = remaining_b[O_index] / formula_matrix[O_index, O2_index]
            n_other = n_other.at[O2_index].set(n_O2)
        # jax.debug.print("n_other after O2 allocation = {out}", out=n_other)

        # -----------------------------------------------------------------------------------
        # ALLOCATION 4. Fallbacks for zero and undefined moles.
        # -----------------------------------------------------------------------------------
        # Zero moles (elements used up, or no mass-constrained element) become the trace value;
        # this includes the condensate entries, which are not used but must have a finite log. NaN,
        # which at this point arises only for O2 when O is not mass constrained, becomes the
        # geometric mean of the valid estimates, which keeps it at a magnitude comparable to the
        # other species.
        unconstrained: Bool[Array, " n_species"] = jnp.isnan(n_other)
        # jax.debug.print("unconstrained = {out}", out=unconstrained)
        exhausted: Bool[Array, " n_species"] = n_other == 0
        # jax.debug.print("exhausted = {out}", out=exhausted)
        valid: Bool[Array, " n_species"] = n_other > 0
        # jax.debug.print("valid = {out}", out=valid)

        log_geometric_mean: Float[Array, ""] = jnp.sum(
            jnp.where(valid, jnp.log(n_other), 0.0)
        ) / jnp.sum(valid).astype(float)
        # jax.debug.print("log_geometric_mean = {out}", out=log_geometric_mean)

        n_other_fallback = jnp.where(unconstrained, jnp.exp(log_geometric_mean), n_other)
        n_other_fallback = jnp.where(exhausted, jnp.exp(self.log_trace_value), n_other_fallback)

        # jax.debug.print("n_condensate after fallback = {out}", out=n_condensate_fallback)
        # jax.debug.print("n_other after fallback = {out}", out=n_other_fallback)

        return n_condensate_fallback, n_other_fallback

    def _get_log_activity_estimate(
        self,
        parameters: Parameters,
        species_abundance: Float[Array, " n_species"],
        pressure: FloatArray,
    ) -> Float[Array, " n_species"]:
        r"""Estimates log activity for gas species based on ideal-gas assumptions.

        For gas species, the log activity is the log partial pressure,
        :math:`\ln a_s = \ln (n_s / N_\mathrm{gas}) + \ln P`. For other species (condensates and
        dissolved species), the log activity is set to 0, i.e. unit activity. This ignores
        dilution (mole fraction < 1) and activity coefficients for melt/solid solution species,
        but is intentional: the pre-screen is a cheap heuristic and calling the EOS/mixing models
        here would be circular and expensive. Activity coefficients are in any case unavailable
        without a complete solution.

        Args:
            parameters: Parameters containing the reaction system and state information
            species_abundance: Estimated abundance of each species
            pressure: Pressure (bar) of the estimate

        Returns:
            Estimated log activity for each species
        """
        gas_mask: Bool[Array, " n_species"] = jnp.asarray(
            parameters.reaction_system.phase_system.gas_species_mask
        )

        n_gas_total: Float[Array, ""] = jnp.nansum(jnp.where(gas_mask, species_abundance, 0.0))
        safe_n_gas_total: Float[Array, ""] = jnp.where(n_gas_total > 0, n_gas_total, 1.0)

        # TODO: Here, activity of dissolved species is also computed as unity. To improve.
        log_activity: Float[Array, " n_species"] = jnp.where(
            gas_mask,
            jnp.log(jnp.where(gas_mask, species_abundance, 1.0))
            - jnp.log(safe_n_gas_total)
            + jnp.log(jnp.where(pressure > 0, pressure, 1.0)),
            0.0,
        )
        # jax.debug.print("log_activity = {out}", out=log_activity)

        # activity: Float[Array, " n_species"] = jnp.exp(log_activity)
        # jax.debug.print("activity = {out}", out=activity)

        return log_activity

    def _get_stability_signal(
        self, parameters: Parameters, species_abundance: Float[Array, " n_species"]
    ) -> Float[Array, "n_reactions n_species"]:
        r"""Computes the stability signal for each reaction and condensate species.

        The stability signal is :math:`\nu^\zeta_{rc} (\ln K_r - \ln Q_r)`, where
        :math:`\nu^\zeta` is the reaction stability matrix (the reaction matrix restricted to
        species with a stability unknown), :math:`K_r` the equilibrium constant and :math:`Q_r`
        the reaction quotient from the estimated activities. A positive signal means that the
        reaction is driven towards forming the condensate.

        Args:
            parameters: Parameters containing the reaction system information
            species_abundance: Estimated abundance of each species, used to estimate activities

        Returns:
            Stability signal for each reaction and condensate species
        """
        reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
            parameters.reaction_system.reaction.reaction_matrix_full
        )
        # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)
        stability_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
            parameters.reaction_system.reaction.reaction_stability_matrix_full
        )
        # jax.debug.print("stability_matrix = {out}", out=stability_matrix)

        temperature: FloatArray = parameters.state.temperature
        pressure: FloatArray = parameters.state.get_pressure(jnp.log(species_abundance))
        log_Kp: Float[Array, " n_reactions"] = parameters.reaction_system.reaction.log_Kp(
            temperature, pressure
        )
        # jax.debug.print("log_Kp = {out}", out=log_Kp)

        log_activity: Float[Array, " n_species"] = self._get_log_activity_estimate(
            parameters, species_abundance, pressure
        )

        # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
        log_Q: Float[Array, " n_reactions"] = reaction_matrix @ log_activity
        stability_signal: Float[Array, "n_reactions n_species"] = (
            stability_matrix * (log_Kp - log_Q)[:, None]
        )
        # jax.debug.print("stability_signal = {out}", out=stability_signal)

        return stability_signal
