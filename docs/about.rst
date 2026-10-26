About
=====

*Atmodeller* is a Python package for thermochemical equilibrium modelling of planetary interiors and atmospheres, built on `JAX <https://docs.jax.dev/en/latest/>`_. It can also be used as a generic thermochemical equilibrium engine for other applications.

*Atmodeller* computes the chemical equilibrium between a planet's interior and atmosphere, solving gas-phase speciation, condensate stability and volatile solubility in silicate melt together, with real-gas equations of state for high pressures. It is designed for researchers in planetary science, exoplanet science and geochemistry who need these calculations, either standalone or as one component of a larger model. Applications include magma oceans, volcanic outgassing, and the atmospheres of super-Earths and sub-Neptunes. Users assemble a model from the species, solubility laws and equations of state they need, so the effect of each modelling choice can be explored. Because it is built on JAX, a batch of conditions is solved in one call and derivatives of the solution with respect to its inputs are available, which supports parameter sweeps, Monte Carlo sampling and gradient-based Bayesian inference.

It is released under the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_. Community development is strongly encouraged. If you use *Atmodeller* in your research, let us know and we'd be happy to include your work in the :ref:`Used In <UsedInFile>` section. This will help us track the impact of *Atmodeller* and also provide a resource for users to see how it has been applied in practice.

Citation
--------

If you use *Atmodeller*, or data from *Atmodeller*, please cite :cite:t:`BTH25`. In addition, please consult the *Atmodeller* source code for citations related to specific model components, such as solubility laws and real gas equations of state. Thermodynamic data are sourced from :cite:t:`MZG02`, with the volumes of graphite and diamond, and the Gibbs energy of diamond relative to graphite, from :cite:t:`Gustafson1986`.

Authors
-------

* Dan J. Bower (lead developer, ETH Zurich)
* Maggie A. Thompson (Carnegie Institution for Science)
* Kaustubh Hakim (KU Leuven/Royal Observatory of Belgium)
* Meng Tian (LMU Munich)
* Paolo A. Sossi (ETH Zurich)

Funding
-------

Atmodeller was created as part of a SERI-funded ERC Starting grant '2ATMO' granted to P. Sossi (Contract no. MB22.00033), with additional funding provided through a Swiss National Science Foundation (SNSF) Eccellenza Professorship (#203668).

M\. Thompson was supported by NASA through the NASA Hubble Fellowship grant #HST-HF2-51545 awarded by the Space Telescope Science Institute, United States, which is operated by the Association of Universities for Research in Astronomy, Inc., for NASA under contract NAS5-26555.

K\. Hakim acknowledges the FED-tWIN research programme STELLA (Prf-2021-022) funded by the Belgian Science Policy Office (BELSPO) and the research grant G014425N funded by the Research Foundation Flanders (FWO).