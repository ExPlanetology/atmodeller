"""Solubility laws.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging

import numpy as np

from atmodeller.interfaces import Solubility, limit_solubility
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)

# Solubility limiters.
# Maximum sulfur solubility.
SULFUR_MAXIMUM_PPMW: float = UnitConversion.weight_percent_to_ppmw(1)  # 1% by weight

# region Andesite solubility


class AndesiteH2(Solubility):
    """Hirschmann et al. 2012, H2 solubility in silicate melts

    https://ui.adsabs.harvard.edu/abs/2012E%26PSL.345...38H/abstract

    Log-scale linear fit to fH2 vs. H2 concentration for andesite in Table 2
    Experiments conducted from 0.7-3 GPa, 1400 C
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        ppmw: float = 10 ** (0.60128868 * np.log10(fugacity) + 1.01058631)
        return ppmw


class AndesiteS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    https://ui.adsabs.harvard.edu/abs/2022GeCoA.336..150B/abstract

    Using the first equation in the abstract and the corrected expression for sulfate capacity (C_S6+)
    in corrigendum (https://ui.adsabs.harvard.edu/abs/2023GeCoA.343..420B/abstract).
    Composition for Andesite from Table 1.
    Experiments conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated
    with Air/SO2 mixtures
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Fugacity is fS2."""
        del pressure
        logCs: float = -12.948 + (31586.2393 / temperature)
        logS_wtp: float = logCs + (0.5 * np.log10(fugacity)) + (1.5 * log10_fugacities_dict["O2"])
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class AndesiteS2_Sulfide(Solubility):
    """Boulliung & Wood 2023. Solubility of sulfur as sulfide (S^2-)

    https://ui.adsabs.harvard.edu/abs/2023CoMP..178...56B/abstract

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition
    for Andesite from Table 1.
    Experiments conducted at 1 atm pressure and temperatures from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere
    fO2 conditions were greater than 1 log unit below FMQ
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """fugacity is fS2."""
        del pressure
        logCs: float = 0.225 - (8921.0927 / temperature)
        logS_wtp: float = logCs - (0.5 * (log10_fugacities_dict["O2"] - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class AndesiteS2(Solubility):
    """Boulliung & Wood 2022, 2023. Total S solubility accounting for both sulfide and sulfate dissolution.

    Adding sufate solubility (Boulliung & Wood 2022) and sulfide solubility (Boulliun & Wood 2023)
    """

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteS2_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteS2_Sulfate()

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)

        return solubility


# endregion


class AnorthiteDiopsideH2O(Solubility):
    """Newcombe et al. (2017). Water solubility in lunar basalt and Anorthite-Diopside-Eutectic compositions.

    https://ui.adsabs.harvard.edu/abs/2017GeCoA.200..330N/abstract

    Power law from Figure 5(A) for Anorthite-Diopside glass
    Experiments conducted at 1 atm and 1350 C. Melts equilibrated in 1-atm furnace with H2/CO2 gas mixtures
    that spanned fO2 from IW-3 to IW+4.8 and pH2/pH2O from 0.003-24
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure

        return self.power_law(fugacity, 727, 0.5)


# region Basalt solubility


class BasaltDixonCO2(Solubility):
    """Dixon et al. (1995). H2O and CO2 solubilities in MORB liquids

    https://academic.oup.com/petrology/article/36/6/1607/1493308?login=true

    Equation 6 for mole fraction of dissolved carbonate (CO3^2-) and then converting to ppmw for CO2
    Experiments conducted at 1200 C, 210-980 bars with mixed H2O-CO2 vapor phase (CO2 vapor mole fraction varied from 0.42-0.97)
    """

    @limit_solubility()
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del log10_fugacities_dict
        del pressure
        ppmw: float = (3.8e-7) * fugacity * np.exp(-23 * (fugacity - 1) / (83.15 * temperature))
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)

        return ppmw


class BasaltDixonH2O(Solubility):
    """Dixon et al. (1995). H2O and CO2 solubilities in MORB liquids

    https://academic.oup.com/petrology/article/36/6/1607/1493308?login=true

    Refit data to a power law by Paolo Sossi (fitting Figure 4, CHECK)
    Experiments conducted at 1200 C, 200-717 bars with pure H2O
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure

        return self.power_law(fugacity, 965, 0.5)


class BasaltH2(Solubility):
    """Hirschmann et al. 2012, H2 solubility in silicate melts

    https://ui.adsabs.harvard.edu/abs/2012E%26PSL.345...38H/abstract

    Log-scale linear fit to fH2 vs. H2 concentration for basalt in Table 2
    Experiments conducted from 0.7-3 GPa, 1400 C
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        ppmw: float = 10 ** (0.52413928 * np.log10(fugacity) + 1.10083602)

        return ppmw


class BasaltLibourelN2(Solubility):
    """Libourel et al. (2003), basalt (tholeiitic) magmas.

    https://ui.adsabs.harvard.edu/abs/2003GeCoA..67.4123L/abstract

    Equation 23, includes dependencies on fN2 and fO2.
    Experiments conducted at 1 atm and 1425 C (two experiments at 1400 C), fO2 from IW-8.3 to IW+8.7
    using mixtures of CO, CO2 and N2 gases
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del pressure
        ppmw: float = self.power_law(fugacity, 0.0611, 1.0)
        # Below is correct, i.e. fO2 and NOT log10(fO2), unlike most other formulations
        constant: float = ((10 ** log10_fugacities_dict["O2"]) ** -0.75) * 5.97e-10
        ppmw += self.power_law(fugacity, constant, 0.5)

        return ppmw


class BasaltS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    https://ui.adsabs.harvard.edu/abs/2022GeCoA.336..150B/abstract

    Using the first equation in the abstract and the corrected expression for sulfate capacity (C_S6+)
    in corrigendum (https://ui.adsabs.harvard.edu/abs/2023GeCoA.343..420B/abstract).
    Composition for Basalt from Table 1.
    Experiments conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated
    with Air/SO2 mixtures
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Fugacity is fS2."""
        del pressure
        logCs: float = -12.948 + (32333.5635 / temperature)
        logSO4_wtp: float = (
            logCs + (0.5 * np.log10(fugacity)) + (1.5 * log10_fugacities_dict["O2"])
        )
        SO4_wtp: float = 10**logSO4_wtp
        S_wtp: float = SO4_wtp * (32.065 / 96.06)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class BasaltS2_Sulfide(Solubility):
    """Boulliung & Wood 2023. Solubility of sulfur as sulfide (S^2-)

    https://ui.adsabs.harvard.edu/abs/2023CoMP..178...56B/abstract

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition
    for Basalt from Table 1.
    Experiments conducted at 1 atm pressure and temperatures from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere
    fO2 conditions were greater than 1 log unit below FMQ
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Fugacity is fS2."""
        del pressure
        logCs: float = 0.225 - (8045.7465 / temperature)
        logS_wtp: float = logCs - (0.5 * (log10_fugacities_dict["O2"] - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class BasaltS2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution.

    Adding sufate solubility (Boulliung & Wood 2022) and sulfide solubility (Boulliun & Wood 2023)
    """

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltS2_Sulfide()
        self.sulfate_solubility: Solubility = BasaltS2_Sulfate()

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)

        return solubility


class BasaltWilsonH2O(Solubility):
    """Wilson and Head (1981) and Hamilton et al. (1964)

    https://ui.adsabs.harvard.edu/abs/1981JGR....86.2971W/abstract
    https://doi.org/10.1093/petrology/5.1.21

    Equation 30, and converting from weight % to ppmw.
    Not clear what all experimental data is used to derive this fit,
    but it fits data at 1100 C and 1000-6000 bars H2O from Hamilton et al. 1964 decently well (their Table 3)
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        return self.power_law(fugacity, 215, 0.7)


class TBasaltS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    https://ui.adsabs.harvard.edu/abs/2022GeCoA.336..150B/abstract

    Using the first equation in the abstract and the corrected expression for sulfate capacity (C_S6+)
    in corrigendum (https://ui.adsabs.harvard.edu/abs/2023GeCoA.343..420B/abstract).
    Composition for Basalt from Table 1.
    Experiments conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated
    with Air/SO2 mixtures. Composition for Trachy-Basalt from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Fugacity is fS2."""
        del pressure
        logCs: float = -12.948 + (32446.366 / temperature)
        logS_wtp: float = logCs + (0.5 * np.log10(fugacity)) + (1.5 * log10_fugacities_dict["O2"])
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class TBasaltS2_Sulfide(Solubility):
    """Boulliung & Wood 2023. Solubility of sulfur as sulfide (S^2-)

    https://ui.adsabs.harvard.edu/abs/2023CoMP..178...56B/abstract

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition
    for Basalt from Table 1.
    Experiments conducted at 1 atm pressure and temperatures from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere
    fO2 conditions were greater than 1 log unit below FMQ. Composition for Trachy-basalt from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Fugacity is fS2."""
        del pressure
        logCs: float = 0.225 - (7842.5 / temperature)
        logS_wtp: float = logCs - (0.5 * (log10_fugacities_dict["O2"] - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


# endregion


class LunarGlassH2O(Solubility):
    """Newcombe et al. (2017). Water solubility in lunar basalt and Anorthite-Diopside-Eutectic compositions.

    https://ui.adsabs.harvard.edu/abs/2017GeCoA.200..330N/abstract

    Power law from Figure 5(A) for Lunar glass
    Experiments conducted at 1 atm and 1350 C. Melts equilibrated in 1-atm furnace with H2/CO2 gas mixtures
    that spanned fO2 from IW-3 to IW+4.8
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        return self.power_law(fugacity, 683, 0.5)


class MercuryMagmaS(Solubility):
    """Namur et al. 2016. Sulfur solubility in reduced mafic silicate melts relevant for Mercury

    https://ui.adsabs.harvard.edu/abs/2016E%26PSL.448..102N/abstract

    Dissolved S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like magmas
    Equation 10, with coefficients from Table 2, assumed composition is Northern Volcanic Plains (NVP)
    Experiments on Mercurian lavas and enstatite chondrites at 1200-1750 C and pressures from 1 bar to 4 GPa
    Equilibrated silicate melts with sulfide and metallic melts at reducing conditions (fO2 at IW-1.5 to IW-9.4)
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del pressure
        a, b, c, d = [7.25, -2.54e4, 0.04, -0.551]
        wt_perc: float = np.exp(
            a
            + (b / temperature)
            + ((c * fugacity) / temperature)
            + (d * log10_fugacities_dict["O2"])
            - 0.136
        )
        ppmw: float = UnitConversion.weight_percent_to_ppmw(wt_perc)

        return ppmw


class PeridotiteH2O(Solubility):
    """Sossi et al. (2023). Solubility of water in peridotite liquids

    https://ui.adsabs.harvard.edu/abs/2023E%26PSL.60117894S/abstract

    Power law parameters in the abstract for peridotitic glasses.
    Experiments conducted at 2173 K and 1 bar and range of fO2 from IW-1.9 to IW+6.0
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        return self.power_law(fugacity, 647, 0.5)


class SilicicMeltsH2(Solubility):
    """Gaillard et al. 2003. Fe-H redox exchange in silicate glasses

    https://ui.adsabs.harvard.edu/abs/2003GeCoA..67.2427G/abstract

    Power law fit for fH2 vs. H2 (ppm-wt) from Table 4 data
    Experiments at pressures from 0.02-70 bar, temperatures from 300-1000C.
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        ppmw: float = self.power_law(fugacity, 0.163, 1.252)
        return ppmw


class BasaltCO(Solubility):
    """Yoshioka et al. 2019. Carbon solubility in silicate melts

    https://ui.adsabs.harvard.edu/abs/2019GeCoA.259..129Y/abstract

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C
    Henry's Law, their expression for solubility of CO in MORB in the abstract.
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        CO_wtp: float = 10 ** (-5.20 + (0.8 * np.log10(fugacity)))
        ppmw: float = UnitConversion.weight_percent_to_ppmw(CO_wtp)
        return ppmw


class RhyoliteCO(Solubility):
    """Yoshioka et al. 2019. Carbon solubility in silicate melts

    https://ui.adsabs.harvard.edu/abs/2019GeCoA.259..129Y/abstract

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C
    Henry's Law, their expression for solubility of CO in rhyolite in the abstract.
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        CO_wtp: float = 10 ** (-4.08 + (0.52 * np.log10(fugacity)))
        ppmw: float = UnitConversion.weight_percent_to_ppmw(CO_wtp)
        return ppmw


class BasaltArmstrongCO(Solubility):
    """Armstrong et al. 2015. Solubility of volatiles in mafic melts under reduced conditions

    https://ui.adsabs.harvard.edu/abs/2015GeCoA.171..283A/abstract

    Experiments on Martian and terrestrial basalts at 1.2 GPa and 1400 C with variable fO2 from
    IW-3.65 to IW+1.46
    Equation 10, log-scale linear fit for CO and includes dependence on total pressure
    The fitting coefficients also use data from Stanley et al. 2014 (experiments from 1-1.2 GPa)
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        logCO_ppm: float = -0.738 + (0.876 * np.log10(fugacity)) - (5.44e-5 * pressure)
        ppmw: float = 10**logCO_ppm
        return ppmw


class BasaltCH4(Solubility):
    """Ardia et al. 2013, CH4 solubility in haplobasalt (Fe-free) silicate melt.

    https://ui.adsabs.harvard.edu/abs/2013GeCoA.114...52A/abstract

    Experiments conducted at 0.7-3 GPa and 1400-1450 C
    Equations 7a and 8, values for lnK0 and deltaV from the text.
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        P_GPa: float = UnitConversion.bar_to_GPa(pressure)
        one_bar_in_GPa: float = UnitConversion.bar_to_GPa(1)
        K: float = np.exp(4.93 - (1.93 * (P_GPa - one_bar_in_GPa)))
        ppmw: float = K * UnitConversion.bar_to_GPa(fugacity)
        return ppmw


class BasaltHe(Solubility):
    """Jambon et al. 1986, Solubility of He in tholeittic basalt melt

    https://ui.adsabs.harvard.edu/abs/1986GeCoA..50..401J/abstract

    Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and 1250-1600 C
    Using Henry's Law solubility constant for He from the abstract, convert from STP units to mol/g*bar
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        Henry_sol_constant: float = 56e-5  # cm3*STP/g*bar
        He_conc: float = (
            Henry_sol_constant / 2.24e4
        ) * fugacity  # converts Henry sol constant to mol/g*bar, 2.24e4 cm^3/mol at STP
        ppmw: float = (
            He_conc * 4.0026 * 1e6
        )  # converts He conc from mol/g to g H2/g Total and then to ppmw
        return ppmw


class BasaltCl2(Solubility):
    """Thomas & Wood 2021. Solubility of chlorine in silicate melts

    https://ui.adsabs.harvard.edu/abs/2021GeCoA.294...28T/abstract

    Solubility law from Figure 4 showing relation between dissolved Cl concentration and Cl fugacity
    for Icelandic basalt at 1400 C and 1.5 GPa
    Experiments from 0.5-2 GPa and 1200-1500 C
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        Cl_wtp: float = 78.56 * np.sqrt(fugacity)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(Cl_wtp)
        return ppmw


class AnorthiteDiopsideForsteriteCl2(Solubility):
    """Thomas & Wood 2021. Solubility of chlorine in silicate melts

    https://ui.adsabs.harvard.edu/abs/2021GeCoA.294...28T/abstract

    Solubility law from Figure 4 showing relation between dissolved Cl concentration and Cl fugacity
    for CMAS composition (An50Di28Fo22 (anorthite-diopside-forsterite), Fe-free low-degree mantle melt)
    at 1400 C and 1.5 GPa
    Experiments from 0.5-2 GPa and 1200-1500 C
    """

    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        del temperature
        del log10_fugacities_dict
        del pressure
        Cl_wtp: float = 140.52 * np.sqrt(fugacity)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(Cl_wtp)
        return ppmw


# Dictionaries of self-consistent solubility laws for a given composition.
andesite_solubilities: dict[str, Solubility] = {
    "H2": AndesiteH2(),
    "S2": AndesiteS2(),
}

anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltDixonH2O(),
    "CO2": BasaltDixonCO2(),
    "H2": BasaltH2(),
    "N2": BasaltLibourelN2(),
    "S2": BasaltS2(),
    "CO": BasaltCO(),
    "He": BasaltHe(),
    "Cl2": BasaltCl2(),
    "CH4": BasaltCH4(),
}
rhyolite_solubilities: dict[str, Solubility] = {
    "CO": RhyoliteCO(),
}
peridotite_solubilities: dict[str, Solubility] = {"H2O": PeridotiteH2O()}
reducedmagma_solubilities: dict[str, Solubility] = {"H2S": MercuryMagmaS()}

# Dictionary of all the composition solubilities. Lowercase key name by convention. All of the
# dictionaries with self-consistent solubility laws for a given composition (above) should be
# included in this dictionary.
composition_solubilities: dict[str, dict[str, Solubility]] = {
    "basalt": basalt_solubilities,
    "andesite": andesite_solubilities,
    "peridotite": peridotite_solubilities,
    "anorthiteDiopsideEuctectic": anorthdiop_solubilities,
    "reducedmagma": reducedmagma_solubilities,
    "rhyolite": rhyolite_solubilities,
}
