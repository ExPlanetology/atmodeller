#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Solubility laws for other species

For every law there should be a test in the test suite.
"""

import sys

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.interfaces import RedoxBufferProtocol, SolubilityProtocol
from atmodeller.solubility.classes import Solubility, SolubilityPowerLaw, power_law
from atmodeller.solubility.core import fO2_temperature_correction
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import PyTreeNoData, safe_exp, unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

Cl2_ano_dio_for_thomas21: SolubilityProtocol = SolubilityPowerLaw(
    140.52 * unit_conversion.percent_to_ppm, 0.5
)
"""Cl in silicate melts :cite:p:`TW21`

Solubility law from :cite:t:`TW21{Figure 4}` showing relation between dissolved Cl concentration
and Cl fugacity for CMAS composition (An50Di28Fo22 (anorthite-diopside-forsterite), Fe-free 
low-degree mantle melt) at 1400 C and 1.5 GPa. Experiments from 0.5-2 GPa and 1200-1500 C.
"""

Cl2_basalt_thomas21: SolubilityProtocol = SolubilityPowerLaw(
    78.56 * unit_conversion.percent_to_ppm, 0.5
)
"""Cl in silicate melts :cite:p:`TW21`

Solubility law from :cite:t:`TW21{Figure 4}` showing relation between dissolved Cl concentration
and Cl fugacity for Icelandic basalt at 1400 C and 1.5 GPa. Experiments from 0.5-2 GPa and 
1200-1500 C.
"""

_He_henry_sol_constant_jambon86: float = 56e-5  # cm3*STP/g*bar
# Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
# Convert He conc from mol/g to g He/g total and then to ppmw
_He_henry_sol_constant_jambon86 = (
    (_He_henry_sol_constant_jambon86 / 2.24e4) * 4.0026 * unit_conversion.fraction_to_ppm
)
He_basalt_jambon86: SolubilityProtocol = SolubilityPowerLaw(_He_henry_sol_constant_jambon86, 1)
"""Solubility of He in tholeittic basalt melt :cite:p:`JWB86`

Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
1250-1600 C. Using Henry's Law solubility constant for He from the abstract, convert from STP
units to mol/g*bar.
"""

_Ar_henry_sol_constant_jambon86: float = 5.9e-5  # cm3*STP/g*bar
# Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
# Convert Ar conc from mol/g to g Ar/g total and then to ppmw
_Ar_henry_sol_constant_jambon86 = (
    (_Ar_henry_sol_constant_jambon86 / 2.24e4) * 39.948 * unit_conversion.fraction_to_ppm
)
Ar_basalt_jambon86: SolubilityProtocol = SolubilityPowerLaw(_Ar_henry_sol_constant_jambon86, 1)
"""Solubility of Ar in tholeittic basalt melt :cite:p:`JWB86`

Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
1250-1600 C. Using Henry's Law solubility constant for Ar from the abstract, convert from STP
units to mol/g*bar.
"""

_Ne_henry_sol_constant_jambon86: float = 25e-5  # cm3*STP/g*bar
# Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
# Convert Ne conc from mol/g to g Ne/g total and then to ppmw
_Ne_henry_sol_constant_jambon86 = (
    (_Ne_henry_sol_constant_jambon86 / 2.24e4) * 20.1797 * unit_conversion.fraction_to_ppm
)
Ne_basalt_jambon86: SolubilityProtocol = SolubilityPowerLaw(_Ne_henry_sol_constant_jambon86, 1)
"""Solubility of Ne in tholeittic basalt melt :cite:p:`JWB86`

Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
1250-1600 C. Using Henry's Law solubility constant for Ne from the abstract, convert from STP
units to mol/g*bar.
"""

_Kr_henry_sol_constant_jambon86: float = 3.0e-5  # cm3*STP/g*bar
# Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
# Convert Kr conc from mol/g to g Kr/g total and then to ppmw
_Kr_henry_sol_constant_jambon86 = (
    (_Kr_henry_sol_constant_jambon86 / 2.24e4) * 83.798 * unit_conversion.fraction_to_ppm
)
Kr_basalt_jambon86: SolubilityProtocol = SolubilityPowerLaw(_Kr_henry_sol_constant_jambon86, 1)
"""Solubility of Kr in tholeittic basalt melt :cite:p:`JWB86`

Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
1250-1600 C. Using Henry's Law solubility constant for Kr from the abstract, convert from STP
units to mol/g*bar.
"""

_Xe_henry_sol_constant_jambon86: float = 1.7e-5  # cm3*STP/g*bar
# Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
# Convert Xe conc from mol/g to g Xe/g total and then to ppmw
_Xe_henry_sol_constant_jambon86 = (
    (_Xe_henry_sol_constant_jambon86 / 2.24e4) * 131.293 * unit_conversion.fraction_to_ppm
)
Xe_basalt_jambon86: SolubilityProtocol = SolubilityPowerLaw(_Xe_henry_sol_constant_jambon86, 1)
"""Solubility of Xe in tholeittic basalt melt :cite:p:`JWB86`

Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
1250-1600 C. Using Henry's Law solubility constant for Xe from the abstract, convert from STP
units to mol/g*bar.
"""


@register_pytree_node_class
class _N2_basalt_bernadou21(PyTreeNoData, Solubility):
    """N2 in basaltic silicate melt :cite:p:`BGF21`

    :cite:t:`BGF21{Equation 18}` and using :cite:t:`BGF21{Equations 19-20}` and the values for the
    thermodynamic constants from :cite:t:`BGF21{Table 6}`. Experiments on basaltic samples at fluid
    saturation in C-H-O-N system, pressure range: 0.8-10 kbar, temperature range: 1200-1300 C;
    fO2 range: IW+4.9 to IW-4.7. Using their experimental results and a database for N
    concentrations at fluid saturation from 1 bar to 10 kbar, calibrated their solubility law.
    """

    @override
    @jit
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        # Numerator and denominator of k13 and k14 should both have units of J/mol so that k13 and
        # k14 are unitless
        k13: Array = safe_exp(
            -(29344 + 121 * temperature + 4 * pressure) / (GAS_CONSTANT_BAR * 1.0e5 * temperature)
        )
        k14: Array = safe_exp(
            -(183733 + 172 * temperature - 5 * pressure) / (GAS_CONSTANT_BAR * 1.0e5 * temperature)
        )
        molfrac: Array = (k13 * fugacity) + ((fO2 ** (-3 / 4)) * k14 * (fugacity**0.5))
        ppmw: Array = molfrac * unit_conversion.fraction_to_ppm

        return ppmw


N2_basalt_bernadou21: SolubilityProtocol = _N2_basalt_bernadou21()


@register_pytree_node_class
class _N2_basalt_dasgupta22(Solubility):
    """N2 in silicate melts :cite:p:`DFP22`

    Using :cite:t:`DFP22{Equation 10}`, composition parameters from :cite:t:`DFP22{Figure 8}`, and
    Iron-wustite buffer (logIW_fugacity) from :cite:t:`OP93,HGD08`.

    Performed experiments on 80:20 synthetic basalt-Si3N4 mixture at 1.5-3.0 GPa and 1300-1600 C
    fO2 from ~IW-3 to IW-4. Combined this high pressure data with lower pressure studies to derive
    their N solubility law.

    Args:
        xsio2: Mole fraction of SiO2. Defaults to 0.56.
        xal2o3: Mole fraction of Al2O3. Defaults to 0.11.
        xtio2: Mole fraction of TiO2. Defaults to 0.01.

    Attributes:
        xsio2: Mole fraction of SiO2
        xal2o3: Mole fraction of Al2O3
        xtio2: Mole fraction of TiO2
    """

    def __init__(self, xsio2: float = 0.56, xal2o3: float = 0.11, xtio2: float = 0.01):
        self.xsio2: float = xsio2
        self.xal2o3: float = xal2o3
        self.xtio2: float = xtio2
        # The buffer is evaluated at the pressure of interest, not the default of 1 bar
        self._buffer: RedoxBufferProtocol = IronWustiteBuffer(evaluation_pressure=None)

    @override
    @jit
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        fugacity_gpa: ArrayLike = fugacity * unit_conversion.bar_to_GPa
        pressure_gpa: ArrayLike = pressure * unit_conversion.bar_to_GPa

        fo2_shift: Array = jnp.log10(fO2) - self._buffer.log10_fugacity_buffer(
            temperature, pressure
        )
        ppmw: Array = safe_exp(
            (5908.0 * jnp.sqrt(pressure_gpa) / temperature) - (1.6 * fo2_shift)
        ) * jnp.sqrt(fugacity_gpa)
        ppmw = ppmw + fugacity_gpa * safe_exp(
            4.67 + (7.11 * self.xsio2) - (13.06 * self.xal2o3) - (120.67 * self.xtio2)
        )

        return ppmw

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {"xsio2": self.xsio2, "xal2o3": self.xal2o3, "xtio2": self.xtio2}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


N2_basalt_dasgupta22: SolubilityProtocol = _N2_basalt_dasgupta22()
"""N2 in basaltic silicate melt :cite:p:`BGF21`

:cite:t:`BGF21{Equation 18}` and using :cite:t:`BGF21{Equations 19-20}` and the values for the
thermodynamic constants from :cite:t:`BGF21{Table 6}`. Experiments on basaltic samples at fluid
saturation in C-H-O-N system, pressure range: 0.8-10 kbar, temperature range: 1200-1300 C;
fO2 range: IW+4.9 to IW-4.7. Using their experimental results and a database for N
concentrations at fluid saturation from 1 bar to 10 kbar, calibrated their solubility law.
"""


@register_pytree_node_class
class _N2_basalt_libourel03(PyTreeNoData, Solubility):
    """N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`

    :cite:t:`LMH03{Equation 23}`, includes dependencies on fN2 and fO2. Experiments conducted at 1
    atm and 1425 C (two experiments at 1400 C), fO2 from IW-8.3 to IW+8.7 using mixtures of CO, CO2
    and N2 gases.
    """

    @override
    @jit
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        # Libourel performed the experiment at 1698 K and fitted for fO2 at this temperature hence
        # a correction is necessary.
        libourel_temperature: ArrayLike = 1698.15
        adjusted_fO2: Array = fO2_temperature_correction(
            fO2,
            temperature=temperature,
            pressure=pressure,
            reference_temperature=libourel_temperature,
        )

        ppmw: Array = power_law(fugacity, 0.0611, 1)
        constant: Array = power_law(adjusted_fO2, 5.97e-10, -0.75)
        ppmw2: Array = power_law(fugacity, constant, 0.5)
        ppmw = ppmw + ppmw2

        return ppmw


N2_basalt_libourel03: SolubilityProtocol = _N2_basalt_libourel03()
"""N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`

:cite:t:`LMH03{Equation 23}`, includes dependencies on fN2 and fO2. Experiments conducted at 1
atm and 1425 C (two experiments at 1400 C), fO2 from IW-8.3 to IW+8.7 using mixtures of CO, CO2
and N2 gases.
"""
