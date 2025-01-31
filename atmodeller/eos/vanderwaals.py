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
"""Real gas EOSs from :cite:t:`L05`

Examples:
    Get the preferred EOS models for various species from the van der Waals models::
    
        from atmodeller.eos.vanderwaals import get_vanderwaals_eos_models
        models = get_vanderwaals_eos_models()
        # List the available species
        models.keys()
        # Get the EOS model for He
        H2_model = models['H2']
        # Determine the fugacity coefficient at 1000 K and 100 bar
        fugacity_coefficient = he_model.fugacity_coefficient(temperature=1000, pressure=100)
        print(fugacity_coefficient)
"""

# Use symbols from the paper for consistency so pylint: disable=C0103

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy.polynomial.polynomial import Polynomial

from atmodeller import ATMOSPHERE, GAS_CONSTANT_BAR
from atmodeller.eos.interfaces import ExperimentalCalibration, RealGas
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class vanderWaals(RealGas):
    """van der Waals equation

    Args:
        a: a constant units in :math:`\mathrm{m}^6\mathrm{bar}\mathrm{mol}^{-2}` 
        b: b constant units in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    a: float
    """a constant units in :math:`\mathrm{m}^6\mathrm{bar}\mathrm{mol}^{-2}` """
    b: float
    """b constant units :math:`\mathrm{m}^3\mathrm{mol}^{-1}` """
    standard_state_pressure: float = field(init=False, default=1)
    """Standard state pressure with the appropriate units"""

    def volume_roots(self, temperature: float, pressure: float) -> npt.NDArray:
        r"""Real and potentially physically meaningful volume solutions :cite:p:`L05{equation 2}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume solutions in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        coefficients: list[float] = []
        # van  der Waals
        coefficients.append(- self.a * self.b / pressure)
        coefficients.append(- self.a / pressure)
        coefficients.append(- self.b - GAS_CONSTANT_BAR * temperature / pressure)
        coefficients.append(1)

        polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="V")
        logger.debug("van der Waals equation = %s", polynomial)
        volume_roots: npt.NDArray = polynomial.roots()
        logger.debug("volume_roots = %s", volume_roots)
        # Numerical solution could result in a small imaginery component, even though the real
        # root is purely real.
        real_roots: npt.NDArray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
        # Physically meaningful volumes must be positive.
        positive_roots: npt.NDArray = real_roots[real_roots > 0]
        # In general, several roots could be returned, and subclasses will need to determine which
        # is the correct volume to use.
        logger.debug("V = %s", positive_roots)

        return positive_roots

    @override
    def volume(self, *args, **kwargs) -> float:
        r"""Volume

        :cite:t:`L05` doesn't say which root to take, but one root is very small and the maximum
        root gives a volume that agrees with the tabulated compressibility factor for all species.

        Args:
            *args: Positional arguments to pass to :func:`volume_roots`
            **kwargs: Keyword arguments to pass to :func:`volume_roots`

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        volume_roots: npt.NDArray = self.volume_roots(*args, **kwargs)

        return np.max(volume_roots)
    
    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        r"""Volume integral :cite:p:`L05{Equation 11}`.

        It is necessary to multiply :math:`\ln f` by :math:`RT` to obtain the volume integral.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        T: float = temperature
        P: float = pressure
        P0: float = self.standard_state_pressure

        vol: float = self.volume(T, P)
        vol0: float = self.volume(T, P0)
        volume_integral: float = (
            (
                self.b * (vol0 - vol) / ((vol - self.b) * (vol0 - self.b)) 
                - np.log((vol - self.b) / (vol0 - self.b))
            )
            * GAS_CONSTANT_BAR
            * T
            - 2 * self.a * (1/vol - 1/vol0)
        )

        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral

# van der Waals cefficients from David R. Lide, ed., CRC Handbook of Chemistry and Physics,
# Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005

## TO DO: Check calibration range and provide the right citation instead of "L05"

H2_Lide_vanderwaals: RealGas = vanderWaals(
    a=2.452e-7,
    b=2.65e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""H2 van der Waals :cite:p:`L05`"""
He_Lide_vanderwaals: RealGas = vanderWaals(
    a=3.46e-8,
    b=2.38e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""He van der Waals :cite:p:`L05`"""
N2_Lide_vanderwaals: RealGas = vanderWaals(
    a=1.37e-6,
    b=3.87e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""N2 van der Waals :cite:p:`L05`"""
H4Si_Lide_vanderwaals: RealGas = vanderWaals(
    a=4.38e-6,
    b=5.79e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""SiH4 van der Waals :cite:p:`L05`"""
H2O_Lide_vanderwaals: RealGas = vanderWaals(
    a=5.537e-6,
    b=3.05e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""H2O van der Waals :cite:p:`L05`"""
CH4_Lide_vanderwaals: RealGas = vanderWaals(
    a=2.303e-6,
    b=4.31e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""CH4 van der Waals :cite:p:`L05`"""
H3N_Lide_vanderwaals: RealGas = vanderWaals(
    a=4.225e-6,
    b=3.71e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""NH3 van der Waals :cite:p:`L05`"""
CHN_Lide_vanderwaals: RealGas = vanderWaals(
    a=1.29e-5,
    b=8.81e-5,
    calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
)
"""HCN van der Waals :cite:p:`L05`"""
# H4Si_Isham_vanderwaals: RealGas = vanderWaals(
#     a=2.478e-6,
#     b=3.275e-5,
#     calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
# )
# """SiH4 van der Waals :cite:p:`L05`"""
# OSi_Isham_vanderwaals: RealGas = vanderWaals(
#     a=8.698e-6,
#     b=8.582e-6,
#     calibration=ExperimentalCalibration(100, 1000, 0.1, 1000),
# )
# """SiH4 van der Waals :cite:p:`L05`"""

def get_vanderwaals_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred van der Waals EOS models for each species.

    Returns:
        Dictionary of EOS models for each species
    """
    models: dict[str, RealGas] = {}

    models["H2"] = H2_Lide_vanderwaals
    models["He"] = He_Lide_vanderwaals
    models["N2"] = N2_Lide_vanderwaals
    models["H4Si"] = H4Si_Lide_vanderwaals
    models["H2O"] = H2O_Lide_vanderwaals
    models["CH4"] = CH4_Lide_vanderwaals
    models["H3N"] = H3N_Lide_vanderwaals
    models["CHN"] = CHN_Lide_vanderwaals

    return models
