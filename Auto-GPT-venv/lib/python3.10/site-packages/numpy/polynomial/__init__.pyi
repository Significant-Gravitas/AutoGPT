from numpy._pytesttester import PytestTester

from numpy.polynomial import (
    chebyshev as chebyshev,
    hermite as hermite,
    hermite_e as hermite_e,
    laguerre as laguerre,
    legendre as legendre,
    polynomial as polynomial,
)
from numpy.polynomial.chebyshev import Chebyshev as Chebyshev
from numpy.polynomial.hermite import Hermite as Hermite
from numpy.polynomial.hermite_e import HermiteE as HermiteE
from numpy.polynomial.laguerre import Laguerre as Laguerre
from numpy.polynomial.legendre import Legendre as Legendre
from numpy.polynomial.polynomial import Polynomial as Polynomial

__all__: list[str]
__path__: list[str]
test: PytestTester

def set_default_printstyle(style): ...
