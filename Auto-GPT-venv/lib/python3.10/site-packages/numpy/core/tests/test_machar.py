"""
Test machar. Given recent changes to hardcode type data, we might want to get
rid of both MachAr and this test at some point.

"""
from numpy.core._machar import MachAr
import numpy.core.numerictypes as ntypes
from numpy import errstate, array


class TestMachAr:
    def _run_machar_highprec(self):
        # Instantiate MachAr instance with high enough precision to cause
        # underflow
        try:
            hiprec = ntypes.float96
            MachAr(lambda v: array(v, hiprec))
        except AttributeError:
            # Fixme, this needs to raise a 'skip' exception.
            "Skipping test: no ntypes.float96 available on this platform."

    def test_underlow(self):
        # Regression test for #759:
        # instantiating MachAr for dtype = np.float96 raises spurious warning.
        with errstate(all='raise'):
            try:
                self._run_machar_highprec()
            except FloatingPointError as e:
                msg = "Caught %s exception, should not have been raised." % e
                raise AssertionError(msg)
