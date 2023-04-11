"""Common test support for all numpy test scripts.

This single module should provide all the common functionality for numpy tests
in a single location, so that test scripts can just import it and work right
away.

"""
from unittest import TestCase

from . import _private
from ._private.utils import *
from ._private.utils import (_assert_valid_refcount, _gen_alignment_data)
from ._private import extbuild, decorators as dec
from ._private.nosetester import (
    run_module_suite, NoseTester as Tester
    )

__all__ = _private.utils.__all__ + ['TestCase', 'run_module_suite']

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
