"""
Check the numpy version is valid.

Note that a development version is marked by the presence of 'dev0' or '+'
in the version string, all else is treated as a release. The version string
itself is set from the output of ``git describe`` which relies on tags.

Examples
--------

Valid Development: 1.22.0.dev0 1.22.0.dev0+5-g7999db4df2 1.22.0+5-g7999db4df2
Valid Release: 1.21.0.rc1, 1.21.0.b1, 1.21.0
Invalid: 1.22.0.dev, 1.22.0.dev0-5-g7999db4dfB, 1.21.0.d1, 1.21.a

Note that a release is determined by the version string, which in turn
is controlled by the result of the ``git describe`` command.
"""
import re

import numpy as np
from numpy.testing import assert_


def test_valid_numpy_version():
    # Verify that the numpy version is a valid one (no .post suffix or other
    # nonsense).  See gh-6431 for an issue caused by an invalid version.
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(a[0-9]|b[0-9]|rc[0-9]|)"
    dev_suffix = r"(\.dev0|)(\+[0-9]*\.g[0-9a-f]+|)"
    if np.version.release:
        res = re.match(version_pattern + '$', np.__version__)
    else:
        res = re.match(version_pattern + dev_suffix + '$', np.__version__)

    assert_(res is not None, np.__version__)


def test_short_version():
    # Check numpy.short_version actually exists
    if np.version.release:
        assert_(np.__version__ == np.version.short_version,
                "short_version mismatch in release version")
    else:
        assert_(np.__version__.split("+")[0] == np.version.short_version,
                "short_version mismatch in development version")
