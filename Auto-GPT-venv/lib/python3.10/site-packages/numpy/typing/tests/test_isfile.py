import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_

ROOT = Path(np.__file__).parents[0]
FILES = [
    ROOT / "py.typed",
    ROOT / "__init__.pyi",
    ROOT / "ctypeslib.pyi",
    ROOT / "core" / "__init__.pyi",
    ROOT / "distutils" / "__init__.pyi",
    ROOT / "f2py" / "__init__.pyi",
    ROOT / "fft" / "__init__.pyi",
    ROOT / "lib" / "__init__.pyi",
    ROOT / "linalg" / "__init__.pyi",
    ROOT / "ma" / "__init__.pyi",
    ROOT / "matrixlib" / "__init__.pyi",
    ROOT / "polynomial" / "__init__.pyi",
    ROOT / "random" / "__init__.pyi",
    ROOT / "testing" / "__init__.pyi",
]


class TestIsFile:
    def test_isfile(self):
        """Test if all ``.pyi`` files are properly installed."""
        for file in FILES:
            assert_(os.path.isfile(file))
