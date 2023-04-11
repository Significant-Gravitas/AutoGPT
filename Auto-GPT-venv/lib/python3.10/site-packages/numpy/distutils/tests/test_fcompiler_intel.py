import numpy.distutils.fcompiler
from numpy.testing import assert_


intel_32bit_version_strings = [
    ("Intel(R) Fortran Intel(R) 32-bit Compiler Professional for applications"
     "running on Intel(R) 32, Version 11.1", '11.1'),
]

intel_64bit_version_strings = [
    ("Intel(R) Fortran IA-64 Compiler Professional for applications"
     "running on IA-64, Version 11.0", '11.0'),
    ("Intel(R) Fortran Intel(R) 64 Compiler Professional for applications"
     "running on Intel(R) 64, Version 11.1", '11.1')
]

class TestIntelFCompilerVersions:
    def test_32bit_version(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='intel')
        for vs, version in intel_32bit_version_strings:
            v = fc.version_match(vs)
            assert_(v == version)


class TestIntelEM64TFCompilerVersions:
    def test_64bit_version(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='intelem')
        for vs, version in intel_64bit_version_strings:
            v = fc.version_match(vs)
            assert_(v == version)
