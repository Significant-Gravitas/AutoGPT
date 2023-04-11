from numpy.testing import assert_
import numpy.distutils.fcompiler

nag_version_strings = [('nagfor', 'NAG Fortran Compiler Release '
                        '6.2(Chiyoda) Build 6200', '6.2'),
                       ('nagfor', 'NAG Fortran Compiler Release '
                        '6.1(Tozai) Build 6136', '6.1'),
                       ('nagfor', 'NAG Fortran Compiler Release '
                        '6.0(Hibiya) Build 1021', '6.0'),
                       ('nagfor', 'NAG Fortran Compiler Release '
                        '5.3.2(971)', '5.3.2'),
                       ('nag', 'NAGWare Fortran 95 compiler Release 5.1'
                        '(347,355-367,375,380-383,389,394,399,401-402,407,'
                        '431,435,437,446,459-460,463,472,494,496,503,508,'
                        '511,517,529,555,557,565)', '5.1')]

class TestNagFCompilerVersions:
    def test_version_match(self):
        for comp, vs, version in nag_version_strings:
            fc = numpy.distutils.fcompiler.new_fcompiler(compiler=comp)
            v = fc.version_match(vs)
            assert_(v == version)
