""" Test printing of scalar types.

"""
import code
import platform
import pytest
import sys

from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises

class TestRealScalars:
    def test_str(self):
        svals = [0.0, -0.0, 1, -1, np.inf, -np.inf, np.nan]
        styps = [np.float16, np.float32, np.float64, np.longdouble]
        wanted = [
             ['0.0',  '0.0',  '0.0',  '0.0' ],
             ['-0.0', '-0.0', '-0.0', '-0.0'],
             ['1.0',  '1.0',  '1.0',  '1.0' ],
             ['-1.0', '-1.0', '-1.0', '-1.0'],
             ['inf',  'inf',  'inf',  'inf' ],
             ['-inf', '-inf', '-inf', '-inf'],
             ['nan',  'nan',  'nan',  'nan']]

        for wants, val in zip(wanted, svals):
            for want, styp in zip(wants, styps):
                msg = 'for str({}({}))'.format(np.dtype(styp).name, repr(val))
                assert_equal(str(styp(val)), want, err_msg=msg)

    def test_scalar_cutoffs(self):
        # test that both the str and repr of np.float64 behaves
        # like python floats in python3.
        def check(v):
            assert_equal(str(np.float64(v)), str(v))
            assert_equal(str(np.float64(v)), repr(v))
            assert_equal(repr(np.float64(v)), repr(v))
            assert_equal(repr(np.float64(v)), str(v))

        # check we use the same number of significant digits
        check(1.12345678901234567890)
        check(0.0112345678901234567890)

        # check switch from scientific output to positional and back
        check(1e-5)
        check(1e-4)
        check(1e15)
        check(1e16)

    def test_py2_float_print(self):
        # gh-10753
        # In python2, the python float type implements an obsolete method
        # tp_print, which overrides tp_repr and tp_str when using "print" to
        # output to a "real file" (ie, not a StringIO). Make sure we don't
        # inherit it.
        x = np.double(0.1999999999999)
        with TemporaryFile('r+t') as f:
            print(x, file=f)
            f.seek(0)
            output = f.read()
        assert_equal(output, str(x) + '\n')
        # In python2 the value float('0.1999999999999') prints with reduced
        # precision as '0.2', but we want numpy's np.double('0.1999999999999')
        # to print the unique value, '0.1999999999999'.

        # gh-11031
        # Only in the python2 interactive shell and when stdout is a "real"
        # file, the output of the last command is printed to stdout without
        # Py_PRINT_RAW (unlike the print statement) so `>>> x` and `>>> print
        # x` are potentially different. Make sure they are the same. The only
        # way I found to get prompt-like output is using an actual prompt from
        # the 'code' module. Again, must use tempfile to get a "real" file.

        # dummy user-input which enters one line and then ctrl-Ds.
        def userinput():
            yield 'np.sqrt(2)'
            raise EOFError
        gen = userinput()
        input_func = lambda prompt="": next(gen)

        with TemporaryFile('r+t') as fo, TemporaryFile('r+t') as fe:
            orig_stdout, orig_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = fo, fe

            code.interact(local={'np': np}, readfunc=input_func, banner='')

            sys.stdout, sys.stderr = orig_stdout, orig_stderr

            fo.seek(0)
            capture = fo.read().strip()

        assert_equal(capture, repr(np.sqrt(2)))

    def test_dragon4(self):
        # these tests are adapted from Ryan Juckett's dragon4 implementation,
        # see dragon4.c for details.

        fpos32 = lambda x, **k: np.format_float_positional(np.float32(x), **k)
        fsci32 = lambda x, **k: np.format_float_scientific(np.float32(x), **k)
        fpos64 = lambda x, **k: np.format_float_positional(np.float64(x), **k)
        fsci64 = lambda x, **k: np.format_float_scientific(np.float64(x), **k)

        preckwd = lambda prec: {'unique': False, 'precision': prec}

        assert_equal(fpos32('1.0'), "1.")
        assert_equal(fsci32('1.0'), "1.e+00")
        assert_equal(fpos32('10.234'), "10.234")
        assert_equal(fpos32('-10.234'), "-10.234")
        assert_equal(fsci32('10.234'), "1.0234e+01")
        assert_equal(fsci32('-10.234'), "-1.0234e+01")
        assert_equal(fpos32('1000.0'), "1000.")
        assert_equal(fpos32('1.0', precision=0), "1.")
        assert_equal(fsci32('1.0', precision=0), "1.e+00")
        assert_equal(fpos32('10.234', precision=0), "10.")
        assert_equal(fpos32('-10.234', precision=0), "-10.")
        assert_equal(fsci32('10.234', precision=0), "1.e+01")
        assert_equal(fsci32('-10.234', precision=0), "-1.e+01")
        assert_equal(fpos32('10.234', precision=2), "10.23")
        assert_equal(fsci32('-10.234', precision=2), "-1.02e+01")
        assert_equal(fsci64('9.9999999999999995e-08', **preckwd(16)),
                            '9.9999999999999995e-08')
        assert_equal(fsci64('9.8813129168249309e-324', **preckwd(16)),
                            '9.8813129168249309e-324')
        assert_equal(fsci64('9.9999999999999694e-311', **preckwd(16)),
                            '9.9999999999999694e-311')


        # test rounding
        # 3.1415927410 is closest float32 to np.pi
        assert_equal(fpos32('3.14159265358979323846', **preckwd(10)),
                            "3.1415927410")
        assert_equal(fsci32('3.14159265358979323846', **preckwd(10)),
                            "3.1415927410e+00")
        assert_equal(fpos64('3.14159265358979323846', **preckwd(10)),
                            "3.1415926536")
        assert_equal(fsci64('3.14159265358979323846', **preckwd(10)),
                            "3.1415926536e+00")
        # 299792448 is closest float32 to 299792458
        assert_equal(fpos32('299792458.0', **preckwd(5)), "299792448.00000")
        assert_equal(fsci32('299792458.0', **preckwd(5)), "2.99792e+08")
        assert_equal(fpos64('299792458.0', **preckwd(5)), "299792458.00000")
        assert_equal(fsci64('299792458.0', **preckwd(5)), "2.99792e+08")

        assert_equal(fpos32('3.14159265358979323846', **preckwd(25)),
                            "3.1415927410125732421875000")
        assert_equal(fpos64('3.14159265358979323846', **preckwd(50)),
                         "3.14159265358979311599796346854418516159057617187500")
        assert_equal(fpos64('3.14159265358979323846'), "3.141592653589793")


        # smallest numbers
        assert_equal(fpos32(0.5**(126 + 23), unique=False, precision=149),
                    "0.00000000000000000000000000000000000000000000140129846432"
                    "4817070923729583289916131280261941876515771757068283889791"
                    "08268586060148663818836212158203125")
        
        assert_equal(fpos64(5e-324, unique=False, precision=1074),
                    "0.00000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000049406564584124654417656"
                    "8792868221372365059802614324764425585682500675507270208751"
                    "8652998363616359923797965646954457177309266567103559397963"
                    "9877479601078187812630071319031140452784581716784898210368"
                    "8718636056998730723050006387409153564984387312473397273169"
                    "6151400317153853980741262385655911710266585566867681870395"
                    "6031062493194527159149245532930545654440112748012970999954"
                    "1931989409080416563324524757147869014726780159355238611550"
                    "1348035264934720193790268107107491703332226844753335720832"
                    "4319360923828934583680601060115061698097530783422773183292"
                    "4790498252473077637592724787465608477820373446969953364701"
                    "7972677717585125660551199131504891101451037862738167250955"
                    "8373897335989936648099411642057026370902792427675445652290"
                    "87538682506419718265533447265625")

        # largest numbers
        f32x = np.finfo(np.float32).max
        assert_equal(fpos32(f32x, **preckwd(0)),
                    "340282346638528859811704183484516925440.")
        assert_equal(fpos64(np.finfo(np.float64).max, **preckwd(0)),
                    "1797693134862315708145274237317043567980705675258449965989"
                    "1747680315726078002853876058955863276687817154045895351438"
                    "2464234321326889464182768467546703537516986049910576551282"
                    "0762454900903893289440758685084551339423045832369032229481"
                    "6580855933212334827479782620414472316873817718091929988125"
                    "0404026184124858368.")
        # Warning: In unique mode only the integer digits necessary for
        # uniqueness are computed, the rest are 0.
        assert_equal(fpos32(f32x),
                    "340282350000000000000000000000000000000.")

        # Further tests of zero-padding vs rounding in different combinations
        # of unique, fractional, precision, min_digits
        # precision can only reduce digits, not add them.
        # min_digits can only extend digits, not reduce them.
        assert_equal(fpos32(f32x, unique=True, fractional=True, precision=0),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=True, precision=4),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=0),
                    "340282346638528859811704183484516925440.")
        assert_equal(fpos32(f32x, unique=True, fractional=True, min_digits=4),
                    "340282346638528859811704183484516925440.0000")
        assert_equal(fpos32(f32x, unique=True, fractional=True,
                                    min_digits=4, precision=4),
                    "340282346638528859811704183484516925440.0000")
        assert_raises(ValueError, fpos32, f32x, unique=True, fractional=False,
                                          precision=0)
        assert_equal(fpos32(f32x, unique=True, fractional=False, precision=4),
                    "340300000000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False, precision=20),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False, min_digits=4),
                    "340282350000000000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False,
                                  min_digits=20),
                    "340282346638528859810000000000000000000.")
        assert_equal(fpos32(f32x, unique=True, fractional=False,
                                  min_digits=15),
                    "340282346638529000000000000000000000000.")
        assert_equal(fpos32(f32x, unique=False, fractional=False, precision=4),
                    "340300000000000000000000000000000000000.")
        # test that unique rounding is preserved when precision is supplied
        # but no extra digits need to be printed (gh-18609)
        a = np.float64.fromhex('-1p-97')
        assert_equal(fsci64(a, unique=True), '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=False, precision=15),
                     '-6.310887241768094e-30')
        assert_equal(fsci64(a, unique=True, precision=15),
                     '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, min_digits=15),
                     '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, precision=15, min_digits=15),
                     '-6.310887241768095e-30')
        # adds/remove digits in unique mode with unbiased rnding
        assert_equal(fsci64(a, unique=True, precision=14),
                     '-6.31088724176809e-30')
        assert_equal(fsci64(a, unique=True, min_digits=16),
                     '-6.3108872417680944e-30')
        assert_equal(fsci64(a, unique=True, precision=16),
                     '-6.310887241768095e-30')
        assert_equal(fsci64(a, unique=True, min_digits=14),
                     '-6.310887241768095e-30')
        # test min_digits in unique mode with different rounding cases
        assert_equal(fsci64('1e120', min_digits=3), '1.000e+120')
        assert_equal(fsci64('1e100', min_digits=3), '1.000e+100')

        # test trailing zeros
        assert_equal(fpos32('1.0', unique=False, precision=3), "1.000")
        assert_equal(fpos64('1.0', unique=False, precision=3), "1.000")
        assert_equal(fsci32('1.0', unique=False, precision=3), "1.000e+00")
        assert_equal(fsci64('1.0', unique=False, precision=3), "1.000e+00")
        assert_equal(fpos32('1.5', unique=False, precision=3), "1.500")
        assert_equal(fpos64('1.5', unique=False, precision=3), "1.500")
        assert_equal(fsci32('1.5', unique=False, precision=3), "1.500e+00")
        assert_equal(fsci64('1.5', unique=False, precision=3), "1.500e+00")
        # gh-10713
        assert_equal(fpos64('324', unique=False, precision=5,
                                   fractional=False), "324.00")


    def test_dragon4_interface(self):
        tps = [np.float16, np.float32, np.float64]
        if hasattr(np, 'float128'):
            tps.append(np.float128)

        fpos = np.format_float_positional
        fsci = np.format_float_scientific

        for tp in tps:
            # test padding
            assert_equal(fpos(tp('1.0'), pad_left=4, pad_right=4), "   1.    ")
            assert_equal(fpos(tp('-1.0'), pad_left=4, pad_right=4), "  -1.    ")
            assert_equal(fpos(tp('-10.2'),
                         pad_left=4, pad_right=4), " -10.2   ")

            # test exp_digits
            assert_equal(fsci(tp('1.23e1'), exp_digits=5), "1.23e+00001")

            # test fixed (non-unique) mode
            assert_equal(fpos(tp('1.0'), unique=False, precision=4), "1.0000")
            assert_equal(fsci(tp('1.0'), unique=False, precision=4),
                         "1.0000e+00")

            # test trimming
            # trim of 'k' or '.' only affects non-unique mode, since unique
            # mode will not output trailing 0s.
            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='k'),
                         "1.0000")

            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='.'),
                         "1.")
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='.'),
                         "1.2" if tp != np.float16 else "1.2002")

            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='0'),
                         "1.0")
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='0'),
                         "1.2" if tp != np.float16 else "1.2002")
            assert_equal(fpos(tp('1.'), trim='0'), "1.0")

            assert_equal(fpos(tp('1.'), unique=False, precision=4, trim='-'),
                         "1")
            assert_equal(fpos(tp('1.2'), unique=False, precision=4, trim='-'),
                         "1.2" if tp != np.float16 else "1.2002")
            assert_equal(fpos(tp('1.'), trim='-'), "1")
            assert_equal(fpos(tp('1.001'), precision=1, trim='-'), "1")

    @pytest.mark.skipif(not platform.machine().startswith("ppc64"),
                        reason="only applies to ppc float128 values")
    def test_ppc64_ibm_double_double128(self):
        # check that the precision decreases once we get into the subnormal
        # range. Unlike float64, this starts around 1e-292 instead of 1e-308,
        # which happens when the first double is normal and the second is
        # subnormal.
        x = np.float128('2.123123123123123123123123123123123e-286')
        got = [str(x/np.float128('2e' + str(i))) for i in range(0,40)]
        expected = [
            "1.06156156156156156156156156156157e-286",
            "1.06156156156156156156156156156158e-287",
            "1.06156156156156156156156156156159e-288",
            "1.0615615615615615615615615615616e-289",
            "1.06156156156156156156156156156157e-290",
            "1.06156156156156156156156156156156e-291",
            "1.0615615615615615615615615615616e-292",
            "1.0615615615615615615615615615615e-293",
            "1.061561561561561561561561561562e-294",
            "1.06156156156156156156156156155e-295",
            "1.0615615615615615615615615616e-296",
            "1.06156156156156156156156156e-297",
            "1.06156156156156156156156157e-298",
            "1.0615615615615615615615616e-299",
            "1.06156156156156156156156e-300",
            "1.06156156156156156156155e-301",
            "1.0615615615615615615616e-302",
            "1.061561561561561561562e-303",
            "1.06156156156156156156e-304",
            "1.0615615615615615618e-305",
            "1.06156156156156156e-306",
            "1.06156156156156157e-307",
            "1.0615615615615616e-308",
            "1.06156156156156e-309",
            "1.06156156156157e-310",
            "1.0615615615616e-311",
            "1.06156156156e-312",
            "1.06156156154e-313",
            "1.0615615616e-314",
            "1.06156156e-315",
            "1.06156155e-316",
            "1.061562e-317",
            "1.06156e-318",
            "1.06155e-319",
            "1.0617e-320",
            "1.06e-321",
            "1.04e-322",
            "1e-323",
            "0.0",
            "0.0"]
        assert_equal(got, expected)

        # Note: we follow glibc behavior, but it (or gcc) might not be right.
        # In particular we can get two values that print the same but are not
        # equal:
        a = np.float128('2')/np.float128('3')
        b = np.float128(str(a))
        assert_equal(str(a), str(b))
        assert_(a != b)

    def float32_roundtrip(self):
        # gh-9360
        x = np.float32(1024 - 2**-14)
        y = np.float32(1024 - 2**-13)
        assert_(repr(x) != repr(y))
        assert_equal(np.float32(repr(x)), x)
        assert_equal(np.float32(repr(y)), y)

    def float64_vs_python(self):
        # gh-2643, gh-6136, gh-6908
        assert_equal(repr(np.float64(0.1)), repr(0.1))
        assert_(repr(np.float64(0.20000000000000004)) != repr(0.2))
