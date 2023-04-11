from numpy import (
    logspace, linspace, geomspace, dtype, array, sctypes, arange, isnan,
    ndarray, sqrt, nextafter, stack, errstate
    )
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal, assert_allclose,
    )


class PhysicalQuantity(float):
    def __new__(cls, value):
        return float.__new__(cls, value)

    def __add__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) + float(self))
    __radd__ = __add__

    def __sub__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(self) - float(x))

    def __rsub__(self, x):
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) - float(self))

    def __mul__(self, x):
        return PhysicalQuantity(float(x) * float(self))
    __rmul__ = __mul__

    def __div__(self, x):
        return PhysicalQuantity(float(self) / float(x))

    def __rdiv__(self, x):
        return PhysicalQuantity(float(x) / float(self))


class PhysicalQuantity2(ndarray):
    __array_priority__ = 10


class TestLogspace:

    def test_basic(self):
        y = logspace(0, 6)
        assert_(len(y) == 50)
        y = logspace(0, 6, num=100)
        assert_(y[-1] == 10 ** 6)
        y = logspace(0, 6, endpoint=False)
        assert_(y[-1] < 10 ** 6)
        y = logspace(0, 6, num=7)
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])

    def test_start_stop_array(self):
        start = array([0., 1.])
        stop = array([6., 7.])
        t1 = logspace(start, stop, 6)
        t2 = stack([logspace(_start, _stop, 6)
                    for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = logspace(start, stop[0], 6)
        t4 = stack([logspace(_start, stop[0], 6)
                    for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = logspace(start, stop, 6, axis=-1)
        assert_equal(t5, t2.T)

    def test_dtype(self):
        y = logspace(0, 6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = logspace(0, 6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = logspace(0, 6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

    def test_physical_quantities(self):
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        assert_equal(logspace(a, b), logspace(1.0, 5.0))

    def test_subclass(self):
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        ls = logspace(a, b)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, logspace(1.0, 7.0))
        ls = logspace(a, b, 1)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, logspace(1.0, 7.0, 1))


class TestGeomspace:

    def test_basic(self):
        y = geomspace(1, 1e6)
        assert_(len(y) == 50)
        y = geomspace(1, 1e6, num=100)
        assert_(y[-1] == 10 ** 6)
        y = geomspace(1, 1e6, endpoint=False)
        assert_(y[-1] < 10 ** 6)
        y = geomspace(1, 1e6, num=7)
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])

        y = geomspace(8, 2, num=3)
        assert_allclose(y, [8, 4, 2])
        assert_array_equal(y.imag, 0)

        y = geomspace(-1, -100, num=3)
        assert_array_equal(y, [-1, -10, -100])
        assert_array_equal(y.imag, 0)

        y = geomspace(-100, -1, num=3)
        assert_array_equal(y, [-100, -10, -1])
        assert_array_equal(y.imag, 0)

    def test_boundaries_match_start_and_stop_exactly(self):
        # make sure that the boundaries of the returned array exactly
        # equal 'start' and 'stop' - this isn't obvious because
        # np.exp(np.log(x)) isn't necessarily exactly equal to x
        start = 0.3
        stop = 20.3

        y = geomspace(start, stop, num=1)
        assert_equal(y[0], start)

        y = geomspace(start, stop, num=1, endpoint=False)
        assert_equal(y[0], start)

        y = geomspace(start, stop, num=3)
        assert_equal(y[0], start)
        assert_equal(y[-1], stop)

        y = geomspace(start, stop, num=3, endpoint=False)
        assert_equal(y[0], start)

    def test_nan_interior(self):
        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4)

        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:-1]).all())
        assert_equal(y[3], 3.0)

        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4, endpoint=False)

        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:]).all())

    def test_complex(self):
        # Purely imaginary
        y = geomspace(1j, 16j, num=5)
        assert_allclose(y, [1j, 2j, 4j, 8j, 16j])
        assert_array_equal(y.real, 0)

        y = geomspace(-4j, -324j, num=5)
        assert_allclose(y, [-4j, -12j, -36j, -108j, -324j])
        assert_array_equal(y.real, 0)

        y = geomspace(1+1j, 1000+1000j, num=4)
        assert_allclose(y, [1+1j, 10+10j, 100+100j, 1000+1000j])

        y = geomspace(-1+1j, -1000+1000j, num=4)
        assert_allclose(y, [-1+1j, -10+10j, -100+100j, -1000+1000j])

        # Logarithmic spirals
        y = geomspace(-1, 1, num=3, dtype=complex)
        assert_allclose(y, [-1, 1j, +1])

        y = geomspace(0+3j, -3+0j, 3)
        assert_allclose(y, [0+3j, -3/sqrt(2)+3j/sqrt(2), -3+0j])
        y = geomspace(0+3j, 3+0j, 3)
        assert_allclose(y, [0+3j, 3/sqrt(2)+3j/sqrt(2), 3+0j])
        y = geomspace(-3+0j, 0-3j, 3)
        assert_allclose(y, [-3+0j, -3/sqrt(2)-3j/sqrt(2), 0-3j])
        y = geomspace(0+3j, -3+0j, 3)
        assert_allclose(y, [0+3j, -3/sqrt(2)+3j/sqrt(2), -3+0j])
        y = geomspace(-2-3j, 5+7j, 7)
        assert_allclose(y, [-2-3j, -0.29058977-4.15771027j,
                            2.08885354-4.34146838j, 4.58345529-3.16355218j,
                            6.41401745-0.55233457j, 6.75707386+3.11795092j,
                            5+7j])

        # Type promotion should prevent the -5 from becoming a NaN
        y = geomspace(3j, -5, 2)
        assert_allclose(y, [3j, -5])
        y = geomspace(-5, 3j, 2)
        assert_allclose(y, [-5, 3j])

    def test_dtype(self):
        y = geomspace(1, 1e6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = geomspace(1, 1e6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = geomspace(1, 1e6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

        # Native types
        y = geomspace(1, 1e6, dtype=float)
        assert_equal(y.dtype, dtype('float_'))
        y = geomspace(1, 1e6, dtype=complex)
        assert_equal(y.dtype, dtype('complex'))

    def test_start_stop_array_scalar(self):
        lim1 = array([120, 100], dtype="int8")
        lim2 = array([-120, -100], dtype="int8")
        lim3 = array([1200, 1000], dtype="uint16")
        t1 = geomspace(lim1[0], lim1[1], 5)
        t2 = geomspace(lim2[0], lim2[1], 5)
        t3 = geomspace(lim3[0], lim3[1], 5)
        t4 = geomspace(120.0, 100.0, 5)
        t5 = geomspace(-120.0, -100.0, 5)
        t6 = geomspace(1200.0, 1000.0, 5)

        # t3 uses float32, t6 uses float64
        assert_allclose(t1, t4, rtol=1e-2)
        assert_allclose(t2, t5, rtol=1e-2)
        assert_allclose(t3, t6, rtol=1e-5)

    def test_start_stop_array(self):
        # Try to use all special cases.
        start = array([1.e0, 32., 1j, -4j, 1+1j, -1])
        stop = array([1.e4, 2., 16j, -324j, 10000+10000j, 1])
        t1 = geomspace(start, stop, 5)
        t2 = stack([geomspace(_start, _stop, 5)
                    for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = geomspace(start, stop[0], 5)
        t4 = stack([geomspace(_start, stop[0], 5)
                    for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = geomspace(start, stop, 5, axis=-1)
        assert_equal(t5, t2.T)

    def test_physical_quantities(self):
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        assert_equal(geomspace(a, b), geomspace(1.0, 5.0))

    def test_subclass(self):
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        gs = geomspace(a, b)
        assert type(gs) is PhysicalQuantity2
        assert_equal(gs, geomspace(1.0, 7.0))
        gs = geomspace(a, b, 1)
        assert type(gs) is PhysicalQuantity2
        assert_equal(gs, geomspace(1.0, 7.0, 1))

    def test_bounds(self):
        assert_raises(ValueError, geomspace, 0, 10)
        assert_raises(ValueError, geomspace, 10, 0)
        assert_raises(ValueError, geomspace, 0, 0)


class TestLinspace:

    def test_basic(self):
        y = linspace(0, 10)
        assert_(len(y) == 50)
        y = linspace(2, 10, num=100)
        assert_(y[-1] == 10)
        y = linspace(2, 10, endpoint=False)
        assert_(y[-1] < 10)
        assert_raises(ValueError, linspace, 0, 10, num=-1)

    def test_corner(self):
        y = list(linspace(0, 1, 1))
        assert_(y == [0.0], y)
        assert_raises(TypeError, linspace, 0, 1, num=2.5)

    def test_type(self):
        t1 = linspace(0, 1, 0).dtype
        t2 = linspace(0, 1, 1).dtype
        t3 = linspace(0, 1, 2).dtype
        assert_equal(t1, t2)
        assert_equal(t2, t3)

    def test_dtype(self):
        y = linspace(0, 6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = linspace(0, 6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = linspace(0, 6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

    def test_start_stop_array_scalar(self):
        lim1 = array([-120, 100], dtype="int8")
        lim2 = array([120, -100], dtype="int8")
        lim3 = array([1200, 1000], dtype="uint16")
        t1 = linspace(lim1[0], lim1[1], 5)
        t2 = linspace(lim2[0], lim2[1], 5)
        t3 = linspace(lim3[0], lim3[1], 5)
        t4 = linspace(-120.0, 100.0, 5)
        t5 = linspace(120.0, -100.0, 5)
        t6 = linspace(1200.0, 1000.0, 5)
        assert_equal(t1, t4)
        assert_equal(t2, t5)
        assert_equal(t3, t6)

    def test_start_stop_array(self):
        start = array([-120, 120], dtype="int8")
        stop = array([100, -100], dtype="int8")
        t1 = linspace(start, stop, 5)
        t2 = stack([linspace(_start, _stop, 5)
                    for _start, _stop in zip(start, stop)], axis=1)
        assert_equal(t1, t2)
        t3 = linspace(start, stop[0], 5)
        t4 = stack([linspace(_start, stop[0], 5)
                    for _start in start], axis=1)
        assert_equal(t3, t4)
        t5 = linspace(start, stop, 5, axis=-1)
        assert_equal(t5, t2.T)

    def test_complex(self):
        lim1 = linspace(1 + 2j, 3 + 4j, 5)
        t1 = array([1.0+2.j, 1.5+2.5j,  2.0+3j, 2.5+3.5j, 3.0+4j])
        lim2 = linspace(1j, 10, 5)
        t2 = array([0.0+1.j, 2.5+0.75j, 5.0+0.5j, 7.5+0.25j, 10.0+0j])
        assert_equal(lim1, t1)
        assert_equal(lim2, t2)

    def test_physical_quantities(self):
        a = PhysicalQuantity(0.0)
        b = PhysicalQuantity(1.0)
        assert_equal(linspace(a, b), linspace(0.0, 1.0))

    def test_subclass(self):
        a = array(0).view(PhysicalQuantity2)
        b = array(1).view(PhysicalQuantity2)
        ls = linspace(a, b)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, linspace(0.0, 1.0))
        ls = linspace(a, b, 1)
        assert type(ls) is PhysicalQuantity2
        assert_equal(ls, linspace(0.0, 1.0, 1))

    def test_array_interface(self):
        # Regression test for https://github.com/numpy/numpy/pull/6659
        # Ensure that start/stop can be objects that implement
        # __array_interface__ and are convertible to numeric scalars

        class Arrayish:
            """
            A generic object that supports the __array_interface__ and hence
            can in principle be converted to a numeric scalar, but is not
            otherwise recognized as numeric, but also happens to support
            multiplication by floats.

            Data should be an object that implements the buffer interface,
            and contains at least 4 bytes.
            """

            def __init__(self, data):
                self._data = data

            @property
            def __array_interface__(self):
                return {'shape': (), 'typestr': '<i4', 'data': self._data,
                        'version': 3}

            def __mul__(self, other):
                # For the purposes of this test any multiplication is an
                # identity operation :)
                return self

        one = Arrayish(array(1, dtype='<i4'))
        five = Arrayish(array(5, dtype='<i4'))

        assert_equal(linspace(one, five), linspace(1, 5))

    def test_denormal_numbers(self):
        # Regression test for gh-5437. Will probably fail when compiled
        # with ICC, which flushes denormals to zero
        for ftype in sctypes['float']:
            stop = nextafter(ftype(0), ftype(1)) * 5  # A denormal number
            assert_(any(linspace(0, stop, 10, endpoint=False, dtype=ftype)))

    def test_equivalent_to_arange(self):
        for j in range(1000):
            assert_equal(linspace(0, j, j+1, dtype=int),
                         arange(j+1, dtype=int))

    def test_retstep(self):
        for num in [0, 1, 2]:
            for ept in [False, True]:
                y = linspace(0, 1, num, endpoint=ept, retstep=True)
                assert isinstance(y, tuple) and len(y) == 2
                if num == 2:
                    y0_expect = [0.0, 1.0] if ept else [0.0, 0.5]
                    assert_array_equal(y[0], y0_expect)
                    assert_equal(y[1], y0_expect[1])
                elif num == 1 and not ept:
                    assert_array_equal(y[0], [0.0])
                    assert_equal(y[1], 1.0)
                else:
                    assert_array_equal(y[0], [0.0][:num])
                    assert isnan(y[1])

    def test_object(self):
        start = array(1, dtype='O')
        stop = array(2, dtype='O')
        y = linspace(start, stop, 3)
        assert_array_equal(y, array([1., 1.5, 2.]))
                    
    def test_round_negative(self):
        y = linspace(-1, 3, num=8, dtype=int)
        t = array([-1, -1, 0, 0, 1, 1, 2, 3], dtype=int)
        assert_array_equal(y, t)
