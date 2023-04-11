import itertools
import pytest

import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
    assert_, assert_raises, assert_equal, assert_array_equal
    )


ndims = 2
size = 10
shape = tuple([size] * ndims)

MAY_SHARE_BOUNDS = 0
MAY_SHARE_EXACT = -1


def _indices_for_nelems(nelems):
    """Returns slices of length nelems, from start onwards, in direction sign."""

    if nelems == 0:
        return [size // 2]  # int index

    res = []
    for step in (1, 2):
        for sign in (-1, 1):
            start = size // 2 - nelems * step * sign // 2
            stop = start + nelems * step * sign
            res.append(slice(start, stop, step * sign))

    return res


def _indices_for_axis():
    """Returns (src, dst) pairs of indices."""

    res = []
    for nelems in (0, 2, 3):
        ind = _indices_for_nelems(nelems)
        res.extend(itertools.product(ind, ind))  # all assignments of size "nelems"

    return res


def _indices(ndims):
    """Returns ((axis0_src, axis0_dst), (axis1_src, axis1_dst), ... ) index pairs."""

    ind = _indices_for_axis()
    return itertools.product(ind, repeat=ndims)


def _check_assignment(srcidx, dstidx):
    """Check assignment arr[dstidx] = arr[srcidx] works."""

    arr = np.arange(np.product(shape)).reshape(shape)

    cpy = arr.copy()

    cpy[dstidx] = arr[srcidx]
    arr[dstidx] = arr[srcidx]

    assert_(np.all(arr == cpy),
            'assigning arr[%s] = arr[%s]' % (dstidx, srcidx))


def test_overlapping_assignments():
    # Test automatically generated assignments which overlap in memory.

    inds = _indices(ndims)

    for ind in inds:
        srcidx = tuple([a[0] for a in ind])
        dstidx = tuple([a[1] for a in ind])

        _check_assignment(srcidx, dstidx)


@pytest.mark.slow
def test_diophantine_fuzz():
    # Fuzz test the diophantine solver
    rng = np.random.RandomState(1234)

    max_int = np.iinfo(np.intp).max

    for ndim in range(10):
        feasible_count = 0
        infeasible_count = 0

        min_count = 500//(ndim + 1)

        while min(feasible_count, infeasible_count) < min_count:
            # Ensure big and small integer problems
            A_max = 1 + rng.randint(0, 11, dtype=np.intp)**6
            U_max = rng.randint(0, 11, dtype=np.intp)**6

            A_max = min(max_int, A_max)
            U_max = min(max_int-1, U_max)

            A = tuple(int(rng.randint(1, A_max+1, dtype=np.intp))
                      for j in range(ndim))
            U = tuple(int(rng.randint(0, U_max+2, dtype=np.intp))
                      for j in range(ndim))

            b_ub = min(max_int-2, sum(a*ub for a, ub in zip(A, U)))
            b = int(rng.randint(-1, b_ub+2, dtype=np.intp))

            if ndim == 0 and feasible_count < min_count:
                b = 0

            X = solve_diophantine(A, U, b)

            if X is None:
                # Check the simplified decision problem agrees
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is None, (A, U, b, X_simplified))

                # Check no solution exists (provided the problem is
                # small enough so that brute force checking doesn't
                # take too long)
                ranges = tuple(range(0, a*ub+1, a) for a, ub in zip(A, U))

                size = 1
                for r in ranges:
                    size *= len(r)
                if size < 100000:
                    assert_(not any(sum(w) == b for w in itertools.product(*ranges)))
                    infeasible_count += 1
            else:
                # Check the simplified decision problem agrees
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is not None, (A, U, b, X_simplified))

                # Check validity
                assert_(sum(a*x for a, x in zip(A, X)) == b)
                assert_(all(0 <= x <= ub for x, ub in zip(X, U)))
                feasible_count += 1


def test_diophantine_overflow():
    # Smoke test integer overflow detection
    max_intp = np.iinfo(np.intp).max
    max_int64 = np.iinfo(np.int64).max

    if max_int64 <= max_intp:
        # Check that the algorithm works internally in 128-bit;
        # solving this problem requires large intermediate numbers
        A = (max_int64//2, max_int64//2 - 10)
        U = (max_int64//2, max_int64//2 - 10)
        b = 2*(max_int64//2) - 10

        assert_equal(solve_diophantine(A, U, b), (1, 1))


def check_may_share_memory_exact(a, b):
    got = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

    assert_equal(np.may_share_memory(a, b),
                 np.may_share_memory(a, b, max_work=MAY_SHARE_BOUNDS))

    a.fill(0)
    b.fill(0)
    a.fill(1)
    exact = b.any()

    err_msg = ""
    if got != exact:
        err_msg = "    " + "\n    ".join([
            "base_a - base_b = %r" % (a.__array_interface__['data'][0] - b.__array_interface__['data'][0],),
            "shape_a = %r" % (a.shape,),
            "shape_b = %r" % (b.shape,),
            "strides_a = %r" % (a.strides,),
            "strides_b = %r" % (b.strides,),
            "size_a = %r" % (a.size,),
            "size_b = %r" % (b.size,)
        ])

    assert_equal(got, exact, err_msg=err_msg)


def test_may_share_memory_manual():
    # Manual test cases for may_share_memory

    # Base arrays
    xs0 = [
        np.zeros([13, 21, 23, 22], dtype=np.int8),
        np.zeros([13, 21, 23*2, 22], dtype=np.int8)[:,:,::2,:]
    ]

    # Generate all negative stride combinations
    xs = []
    for x in xs0:
        for ss in itertools.product(*(([slice(None), slice(None, None, -1)],)*4)):
            xp = x[ss]
            xs.append(xp)

    for x in xs:
        # The default is a simple extent check
        assert_(np.may_share_memory(x[:,0,:], x[:,1,:]))
        assert_(np.may_share_memory(x[:,0,:], x[:,1,:], max_work=None))

        # Exact checks
        check_may_share_memory_exact(x[:,0,:], x[:,1,:])
        check_may_share_memory_exact(x[:,::7], x[:,3::3])

        try:
            xp = x.ravel()
            if xp.flags.owndata:
                continue
            xp = xp.view(np.int16)
        except ValueError:
            continue

        # 0-size arrays cannot overlap
        check_may_share_memory_exact(x.ravel()[6:6],
                                     xp.reshape(13, 21, 23, 11)[:,::7])

        # Test itemsize is dealt with
        check_may_share_memory_exact(x[:,::7],
                                     xp.reshape(13, 21, 23, 11))
        check_may_share_memory_exact(x[:,::7],
                                     xp.reshape(13, 21, 23, 11)[:,3::3])
        check_may_share_memory_exact(x.ravel()[6:7],
                                     xp.reshape(13, 21, 23, 11)[:,::7])

    # Check unit size
    x = np.zeros([1], dtype=np.int8)
    check_may_share_memory_exact(x, x)
    check_may_share_memory_exact(x, x.copy())


def iter_random_view_pairs(x, same_steps=True, equal_size=False):
    rng = np.random.RandomState(1234)

    if equal_size and same_steps:
        raise ValueError()

    def random_slice(n, step):
        start = rng.randint(0, n+1, dtype=np.intp)
        stop = rng.randint(start, n+1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    def random_slice_fixed_size(n, step, size):
        start = rng.randint(0, n+1 - size*step)
        stop = start + (size-1)*step + 1
        if rng.randint(0, 2) == 0:
            stop, start = start-1, stop-1
            if stop < 0:
                stop = None
            step *= -1
        return slice(start, stop, step)

    # First a few regular views
    yield x, x
    for j in range(1, 7, 3):
        yield x[j:], x[:-j]
        yield x[...,j:], x[...,:-j]

    # An array with zero stride internal overlap
    strides = list(x.strides)
    strides[0] = 0
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield x, xp
    yield xp, xp

    # An array with non-zero stride internal overlap
    strides = list(x.strides)
    if strides[0] > 1:
        strides[0] = 1
    xp = as_strided(x, shape=x.shape, strides=strides)
    yield x, xp
    yield xp, xp

    # Then discontiguous views
    while True:
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))

        t1 = np.arange(x.ndim)
        rng.shuffle(t1)

        if equal_size:
            t2 = t1
        else:
            t2 = np.arange(x.ndim)
            rng.shuffle(t2)

        a = x[s1]

        if equal_size:
            if a.size == 0:
                continue

            steps2 = tuple(rng.randint(1, max(2, p//(1+pa)))
                           if rng.randint(0, 5) == 0 else 1
                           for p, s, pa in zip(x.shape, s1, a.shape))
            s2 = tuple(random_slice_fixed_size(p, s, pa)
                       for p, s, pa in zip(x.shape, steps2, a.shape))
        elif same_steps:
            steps2 = steps
        else:
            steps2 = tuple(rng.randint(1, 11, dtype=np.intp)
                           if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                           for j in range(x.ndim))

        if not equal_size:
            s2 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps2))

        a = a.transpose(t1)
        b = x[s2].transpose(t2)

        yield a, b


def check_may_share_memory_easy_fuzz(get_max_work, same_steps, min_count):
    # Check that overlap problems with common strides are solved with
    # little work.
    x = np.zeros([17,34,71,97], dtype=np.int16)

    feasible = 0
    infeasible = 0

    pair_iter = iter_random_view_pairs(x, same_steps)

    while min(feasible, infeasible) < min_count:
        a, b = next(pair_iter)

        bounds_overlap = np.may_share_memory(a, b)
        may_share_answer = np.may_share_memory(a, b)
        easy_answer = np.may_share_memory(a, b, max_work=get_max_work(a, b))
        exact_answer = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

        if easy_answer != exact_answer:
            # assert_equal is slow...
            assert_equal(easy_answer, exact_answer)

        if may_share_answer != bounds_overlap:
            assert_equal(may_share_answer, bounds_overlap)

        if bounds_overlap:
            if exact_answer:
                feasible += 1
            else:
                infeasible += 1


@pytest.mark.slow
def test_may_share_memory_easy_fuzz():
    # Check that overlap problems with common strides are always
    # solved with little work.

    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: 1,
                                     same_steps=True,
                                     min_count=2000)


@pytest.mark.slow
def test_may_share_memory_harder_fuzz():
    # Overlap problems with not necessarily common strides take more
    # work.
    #
    # The work bound below can't be reduced much. Harder problems can
    # also exist but not be detected here, as the set of problems
    # comes from RNG.

    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: max(a.size, b.size)//2,
                                     same_steps=False,
                                     min_count=2000)


def test_shares_memory_api():
    x = np.zeros([4, 5, 6], dtype=np.int8)

    assert_equal(np.shares_memory(x, x), True)
    assert_equal(np.shares_memory(x, x.copy()), False)

    a = x[:,::2,::3]
    b = x[:,::3,::2]
    assert_equal(np.shares_memory(a, b), True)
    assert_equal(np.shares_memory(a, b, max_work=None), True)
    assert_raises(np.TooHardError, np.shares_memory, a, b, max_work=1)


def test_may_share_memory_bad_max_work():
    x = np.zeros([1])
    assert_raises(OverflowError, np.may_share_memory, x, x, max_work=10**100)
    assert_raises(OverflowError, np.shares_memory, x, x, max_work=10**100)


def test_internal_overlap_diophantine():
    def check(A, U, exists=None):
        X = solve_diophantine(A, U, 0, require_ub_nontrivial=1)

        if exists is None:
            exists = (X is not None)

        if X is not None:
            assert_(sum(a*x for a, x in zip(A, X)) == sum(a*u//2 for a, u in zip(A, U)))
            assert_(all(0 <= x <= u for x, u in zip(X, U)))
            assert_(any(x != u//2 for x, u in zip(X, U)))

        if exists:
            assert_(X is not None, repr(X))
        else:
            assert_(X is None, repr(X))

    # Smoke tests
    check((3, 2), (2*2, 3*2), exists=True)
    check((3*2, 2), (15*2, (3-1)*2), exists=False)


def test_internal_overlap_slices():
    # Slicing an array never generates internal overlap

    x = np.zeros([17,34,71,97], dtype=np.int16)

    rng = np.random.RandomState(1234)

    def random_slice(n, step):
        start = rng.randint(0, n+1, dtype=np.intp)
        stop = rng.randint(start, n+1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    cases = 0
    min_count = 5000

    while cases < min_count:
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        t1 = np.arange(x.ndim)
        rng.shuffle(t1)
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))
        a = x[s1].transpose(t1)

        assert_(not internal_overlap(a))
        cases += 1


def check_internal_overlap(a, manual_expected=None):
    got = internal_overlap(a)

    # Brute-force check
    m = set()
    ranges = tuple(range(n) for n in a.shape)
    for v in itertools.product(*ranges):
        offset = sum(s*w for s, w in zip(a.strides, v))
        if offset in m:
            expected = True
            break
        else:
            m.add(offset)
    else:
        expected = False

    # Compare
    if got != expected:
        assert_equal(got, expected, err_msg=repr((a.strides, a.shape)))
    if manual_expected is not None and expected != manual_expected:
        assert_equal(expected, manual_expected)
    return got


def test_internal_overlap_manual():
    # Stride tricks can construct arrays with internal overlap

    # We don't care about memory bounds, the array is not
    # read/write accessed
    x = np.arange(1).astype(np.int8)

    # Check low-dimensional special cases

    check_internal_overlap(x, False) # 1-dim
    check_internal_overlap(x.reshape([]), False) # 0-dim

    a = as_strided(x, strides=(3, 4), shape=(4, 4))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(3, 4), shape=(5, 4))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0,), shape=(0,))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0,), shape=(1,))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0,), shape=(2,))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0, -9993), shape=(87, 22))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0, -9993), shape=(1, 22))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0, -9993), shape=(0, 22))
    check_internal_overlap(a, False)


def test_internal_overlap_fuzz():
    # Fuzz check; the brute-force check is fairly slow

    x = np.arange(1).astype(np.int8)

    overlap = 0
    no_overlap = 0
    min_count = 100

    rng = np.random.RandomState(1234)

    while min(overlap, no_overlap) < min_count:
        ndim = rng.randint(1, 4, dtype=np.intp)

        strides = tuple(rng.randint(-1000, 1000, dtype=np.intp)
                        for j in range(ndim))
        shape = tuple(rng.randint(1, 30, dtype=np.intp)
                      for j in range(ndim))

        a = as_strided(x, strides=strides, shape=shape)
        result = check_internal_overlap(a)

        if result:
            overlap += 1
        else:
            no_overlap += 1


def test_non_ndarray_inputs():
    # Regression check for gh-5604

    class MyArray:
        def __init__(self, data):
            self.data = data

        @property
        def __array_interface__(self):
            return self.data.__array_interface__

    class MyArray2:
        def __init__(self, data):
            self.data = data

        def __array__(self):
            return self.data

    for cls in [MyArray, MyArray2]:
        x = np.arange(5)

        assert_(np.may_share_memory(cls(x[::2]), x[1::2]))
        assert_(not np.shares_memory(cls(x[::2]), x[1::2]))

        assert_(np.shares_memory(cls(x[1::3]), x[::2]))
        assert_(np.may_share_memory(cls(x[1::3]), x[::2]))


def view_element_first_byte(x):
    """Construct an array viewing the first byte of each element of `x`"""
    from numpy.lib.stride_tricks import DummyArray
    interface = dict(x.__array_interface__)
    interface['typestr'] = '|b1'
    interface['descr'] = [('', '|b1')]
    return np.asarray(DummyArray(interface, x))


def assert_copy_equivalent(operation, args, out, **kwargs):
    """
    Check that operation(*args, out=out) produces results
    equivalent to out[...] = operation(*args, out=out.copy())
    """

    kwargs['out'] = out
    kwargs2 = dict(kwargs)
    kwargs2['out'] = out.copy()

    out_orig = out.copy()
    out[...] = operation(*args, **kwargs2)
    expected = out.copy()
    out[...] = out_orig

    got = operation(*args, **kwargs).copy()

    if (got != expected).any():
        assert_equal(got, expected)


class TestUFunc:
    """
    Test ufunc call memory overlap handling
    """

    def check_unary_fuzz(self, operation, get_out_axis_size, dtype=np.int16,
                             count=5000):
        shapes = [7, 13, 8, 21, 29, 32]

        rng = np.random.RandomState(1234)

        for ndim in range(1, 6):
            x = rng.randint(0, 2**16, size=shapes[:ndim]).astype(dtype)

            it = iter_random_view_pairs(x, same_steps=False, equal_size=True)

            min_count = count // (ndim + 1)**2

            overlapping = 0
            while overlapping < min_count:
                a, b = next(it)

                a_orig = a.copy()
                b_orig = b.copy()

                if get_out_axis_size is None:
                    assert_copy_equivalent(operation, [a], out=b)

                    if np.shares_memory(a, b):
                        overlapping += 1
                else:
                    for axis in itertools.chain(range(ndim), [None]):
                        a[...] = a_orig
                        b[...] = b_orig

                        # Determine size for reduction axis (None if scalar)
                        outsize, scalarize = get_out_axis_size(a, b, axis)
                        if outsize == 'skip':
                            continue

                        # Slice b to get an output array of the correct size
                        sl = [slice(None)] * ndim
                        if axis is None:
                            if outsize is None:
                                sl = [slice(0, 1)] + [0]*(ndim - 1)
                            else:
                                sl = [slice(0, outsize)] + [0]*(ndim - 1)
                        else:
                            if outsize is None:
                                k = b.shape[axis]//2
                                if ndim == 1:
                                    sl[axis] = slice(k, k + 1)
                                else:
                                    sl[axis] = k
                            else:
                                assert b.shape[axis] >= outsize
                                sl[axis] = slice(0, outsize)
                        b_out = b[tuple(sl)]

                        if scalarize:
                            b_out = b_out.reshape([])

                        if np.shares_memory(a, b_out):
                            overlapping += 1

                        # Check result
                        assert_copy_equivalent(operation, [a], out=b_out, axis=axis)

    @pytest.mark.slow
    def test_unary_ufunc_call_fuzz(self):
        self.check_unary_fuzz(np.invert, None, np.int16)

    @pytest.mark.slow
    def test_unary_ufunc_call_complex_fuzz(self):
        # Complex typically has a smaller alignment than itemsize
        self.check_unary_fuzz(np.negative, None, np.complex128, count=500)

    def test_binary_ufunc_accumulate_fuzz(self):
        def get_out_axis_size(a, b, axis):
            if axis is None:
                if a.ndim == 1:
                    return a.size, False
                else:
                    return 'skip', False  # accumulate doesn't support this
            else:
                return a.shape[axis], False

        self.check_unary_fuzz(np.add.accumulate, get_out_axis_size,
                              dtype=np.int16, count=500)

    def test_binary_ufunc_reduce_fuzz(self):
        def get_out_axis_size(a, b, axis):
            return None, (axis is None or a.ndim == 1)

        self.check_unary_fuzz(np.add.reduce, get_out_axis_size,
                              dtype=np.int16, count=500)

    def test_binary_ufunc_reduceat_fuzz(self):
        def get_out_axis_size(a, b, axis):
            if axis is None:
                if a.ndim == 1:
                    return a.size, False
                else:
                    return 'skip', False  # reduceat doesn't support this
            else:
                return a.shape[axis], False

        def do_reduceat(a, out, axis):
            if axis is None:
                size = len(a)
                step = size//len(out)
            else:
                size = a.shape[axis]
                step = a.shape[axis] // out.shape[axis]
            idx = np.arange(0, size, step)
            return np.add.reduceat(a, idx, out=out, axis=axis)

        self.check_unary_fuzz(do_reduceat, get_out_axis_size,
                              dtype=np.int16, count=500)

    def test_binary_ufunc_reduceat_manual(self):
        def check(ufunc, a, ind, out):
            c1 = ufunc.reduceat(a.copy(), ind.copy(), out=out.copy())
            c2 = ufunc.reduceat(a, ind, out=out)
            assert_array_equal(c1, c2)

        # Exactly same input/output arrays
        a = np.arange(10000, dtype=np.int16)
        check(np.add, a, a[::-1].copy(), a)

        # Overlap with index
        a = np.arange(10000, dtype=np.int16)
        check(np.add, a, a[::-1], a)

    @pytest.mark.slow
    def test_unary_gufunc_fuzz(self):
        shapes = [7, 13, 8, 21, 29, 32]
        gufunc = _umath_tests.euclidean_pdist

        rng = np.random.RandomState(1234)

        for ndim in range(2, 6):
            x = rng.rand(*shapes[:ndim])

            it = iter_random_view_pairs(x, same_steps=False, equal_size=True)

            min_count = 500 // (ndim + 1)**2

            overlapping = 0
            while overlapping < min_count:
                a, b = next(it)

                if min(a.shape[-2:]) < 2 or min(b.shape[-2:]) < 2 or a.shape[-1] < 2:
                    continue

                # Ensure the shapes are so that euclidean_pdist is happy
                if b.shape[-1] > b.shape[-2]:
                    b = b[...,0,:]
                else:
                    b = b[...,:,0]

                n = a.shape[-2]
                p = n * (n - 1) // 2
                if p <= b.shape[-1] and p > 0:
                    b = b[...,:p]
                else:
                    n = max(2, int(np.sqrt(b.shape[-1]))//2)
                    p = n * (n - 1) // 2
                    a = a[...,:n,:]
                    b = b[...,:p]

                # Call
                if np.shares_memory(a, b):
                    overlapping += 1

                with np.errstate(over='ignore', invalid='ignore'):
                    assert_copy_equivalent(gufunc, [a], out=b)

    def test_ufunc_at_manual(self):
        def check(ufunc, a, ind, b=None):
            a0 = a.copy()
            if b is None:
                ufunc.at(a0, ind.copy())
                c1 = a0.copy()
                ufunc.at(a, ind)
                c2 = a.copy()
            else:
                ufunc.at(a0, ind.copy(), b.copy())
                c1 = a0.copy()
                ufunc.at(a, ind, b)
                c2 = a.copy()
            assert_array_equal(c1, c2)

        # Overlap with index
        a = np.arange(10000, dtype=np.int16)
        check(np.invert, a[::-1], a)

        # Overlap with second data array
        a = np.arange(100, dtype=np.int16)
        ind = np.arange(0, 100, 2, dtype=np.int16)
        check(np.add, a, ind, a[25:75])

    def test_unary_ufunc_1d_manual(self):
        # Exercise ufunc fast-paths (that avoid creation of an `np.nditer`)

        def check(a, b):
            a_orig = a.copy()
            b_orig = b.copy()

            b0 = b.copy()
            c1 = ufunc(a, out=b0)
            c2 = ufunc(a, out=b)
            assert_array_equal(c1, c2)

            # Trigger "fancy ufunc loop" code path
            mask = view_element_first_byte(b).view(np.bool_)

            a[...] = a_orig
            b[...] = b_orig
            c1 = ufunc(a, out=b.copy(), where=mask.copy()).copy()

            a[...] = a_orig
            b[...] = b_orig
            c2 = ufunc(a, out=b, where=mask.copy()).copy()

            # Also, mask overlapping with output
            a[...] = a_orig
            b[...] = b_orig
            c3 = ufunc(a, out=b, where=mask).copy()

            assert_array_equal(c1, c2)
            assert_array_equal(c1, c3)

        dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32,
                  np.float64, np.complex64, np.complex128]
        dtypes = [np.dtype(x) for x in dtypes]

        for dtype in dtypes:
            if np.issubdtype(dtype, np.integer):
                ufunc = np.invert
            else:
                ufunc = np.reciprocal

            n = 1000
            k = 10
            indices = [
                np.index_exp[:n],
                np.index_exp[k:k+n],
                np.index_exp[n-1::-1],
                np.index_exp[k+n-1:k-1:-1],
                np.index_exp[:2*n:2],
                np.index_exp[k:k+2*n:2],
                np.index_exp[2*n-1::-2],
                np.index_exp[k+2*n-1:k-1:-2],
            ]

            for xi, yi in itertools.product(indices, indices):
                v = np.arange(1, 1 + n*2 + k, dtype=dtype)
                x = v[xi]
                y = v[yi]

                with np.errstate(all='ignore'):
                    check(x, y)

                    # Scalar cases
                    check(x[:1], y)
                    check(x[-1:], y)
                    check(x[:1].reshape([]), y)
                    check(x[-1:].reshape([]), y)

    def test_unary_ufunc_where_same(self):
        # Check behavior at wheremask overlap
        ufunc = np.invert

        def check(a, out, mask):
            c1 = ufunc(a, out=out.copy(), where=mask.copy())
            c2 = ufunc(a, out=out, where=mask)
            assert_array_equal(c1, c2)

        # Check behavior with same input and output arrays
        x = np.arange(100).astype(np.bool_)
        check(x, x, x)
        check(x, x.copy(), x)
        check(x, x, x.copy())

    @pytest.mark.slow
    def test_binary_ufunc_1d_manual(self):
        ufunc = np.add

        def check(a, b, c):
            c0 = c.copy()
            c1 = ufunc(a, b, out=c0)
            c2 = ufunc(a, b, out=c)
            assert_array_equal(c1, c2)

        for dtype in [np.int8, np.int16, np.int32, np.int64,
                      np.float32, np.float64, np.complex64, np.complex128]:
            # Check different data dependency orders

            n = 1000
            k = 10

            indices = []
            for p in [1, 2]:
                indices.extend([
                    np.index_exp[:p*n:p],
                    np.index_exp[k:k+p*n:p],
                    np.index_exp[p*n-1::-p],
                    np.index_exp[k+p*n-1:k-1:-p],
                ])

            for x, y, z in itertools.product(indices, indices, indices):
                v = np.arange(6*n).astype(dtype)
                x = v[x]
                y = v[y]
                z = v[z]

                check(x, y, z)

                # Scalar cases
                check(x[:1], y, z)
                check(x[-1:], y, z)
                check(x[:1].reshape([]), y, z)
                check(x[-1:].reshape([]), y, z)
                check(x, y[:1], z)
                check(x, y[-1:], z)
                check(x, y[:1].reshape([]), z)
                check(x, y[-1:].reshape([]), z)

    def test_inplace_op_simple_manual(self):
        rng = np.random.RandomState(1234)
        x = rng.rand(200, 200)  # bigger than bufsize

        x += x.T
        assert_array_equal(x - x.T, 0)
