# NOTE: Please avoid the use of numpy.testing since NPYV intrinsics
# may be involved in their functionality.
import pytest, math, re
import itertools
import operator
from numpy.core._simd import targets, clear_floatstatus, get_floatstatus
from numpy.core._multiarray_umath import __cpu_baseline__

def check_floatstatus(divbyzero=False, overflow=False,
                      underflow=False, invalid=False,
                      all=False):
    #define NPY_FPE_DIVIDEBYZERO  1
    #define NPY_FPE_OVERFLOW      2
    #define NPY_FPE_UNDERFLOW     4
    #define NPY_FPE_INVALID       8
    err = get_floatstatus()
    ret = (all or divbyzero) and (err & 1) != 0
    ret |= (all or overflow) and (err & 2) != 0
    ret |= (all or underflow) and (err & 4) != 0
    ret |= (all or invalid) and (err & 8) != 0
    return ret

class _Test_Utility:
    # submodule of the desired SIMD extension, e.g. targets["AVX512F"]
    npyv = None
    # the current data type suffix e.g. 's8'
    sfx  = None
    # target name can be 'baseline' or one or more of CPU features
    target_name = None

    def __getattr__(self, attr):
        """
        To call NPV intrinsics without the attribute 'npyv' and
        auto suffixing intrinsics according to class attribute 'sfx'
        """
        return getattr(self.npyv, attr + "_" + self.sfx)

    def _data(self, start=None, count=None, reverse=False):
        """
        Create list of consecutive numbers according to number of vector's lanes.
        """
        if start is None:
            start = 1
        if count is None:
            count = self.nlanes
        rng = range(start, start + count)
        if reverse:
            rng = reversed(rng)
        if self._is_fp():
            return [x / 1.0 for x in rng]
        return list(rng)

    def _is_unsigned(self):
        return self.sfx[0] == 'u'

    def _is_signed(self):
        return self.sfx[0] == 's'

    def _is_fp(self):
        return self.sfx[0] == 'f'

    def _scalar_size(self):
        return int(self.sfx[1:])

    def _int_clip(self, seq):
        if self._is_fp():
            return seq
        max_int = self._int_max()
        min_int = self._int_min()
        return [min(max(v, min_int), max_int) for v in seq]

    def _int_max(self):
        if self._is_fp():
            return None
        max_u = self._to_unsigned(self.setall(-1))[0]
        if self._is_signed():
            return max_u // 2
        return max_u

    def _int_min(self):
        if self._is_fp():
            return None
        if self._is_unsigned():
            return 0
        return -(self._int_max() + 1)

    def _true_mask(self):
        max_unsig = getattr(self.npyv, "setall_u" + self.sfx[1:])(-1)
        return max_unsig[0]

    def _to_unsigned(self, vector):
        if isinstance(vector, (list, tuple)):
            return getattr(self.npyv, "load_u" + self.sfx[1:])(vector)
        else:
            sfx = vector.__name__.replace("npyv_", "")
            if sfx[0] == "b":
                cvt_intrin = "cvt_u{0}_b{0}"
            else:
                cvt_intrin = "reinterpret_u{0}_{1}"
            return getattr(self.npyv, cvt_intrin.format(sfx[1:], sfx))(vector)

    def _pinfinity(self):
        return float("inf")

    def _ninfinity(self):
        return -float("inf")

    def _nan(self):
        return float("nan")

    def _cpu_features(self):
        target = self.target_name
        if target == "baseline":
            target = __cpu_baseline__
        else:
            target = target.split('__') # multi-target separator
        return ' '.join(target)

class _SIMD_BOOL(_Test_Utility):
    """
    To test all boolean vector types at once
    """
    def _nlanes(self):
        return getattr(self.npyv, "nlanes_u" + self.sfx[1:])

    def _data(self, start=None, count=None, reverse=False):
        true_mask = self._true_mask()
        rng = range(self._nlanes())
        if reverse:
            rng = reversed(rng)
        return [true_mask if x % 2 else 0 for x in rng]

    def _load_b(self, data):
        len_str = self.sfx[1:]
        load = getattr(self.npyv, "load_u" + len_str)
        cvt = getattr(self.npyv, f"cvt_b{len_str}_u{len_str}")
        return cvt(load(data))

    def test_operators_logical(self):
        """
        Logical operations for boolean types.
        Test intrinsics:
            npyv_xor_##SFX, npyv_and_##SFX, npyv_or_##SFX, npyv_not_##SFX,
            npyv_andc_b8, npvy_orc_b8, nvpy_xnor_b8
        """
        data_a = self._data()
        data_b = self._data(reverse=True)
        vdata_a = self._load_b(data_a)
        vdata_b = self._load_b(data_b)

        data_and = [a & b for a, b in zip(data_a, data_b)]
        vand = getattr(self, "and")(vdata_a, vdata_b)
        assert vand == data_and

        data_or = [a | b for a, b in zip(data_a, data_b)]
        vor = getattr(self, "or")(vdata_a, vdata_b)
        assert vor == data_or

        data_xor = [a ^ b for a, b in zip(data_a, data_b)]
        vxor = getattr(self, "xor")(vdata_a, vdata_b)
        assert vxor == data_xor

        vnot = getattr(self, "not")(vdata_a)
        assert vnot == data_b

        # among the boolean types, andc, orc and xnor only support b8
        if self.sfx not in ("b8"):
            return

        data_andc = [(a & ~b) & 0xFF for a, b in zip(data_a, data_b)]
        vandc = getattr(self, "andc")(vdata_a, vdata_b)
        assert data_andc == vandc

        data_orc = [(a | ~b) & 0xFF for a, b in zip(data_a, data_b)]
        vorc = getattr(self, "orc")(vdata_a, vdata_b)
        assert data_orc == vorc

        data_xnor = [~(a ^ b) & 0xFF for a, b in zip(data_a, data_b)]
        vxnor = getattr(self, "xnor")(vdata_a, vdata_b)
        assert data_xnor == vxnor

    def test_tobits(self):
        data2bits = lambda data: sum([int(x != 0) << i for i, x in enumerate(data, 0)])
        for data in (self._data(), self._data(reverse=True)):
            vdata = self._load_b(data)
            data_bits = data2bits(data)
            tobits = self.tobits(vdata)
            bin_tobits = bin(tobits)
            assert bin_tobits == bin(data_bits)

    def test_pack(self):
        """
        Pack multiple vectors into one
        Test intrinsics:
            npyv_pack_b8_b16
            npyv_pack_b8_b32
            npyv_pack_b8_b64
        """
        if self.sfx not in ("b16", "b32", "b64"):
            return
        # create the vectors
        data = self._data()
        rdata = self._data(reverse=True)
        vdata = self._load_b(data)
        vrdata = self._load_b(rdata)
        pack_simd = getattr(self.npyv, f"pack_b8_{self.sfx}")
        # for scalar execution, concatenate the elements of the multiple lists
        # into a single list (spack) and then iterate over the elements of
        # the created list applying a mask to capture the first byte of them.
        if self.sfx == "b16":
            spack = [(i & 0xFF) for i in (list(rdata) + list(data))]
            vpack = pack_simd(vrdata, vdata)
        elif self.sfx == "b32":
            spack = [(i & 0xFF) for i in (2*list(rdata) + 2*list(data))]
            vpack = pack_simd(vrdata, vrdata, vdata, vdata)
        elif self.sfx == "b64":
            spack = [(i & 0xFF) for i in (4*list(rdata) + 4*list(data))]
            vpack = pack_simd(vrdata, vrdata, vrdata, vrdata,
                               vdata,  vdata,  vdata,  vdata)
        assert vpack == spack

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [-1, 0],
        [0, -1],
        [-1],
        [0]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self._load_b(data * self._nlanes())
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired

class _SIMD_INT(_Test_Utility):
    """
    To test all integer vector types at once
    """
    def test_operators_shift(self):
        if self.sfx in ("u8", "s8"):
            return

        data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        for count in range(self._scalar_size()):
            # load to cast
            data_shl_a = self.load([a << count for a in data_a])
            # left shift
            shl = self.shl(vdata_a, count)
            assert shl == data_shl_a
            # load to cast
            data_shr_a = self.load([a >> count for a in data_a])
            # right shift
            shr = self.shr(vdata_a, count)
            assert shr == data_shr_a

        # shift by zero or max or out-range immediate constant is not applicable and illogical
        for count in range(1, self._scalar_size()):
            # load to cast
            data_shl_a = self.load([a << count for a in data_a])
            # left shift by an immediate constant
            shli = self.shli(vdata_a, count)
            assert shli == data_shl_a
            # load to cast
            data_shr_a = self.load([a >> count for a in data_a])
            # right shift by an immediate constant
            shri = self.shri(vdata_a, count)
            assert shri == data_shr_a

    def test_arithmetic_subadd_saturated(self):
        if self.sfx in ("u32", "s32", "u64", "s64"):
            return

        data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        data_adds = self._int_clip([a + b for a, b in zip(data_a, data_b)])
        adds = self.adds(vdata_a, vdata_b)
        assert adds == data_adds

        data_subs = self._int_clip([a - b for a, b in zip(data_a, data_b)])
        subs = self.subs(vdata_a, vdata_b)
        assert subs == data_subs

    def test_math_max_min(self):
        data_a = self._data()
        data_b = self._data(self.nlanes)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        data_max = [max(a, b) for a, b in zip(data_a, data_b)]
        simd_max = self.max(vdata_a, vdata_b)
        assert simd_max == data_max

        data_min = [min(a, b) for a, b in zip(data_a, data_b)]
        simd_min = self.min(vdata_a, vdata_b)
        assert simd_min == data_min

    @pytest.mark.parametrize("start", [-100, -10000, 0, 100, 10000])
    def test_reduce_max_min(self, start):
        """
        Test intrinsics:
            npyv_reduce_max_##sfx
            npyv_reduce_min_##sfx
        """
        vdata_a = self.load(self._data(start))
        assert self.reduce_max(vdata_a) == max(vdata_a)
        assert self.reduce_min(vdata_a) == min(vdata_a)


class _SIMD_FP32(_Test_Utility):
    """
    To only test single precision
    """
    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        features = self._cpu_features()
        if not self.npyv.simd_f64 and re.match(r".*(NEON|ASIMD)", features):
            # very costly to emulate nearest even on Armv7
            # instead we round halves to up. e.g. 0.5 -> 1, -0.5 -> -1
            _round = lambda v: int(v + (0.5 if v >= 0 else -0.5))
        else:
            _round = round
        vdata_a = self.load(self._data())
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        data_round = [_round(x) for x in vdata_a]
        vround = self.round_s32(vdata_a)
        assert vround == data_round

class _SIMD_FP64(_Test_Utility):
    """
    To only test double precision
    """
    def test_conversions(self):
        """
        Round to nearest even integer, assume CPU control register is set to rounding.
        Test intrinsics:
            npyv_round_s32_##SFX
        """
        vdata_a = self.load(self._data())
        vdata_a = self.sub(vdata_a, self.setall(0.5))
        vdata_b = self.mul(vdata_a, self.setall(-1.5))
        data_round = [round(x) for x in list(vdata_a) + list(vdata_b)]
        vround = self.round_s32(vdata_a, vdata_b)
        assert vround == data_round

class _SIMD_FP(_Test_Utility):
    """
    To test all float vector types at once
    """
    def test_arithmetic_fused(self):
        vdata_a, vdata_b, vdata_c = [self.load(self._data())]*3
        vdata_cx2 = self.add(vdata_c, vdata_c)
        # multiply and add, a*b + c
        data_fma = self.load([a * b + c for a, b, c in zip(vdata_a, vdata_b, vdata_c)])
        fma = self.muladd(vdata_a, vdata_b, vdata_c)
        assert fma == data_fma
        # multiply and subtract, a*b - c
        fms = self.mulsub(vdata_a, vdata_b, vdata_c)
        data_fms = self.sub(data_fma, vdata_cx2)
        assert fms == data_fms
        # negate multiply and add, -(a*b) + c
        nfma = self.nmuladd(vdata_a, vdata_b, vdata_c)
        data_nfma = self.sub(vdata_cx2, data_fma)
        assert nfma == data_nfma
        # negate multiply and subtract, -(a*b) - c
        nfms = self.nmulsub(vdata_a, vdata_b, vdata_c)
        data_nfms = self.mul(data_fma, self.setall(-1))
        assert nfms == data_nfms

    def test_abs(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        abs_cases = ((-0, 0), (ninf, pinf), (pinf, pinf), (nan, nan))
        for case, desired in abs_cases:
            data_abs = [desired]*self.nlanes
            vabs = self.abs(self.setall(case))
            assert vabs == pytest.approx(data_abs, nan_ok=True)

        vabs = self.abs(self.mul(vdata, self.setall(-1)))
        assert vabs == data

    def test_sqrt(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        sqrt_cases = ((-0.0, -0.0), (0.0, 0.0), (-1.0, nan), (ninf, nan), (pinf, pinf))
        for case, desired in sqrt_cases:
            data_sqrt = [desired]*self.nlanes
            sqrt  = self.sqrt(self.setall(case))
            assert sqrt == pytest.approx(data_sqrt, nan_ok=True)

        data_sqrt = self.load([math.sqrt(x) for x in data]) # load to truncate precision
        sqrt = self.sqrt(vdata)
        assert sqrt == data_sqrt

    def test_square(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())
        # square
        square_cases = ((nan, nan), (pinf, pinf), (ninf, pinf))
        for case, desired in square_cases:
            data_square = [desired]*self.nlanes
            square  = self.square(self.setall(case))
            assert square == pytest.approx(data_square, nan_ok=True)

        data_square = [x*x for x in data]
        square = self.square(vdata)
        assert square == data_square

    @pytest.mark.parametrize("intrin, func", [("ceil", math.ceil),
    ("trunc", math.trunc), ("floor", math.floor), ("rint", round)])
    def test_rounding(self, intrin, func):
        """
        Test intrinsics:
            npyv_rint_##SFX
            npyv_ceil_##SFX
            npyv_trunc_##SFX
            npyv_floor##SFX
        """
        intrin_name = intrin
        intrin = getattr(self, intrin)
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        # special cases
        round_cases = ((nan, nan), (pinf, pinf), (ninf, ninf))
        for case, desired in round_cases:
            data_round = [desired]*self.nlanes
            _round = intrin(self.setall(case))
            assert _round == pytest.approx(data_round, nan_ok=True)

        for x in range(0, 2**20, 256**2):
            for w in (-1.05, -1.10, -1.15, 1.05, 1.10, 1.15):
                data = self.load([(x+a)*w for a in range(self.nlanes)])
                data_round = [func(x) for x in data]
                _round = intrin(data)
                assert _round == data_round

        # test large numbers
        for i in (
            1.1529215045988576e+18, 4.6116860183954304e+18,
            5.902958103546122e+20, 2.3611832414184488e+21
        ):
            x = self.setall(i)
            y = intrin(x)
            data_round = [func(n) for n in x]
            assert y == data_round

        # signed zero
        if intrin_name == "floor":
            data_szero = (-0.0,)
        else:
            data_szero = (-0.0, -0.25, -0.30, -0.45, -0.5)

        for w in data_szero:
            _round = self._to_unsigned(intrin(self.setall(w)))
            data_round = self._to_unsigned(self.setall(-0.0))
            assert _round == data_round

    @pytest.mark.parametrize("intrin", [
        "max", "maxp", "maxn", "min", "minp", "minn"
    ])
    def test_max_min(self, intrin):
        """
        Test intrinsics:
            npyv_max_##sfx
            npyv_maxp_##sfx
            npyv_maxn_##sfx
            npyv_min_##sfx
            npyv_minp_##sfx
            npyv_minn_##sfx
            npyv_reduce_max_##sfx
            npyv_reduce_maxp_##sfx
            npyv_reduce_maxn_##sfx
            npyv_reduce_min_##sfx
            npyv_reduce_minp_##sfx
            npyv_reduce_minn_##sfx
        """
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        chk_nan = {"xp": 1, "np": 1, "nn": 2, "xn": 2}.get(intrin[-2:], 0)
        func = eval(intrin[:3])
        reduce_intrin = getattr(self, "reduce_" + intrin)
        intrin = getattr(self, intrin)
        hf_nlanes = self.nlanes//2

        cases = (
            ([0.0, -0.0], [-0.0, 0.0]),
            ([10, -10],  [10, -10]),
            ([pinf, 10], [10, ninf]),
            ([10, pinf], [ninf, 10]),
            ([10, -10], [10, -10]),
            ([-10, 10], [-10, 10])
        )
        for op1, op2 in cases:
            vdata_a = self.load(op1*hf_nlanes)
            vdata_b = self.load(op2*hf_nlanes)
            data = func(vdata_a, vdata_b)
            simd = intrin(vdata_a, vdata_b)
            assert simd == data
            data = func(vdata_a)
            simd = reduce_intrin(vdata_a)
            assert simd == data

        if not chk_nan:
            return
        if chk_nan == 1:
            test_nan = lambda a, b: (
                b if math.isnan(a) else a if math.isnan(b) else b
            )
        else:
            test_nan = lambda a, b: (
                nan if math.isnan(a) or math.isnan(b) else b
            )
        cases = (
            (nan, 10),
            (10, nan),
            (nan, pinf),
            (pinf, nan),
            (nan, nan)
        )
        for op1, op2 in cases:
            vdata_ab = self.load([op1, op2]*hf_nlanes)
            data = test_nan(op1, op2)
            simd = reduce_intrin(vdata_ab)
            assert simd == pytest.approx(data, nan_ok=True)
            vdata_a = self.setall(op1)
            vdata_b = self.setall(op2)
            data = [data] * self.nlanes
            simd = intrin(vdata_a, vdata_b)
            assert simd == pytest.approx(data, nan_ok=True)

    def test_reciprocal(self):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        data = self._data()
        vdata = self.load(self._data())

        recip_cases = ((nan, nan), (pinf, 0.0), (ninf, -0.0), (0.0, pinf), (-0.0, ninf))
        for case, desired in recip_cases:
            data_recip = [desired]*self.nlanes
            recip = self.recip(self.setall(case))
            assert recip == pytest.approx(data_recip, nan_ok=True)

        data_recip = self.load([1/x for x in data]) # load to truncate precision
        recip = self.recip(vdata)
        assert recip == data_recip

    def test_special_cases(self):
        """
        Compare Not NaN. Test intrinsics:
            npyv_notnan_##SFX
        """
        nnan = self.notnan(self.setall(self._nan()))
        assert nnan == [0]*self.nlanes

    @pytest.mark.parametrize("intrin_name", [
        "rint", "trunc", "ceil", "floor"
    ])
    def test_unary_invalid_fpexception(self, intrin_name):
        intrin = getattr(self, intrin_name)
        for d in [float("nan"), float("inf"), -float("inf")]:
            v = self.setall(d)
            clear_floatstatus()
            intrin(v)
            assert check_floatstatus(invalid=True) == False

    @pytest.mark.parametrize('py_comp,np_comp', [
        (operator.lt, "cmplt"),
        (operator.le, "cmple"),
        (operator.gt, "cmpgt"),
        (operator.ge, "cmpge"),
        (operator.eq, "cmpeq"),
        (operator.ne, "cmpneq")
    ])
    def test_comparison_with_nan(self, py_comp, np_comp):
        pinf, ninf, nan = self._pinfinity(), self._ninfinity(), self._nan()
        mask_true = self._true_mask()

        def to_bool(vector):
            return [lane == mask_true for lane in vector]

        intrin = getattr(self, np_comp)
        cmp_cases = ((0, nan), (nan, 0), (nan, nan), (pinf, nan),
                     (ninf, nan), (-0.0, +0.0))
        for case_operand1, case_operand2 in cmp_cases:
            data_a = [case_operand1]*self.nlanes
            data_b = [case_operand2]*self.nlanes
            vdata_a = self.setall(case_operand1)
            vdata_b = self.setall(case_operand2)
            vcmp = to_bool(intrin(vdata_a, vdata_b))
            data_cmp = [py_comp(a, b) for a, b in zip(data_a, data_b)]
            assert vcmp == data_cmp

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [float("nan"), 0],
        [0, float("nan")],
        [float("nan"), 1],
        [1, float("nan")],
        [float("nan"), float("nan")],
        [0.0, -0.0],
        [-0.0, 0.0],
        [1.0, -0.0]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self.load(data * self.nlanes)
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired

class _SIMD_ALL(_Test_Utility):
    """
    To test all vector types at once
    """
    def test_memory_load(self):
        data = self._data()
        # unaligned load
        load_data = self.load(data)
        assert load_data == data
        # aligned load
        loada_data = self.loada(data)
        assert loada_data == data
        # stream load
        loads_data = self.loads(data)
        assert loads_data == data
        # load lower part
        loadl = self.loadl(data)
        loadl_half = list(loadl)[:self.nlanes//2]
        data_half = data[:self.nlanes//2]
        assert loadl_half == data_half
        assert loadl != data # detect overflow

    def test_memory_store(self):
        data = self._data()
        vdata = self.load(data)
        # unaligned store
        store = [0] * self.nlanes
        self.store(store, vdata)
        assert store == data
        # aligned store
        store_a = [0] * self.nlanes
        self.storea(store_a, vdata)
        assert store_a == data
        # stream store
        store_s = [0] * self.nlanes
        self.stores(store_s, vdata)
        assert store_s == data
        # store lower part
        store_l = [0] * self.nlanes
        self.storel(store_l, vdata)
        assert store_l[:self.nlanes//2] == data[:self.nlanes//2]
        assert store_l != vdata # detect overflow
        # store higher part
        store_h = [0] * self.nlanes
        self.storeh(store_h, vdata)
        assert store_h[:self.nlanes//2] == data[self.nlanes//2:]
        assert store_h != vdata  # detect overflow

    def test_memory_partial_load(self):
        if self.sfx in ("u8", "s8", "u16", "s16"):
            return

        data = self._data()
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4] # test out of range
        for n in lanes:
            load_till  = self.load_till(data, n, 15)
            data_till  = data[:n] + [15] * (self.nlanes-n)
            assert load_till == data_till
            load_tillz = self.load_tillz(data, n)
            data_tillz = data[:n] + [0] * (self.nlanes-n)
            assert load_tillz == data_tillz

    def test_memory_partial_store(self):
        if self.sfx in ("u8", "s8", "u16", "s16"):
            return

        data = self._data()
        data_rev = self._data(reverse=True)
        vdata = self.load(data)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        for n in lanes:
            data_till = data_rev.copy()
            data_till[:n] = data[:n]
            store_till = self._data(reverse=True)
            self.store_till(store_till, n, vdata)
            assert store_till == data_till

    def test_memory_noncont_load(self):
        if self.sfx in ("u8", "s8", "u16", "s16"):
            return

        for stride in range(1, 64):
            data = self._data(count=stride*self.nlanes)
            data_stride = data[::stride]
            loadn = self.loadn(data, stride)
            assert loadn == data_stride

        for stride in range(-64, 0):
            data = self._data(stride, -stride*self.nlanes)
            data_stride = self.load(data[::stride]) # cast unsigned
            loadn = self.loadn(data, stride)
            assert loadn == data_stride

    def test_memory_noncont_partial_load(self):
        if self.sfx in ("u8", "s8", "u16", "s16"):
            return

        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        for stride in range(1, 64):
            data = self._data(count=stride*self.nlanes)
            data_stride = data[::stride]
            for n in lanes:
                data_stride_till = data_stride[:n] + [15] * (self.nlanes-n)
                loadn_till = self.loadn_till(data, stride, n, 15)
                assert loadn_till == data_stride_till
                data_stride_tillz = data_stride[:n] + [0] * (self.nlanes-n)
                loadn_tillz = self.loadn_tillz(data, stride, n)
                assert loadn_tillz == data_stride_tillz

        for stride in range(-64, 0):
            data = self._data(stride, -stride*self.nlanes)
            data_stride = list(self.load(data[::stride])) # cast unsigned
            for n in lanes:
                data_stride_till = data_stride[:n] + [15] * (self.nlanes-n)
                loadn_till = self.loadn_till(data, stride, n, 15)
                assert loadn_till == data_stride_till
                data_stride_tillz = data_stride[:n] + [0] * (self.nlanes-n)
                loadn_tillz = self.loadn_tillz(data, stride, n)
                assert loadn_tillz == data_stride_tillz

    def test_memory_noncont_store(self):
        if self.sfx in ("u8", "s8", "u16", "s16"):
            return

        vdata = self.load(self._data())
        for stride in range(1, 64):
            data = [15] * stride * self.nlanes
            data[::stride] = vdata
            storen = [15] * stride * self.nlanes
            storen += [127]*64
            self.storen(storen, stride, vdata)
            assert storen[:-64] == data
            assert storen[-64:] == [127]*64 # detect overflow

        for stride in range(-64, 0):
            data = [15] * -stride * self.nlanes
            data[::stride] = vdata
            storen = [127]*64
            storen += [15] * -stride * self.nlanes
            self.storen(storen, stride, vdata)
            assert storen[64:] == data
            assert storen[:64] == [127]*64 # detect overflow

    def test_memory_noncont_partial_store(self):
        if self.sfx in ("u8", "s8", "u16", "s16"):
            return

        data = self._data()
        vdata = self.load(data)
        lanes = list(range(1, self.nlanes + 1))
        lanes += [self.nlanes**2, self.nlanes**4]
        for stride in range(1, 64):
            for n in lanes:
                data_till = [15] * stride * self.nlanes
                data_till[::stride] = data[:n] + [15] * (self.nlanes-n)
                storen_till = [15] * stride * self.nlanes
                storen_till += [127]*64
                self.storen_till(storen_till, stride, n, vdata)
                assert storen_till[:-64] == data_till
                assert storen_till[-64:] == [127]*64 # detect overflow

        for stride in range(-64, 0):
            for n in lanes:
                data_till = [15] * -stride * self.nlanes
                data_till[::stride] = data[:n] + [15] * (self.nlanes-n)
                storen_till = [127]*64
                storen_till += [15] * -stride * self.nlanes
                self.storen_till(storen_till, stride, n, vdata)
                assert storen_till[64:] == data_till
                assert storen_till[:64] == [127]*64 # detect overflow

    @pytest.mark.parametrize("intrin, table_size, elsize", [
        ("self.lut32", 32, 32),
        ("self.lut16", 16, 64)
    ])
    def test_lut(self, intrin, table_size, elsize):
        """
        Test lookup table intrinsics:
            npyv_lut32_##sfx
            npyv_lut16_##sfx
        """
        if elsize != self._scalar_size():
            return
        intrin = eval(intrin)
        idx_itrin = getattr(self.npyv, f"setall_u{elsize}")
        table = range(0, table_size)
        for i in table:
            broadi = self.setall(i)
            idx = idx_itrin(i)
            lut = intrin(table, idx)
            assert lut == broadi

    def test_misc(self):
        broadcast_zero = self.zero()
        assert broadcast_zero == [0] * self.nlanes
        for i in range(1, 10):
            broadcasti = self.setall(i)
            assert broadcasti == [i] * self.nlanes

        data_a, data_b = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # py level of npyv_set_* don't support ignoring the extra specified lanes or
        # fill non-specified lanes with zero.
        vset = self.set(*data_a)
        assert vset == data_a
        # py level of npyv_setf_* don't support ignoring the extra specified lanes or
        # fill non-specified lanes with the specified scalar.
        vsetf = self.setf(10, *data_a)
        assert vsetf == data_a

        # We're testing the sanity of _simd's type-vector,
        # reinterpret* intrinsics itself are tested via compiler
        # during the build of _simd module
        sfxes = ["u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64"]
        if self.npyv.simd_f64:
            sfxes.append("f64")
        if self.npyv.simd_f32:
            sfxes.append("f32")
        for sfx in sfxes:
            vec_name = getattr(self, "reinterpret_" + sfx)(vdata_a).__name__
            assert vec_name == "npyv_" + sfx

        # select & mask operations
        select_a = self.select(self.cmpeq(self.zero(), self.zero()), vdata_a, vdata_b)
        assert select_a == data_a
        select_b = self.select(self.cmpneq(self.zero(), self.zero()), vdata_a, vdata_b)
        assert select_b == data_b

        # test extract elements
        assert self.extract0(vdata_b) == vdata_b[0]

        # cleanup intrinsic is only used with AVX for
        # zeroing registers to avoid the AVX-SSE transition penalty,
        # so nothing to test here
        self.npyv.cleanup()

    def test_reorder(self):
        data_a, data_b  = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)
        # lower half part
        data_a_lo = data_a[:self.nlanes//2]
        data_b_lo = data_b[:self.nlanes//2]
        # higher half part
        data_a_hi = data_a[self.nlanes//2:]
        data_b_hi = data_b[self.nlanes//2:]
        # combine two lower parts
        combinel = self.combinel(vdata_a, vdata_b)
        assert combinel == data_a_lo + data_b_lo
        # combine two higher parts
        combineh = self.combineh(vdata_a, vdata_b)
        assert combineh == data_a_hi + data_b_hi
        # combine x2
        combine  = self.combine(vdata_a, vdata_b)
        assert combine == (data_a_lo + data_b_lo, data_a_hi + data_b_hi)
        # zip(interleave)
        data_zipl = [v for p in zip(data_a_lo, data_b_lo) for v in p]
        data_ziph = [v for p in zip(data_a_hi, data_b_hi) for v in p]
        vzip  = self.zip(vdata_a, vdata_b)
        assert vzip == (data_zipl, data_ziph)

    def test_reorder_rev64(self):
        # Reverse elements of each 64-bit lane
        ssize = self._scalar_size()
        if ssize == 64:
            return
        data_rev64 = [
            y for x in range(0, self.nlanes, 64//ssize)
              for y in reversed(range(x, x + 64//ssize))
        ]
        rev64 = self.rev64(self.load(range(self.nlanes)))
        assert rev64 == data_rev64

    @pytest.mark.parametrize('func, intrin', [
        (operator.lt, "cmplt"),
        (operator.le, "cmple"),
        (operator.gt, "cmpgt"),
        (operator.ge, "cmpge"),
        (operator.eq, "cmpeq")
    ])
    def test_operators_comparison(self, func, intrin):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)
        intrin = getattr(self, intrin)

        mask_true = self._true_mask()
        def to_bool(vector):
            return [lane == mask_true for lane in vector]

        data_cmp = [func(a, b) for a, b in zip(data_a, data_b)]
        cmp = to_bool(intrin(vdata_a, vdata_b))
        assert cmp == data_cmp

    def test_operators_logical(self):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        if self._is_fp():
            data_cast_a = self._to_unsigned(vdata_a)
            data_cast_b = self._to_unsigned(vdata_b)
            cast, cast_data = self._to_unsigned, self._to_unsigned
        else:
            data_cast_a, data_cast_b = data_a, data_b
            cast, cast_data = lambda a: a, self.load

        data_xor = cast_data([a ^ b for a, b in zip(data_cast_a, data_cast_b)])
        vxor = cast(self.xor(vdata_a, vdata_b))
        assert vxor == data_xor

        data_or  = cast_data([a | b for a, b in zip(data_cast_a, data_cast_b)])
        vor  = cast(getattr(self, "or")(vdata_a, vdata_b))
        assert vor == data_or

        data_and = cast_data([a & b for a, b in zip(data_cast_a, data_cast_b)])
        vand = cast(getattr(self, "and")(vdata_a, vdata_b))
        assert vand == data_and

        data_not = cast_data([~a for a in data_cast_a])
        vnot = cast(getattr(self, "not")(vdata_a))
        assert vnot == data_not

        if self.sfx not in ("u8"):
            return
        data_andc = [a & ~b for a, b in zip(data_cast_a, data_cast_b)]
        vandc = cast(getattr(self, "andc")(vdata_a, vdata_b))
        assert vandc == data_andc

    @pytest.mark.parametrize("intrin", ["any", "all"])
    @pytest.mark.parametrize("data", (
        [1, 2, 3, 4],
        [-1, -2, -3, -4],
        [0, 1, 2, 3, 4],
        [0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff],
        [0, -1, -2, -3, 4],
        [0],
        [1],
        [-1]
    ))
    def test_operators_crosstest(self, intrin, data):
        """
        Test intrinsics:
            npyv_any_##SFX
            npyv_all_##SFX
        """
        data_a = self.load(data * self.nlanes)
        func = eval(intrin)
        intrin = getattr(self, intrin)
        desired = func(data_a)
        simd = intrin(data_a)
        assert not not simd == desired

    def test_conversion_boolean(self):
        bsfx = "b" + self.sfx[1:]
        to_boolean = getattr(self.npyv, "cvt_%s_%s" % (bsfx, self.sfx))
        from_boolean = getattr(self.npyv, "cvt_%s_%s" % (self.sfx, bsfx))

        false_vb = to_boolean(self.setall(0))
        true_vb  = self.cmpeq(self.setall(0), self.setall(0))
        assert false_vb != true_vb

        false_vsfx = from_boolean(false_vb)
        true_vsfx = from_boolean(true_vb)
        assert false_vsfx != true_vsfx

    def test_conversion_expand(self):
        """
        Test expand intrinsics:
            npyv_expand_u16_u8
            npyv_expand_u32_u16
        """
        if self.sfx not in ("u8", "u16"):
            return
        totype = self.sfx[0]+str(int(self.sfx[1:])*2)
        expand = getattr(self.npyv, f"expand_{totype}_{self.sfx}")
        # close enough from the edge to detect any deviation
        data  = self._data(self._int_max() - self.nlanes)
        vdata = self.load(data)
        edata = expand(vdata)
        # lower half part
        data_lo = data[:self.nlanes//2]
        # higher half part
        data_hi = data[self.nlanes//2:]
        assert edata == (data_lo, data_hi)

    def test_arithmetic_subadd(self):
        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # non-saturated
        data_add = self.load([a + b for a, b in zip(data_a, data_b)]) # load to cast
        add  = self.add(vdata_a, vdata_b)
        assert add == data_add
        data_sub  = self.load([a - b for a, b in zip(data_a, data_b)])
        sub  = self.sub(vdata_a, vdata_b)
        assert sub == data_sub

    def test_arithmetic_mul(self):
        if self.sfx in ("u64", "s64"):
            return

        if self._is_fp():
            data_a = self._data()
        else:
            data_a = self._data(self._int_max() - self.nlanes)
        data_b = self._data(self._int_min(), reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        data_mul = self.load([a * b for a, b in zip(data_a, data_b)])
        mul = self.mul(vdata_a, vdata_b)
        assert mul == data_mul

    def test_arithmetic_div(self):
        if not self._is_fp():
            return

        data_a, data_b = self._data(), self._data(reverse=True)
        vdata_a, vdata_b = self.load(data_a), self.load(data_b)

        # load to truncate f64 to precision of f32
        data_div = self.load([a / b for a, b in zip(data_a, data_b)])
        div = self.div(vdata_a, vdata_b)
        assert div == data_div

    def test_arithmetic_intdiv(self):
        """
        Test integer division intrinsics:
            npyv_divisor_##sfx
            npyv_divc_##sfx
        """
        if self._is_fp():
            return

        int_min = self._int_min()
        def trunc_div(a, d):
            """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
            if d == -1 and a == int_min:
                return a
            sign_a, sign_d = a < 0, d < 0
            if a == 0 or sign_a == sign_d:
                return a // d
            return (a + sign_d - sign_a) // d + 1

        data = [1, -int_min]  # to test overflow
        data += range(0, 2**8, 2**5)
        data += range(0, 2**8, 2**5-1)
        bsize = self._scalar_size()
        if bsize > 8:
            data += range(2**8, 2**16, 2**13)
            data += range(2**8, 2**16, 2**13-1)
        if bsize > 16:
            data += range(2**16, 2**32, 2**29)
            data += range(2**16, 2**32, 2**29-1)
        if bsize > 32:
            data += range(2**32, 2**64, 2**61)
            data += range(2**32, 2**64, 2**61-1)
        # negate
        data += [-x for x in data]
        for dividend, divisor in itertools.product(data, data):
            divisor = self.setall(divisor)[0]  # cast
            if divisor == 0:
                continue
            dividend = self.load(self._data(dividend))
            data_divc = [trunc_div(a, divisor) for a in dividend]
            divisor_parms = self.divisor(divisor)
            divc = self.divc(dividend, divisor_parms)
            assert divc == data_divc

    def test_arithmetic_reduce_sum(self):
        """
        Test reduce sum intrinsics:
            npyv_sum_##sfx
        """
        if self.sfx not in ("u32", "u64", "f32", "f64"):
            return
        # reduce sum
        data = self._data()
        vdata = self.load(data)

        data_sum = sum(data)
        vsum = self.sum(vdata)
        assert vsum == data_sum

    def test_arithmetic_reduce_sumup(self):
        """
        Test extend reduce sum intrinsics:
            npyv_sumup_##sfx
        """
        if self.sfx not in ("u8", "u16"):
            return
        rdata = (0, self.nlanes, self._int_min(), self._int_max()-self.nlanes)
        for r in rdata:
            data = self._data(r)
            vdata = self.load(data)
            data_sum = sum(data)
            vsum = self.sumup(vdata)
            assert vsum == data_sum

    def test_mask_conditional(self):
        """
        Conditional addition and subtraction for all supported data types.
        Test intrinsics:
            npyv_ifadd_##SFX, npyv_ifsub_##SFX
        """
        vdata_a = self.load(self._data())
        vdata_b = self.load(self._data(reverse=True))
        true_mask  = self.cmpeq(self.zero(), self.zero())
        false_mask = self.cmpneq(self.zero(), self.zero())

        data_sub = self.sub(vdata_b, vdata_a)
        ifsub = self.ifsub(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifsub == data_sub
        ifsub = self.ifsub(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifsub == vdata_b

        data_add = self.add(vdata_b, vdata_a)
        ifadd = self.ifadd(true_mask, vdata_b, vdata_a, vdata_b)
        assert ifadd == data_add
        ifadd = self.ifadd(false_mask, vdata_a, vdata_b, vdata_b)
        assert ifadd == vdata_b

bool_sfx = ("b8", "b16", "b32", "b64")
int_sfx = ("u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64")
fp_sfx  = ("f32", "f64")
all_sfx = int_sfx + fp_sfx
tests_registry = {
    bool_sfx: _SIMD_BOOL,
    int_sfx : _SIMD_INT,
    fp_sfx  : _SIMD_FP,
    ("f32",): _SIMD_FP32,
    ("f64",): _SIMD_FP64,
    all_sfx : _SIMD_ALL
}
for target_name, npyv in targets.items():
    simd_width = npyv.simd if npyv else ''
    pretty_name = target_name.split('__') # multi-target separator
    if len(pretty_name) > 1:
        # multi-target
        pretty_name = f"({' '.join(pretty_name)})"
    else:
        pretty_name = pretty_name[0]

    skip = ""
    skip_sfx = dict()
    if not npyv:
        skip = f"target '{pretty_name}' isn't supported by current machine"
    elif not npyv.simd:
        skip = f"target '{pretty_name}' isn't supported by NPYV"
    else:
        if not npyv.simd_f32:
            skip_sfx["f32"] = f"target '{pretty_name}' "\
                               "doesn't support single-precision"
        if not npyv.simd_f64:
            skip_sfx["f64"] = f"target '{pretty_name}' doesn't"\
                               "support double-precision"

    for sfxes, cls in tests_registry.items():
        for sfx in sfxes:
            skip_m = skip_sfx.get(sfx, skip)
            inhr = (cls,)
            attr = dict(npyv=targets[target_name], sfx=sfx, target_name=target_name)
            tcls = type(f"Test{cls.__name__}_{simd_width}_{target_name}_{sfx}", inhr, attr)
            if skip_m:
                pytest.mark.skip(reason=skip_m)(tcls)
            globals()[tcls.__name__] = tcls
