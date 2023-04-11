import pytest
from numpy.core._simd import targets
"""
This testing unit only for checking the sanity of common functionality,
therefore all we need is just to take one submodule that represents any
of enabled SIMD extensions to run the test on it and the second submodule
required to run only one check related to the possibility of mixing
the data types among each submodule.
"""
npyvs = [npyv_mod for npyv_mod in targets.values() if npyv_mod and npyv_mod.simd]
npyv, npyv2 = (npyvs + [None, None])[:2]

unsigned_sfx = ["u8", "u16", "u32", "u64"]
signed_sfx = ["s8", "s16", "s32", "s64"]
fp_sfx = []
if npyv and npyv.simd_f32:
    fp_sfx.append("f32")
if npyv and npyv.simd_f64:
    fp_sfx.append("f64")

int_sfx = unsigned_sfx + signed_sfx
all_sfx = unsigned_sfx + int_sfx

@pytest.mark.skipif(not npyv, reason="could not find any SIMD extension with NPYV support")
class Test_SIMD_MODULE:

    @pytest.mark.parametrize('sfx', all_sfx)
    def test_num_lanes(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        vector = getattr(npyv, "setall_" + sfx)(1)
        assert len(vector) == nlanes

    @pytest.mark.parametrize('sfx', all_sfx)
    def test_type_name(self, sfx):
        vector = getattr(npyv, "setall_" + sfx)(1)
        assert vector.__name__ == "npyv_" + sfx

    def test_raises(self):
        a, b = [npyv.setall_u32(1)]*2
        for sfx in all_sfx:
            vcb = lambda intrin: getattr(npyv, f"{intrin}_{sfx}")
            pytest.raises(TypeError, vcb("add"), a)
            pytest.raises(TypeError, vcb("add"), a, b, a)
            pytest.raises(TypeError, vcb("setall"))
            pytest.raises(TypeError, vcb("setall"), [1])
            pytest.raises(TypeError, vcb("load"), 1)
            pytest.raises(ValueError, vcb("load"), [1])
            pytest.raises(ValueError, vcb("store"), [1], getattr(npyv, f"reinterpret_{sfx}_u32")(a))

    @pytest.mark.skipif(not npyv2, reason=(
        "could not find a second SIMD extension with NPYV support"
    ))
    def test_nomix(self):
        # mix among submodules isn't allowed
        a = npyv.setall_u32(1)
        a2 = npyv2.setall_u32(1)
        pytest.raises(TypeError, npyv.add_u32, a2, a2)
        pytest.raises(TypeError, npyv2.add_u32, a, a)

    @pytest.mark.parametrize('sfx', unsigned_sfx)
    def test_unsigned_overflow(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        maxu = (1 << int(sfx[1:])) - 1
        maxu_72 = (1 << 72) - 1
        lane = getattr(npyv, "setall_" + sfx)(maxu_72)[0]
        assert lane == maxu
        lanes = getattr(npyv, "load_" + sfx)([maxu_72] * nlanes)
        assert lanes == [maxu] * nlanes
        lane = getattr(npyv, "setall_" + sfx)(-1)[0]
        assert lane == maxu
        lanes = getattr(npyv, "load_" + sfx)([-1] * nlanes)
        assert lanes == [maxu] * nlanes

    @pytest.mark.parametrize('sfx', signed_sfx)
    def test_signed_overflow(self, sfx):
        nlanes = getattr(npyv, "nlanes_" + sfx)
        maxs_72 = (1 << 71) - 1
        lane = getattr(npyv, "setall_" + sfx)(maxs_72)[0]
        assert lane == -1
        lanes = getattr(npyv, "load_" + sfx)([maxs_72] * nlanes)
        assert lanes == [-1] * nlanes
        mins_72 = -1 << 71
        lane = getattr(npyv, "setall_" + sfx)(mins_72)[0]
        assert lane == 0
        lanes = getattr(npyv, "load_" + sfx)([mins_72] * nlanes)
        assert lanes == [0] * nlanes

    def test_truncate_f32(self):
        f32 = npyv.setall_f32(0.1)[0]
        assert f32 != 0.1
        assert round(f32, 1) == 0.1

    def test_compare(self):
        data_range = range(0, npyv.nlanes_u32)
        vdata = npyv.load_u32(data_range)
        assert vdata == list(data_range)
        assert vdata == tuple(data_range)
        for i in data_range:
            assert vdata[i] == data_range[i]
