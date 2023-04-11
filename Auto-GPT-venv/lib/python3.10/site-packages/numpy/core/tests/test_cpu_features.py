import sys, platform, re, pytest
from numpy.core._multiarray_umath import __cpu_features__

def assert_features_equal(actual, desired, fname):
    __tracebackhide__ = True  # Hide traceback for py.test
    actual, desired = str(actual), str(desired)
    if actual == desired:
        return
    detected = str(__cpu_features__).replace("'", "")
    try:
        with open("/proc/cpuinfo", "r") as fd:
            cpuinfo = fd.read(2048)
    except Exception as err:
        cpuinfo = str(err)

    try:
        import subprocess
        auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV="1"))
        auxv = auxv.decode()
    except Exception as err:
        auxv = str(err)

    import textwrap
    error_report = textwrap.indent(
"""
###########################################
### Extra debugging information
###########################################
-------------------------------------------
--- NumPy Detections
-------------------------------------------
%s
-------------------------------------------
--- SYS / CPUINFO
-------------------------------------------
%s....
-------------------------------------------
--- SYS / AUXV
-------------------------------------------
%s
""" % (detected, cpuinfo, auxv), prefix='\r')

    raise AssertionError((
        "Failure Detection\n"
        " NAME: '%s'\n"
        " ACTUAL: %s\n"
        " DESIRED: %s\n"
        "%s"
    ) % (fname, actual, desired, error_report))

class AbstractTest:
    features = []
    features_groups = {}
    features_map = {}
    features_flags = set()

    def load_flags(self):
        # a hook
        pass
    def test_features(self):
        self.load_flags()
        for gname, features in self.features_groups.items():
            test_features = [self.cpu_have(f) for f in features]
            assert_features_equal(__cpu_features__.get(gname), all(test_features), gname)

        for feature_name in self.features:
            cpu_have = self.cpu_have(feature_name)
            npy_have = __cpu_features__.get(feature_name)
            assert_features_equal(npy_have, cpu_have, feature_name)

    def cpu_have(self, feature_name):
        map_names = self.features_map.get(feature_name, feature_name)
        if isinstance(map_names, str):
            return map_names in self.features_flags
        for f in map_names:
            if f in self.features_flags:
                return True
        return False

    def load_flags_cpuinfo(self, magic_key):
        self.features_flags = self.get_cpuinfo_item(magic_key)

    def get_cpuinfo_item(self, magic_key):
        values = set()
        with open('/proc/cpuinfo') as fd:
            for line in fd:
                if not line.startswith(magic_key):
                    continue
                flags_value = [s.strip() for s in line.split(':', 1)]
                if len(flags_value) == 2:
                    values = values.union(flags_value[1].upper().split())
        return values

    def load_flags_auxv(self):
        import subprocess
        auxv = subprocess.check_output(['/bin/true'], env=dict(LD_SHOW_AUXV="1"))
        for at in auxv.split(b'\n'):
            if not at.startswith(b"AT_HWCAP"):
                continue
            hwcap_value = [s.strip() for s in at.split(b':', 1)]
            if len(hwcap_value) == 2:
                self.features_flags = self.features_flags.union(
                    hwcap_value[1].upper().decode().split()
                )

is_linux = sys.platform.startswith('linux')
is_cygwin = sys.platform.startswith('cygwin')
machine  = platform.machine()
is_x86   = re.match("^(amd64|x86|i386|i686)", machine, re.IGNORECASE)
@pytest.mark.skipif(
    not (is_linux or is_cygwin) or not is_x86, reason="Only for Linux and x86"
)
class Test_X86_Features(AbstractTest):
    features = [
        "MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE41", "POPCNT", "SSE42",
        "AVX", "F16C", "XOP", "FMA4", "FMA3", "AVX2", "AVX512F", "AVX512CD",
        "AVX512ER", "AVX512PF", "AVX5124FMAPS", "AVX5124VNNIW", "AVX512VPOPCNTDQ",
        "AVX512VL", "AVX512BW", "AVX512DQ", "AVX512VNNI", "AVX512IFMA",
        "AVX512VBMI", "AVX512VBMI2", "AVX512BITALG",
    ]
    features_groups = dict(
        AVX512_KNL = ["AVX512F", "AVX512CD", "AVX512ER", "AVX512PF"],
        AVX512_KNM = ["AVX512F", "AVX512CD", "AVX512ER", "AVX512PF", "AVX5124FMAPS",
                      "AVX5124VNNIW", "AVX512VPOPCNTDQ"],
        AVX512_SKX = ["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL"],
        AVX512_CLX = ["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512VNNI"],
        AVX512_CNL = ["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512IFMA",
                      "AVX512VBMI"],
        AVX512_ICL = ["AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL", "AVX512IFMA",
                      "AVX512VBMI", "AVX512VNNI", "AVX512VBMI2", "AVX512BITALG", "AVX512VPOPCNTDQ"],
    )
    features_map = dict(
        SSE3="PNI", SSE41="SSE4_1", SSE42="SSE4_2", FMA3="FMA",
        AVX512VNNI="AVX512_VNNI", AVX512BITALG="AVX512_BITALG", AVX512VBMI2="AVX512_VBMI2",
        AVX5124FMAPS="AVX512_4FMAPS", AVX5124VNNIW="AVX512_4VNNIW", AVX512VPOPCNTDQ="AVX512_VPOPCNTDQ",
    )
    def load_flags(self):
        self.load_flags_cpuinfo("flags")

is_power = re.match("^(powerpc|ppc)64", machine, re.IGNORECASE)
@pytest.mark.skipif(not is_linux or not is_power, reason="Only for Linux and Power")
class Test_POWER_Features(AbstractTest):
    features = ["VSX", "VSX2", "VSX3", "VSX4"]
    features_map = dict(VSX2="ARCH_2_07", VSX3="ARCH_3_00", VSX4="ARCH_3_1")

    def load_flags(self):
        self.load_flags_auxv()


is_zarch = re.match("^(s390x)", machine, re.IGNORECASE)
@pytest.mark.skipif(not is_linux or not is_zarch,
                    reason="Only for Linux and IBM Z")
class Test_ZARCH_Features(AbstractTest):
    features = ["VX", "VXE", "VXE2"]

    def load_flags(self):
        self.load_flags_auxv()


is_arm = re.match("^(arm|aarch64)", machine, re.IGNORECASE)
@pytest.mark.skipif(not is_linux or not is_arm, reason="Only for Linux and ARM")
class Test_ARM_Features(AbstractTest):
    features = [
        "NEON", "ASIMD", "FPHP", "ASIMDHP", "ASIMDDP", "ASIMDFHM"
    ]
    features_groups = dict(
        NEON_FP16  = ["NEON", "HALF"],
        NEON_VFPV4 = ["NEON", "VFPV4"],
    )
    def load_flags(self):
        self.load_flags_cpuinfo("Features")
        arch = self.get_cpuinfo_item("CPU architecture")
        # in case of mounting virtual filesystem of aarch64 kernel
        is_rootfs_v8 = int('0'+next(iter(arch))) > 7 if arch else 0
        if  re.match("^(aarch64|AARCH64)", machine) or is_rootfs_v8:
            self.features_map = dict(
                NEON="ASIMD", HALF="ASIMD", VFPV4="ASIMD"
            )
        else:
            self.features_map = dict(
                # ELF auxiliary vector and /proc/cpuinfo on Linux kernel(armv8 aarch32)
                # doesn't provide information about ASIMD, so we assume that ASIMD is supported
                # if the kernel reports any one of the following ARM8 features.
                ASIMD=("AES", "SHA1", "SHA2", "PMULL", "CRC32")
            )
