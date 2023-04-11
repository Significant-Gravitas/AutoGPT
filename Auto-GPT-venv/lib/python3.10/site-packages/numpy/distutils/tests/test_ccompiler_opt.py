import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError

is_standalone = __name__ == '__main__' and __package__ is None
if is_standalone:
    import unittest, contextlib, tempfile, shutil
    sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))
    from ccompiler_opt import CCompilerOpt

    # from numpy/testing/_private/utils.py
    @contextlib.contextmanager
    def tempdir(*args, **kwargs):
        tmpdir = tempfile.mkdtemp(*args, **kwargs)
        try:
            yield tmpdir
        finally:
            shutil.rmtree(tmpdir)

    def assert_(expr, msg=''):
        if not expr:
            raise AssertionError(msg)
else:
    from numpy.distutils.ccompiler_opt import CCompilerOpt
    from numpy.testing import assert_, tempdir

# architectures and compilers to test
arch_compilers = dict(
    x86 = ("gcc", "clang", "icc", "iccw", "msvc"),
    x64 = ("gcc", "clang", "icc", "iccw", "msvc"),
    ppc64 = ("gcc", "clang"),
    ppc64le = ("gcc", "clang"),
    armhf = ("gcc", "clang"),
    aarch64 = ("gcc", "clang"),
    s390x = ("gcc", "clang"),
    noarch = ("gcc",)
)

class FakeCCompilerOpt(CCompilerOpt):
    fake_info = ""
    def __init__(self, trap_files="", trap_flags="", *args, **kwargs):
        self.fake_trap_files = trap_files
        self.fake_trap_flags = trap_flags
        CCompilerOpt.__init__(self, None, **kwargs)

    def __repr__(self):
        return textwrap.dedent("""\
            <<<<
            march    : {}
            compiler : {}
            ----------------
            {}
            >>>>
        """).format(self.cc_march, self.cc_name, self.report())

    def dist_compile(self, sources, flags, **kwargs):
        assert(isinstance(sources, list))
        assert(isinstance(flags, list))
        if self.fake_trap_files:
            for src in sources:
                if re.match(self.fake_trap_files, src):
                    self.dist_error("source is trapped by a fake interface")
        if self.fake_trap_flags:
            for f in flags:
                if re.match(self.fake_trap_flags, f):
                    self.dist_error("flag is trapped by a fake interface")
        # fake objects
        return zip(sources, [' '.join(flags)] * len(sources))

    def dist_info(self):
        return FakeCCompilerOpt.fake_info

    @staticmethod
    def dist_log(*args, stderr=False):
        pass

class _Test_CCompilerOpt:
    arch = None # x86_64
    cc   = None # gcc

    def setup_class(self):
        FakeCCompilerOpt.conf_nocache = True
        self._opt = None

    def nopt(self, *args, **kwargs):
        FakeCCompilerOpt.fake_info = (self.arch, self.cc, "")
        return FakeCCompilerOpt(*args, **kwargs)

    def opt(self):
        if not self._opt:
            self._opt = self.nopt()
        return self._opt

    def march(self):
        return self.opt().cc_march

    def cc_name(self):
        return self.opt().cc_name

    def get_targets(self, targets, groups, **kwargs):
        FakeCCompilerOpt.conf_target_groups = groups
        opt = self.nopt(
            cpu_baseline=kwargs.get("baseline", "min"),
            cpu_dispatch=kwargs.get("dispatch", "max"),
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        with tempdir() as tmpdir:
            file = os.path.join(tmpdir, "test_targets.c")
            with open(file, 'w') as f:
                f.write(targets)
            gtargets = []
            gflags = {}
            fake_objects = opt.try_dispatch([file])
            for source, flags in fake_objects:
                gtar = path.basename(source).split('.')[1:-1]
                glen = len(gtar)
                if glen == 0:
                    gtar = "baseline"
                elif glen == 1:
                    gtar = gtar[0].upper()
                else:
                    # converting multi-target into parentheses str format to be equivalent
                    # to the configuration statements syntax.
                    gtar = ('('+' '.join(gtar)+')').upper()
                gtargets.append(gtar)
                gflags[gtar] = flags

        has_baseline, targets = opt.sources_status[file]
        targets = targets + ["baseline"] if has_baseline else targets
        # convert tuple that represent multi-target into parentheses str format
        targets = [
            '('+' '.join(tar)+')' if isinstance(tar, tuple) else tar
            for tar in targets
        ]
        if len(targets) != len(gtargets) or not all(t in gtargets for t in targets):
            raise AssertionError(
                "'sources_status' returns different targets than the compiled targets\n"
                "%s != %s" % (targets, gtargets)
            )
        # return targets from 'sources_status' since the order is matters
        return targets, gflags

    def arg_regex(self, **kwargs):
        map2origin = dict(
            x64 = "x86",
            ppc64le = "ppc64",
            aarch64 = "armhf",
            clang = "gcc",
        )
        march = self.march(); cc_name = self.cc_name()
        map_march = map2origin.get(march, march)
        map_cc = map2origin.get(cc_name, cc_name)
        for key in (
            march, cc_name, map_march, map_cc,
            march + '_' + cc_name,
            map_march + '_' + cc_name,
            march + '_' + map_cc,
            map_march + '_' + map_cc,
        ) :
            regex = kwargs.pop(key, None)
            if regex is not None:
                break
        if regex:
            if isinstance(regex, dict):
                for k, v in regex.items():
                    if v[-1:] not in ')}$?\\.+*':
                        regex[k] = v + '$'
            else:
                assert(isinstance(regex, str))
                if regex[-1:] not in ')}$?\\.+*':
                    regex += '$'
        return regex

    def expect(self, dispatch, baseline="", **kwargs):
        match = self.arg_regex(**kwargs)
        if match is None:
            return
        opt = self.nopt(
            cpu_baseline=baseline, cpu_dispatch=dispatch,
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        features = ' '.join(opt.cpu_dispatch_names())
        if not match:
            if len(features) != 0:
                raise AssertionError(
                    'expected empty features, not "%s"' % features
                )
            return
        if not re.match(match, features, re.IGNORECASE):
            raise AssertionError(
                'dispatch features "%s" not match "%s"' % (features, match)
            )

    def expect_baseline(self, baseline, dispatch="", **kwargs):
        match = self.arg_regex(**kwargs)
        if match is None:
            return
        opt = self.nopt(
            cpu_baseline=baseline, cpu_dispatch=dispatch,
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        features = ' '.join(opt.cpu_baseline_names())
        if not match:
            if len(features) != 0:
                raise AssertionError(
                    'expected empty features, not "%s"' % features
                )
            return
        if not re.match(match, features, re.IGNORECASE):
            raise AssertionError(
                'baseline features "%s" not match "%s"' % (features, match)
            )

    def expect_flags(self, baseline, dispatch="", **kwargs):
        match = self.arg_regex(**kwargs)
        if match is None:
            return
        opt = self.nopt(
            cpu_baseline=baseline, cpu_dispatch=dispatch,
            trap_files=kwargs.get("trap_files", ""),
            trap_flags=kwargs.get("trap_flags", "")
        )
        flags = ' '.join(opt.cpu_baseline_flags())
        if not match:
            if len(flags) != 0:
                raise AssertionError(
                    'expected empty flags not "%s"' % flags
                )
            return
        if not re.match(match, flags):
            raise AssertionError(
                'flags "%s" not match "%s"' % (flags, match)
            )

    def expect_targets(self, targets, groups={}, **kwargs):
        match = self.arg_regex(**kwargs)
        if match is None:
            return
        targets, _ = self.get_targets(targets=targets, groups=groups, **kwargs)
        targets = ' '.join(targets)
        if not match:
            if len(targets) != 0:
                raise AssertionError(
                    'expected empty targets, not "%s"' % targets
                )
            return
        if not re.match(match, targets, re.IGNORECASE):
            raise AssertionError(
                'targets "%s" not match "%s"' % (targets, match)
            )

    def expect_target_flags(self, targets, groups={}, **kwargs):
        match_dict = self.arg_regex(**kwargs)
        if match_dict is None:
            return
        assert(isinstance(match_dict, dict))
        _, tar_flags = self.get_targets(targets=targets, groups=groups)

        for match_tar, match_flags in match_dict.items():
            if match_tar not in tar_flags:
                raise AssertionError(
                    'expected to find target "%s"' % match_tar
                )
            flags = tar_flags[match_tar]
            if not match_flags:
                if len(flags) != 0:
                    raise AssertionError(
                        'expected to find empty flags in target "%s"' % match_tar
                    )
            if not re.match(match_flags, flags):
                raise AssertionError(
                    '"%s" flags "%s" not match "%s"' % (match_tar, flags, match_flags)
                )

    def test_interface(self):
        wrong_arch = "ppc64" if self.arch != "ppc64" else "x86"
        wrong_cc   = "clang" if self.cc   != "clang" else "icc"
        opt = self.opt()
        assert_(getattr(opt, "cc_on_" + self.arch))
        assert_(not getattr(opt, "cc_on_" + wrong_arch))
        assert_(getattr(opt, "cc_is_" + self.cc))
        assert_(not getattr(opt, "cc_is_" + wrong_cc))

    def test_args_empty(self):
        for baseline, dispatch in (
            ("", "none"),
            (None, ""),
            ("none +none", "none - none"),
            ("none -max", "min - max"),
            ("+vsx2 -VSX2", "vsx avx2 avx512f -max"),
            ("max -vsx - avx + avx512f neon -MAX ",
             "min -min + max -max -vsx + avx2 -avx2 +NONE")
        ) :
            opt = self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch)
            assert(len(opt.cpu_baseline_names()) == 0)
            assert(len(opt.cpu_dispatch_names()) == 0)

    def test_args_validation(self):
        if self.march() == "unknown":
            return
        # check sanity of argument's validation
        for baseline, dispatch in (
            ("unkown_feature - max +min", "unknown max min"), # unknowing features
            ("#avx2", "$vsx") # groups and polices aren't acceptable
        ) :
            try:
                self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch)
                raise AssertionError("excepted an exception for invalid arguments")
            except DistutilsError:
                pass

    def test_skip(self):
        # only takes what platform supports and skip the others
        # without casing exceptions
        self.expect(
            "sse vsx neon",
            x86="sse", ppc64="vsx", armhf="neon", unknown=""
        )
        self.expect(
            "sse41 avx avx2 vsx2 vsx3 neon_vfpv4 asimd",
            x86   = "sse41 avx avx2",
            ppc64 = "vsx2 vsx3",
            armhf = "neon_vfpv4 asimd",
            unknown = ""
        )
        # any features in cpu_dispatch must be ignored if it's part of baseline
        self.expect(
            "sse neon vsx", baseline="sse neon vsx",
            x86="", ppc64="", armhf=""
        )
        self.expect(
            "avx2 vsx3 asimdhp", baseline="avx2 vsx3 asimdhp",
            x86="", ppc64="", armhf=""
        )

    def test_implies(self):
        # baseline combining implied features, so we count
        # on it instead of testing 'feature_implies()'' directly
        self.expect_baseline(
            "fma3 avx2 asimd vsx3",
            # .* between two spaces can validate features in between
            x86   = "sse .* sse41 .* fma3.*avx2",
            ppc64 = "vsx vsx2 vsx3",
            armhf = "neon neon_fp16 neon_vfpv4 asimd"
        )
        """
        special cases
        """
        # in icc and msvc, FMA3 and AVX2 can't be separated
        # both need to implies each other, same for avx512f & cd
        for f0, f1 in (
            ("fma3",    "avx2"),
            ("avx512f", "avx512cd"),
        ):
            diff = ".* sse42 .* %s .*%s$" % (f0, f1)
            self.expect_baseline(f0,
                x86_gcc=".* sse42 .* %s$" % f0,
                x86_icc=diff, x86_iccw=diff
            )
            self.expect_baseline(f1,
                x86_gcc=".* avx .* %s$" % f1,
                x86_icc=diff, x86_iccw=diff
            )
        # in msvc, following features can't be separated too
        for f in (("fma3", "avx2"), ("avx512f", "avx512cd", "avx512_skx")):
            for ff in f:
                self.expect_baseline(ff,
                    x86_msvc=".*%s" % ' '.join(f)
                )

        # in ppc64le VSX and VSX2 can't be separated
        self.expect_baseline("vsx", ppc64le="vsx vsx2")
        # in aarch64 following features can't be separated
        for f in ("neon", "neon_fp16", "neon_vfpv4", "asimd"):
            self.expect_baseline(f, aarch64="neon neon_fp16 neon_vfpv4 asimd")

    def test_args_options(self):
        # max & native
        for o in ("max", "native"):
            if o == "native" and self.cc_name() == "msvc":
                continue
            self.expect(o,
                trap_files=".*cpu_(sse|vsx|neon|vx).c",
                x86="", ppc64="", armhf="", s390x=""
            )
            self.expect(o,
                trap_files=".*cpu_(sse3|vsx2|neon_vfpv4|vxe).c",
                x86="sse sse2", ppc64="vsx", armhf="neon neon_fp16",
                aarch64="", ppc64le="", s390x="vx"
            )
            self.expect(o,
                trap_files=".*cpu_(popcnt|vsx3).c",
                x86="sse .* sse41", ppc64="vsx vsx2",
                armhf="neon neon_fp16 .* asimd .*",
                s390x="vx vxe vxe2"
            )
            self.expect(o,
                x86_gcc=".* xop fma4 .* avx512f .* avx512_knl avx512_knm avx512_skx .*",
                # in icc, xop and fam4 aren't supported
                x86_icc=".* avx512f .* avx512_knl avx512_knm avx512_skx .*",
                x86_iccw=".* avx512f .* avx512_knl avx512_knm avx512_skx .*",
                # in msvc, avx512_knl avx512_knm aren't supported
                x86_msvc=".* xop fma4 .* avx512f .* avx512_skx .*",
                armhf=".* asimd asimdhp asimddp .*",
                ppc64="vsx vsx2 vsx3 vsx4.*",
                s390x="vx vxe vxe2.*"
            )
        # min
        self.expect("min",
            x86="sse sse2", x64="sse sse2 sse3",
            armhf="", aarch64="neon neon_fp16 .* asimd",
            ppc64="", ppc64le="vsx vsx2", s390x=""
        )
        self.expect(
            "min", trap_files=".*cpu_(sse2|vsx2).c",
            x86="", ppc64le=""
        )
        # an exception must triggered if native flag isn't supported
        # when option "native" is activated through the args
        try:
            self.expect("native",
                trap_flags=".*(-march=native|-xHost|/QxHost).*",
                x86=".*", ppc64=".*", armhf=".*", s390x=".*"
            )
            if self.march() != "unknown":
                raise AssertionError(
                    "excepted an exception for %s" % self.march()
                )
        except DistutilsError:
            if self.march() == "unknown":
                raise AssertionError("excepted no exceptions")

    def test_flags(self):
        self.expect_flags(
            "sse sse2 vsx vsx2 neon neon_fp16 vx vxe",
            x86_gcc="-msse -msse2", x86_icc="-msse -msse2",
            x86_iccw="/arch:SSE2",
            x86_msvc="/arch:SSE2" if self.march() == "x86" else "",
            ppc64_gcc= "-mcpu=power8",
            ppc64_clang="-maltivec -mvsx -mpower8-vector",
            armhf_gcc="-mfpu=neon-fp16 -mfp16-format=ieee",
            aarch64="",
            s390x="-mzvector -march=arch12"
        )
        # testing normalize -march
        self.expect_flags(
            "asimd",
            aarch64="",
            armhf_gcc=r"-mfp16-format=ieee -mfpu=neon-fp-armv8 -march=armv8-a\+simd"
        )
        self.expect_flags(
            "asimdhp",
            aarch64_gcc=r"-march=armv8.2-a\+fp16",
            armhf_gcc=r"-mfp16-format=ieee -mfpu=neon-fp-armv8 -march=armv8.2-a\+fp16"
        )
        self.expect_flags(
            "asimddp", aarch64_gcc=r"-march=armv8.2-a\+dotprod"
        )
        self.expect_flags(
            # asimdfhm implies asimdhp
            "asimdfhm", aarch64_gcc=r"-march=armv8.2-a\+fp16\+fp16fml"
        )
        self.expect_flags(
            "asimddp asimdhp asimdfhm",
            aarch64_gcc=r"-march=armv8.2-a\+dotprod\+fp16\+fp16fml"
        )
        self.expect_flags(
            "vx vxe vxe2",
            s390x=r"-mzvector -march=arch13"
        )

    def test_targets_exceptions(self):
        for targets in (
            "bla bla", "/*@targets",
            "/*@targets */",
            "/*@targets unknown */",
            "/*@targets $unknown_policy avx2 */",
            "/*@targets #unknown_group avx2 */",
            "/*@targets $ */",
            "/*@targets # vsx */",
            "/*@targets #$ vsx */",
            "/*@targets vsx avx2 ) */",
            "/*@targets vsx avx2 (avx2 */",
            "/*@targets vsx avx2 () */",
            "/*@targets vsx avx2 ($autovec) */", # no features
            "/*@targets vsx avx2 (xxx) */",
            "/*@targets vsx avx2 (baseline) */",
        ) :
            try:
                self.expect_targets(
                    targets,
                    x86="", armhf="", ppc64="", s390x=""
                )
                if self.march() != "unknown":
                    raise AssertionError(
                        "excepted an exception for %s" % self.march()
                    )
            except DistutilsError:
                if self.march() == "unknown":
                    raise AssertionError("excepted no exceptions")

    def test_targets_syntax(self):
        for targets in (
            "/*@targets $keep_baseline sse vsx neon vx*/",
            "/*@targets,$keep_baseline,sse,vsx,neon vx*/",
            "/*@targets*$keep_baseline*sse*vsx*neon*vx*/",
            """
            /*
            ** @targets
            ** $keep_baseline, sse vsx,neon, vx
            */
            """,
            """
            /*
            ************@targets****************
            ** $keep_baseline, sse vsx, neon, vx
            ************************************
            */
            """,
            """
            /*
            /////////////@targets/////////////////
            //$keep_baseline//sse//vsx//neon//vx
            /////////////////////////////////////
            */
            """,
            """
            /*
            @targets
            $keep_baseline
            SSE VSX NEON VX*/
            """
        ) :
            self.expect_targets(targets,
                x86="sse", ppc64="vsx", armhf="neon", s390x="vx", unknown=""
            )

    def test_targets(self):
        # test skipping baseline features
        self.expect_targets(
            """
            /*@targets
                sse sse2 sse41 avx avx2 avx512f
                vsx vsx2 vsx3 vsx4
                neon neon_fp16 asimdhp asimddp
                vx vxe vxe2
            */
            """,
            baseline="avx vsx2 asimd vx vxe",
            x86="avx512f avx2", armhf="asimddp asimdhp", ppc64="vsx4 vsx3",
            s390x="vxe2"
        )
        # test skipping non-dispatch features
        self.expect_targets(
            """
            /*@targets
                sse41 avx avx2 avx512f
                vsx2 vsx3 vsx4
                asimd asimdhp asimddp
                vx vxe vxe2
            */
            """,
            baseline="", dispatch="sse41 avx2 vsx2 asimd asimddp vxe2",
            x86="avx2 sse41", armhf="asimddp asimd", ppc64="vsx2", s390x="vxe2"
        )
        # test skipping features that not supported
        self.expect_targets(
            """
            /*@targets
                sse2 sse41 avx2 avx512f
                vsx2 vsx3 vsx4
                neon asimdhp asimddp
                vx vxe vxe2
            */
            """,
            baseline="",
            trap_files=".*(avx2|avx512f|vsx3|vsx4|asimddp|vxe2).c",
            x86="sse41 sse2", ppc64="vsx2", armhf="asimdhp neon",
            s390x="vxe vx"
        )
        # test skipping features that implies each other
        self.expect_targets(
            """
            /*@targets
                sse sse2 avx fma3 avx2 avx512f avx512cd
                vsx vsx2 vsx3
                neon neon_vfpv4 neon_fp16 neon_fp16 asimd asimdhp
                asimddp asimdfhm
            */
            """,
            baseline="",
            x86_gcc="avx512cd avx512f avx2 fma3 avx sse2",
            x86_msvc="avx512cd avx2 avx sse2",
            x86_icc="avx512cd avx2 avx sse2",
            x86_iccw="avx512cd avx2 avx sse2",
            ppc64="vsx3 vsx2 vsx",
            ppc64le="vsx3 vsx2",
            armhf="asimdfhm asimddp asimdhp asimd neon_vfpv4 neon_fp16 neon",
            aarch64="asimdfhm asimddp asimdhp asimd"
        )

    def test_targets_policies(self):
        # 'keep_baseline', generate objects for baseline features
        self.expect_targets(
            """
            /*@targets
                $keep_baseline
                sse2 sse42 avx2 avx512f
                vsx2 vsx3
                neon neon_vfpv4 asimd asimddp
                vx vxe vxe2
            */
            """,
            baseline="sse41 avx2 vsx2 asimd vsx3 vxe",
            x86="avx512f avx2 sse42 sse2",
            ppc64="vsx3 vsx2",
            armhf="asimddp asimd neon_vfpv4 neon",
            # neon, neon_vfpv4, asimd implies each other
            aarch64="asimddp asimd",
            s390x="vxe2 vxe vx"
        )
        # 'keep_sort', leave the sort as-is
        self.expect_targets(
            """
            /*@targets
                $keep_baseline $keep_sort
                avx512f sse42 avx2 sse2
                vsx2 vsx3
                asimd neon neon_vfpv4 asimddp
                vxe vxe2
            */
            """,
            x86="avx512f sse42 avx2 sse2",
            ppc64="vsx2 vsx3",
            armhf="asimd neon neon_vfpv4 asimddp",
            # neon, neon_vfpv4, asimd implies each other
            aarch64="asimd asimddp",
            s390x="vxe vxe2"
        )
        # 'autovec', skipping features that can't be
        # vectorized by the compiler
        self.expect_targets(
            """
            /*@targets
                $keep_baseline $keep_sort $autovec
                avx512f avx2 sse42 sse41 sse2
                vsx3 vsx2
                asimddp asimd neon_vfpv4 neon
            */
            """,
            x86_gcc="avx512f avx2 sse42 sse41 sse2",
            x86_icc="avx512f avx2 sse42 sse41 sse2",
            x86_iccw="avx512f avx2 sse42 sse41 sse2",
            x86_msvc="avx512f avx2 sse2"
                     if self.march() == 'x86' else "avx512f avx2",
            ppc64="vsx3 vsx2",
            armhf="asimddp asimd neon_vfpv4 neon",
            # neon, neon_vfpv4, asimd implies each other
            aarch64="asimddp asimd"
        )
        for policy in ("$maxopt", "$autovec"):
            # 'maxopt' and autovec set the max acceptable optimization flags
            self.expect_target_flags(
                "/*@targets baseline %s */" % policy,
                gcc={"baseline":".*-O3.*"}, icc={"baseline":".*-O3.*"},
                iccw={"baseline":".*/O3.*"}, msvc={"baseline":".*/O2.*"},
                unknown={"baseline":".*"}
            )

        # 'werror', force compilers to treat warnings as errors
        self.expect_target_flags(
            "/*@targets baseline $werror */",
            gcc={"baseline":".*-Werror.*"}, icc={"baseline":".*-Werror.*"},
            iccw={"baseline":".*/Werror.*"}, msvc={"baseline":".*/WX.*"},
            unknown={"baseline":".*"}
        )

    def test_targets_groups(self):
        self.expect_targets(
            """
            /*@targets $keep_baseline baseline #test_group */
            """,
            groups=dict(
                test_group=("""
                    $keep_baseline
                    asimddp sse2 vsx2 avx2 vsx3
                    avx512f asimdhp
                """)
            ),
            x86="avx512f avx2 sse2 baseline",
            ppc64="vsx3 vsx2 baseline",
            armhf="asimddp asimdhp baseline"
        )
        # test skip duplicating and sorting
        self.expect_targets(
            """
            /*@targets
             * sse42 avx avx512f
             * #test_group_1
             * vsx2
             * #test_group_2
             * asimddp asimdfhm
            */
            """,
            groups=dict(
                test_group_1=("""
                    VSX2 vsx3 asimd avx2 SSE41
                """),
                test_group_2=("""
                    vsx2 vsx3 asImd aVx2 sse41
                """)
            ),
            x86="avx512f avx2 avx sse42 sse41",
            ppc64="vsx3 vsx2",
            # vsx2 part of the default baseline of ppc64le, option ("min")
            ppc64le="vsx3",
            armhf="asimdfhm asimddp asimd",
            # asimd part of the default baseline of aarch64, option ("min")
            aarch64="asimdfhm asimddp"
        )

    def test_targets_multi(self):
        self.expect_targets(
            """
            /*@targets
                (avx512_clx avx512_cnl) (asimdhp asimddp)
            */
            """,
            x86=r"\(avx512_clx avx512_cnl\)",
            armhf=r"\(asimdhp asimddp\)",
        )
        # test skipping implied features and auto-sort
        self.expect_targets(
            """
            /*@targets
                f16c (sse41 avx sse42) (sse3 avx2 avx512f)
                vsx2 (vsx vsx3 vsx2)
                (neon neon_vfpv4 asimd asimdhp asimddp)
            */
            """,
            x86="avx512f f16c avx",
            ppc64="vsx3 vsx2",
            ppc64le="vsx3", # vsx2 part of baseline
            armhf=r"\(asimdhp asimddp\)",
        )
        # test skipping implied features and keep sort
        self.expect_targets(
            """
            /*@targets $keep_sort
                (sse41 avx sse42) (sse3 avx2 avx512f)
                (vsx vsx3 vsx2)
                (asimddp neon neon_vfpv4 asimd asimdhp)
                (vx vxe vxe2)
            */
            """,
            x86="avx avx512f",
            ppc64="vsx3",
            armhf=r"\(asimdhp asimddp\)",
            s390x="vxe2"
        )
        # test compiler variety and avoiding duplicating
        self.expect_targets(
            """
            /*@targets $keep_sort
                fma3 avx2 (fma3 avx2) (avx2 fma3) avx2 fma3
            */
            """,
            x86_gcc=r"fma3 avx2 \(fma3 avx2\)",
            x86_icc="avx2", x86_iccw="avx2",
            x86_msvc="avx2"
        )

def new_test(arch, cc):
    if is_standalone: return textwrap.dedent("""\
    class TestCCompilerOpt_{class_name}(_Test_CCompilerOpt, unittest.TestCase):
        arch = '{arch}'
        cc   = '{cc}'
        def __init__(self, methodName="runTest"):
            unittest.TestCase.__init__(self, methodName)
            self.setup_class()
    """).format(
        class_name=arch + '_' + cc, arch=arch, cc=cc
    )
    return textwrap.dedent("""\
    class TestCCompilerOpt_{class_name}(_Test_CCompilerOpt):
        arch = '{arch}'
        cc   = '{cc}'
    """).format(
        class_name=arch + '_' + cc, arch=arch, cc=cc
    )
"""
if 1 and is_standalone:
    FakeCCompilerOpt.fake_info = "x86_icc"
    cco = FakeCCompilerOpt(None, cpu_baseline="avx2")
    print(' '.join(cco.cpu_baseline_names()))
    print(cco.cpu_baseline_flags())
    unittest.main()
    sys.exit()
"""
for arch, compilers in arch_compilers.items():
    for cc in compilers:
        exec(new_test(arch, cc))

if is_standalone:
    unittest.main()
