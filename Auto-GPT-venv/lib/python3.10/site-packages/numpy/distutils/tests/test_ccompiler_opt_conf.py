import unittest
from os import sys, path

is_standalone = __name__ == '__main__' and __package__ is None
if is_standalone:
    sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))
    from ccompiler_opt import CCompilerOpt
else:
    from numpy.distutils.ccompiler_opt import CCompilerOpt

arch_compilers = dict(
    x86 = ("gcc", "clang", "icc", "iccw", "msvc"),
    x64 = ("gcc", "clang", "icc", "iccw", "msvc"),
    ppc64 = ("gcc", "clang"),
    ppc64le = ("gcc", "clang"),
    armhf = ("gcc", "clang"),
    aarch64 = ("gcc", "clang"),
    narch = ("gcc",)
)

class FakeCCompilerOpt(CCompilerOpt):
    fake_info = ("arch", "compiler", "extra_args")
    def __init__(self, *args, **kwargs):
        CCompilerOpt.__init__(self, None, **kwargs)
    def dist_compile(self, sources, flags, **kwargs):
        return sources
    def dist_info(self):
        return FakeCCompilerOpt.fake_info
    @staticmethod
    def dist_log(*args, stderr=False):
        pass

class _TestConfFeatures(FakeCCompilerOpt):
    """A hook to check the sanity of configured features
-   before it called by the abstract class '_Feature'
    """

    def conf_features_partial(self):
        conf_all = self.conf_features
        for feature_name, feature in conf_all.items():
            self.test_feature(
                "attribute conf_features",
                conf_all, feature_name, feature
            )

        conf_partial = FakeCCompilerOpt.conf_features_partial(self)
        for feature_name, feature in conf_partial.items():
            self.test_feature(
                "conf_features_partial()",
                conf_partial, feature_name, feature
            )
        return conf_partial

    def test_feature(self, log, search_in, feature_name, feature_dict):
        error_msg = (
            "during validate '{}' within feature '{}', "
            "march '{}' and compiler '{}'\n>> "
        ).format(log, feature_name, self.cc_march, self.cc_name)

        if not feature_name.isupper():
            raise AssertionError(error_msg + "feature name must be in uppercase")

        for option, val in feature_dict.items():
            self.test_option_types(error_msg, option, val)
            self.test_duplicates(error_msg, option, val)

        self.test_implies(error_msg, search_in, feature_name, feature_dict)
        self.test_group(error_msg, search_in, feature_name, feature_dict)
        self.test_extra_checks(error_msg, search_in, feature_name, feature_dict)

    def test_option_types(self, error_msg, option, val):
        for tp, available in (
            ((str, list), (
                "implies", "headers", "flags", "group", "detect", "extra_checks"
            )),
            ((str,),  ("disable",)),
            ((int,),  ("interest",)),
            ((bool,), ("implies_detect",)),
            ((bool, type(None)), ("autovec",)),
        ) :
            found_it = option in available
            if not found_it:
                continue
            if not isinstance(val, tp):
                error_tp = [t.__name__ for t in (*tp,)]
                error_tp = ' or '.join(error_tp)
                raise AssertionError(error_msg +
                    "expected '%s' type for option '%s' not '%s'" % (
                     error_tp, option, type(val).__name__
                ))
            break

        if not found_it:
            raise AssertionError(error_msg + "invalid option name '%s'" % option)

    def test_duplicates(self, error_msg, option, val):
        if option not in (
            "implies", "headers", "flags", "group", "detect", "extra_checks"
        ) : return

        if isinstance(val, str):
            val = val.split()

        if len(val) != len(set(val)):
            raise AssertionError(error_msg + "duplicated values in option '%s'" % option)

    def test_implies(self, error_msg, search_in, feature_name, feature_dict):
        if feature_dict.get("disabled") is not None:
            return
        implies = feature_dict.get("implies", "")
        if not implies:
            return
        if isinstance(implies, str):
            implies = implies.split()

        if feature_name in implies:
            raise AssertionError(error_msg + "feature implies itself")

        for impl in implies:
            impl_dict = search_in.get(impl)
            if impl_dict is not None:
                if "disable" in impl_dict:
                    raise AssertionError(error_msg + "implies disabled feature '%s'" % impl)
                continue
            raise AssertionError(error_msg + "implies non-exist feature '%s'" % impl)

    def test_group(self, error_msg, search_in, feature_name, feature_dict):
        if feature_dict.get("disabled") is not None:
            return
        group = feature_dict.get("group", "")
        if not group:
            return
        if isinstance(group, str):
            group = group.split()

        for f in group:
            impl_dict = search_in.get(f)
            if not impl_dict or "disable" in impl_dict:
                continue
            raise AssertionError(error_msg +
                "in option 'group', '%s' already exists as a feature name" % f
            )

    def test_extra_checks(self, error_msg, search_in, feature_name, feature_dict):
        if feature_dict.get("disabled") is not None:
            return
        extra_checks = feature_dict.get("extra_checks", "")
        if not extra_checks:
            return
        if isinstance(extra_checks, str):
            extra_checks = extra_checks.split()

        for f in extra_checks:
            impl_dict = search_in.get(f)
            if not impl_dict or "disable" in impl_dict:
                continue
            raise AssertionError(error_msg +
                "in option 'extra_checks', extra test case '%s' already exists as a feature name" % f
            )

class TestConfFeatures(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        unittest.TestCase.__init__(self, methodName)
        self._setup()

    def _setup(self):
        FakeCCompilerOpt.conf_nocache = True

    def test_features(self):
        for arch, compilers in arch_compilers.items():
            for cc in compilers:
                FakeCCompilerOpt.fake_info = (arch, cc, "")
                _TestConfFeatures()

if is_standalone:
    unittest.main()
