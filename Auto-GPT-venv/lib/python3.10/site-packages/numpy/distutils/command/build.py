import os
import sys
from distutils.command.build import build as old_build
from distutils.util import get_platform
from numpy.distutils.command.config_compiler import show_fortran_compilers

class build(old_build):

    sub_commands = [('config_cc',     lambda *args: True),
                    ('config_fc',     lambda *args: True),
                    ('build_src',     old_build.has_ext_modules),
                    ] + old_build.sub_commands

    user_options = old_build.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ('warn-error', None,
         "turn all warnings into errors (-Werror)"),
        ('cpu-baseline=', None,
         "specify a list of enabled baseline CPU optimizations"),
        ('cpu-dispatch=', None,
         "specify a list of dispatched CPU optimizations"),
        ('disable-optimization', None,
         "disable CPU optimized code(dispatch,simd,fast...)"),
        ('simd-test=', None,
         "specify a list of CPU optimizations to be tested against NumPy SIMD interface"),
        ]

    help_options = old_build.help_options + [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
        ]

    def initialize_options(self):
        old_build.initialize_options(self)
        self.fcompiler = None
        self.warn_error = False
        self.cpu_baseline = "min"
        self.cpu_dispatch = "max -xop -fma4" # drop AMD legacy features by default
        self.disable_optimization = False
        """
        the '_simd' module is a very large. Adding more dispatched features
        will increase binary size and compile time. By default we minimize
        the targeted features to those most commonly used by the NumPy SIMD interface(NPYV),
        NOTE: any specified features will be ignored if they're:
            - part of the baseline(--cpu-baseline)
            - not part of dispatch-able features(--cpu-dispatch)
            - not supported by compiler or platform
        """
        self.simd_test = "BASELINE SSE2 SSE42 XOP FMA4 (FMA3 AVX2) AVX512F " \
                         "AVX512_SKX VSX VSX2 VSX3 VSX4 NEON ASIMD VX VXE VXE2"

    def finalize_options(self):
        build_scripts = self.build_scripts
        old_build.finalize_options(self)
        plat_specifier = ".{}-{}.{}".format(get_platform(), *sys.version_info[:2])
        if build_scripts is None:
            self.build_scripts = os.path.join(self.build_base,
                                              'scripts' + plat_specifier)

    def run(self):
        old_build.run(self)
