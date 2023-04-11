# http://www.pgroup.com
import sys

from numpy.distutils.fcompiler import FCompiler
from sys import platform
from os.path import join, dirname, normpath

compilers = ['PGroupFCompiler', 'PGroupFlangCompiler']


class PGroupFCompiler(FCompiler):

    compiler_type = 'pg'
    description = 'Portland Group Fortran Compiler'
    version_pattern = r'\s*pg(f77|f90|hpf|fortran) (?P<version>[\d.-]+).*'

    if platform == 'darwin':
        executables = {
            'version_cmd': ["<F77>", "-V"],
            'compiler_f77': ["pgfortran", "-dynamiclib"],
            'compiler_fix': ["pgfortran", "-Mfixed", "-dynamiclib"],
            'compiler_f90': ["pgfortran", "-dynamiclib"],
            'linker_so': ["libtool"],
            'archiver': ["ar", "-cr"],
            'ranlib': ["ranlib"]
        }
        pic_flags = ['']
    else:
        executables = {
            'version_cmd': ["<F77>", "-V"],
            'compiler_f77': ["pgfortran"],
            'compiler_fix': ["pgfortran", "-Mfixed"],
            'compiler_f90': ["pgfortran"],
            'linker_so': ["<F90>"],
            'archiver': ["ar", "-cr"],
            'ranlib': ["ranlib"]
        }
        pic_flags = ['-fpic']

    module_dir_switch = '-module '
    module_include_switch = '-I'

    def get_flags(self):
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt

    def get_flags_opt(self):
        return ['-fast']

    def get_flags_debug(self):
        return ['-g']

    if platform == 'darwin':
        def get_flags_linker_so(self):
            return ["-dynamic", '-undefined', 'dynamic_lookup']

    else:
        def get_flags_linker_so(self):
            return ["-shared", '-fpic']

    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir


import functools

class PGroupFlangCompiler(FCompiler):
    compiler_type = 'flang'
    description = 'Portland Group Fortran LLVM Compiler'
    version_pattern = r'\s*(flang|clang) version (?P<version>[\d.-]+).*'

    ar_exe = 'lib.exe'
    possible_executables = ['flang']

    executables = {
        'version_cmd': ["<F77>", "--version"],
        'compiler_f77': ["flang"],
        'compiler_fix': ["flang"],
        'compiler_f90': ["flang"],
        'linker_so': [None],
        'archiver': [ar_exe, "/verbose", "/OUT:"],
        'ranlib': None
    }

    library_switch = '/OUT:'  # No space after /OUT:!
    module_dir_switch = '-module '  # Don't remove ending space!

    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['flang', 'flangrti', 'ompstub'])
        return opt

    @functools.lru_cache(maxsize=128)
    def get_library_dirs(self):
        """List of compiler library directories."""
        opt = FCompiler.get_library_dirs(self)
        flang_dir = dirname(self.executables['compiler_f77'][0])
        opt.append(normpath(join(flang_dir, '..', 'lib')))

        return opt

    def get_flags(self):
        return []

    def get_flags_free(self):
        return []

    def get_flags_debug(self):
        return ['-g']

    def get_flags_opt(self):
        return ['-O3']

    def get_flags_arch(self):
        return []

    def runtime_library_dir_option(self, dir):
        raise NotImplementedError


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    if 'flang' in sys.argv:
        print(customized_fcompiler(compiler='flang').get_version())
    else:
        print(customized_fcompiler(compiler='pg').get_version())
