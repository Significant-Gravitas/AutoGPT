import sys
import re
from numpy.distutils.fcompiler import FCompiler

compilers = ['NAGFCompiler', 'NAGFORCompiler']

class BaseNAGFCompiler(FCompiler):
    version_pattern = r'NAG.* Release (?P<version>[^(\s]*)'

    def version_match(self, version_string):
        m = re.search(self.version_pattern, version_string)
        if m:
            return m.group('version')
        else:
            return None

    def get_flags_linker_so(self):
        return ["-Wl,-shared"]
    def get_flags_opt(self):
        return ['-O4']
    def get_flags_arch(self):
        return []

class NAGFCompiler(BaseNAGFCompiler):

    compiler_type = 'nag'
    description = 'NAGWare Fortran 95 Compiler'

    executables = {
        'version_cmd'  : ["<F90>", "-V"],
        'compiler_f77' : ["f95", "-fixed"],
        'compiler_fix' : ["f95", "-fixed"],
        'compiler_f90' : ["f95"],
        'linker_so'    : ["<F90>"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_flags_linker_so(self):
        if sys.platform == 'darwin':
            return ['-unsharedf95', '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return BaseNAGFCompiler.get_flags_linker_so(self)
    def get_flags_arch(self):
        version = self.get_version()
        if version and version < '5.1':
            return ['-target=native']
        else:
            return BaseNAGFCompiler.get_flags_arch(self)
    def get_flags_debug(self):
        return ['-g', '-gline', '-g90', '-nan', '-C']

class NAGFORCompiler(BaseNAGFCompiler):

    compiler_type = 'nagfor'
    description = 'NAG Fortran Compiler'

    executables = {
        'version_cmd'  : ["nagfor", "-V"],
        'compiler_f77' : ["nagfor", "-fixed"],
        'compiler_fix' : ["nagfor", "-fixed"],
        'compiler_f90' : ["nagfor"],
        'linker_so'    : ["nagfor"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    def get_flags_linker_so(self):
        if sys.platform == 'darwin':
            return ['-unsharedrts',
                    '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        return BaseNAGFCompiler.get_flags_linker_so(self)
    def get_flags_debug(self):
        version = self.get_version()
        if version and version > '6.1':
            return ['-g', '-u', '-nan', '-C=all', '-thread_safe',
                    '-kind=unique', '-Warn=allocation', '-Warn=subnormal']
        else:
            return ['-g', '-nan', '-C=all', '-u', '-thread_safe']


if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    compiler = customized_fcompiler(compiler='nagfor')
    print(compiler.get_version())
    print(compiler.get_flags_debug())
