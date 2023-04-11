# http://g95.sourceforge.net/
from numpy.distutils.fcompiler import FCompiler

compilers = ['G95FCompiler']

class G95FCompiler(FCompiler):
    compiler_type = 'g95'
    description = 'G95 Fortran Compiler'

#    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95!\) (?P<version>.*)\).*'
    # $ g95 --version
    # G95 (GCC 4.0.3 (g95!) May 22 2006)

    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95 (?P<version>.*)!\) (?P<date>.*)\).*'
    # $ g95 --version
    # G95 (GCC 4.0.3 (g95 0.90!) Aug 22 2006)

    executables = {
        'version_cmd'  : ["<F90>", "--version"],
        'compiler_f77' : ["g95", "-ffixed-form"],
        'compiler_fix' : ["g95", "-ffixed-form"],
        'compiler_f90' : ["g95"],
        'linker_so'    : ["<F90>", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    pic_flags = ['-fpic']
    module_dir_switch = '-fmod='
    module_include_switch = '-I'

    def get_flags(self):
        return ['-fno-second-underscore']
    def get_flags_opt(self):
        return ['-O']
    def get_flags_debug(self):
        return ['-g']

if __name__ == '__main__':
    from distutils import log
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)
    print(customized_fcompiler('g95').get_version())
