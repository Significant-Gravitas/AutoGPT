from numpy.distutils.fcompiler import FCompiler

compilers = ['PathScaleFCompiler']

class PathScaleFCompiler(FCompiler):

    compiler_type = 'pathf95'
    description = 'PathScale Fortran Compiler'
    version_pattern =  r'PathScale\(TM\) Compiler Suite: Version (?P<version>[\d.]+)'

    executables = {
        'version_cmd'  : ["pathf95", "-version"],
        'compiler_f77' : ["pathf95", "-fixedform"],
        'compiler_fix' : ["pathf95", "-fixedform"],
        'compiler_f90' : ["pathf95"],
        'linker_so'    : ["pathf95", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
    }
    pic_flags = ['-fPIC']
    module_dir_switch = '-module ' # Don't remove ending space!
    module_include_switch = '-I'

    def get_flags_opt(self):
        return ['-O3']
    def get_flags_debug(self):
        return ['-g']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='pathf95').get_version())
