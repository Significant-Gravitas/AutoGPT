from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler

compilers = ['SunFCompiler']

class SunFCompiler(FCompiler):

    compiler_type = 'sun'
    description = 'Sun or Forte Fortran 95 Compiler'
    # ex:
    # f90: Sun WorkShop 6 update 2 Fortran 95 6.2 Patch 111690-10 2003/08/28
    version_match = simple_version_match(
                      start=r'f9[05]: (Sun|Forte|WorkShop).*Fortran 95')

    executables = {
        'version_cmd'  : ["<F90>", "-V"],
        'compiler_f77' : ["f90"],
        'compiler_fix' : ["f90", "-fixed"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["<F90>", "-Bdynamic", "-G"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = '-moddir='
    module_include_switch = '-M'
    pic_flags = ['-xcode=pic32']

    def get_flags_f77(self):
        ret = ["-ftrap=%none"]
        if (self.get_version() or '') >= '7':
            ret.append("-f77")
        else:
            ret.append("-fixed")
        return ret
    def get_opt(self):
        return ['-fast', '-dalign']
    def get_arch(self):
        return ['-xtarget=generic']
    def get_libraries(self):
        opt = []
        opt.extend(['fsu', 'sunmath', 'mvec'])
        return opt

    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='sun').get_version())
