from numpy.distutils.fcompiler import FCompiler

compilers = ['HPUXFCompiler']

class HPUXFCompiler(FCompiler):

    compiler_type = 'hpux'
    description = 'HP Fortran 90 Compiler'
    version_pattern =  r'HP F90 (?P<version>[^\s*,]*)'

    executables = {
        'version_cmd'  : ["f90", "+version"],
        'compiler_f77' : ["f90"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["ld", "-b"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    module_dir_switch = None #XXX: fix me
    module_include_switch = None #XXX: fix me
    pic_flags = ['+Z']
    def get_flags(self):
        return self.pic_flags + ['+ppu', '+DD64']
    def get_flags_opt(self):
        return ['-O3']
    def get_libraries(self):
        return ['m']
    def get_library_dirs(self):
        opt = ['/usr/lib/hpux64']
        return opt
    def get_version(self, force=0, ok_status=[256, 0, 1]):
        # XXX status==256 may indicate 'unrecognized option' or
        #     'no input file'. So, version_cmd needs more work.
        return FCompiler.get_version(self, force, ok_status)

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(10)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='hpux').get_version())
