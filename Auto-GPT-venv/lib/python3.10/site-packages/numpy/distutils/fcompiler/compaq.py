
#http://www.compaq.com/fortran/docs/
import os
import sys

from numpy.distutils.fcompiler import FCompiler
from distutils.errors import DistutilsPlatformError

compilers = ['CompaqFCompiler']
if os.name != 'posix' or sys.platform[:6] == 'cygwin' :
    # Otherwise we'd get a false positive on posix systems with
    # case-insensitive filesystems (like darwin), because we'll pick
    # up /bin/df
    compilers.append('CompaqVisualFCompiler')

class CompaqFCompiler(FCompiler):

    compiler_type = 'compaq'
    description = 'Compaq Fortran Compiler'
    version_pattern = r'Compaq Fortran (?P<version>[^\s]*).*'

    if sys.platform[:5]=='linux':
        fc_exe = 'fort'
    else:
        fc_exe = 'f90'

    executables = {
        'version_cmd'  : ['<F90>', "-version"],
        'compiler_f77' : [fc_exe, "-f77rtl", "-fixed"],
        'compiler_fix' : [fc_exe, "-fixed"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : ['<F90>'],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    module_dir_switch = '-module ' # not tested
    module_include_switch = '-I'

    def get_flags(self):
        return ['-assume no2underscore', '-nomixed_str_len_arg']
    def get_flags_debug(self):
        return ['-g', '-check bounds']
    def get_flags_opt(self):
        return ['-O4', '-align dcommons', '-assume bigarrays',
                '-assume nozsize', '-math_library fast']
    def get_flags_arch(self):
        return ['-arch host', '-tune host']
    def get_flags_linker_so(self):
        if sys.platform[:5]=='linux':
            return ['-shared']
        return ['-shared', '-Wl,-expect_unresolved,*']

class CompaqVisualFCompiler(FCompiler):

    compiler_type = 'compaqv'
    description = 'DIGITAL or Compaq Visual Fortran Compiler'
    version_pattern = (r'(DIGITAL|Compaq) Visual Fortran Optimizing Compiler'
                       r' Version (?P<version>[^\s]*).*')

    compile_switch = '/compile_only'
    object_switch = '/object:'
    library_switch = '/OUT:'      #No space after /OUT:!

    static_lib_extension = ".lib"
    static_lib_format = "%s%s"
    module_dir_switch = '/module:'
    module_include_switch = '/I'

    ar_exe = 'lib.exe'
    fc_exe = 'DF'

    if sys.platform=='win32':
        from numpy.distutils.msvccompiler import MSVCCompiler

        try:
            m = MSVCCompiler()
            m.initialize()
            ar_exe = m.lib
        except DistutilsPlatformError:
            pass
        except AttributeError as e:
            if '_MSVCCompiler__root' in str(e):
                print('Ignoring "%s" (I think it is msvccompiler.py bug)' % (e))
            else:
                raise
        except OSError as e:
            if not "vcvarsall.bat" in str(e):
                print("Unexpected OSError in", __file__)
                raise
        except ValueError as e:
            if not "'path'" in str(e):
                print("Unexpected ValueError in", __file__)
                raise

    executables = {
        'version_cmd'  : ['<F90>', "/what"],
        'compiler_f77' : [fc_exe, "/f77rtl", "/fixed"],
        'compiler_fix' : [fc_exe, "/fixed"],
        'compiler_f90' : [fc_exe],
        'linker_so'    : ['<F90>'],
        'archiver'     : [ar_exe, "/OUT:"],
        'ranlib'       : None
        }

    def get_flags(self):
        return ['/nologo', '/MD', '/WX', '/iface=(cref,nomixed_str_len_arg)',
                '/names:lowercase', '/assume:underscore']
    def get_flags_opt(self):
        return ['/Ox', '/fast', '/optimize:5', '/unroll:0', '/math_library:fast']
    def get_flags_arch(self):
        return ['/threads']
    def get_flags_debug(self):
        return ['/debug']

if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='compaq').get_version())
