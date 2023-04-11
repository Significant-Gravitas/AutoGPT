"""numpy.distutils.fcompiler

Contains FCompiler, an abstract base class that defines the interface
for the numpy.distutils Fortran compiler abstraction model.

Terminology:

To be consistent, where the term 'executable' is used, it means the single
file, like 'gcc', that is executed, and should be a string. In contrast,
'command' means the entire command line, like ['gcc', '-c', 'file.c'], and
should be a list.

But note that FCompiler.executables is actually a dictionary of commands.

"""
__all__ = ['FCompiler', 'new_fcompiler', 'show_fcompilers',
           'dummy_fortran_file']

import os
import sys
import re

from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
     DistutilsExecError, CompileError, LinkError, DistutilsPlatformError
from distutils.util import split_quoted, strtobool

from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
    make_temp_file, get_shared_lib_extension
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils

from .environment import EnvironmentConfig

__metaclass__ = type

class CompilerNotFound(Exception):
    pass

def flaglist(s):
    if is_string(s):
        return split_quoted(s)
    else:
        return s

def str2bool(s):
    if is_string(s):
        return strtobool(s)
    return bool(s)

def is_sequence_of_strings(seq):
    return is_sequence(seq) and all_strings(seq)

class FCompiler(CCompiler):
    """Abstract base class to define the interface that must be implemented
    by real Fortran compiler classes.

    Methods that subclasses may redefine:

        update_executables(), find_executables(), get_version()
        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
        get_flags_arch_f90(), get_flags_debug_f90(),
        get_flags_fix(), get_flags_linker_so()

    DON'T call these methods (except get_version) after
    constructing a compiler instance or inside any other method.
    All methods, except update_executables() and find_executables(),
    may call the get_version() method.

    After constructing a compiler instance, always call customize(dist=None)
    method that finalizes compiler construction and makes the following
    attributes available:
      compiler_f77
      compiler_f90
      compiler_fix
      linker_so
      archiver
      ranlib
      libraries
      library_dirs
    """

    # These are the environment variables and distutils keys used.
    # Each configuration description is
    # (<hook name>, <environment variable>, <key in distutils.cfg>, <convert>, <append>)
    # The hook names are handled by the self._environment_hook method.
    #  - names starting with 'self.' call methods in this class
    #  - names starting with 'exe.' return the key in the executables dict
    #  - names like 'flags.YYY' return self.get_flag_YYY()
    # convert is either None or a function to convert a string to the
    # appropriate type used.

    distutils_vars = EnvironmentConfig(
        distutils_section='config_fc',
        noopt = (None, None, 'noopt', str2bool, False),
        noarch = (None, None, 'noarch', str2bool, False),
        debug = (None, None, 'debug', str2bool, False),
        verbose = (None, None, 'verbose', str2bool, False),
    )

    command_vars = EnvironmentConfig(
        distutils_section='config_fc',
        compiler_f77 = ('exe.compiler_f77', 'F77', 'f77exec', None, False),
        compiler_f90 = ('exe.compiler_f90', 'F90', 'f90exec', None, False),
        compiler_fix = ('exe.compiler_fix', 'F90', 'f90exec', None, False),
        version_cmd = ('exe.version_cmd', None, None, None, False),
        linker_so = ('exe.linker_so', 'LDSHARED', 'ldshared', None, False),
        linker_exe = ('exe.linker_exe', 'LD', 'ld', None, False),
        archiver = (None, 'AR', 'ar', None, False),
        ranlib = (None, 'RANLIB', 'ranlib', None, False),
    )

    flag_vars = EnvironmentConfig(
        distutils_section='config_fc',
        f77 = ('flags.f77', 'F77FLAGS', 'f77flags', flaglist, True),
        f90 = ('flags.f90', 'F90FLAGS', 'f90flags', flaglist, True),
        free = ('flags.free', 'FREEFLAGS', 'freeflags', flaglist, True),
        fix = ('flags.fix', None, None, flaglist, False),
        opt = ('flags.opt', 'FOPT', 'opt', flaglist, True),
        opt_f77 = ('flags.opt_f77', None, None, flaglist, False),
        opt_f90 = ('flags.opt_f90', None, None, flaglist, False),
        arch = ('flags.arch', 'FARCH', 'arch', flaglist, False),
        arch_f77 = ('flags.arch_f77', None, None, flaglist, False),
        arch_f90 = ('flags.arch_f90', None, None, flaglist, False),
        debug = ('flags.debug', 'FDEBUG', 'fdebug', flaglist, True),
        debug_f77 = ('flags.debug_f77', None, None, flaglist, False),
        debug_f90 = ('flags.debug_f90', None, None, flaglist, False),
        flags = ('self.get_flags', 'FFLAGS', 'fflags', flaglist, True),
        linker_so = ('flags.linker_so', 'LDFLAGS', 'ldflags', flaglist, True),
        linker_exe = ('flags.linker_exe', 'LDFLAGS', 'ldflags', flaglist, True),
        ar = ('flags.ar', 'ARFLAGS', 'arflags', flaglist, True),
    )

    language_map = {'.f': 'f77',
                    '.for': 'f77',
                    '.F': 'f77',    # XXX: needs preprocessor
                    '.ftn': 'f77',
                    '.f77': 'f77',
                    '.f90': 'f90',
                    '.F90': 'f90',  # XXX: needs preprocessor
                    '.f95': 'f90',
                    }
    language_order = ['f90', 'f77']


    # These will be set by the subclass

    compiler_type = None
    compiler_aliases = ()
    version_pattern = None

    possible_executables = []
    executables = {
        'version_cmd': ["f77", "-v"],
        'compiler_f77': ["f77"],
        'compiler_f90': ["f90"],
        'compiler_fix': ["f90", "-fixed"],
        'linker_so': ["f90", "-shared"],
        'linker_exe': ["f90"],
        'archiver': ["ar", "-cr"],
        'ranlib': None,
        }

    # If compiler does not support compiling Fortran 90 then it can
    # suggest using another compiler. For example, gnu would suggest
    # gnu95 compiler type when there are F90 sources.
    suggested_f90_compiler = None

    compile_switch = "-c"
    object_switch = "-o "   # Ending space matters! It will be stripped
                            # but if it is missing then object_switch
                            # will be prefixed to object file name by
                            # string concatenation.
    library_switch = "-o "  # Ditto!

    # Switch to specify where module files are created and searched
    # for USE statement.  Normally it is a string and also here ending
    # space matters. See above.
    module_dir_switch = None

    # Switch to specify where module files are searched for USE statement.
    module_include_switch = '-I'

    pic_flags = []           # Flags to create position-independent code

    src_extensions = ['.for', '.ftn', '.f77', '.f', '.f90', '.f95', '.F', '.F90', '.FOR']
    obj_extension = ".o"

    shared_lib_extension = get_shared_lib_extension()
    static_lib_extension = ".a"  # or .lib
    static_lib_format = "lib%s%s" # or %s%s
    shared_lib_format = "%s%s"
    exe_extension = ""

    _exe_cache = {}

    _executable_keys = ['version_cmd', 'compiler_f77', 'compiler_f90',
                        'compiler_fix', 'linker_so', 'linker_exe', 'archiver',
                        'ranlib']

    # This will be set by new_fcompiler when called in
    # command/{build_ext.py, build_clib.py, config.py} files.
    c_compiler = None

    # extra_{f77,f90}_compile_args are set by build_ext.build_extension method
    extra_f77_compile_args = []
    extra_f90_compile_args = []

    def __init__(self, *args, **kw):
        CCompiler.__init__(self, *args, **kw)
        self.distutils_vars = self.distutils_vars.clone(self._environment_hook)
        self.command_vars = self.command_vars.clone(self._environment_hook)
        self.flag_vars = self.flag_vars.clone(self._environment_hook)
        self.executables = self.executables.copy()
        for e in self._executable_keys:
            if e not in self.executables:
                self.executables[e] = None

        # Some methods depend on .customize() being called first, so
        # this keeps track of whether that's happened yet.
        self._is_customised = False

    def __copy__(self):
        obj = self.__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        obj.distutils_vars = obj.distutils_vars.clone(obj._environment_hook)
        obj.command_vars = obj.command_vars.clone(obj._environment_hook)
        obj.flag_vars = obj.flag_vars.clone(obj._environment_hook)
        obj.executables = obj.executables.copy()
        return obj

    def copy(self):
        return self.__copy__()

    # Use properties for the attributes used by CCompiler. Setting them
    # as attributes from the self.executables dictionary is error-prone,
    # so we get them from there each time.
    def _command_property(key):
        def fget(self):
            assert self._is_customised
            return self.executables[key]
        return property(fget=fget)
    version_cmd = _command_property('version_cmd')
    compiler_f77 = _command_property('compiler_f77')
    compiler_f90 = _command_property('compiler_f90')
    compiler_fix = _command_property('compiler_fix')
    linker_so = _command_property('linker_so')
    linker_exe = _command_property('linker_exe')
    archiver = _command_property('archiver')
    ranlib = _command_property('ranlib')

    # Make our terminology consistent.
    def set_executable(self, key, value):
        self.set_command(key, value)

    def set_commands(self, **kw):
        for k, v in kw.items():
            self.set_command(k, v)

    def set_command(self, key, value):
        if not key in self._executable_keys:
            raise ValueError(
                "unknown executable '%s' for class %s" %
                (key, self.__class__.__name__))
        if is_string(value):
            value = split_quoted(value)
        assert value is None or is_sequence_of_strings(value[1:]), (key, value)
        self.executables[key] = value

    ######################################################################
    ## Methods that subclasses may redefine. But don't call these methods!
    ## They are private to FCompiler class and may return unexpected
    ## results if used elsewhere. So, you have been warned..

    def find_executables(self):
        """Go through the self.executables dictionary, and attempt to
        find and assign appropriate executables.

        Executable names are looked for in the environment (environment
        variables, the distutils.cfg, and command line), the 0th-element of
        the command list, and the self.possible_executables list.

        Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77
        or the Fortran 90 compiler executable is used, unless overridden
        by an environment setting.

        Subclasses should call this if overridden.
        """
        assert self._is_customised
        exe_cache = self._exe_cache
        def cached_find_executable(exe):
            if exe in exe_cache:
                return exe_cache[exe]
            fc_exe = find_executable(exe)
            exe_cache[exe] = exe_cache[fc_exe] = fc_exe
            return fc_exe
        def verify_command_form(name, value):
            if value is not None and not is_sequence_of_strings(value):
                raise ValueError(
                    "%s value %r is invalid in class %s" %
                    (name, value, self.__class__.__name__))
        def set_exe(exe_key, f77=None, f90=None):
            cmd = self.executables.get(exe_key, None)
            if not cmd:
                return None
            # Note that we get cmd[0] here if the environment doesn't
            # have anything set
            exe_from_environ = getattr(self.command_vars, exe_key)
            if not exe_from_environ:
                possibles = [f90, f77] + self.possible_executables
            else:
                possibles = [exe_from_environ] + self.possible_executables

            seen = set()
            unique_possibles = []
            for e in possibles:
                if e == '<F77>':
                    e = f77
                elif e == '<F90>':
                    e = f90
                if not e or e in seen:
                    continue
                seen.add(e)
                unique_possibles.append(e)

            for exe in unique_possibles:
                fc_exe = cached_find_executable(exe)
                if fc_exe:
                    cmd[0] = fc_exe
                    return fc_exe
            self.set_command(exe_key, None)
            return None

        ctype = self.compiler_type
        f90 = set_exe('compiler_f90')
        if not f90:
            f77 = set_exe('compiler_f77')
            if f77:
                log.warn('%s: no Fortran 90 compiler found' % ctype)
            else:
                raise CompilerNotFound('%s: f90 nor f77' % ctype)
        else:
            f77 = set_exe('compiler_f77', f90=f90)
            if not f77:
                log.warn('%s: no Fortran 77 compiler found' % ctype)
            set_exe('compiler_fix', f90=f90)

        set_exe('linker_so', f77=f77, f90=f90)
        set_exe('linker_exe', f77=f77, f90=f90)
        set_exe('version_cmd', f77=f77, f90=f90)
        set_exe('archiver')
        set_exe('ranlib')

    def update_executables(self):
        """Called at the beginning of customisation. Subclasses should
        override this if they need to set up the executables dictionary.

        Note that self.find_executables() is run afterwards, so the
        self.executables dictionary values can contain <F77> or <F90> as
        the command, which will be replaced by the found F77 or F90
        compiler.
        """
        pass

    def get_flags(self):
        """List of flags common to all compiler types."""
        return [] + self.pic_flags

    def _get_command_flags(self, key):
        cmd = self.executables.get(key, None)
        if cmd is None:
            return []
        return cmd[1:]

    def get_flags_f77(self):
        """List of Fortran 77 specific flags."""
        return self._get_command_flags('compiler_f77')
    def get_flags_f90(self):
        """List of Fortran 90 specific flags."""
        return self._get_command_flags('compiler_f90')
    def get_flags_free(self):
        """List of Fortran 90 free format specific flags."""
        return []
    def get_flags_fix(self):
        """List of Fortran 90 fixed format specific flags."""
        return self._get_command_flags('compiler_fix')
    def get_flags_linker_so(self):
        """List of linker flags to build a shared library."""
        return self._get_command_flags('linker_so')
    def get_flags_linker_exe(self):
        """List of linker flags to build an executable."""
        return self._get_command_flags('linker_exe')
    def get_flags_ar(self):
        """List of archiver flags. """
        return self._get_command_flags('archiver')
    def get_flags_opt(self):
        """List of architecture independent compiler flags."""
        return []
    def get_flags_arch(self):
        """List of architecture dependent compiler flags."""
        return []
    def get_flags_debug(self):
        """List of compiler flags to compile with debugging information."""
        return []

    get_flags_opt_f77 = get_flags_opt_f90 = get_flags_opt
    get_flags_arch_f77 = get_flags_arch_f90 = get_flags_arch
    get_flags_debug_f77 = get_flags_debug_f90 = get_flags_debug

    def get_libraries(self):
        """List of compiler libraries."""
        return self.libraries[:]
    def get_library_dirs(self):
        """List of compiler library directories."""
        return self.library_dirs[:]

    def get_version(self, force=False, ok_status=[0]):
        assert self._is_customised
        version = CCompiler.get_version(self, force=force, ok_status=ok_status)
        if version is None:
            raise CompilerNotFound()
        return version


    ############################################################

    ## Public methods:

    def customize(self, dist = None):
        """Customize Fortran compiler.

        This method gets Fortran compiler specific information from
        (i) class definition, (ii) environment, (iii) distutils config
        files, and (iv) command line (later overrides earlier).

        This method should be always called after constructing a
        compiler instance. But not in __init__ because Distribution
        instance is needed for (iii) and (iv).
        """
        log.info('customize %s' % (self.__class__.__name__))

        self._is_customised = True

        self.distutils_vars.use_distribution(dist)
        self.command_vars.use_distribution(dist)
        self.flag_vars.use_distribution(dist)

        self.update_executables()

        # find_executables takes care of setting the compiler commands,
        # version_cmd, linker_so, linker_exe, ar, and ranlib
        self.find_executables()

        noopt = self.distutils_vars.get('noopt', False)
        noarch = self.distutils_vars.get('noarch', noopt)
        debug = self.distutils_vars.get('debug', False)

        f77 = self.command_vars.compiler_f77
        f90 = self.command_vars.compiler_f90

        f77flags = []
        f90flags = []
        freeflags = []
        fixflags = []

        if f77:
            f77 = _shell_utils.NativeParser.split(f77)
            f77flags = self.flag_vars.f77
        if f90:
            f90 = _shell_utils.NativeParser.split(f90)
            f90flags = self.flag_vars.f90
            freeflags = self.flag_vars.free
        # XXX Assuming that free format is default for f90 compiler.
        fix = self.command_vars.compiler_fix
        # NOTE: this and similar examples are probably just
        # excluding --coverage flag when F90 = gfortran --coverage
        # instead of putting that flag somewhere more appropriate
        # this and similar examples where a Fortran compiler
        # environment variable has been customized by CI or a user
        # should perhaps eventually be more thoroughly tested and more
        # robustly handled
        if fix:
            fix = _shell_utils.NativeParser.split(fix)
            fixflags = self.flag_vars.fix + f90flags

        oflags, aflags, dflags = [], [], []
        # examine get_flags_<tag>_<compiler> for extra flags
        # only add them if the method is different from get_flags_<tag>
        def get_flags(tag, flags):
            # note that self.flag_vars.<tag> calls self.get_flags_<tag>()
            flags.extend(getattr(self.flag_vars, tag))
            this_get = getattr(self, 'get_flags_' + tag)
            for name, c, flagvar in [('f77', f77, f77flags),
                                     ('f90', f90, f90flags),
                                     ('f90', fix, fixflags)]:
                t = '%s_%s' % (tag, name)
                if c and this_get is not getattr(self, 'get_flags_' + t):
                    flagvar.extend(getattr(self.flag_vars, t))
        if not noopt:
            get_flags('opt', oflags)
            if not noarch:
                get_flags('arch', aflags)
        if debug:
            get_flags('debug', dflags)

        fflags = self.flag_vars.flags + dflags + oflags + aflags

        if f77:
            self.set_commands(compiler_f77=f77+f77flags+fflags)
        if f90:
            self.set_commands(compiler_f90=f90+freeflags+f90flags+fflags)
        if fix:
            self.set_commands(compiler_fix=fix+fixflags+fflags)


        #XXX: Do we need LDSHARED->SOSHARED, LDFLAGS->SOFLAGS
        linker_so = self.linker_so
        if linker_so:
            linker_so_flags = self.flag_vars.linker_so
            if sys.platform.startswith('aix'):
                python_lib = get_python_lib(standard_lib=1)
                ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
                python_exp = os.path.join(python_lib, 'config', 'python.exp')
                linker_so = [ld_so_aix] + linker_so + ['-bI:'+python_exp]
            if sys.platform.startswith('os400'):
                from distutils.sysconfig import get_config_var
                python_config = get_config_var('LIBPL')
                ld_so_aix = os.path.join(python_config, 'ld_so_aix')
                python_exp = os.path.join(python_config, 'python.exp')
                linker_so = [ld_so_aix] + linker_so + ['-bI:'+python_exp]
            self.set_commands(linker_so=linker_so+linker_so_flags)

        linker_exe = self.linker_exe
        if linker_exe:
            linker_exe_flags = self.flag_vars.linker_exe
            self.set_commands(linker_exe=linker_exe+linker_exe_flags)

        ar = self.command_vars.archiver
        if ar:
            arflags = self.flag_vars.ar
            self.set_commands(archiver=[ar]+arflags)

        self.set_library_dirs(self.get_library_dirs())
        self.set_libraries(self.get_libraries())

    def dump_properties(self):
        """Print out the attributes of a compiler instance."""
        props = []
        for key in list(self.executables.keys()) + \
                ['version', 'libraries', 'library_dirs',
                 'object_switch', 'compile_switch']:
            if hasattr(self, key):
                v = getattr(self, key)
                props.append((key, None, '= '+repr(v)))
        props.sort()

        pretty_printer = FancyGetopt(props)
        for l in pretty_printer.generate_help("%s instance properties:" \
                                              % (self.__class__.__name__)):
            if l[:4]=='  --':
                l = '  ' + l[4:]
            print(l)

    ###################

    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compile 'src' to product 'obj'."""
        src_flags = {}
        if is_f_file(src) and not has_f90_header(src):
            flavor = ':f77'
            compiler = self.compiler_f77
            src_flags = get_f77flags(src)
            extra_compile_args = self.extra_f77_compile_args or []
        elif is_free_format(src):
            flavor = ':f90'
            compiler = self.compiler_f90
            if compiler is None:
                raise DistutilsExecError('f90 not supported by %s needed for %s'\
                      % (self.__class__.__name__, src))
            extra_compile_args = self.extra_f90_compile_args or []
        else:
            flavor = ':fix'
            compiler = self.compiler_fix
            if compiler is None:
                raise DistutilsExecError('f90 (fixed) not supported by %s needed for %s'\
                      % (self.__class__.__name__, src))
            extra_compile_args = self.extra_f90_compile_args or []
        if self.object_switch[-1]==' ':
            o_args = [self.object_switch.strip(), obj]
        else:
            o_args = [self.object_switch.strip()+obj]

        assert self.compile_switch.strip()
        s_args = [self.compile_switch, src]

        if extra_compile_args:
            log.info('extra %s options: %r' \
                     % (flavor[1:], ' '.join(extra_compile_args)))

        extra_flags = src_flags.get(self.compiler_type, [])
        if extra_flags:
            log.info('using compile options from source: %r' \
                     % ' '.join(extra_flags))

        command = compiler + cc_args + extra_flags + s_args + o_args \
                  + extra_postargs + extra_compile_args

        display = '%s: %s' % (os.path.basename(compiler[0]) + flavor,
                              src)
        try:
            self.spawn(command, display=display)
        except DistutilsExecError as e:
            msg = str(e)
            raise CompileError(msg) from None

    def module_options(self, module_dirs, module_build_dir):
        options = []
        if self.module_dir_switch is not None:
            if self.module_dir_switch[-1]==' ':
                options.extend([self.module_dir_switch.strip(), module_build_dir])
            else:
                options.append(self.module_dir_switch.strip()+module_build_dir)
        else:
            print('XXX: module_build_dir=%r option ignored' % (module_build_dir))
            print('XXX: Fix module_dir_switch for ', self.__class__.__name__)
        if self.module_include_switch is not None:
            for d in [module_build_dir]+module_dirs:
                options.append('%s%s' % (self.module_include_switch, d))
        else:
            print('XXX: module_dirs=%r option ignored' % (module_dirs))
            print('XXX: Fix module_include_switch for ', self.__class__.__name__)
        return options

    def library_option(self, lib):
        return "-l" + lib
    def library_dir_option(self, dir):
        return "-L" + dir

    def link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        objects, output_dir = self._fix_object_args(objects, output_dir)
        libraries, library_dirs, runtime_library_dirs = \
            self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)

        lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
                                   libraries)
        if is_string(output_dir):
            output_filename = os.path.join(output_dir, output_filename)
        elif output_dir is not None:
            raise TypeError("'output_dir' must be a string or None")

        if self._need_link(objects, output_filename):
            if self.library_switch[-1]==' ':
                o_args = [self.library_switch.strip(), output_filename]
            else:
                o_args = [self.library_switch.strip()+output_filename]

            if is_string(self.objects):
                ld_args = objects + [self.objects]
            else:
                ld_args = objects + self.objects
            ld_args = ld_args + lib_opts + o_args
            if debug:
                ld_args[:0] = ['-g']
            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)
            self.mkpath(os.path.dirname(output_filename))
            if target_desc == CCompiler.EXECUTABLE:
                linker = self.linker_exe[:]
            else:
                linker = self.linker_so[:]
            command = linker + ld_args
            try:
                self.spawn(command)
            except DistutilsExecError as e:
                msg = str(e)
                raise LinkError(msg) from None
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def _environment_hook(self, name, hook_name):
        if hook_name is None:
            return None
        if is_string(hook_name):
            if hook_name.startswith('self.'):
                hook_name = hook_name[5:]
                hook = getattr(self, hook_name)
                return hook()
            elif hook_name.startswith('exe.'):
                hook_name = hook_name[4:]
                var = self.executables[hook_name]
                if var:
                    return var[0]
                else:
                    return None
            elif hook_name.startswith('flags.'):
                hook_name = hook_name[6:]
                hook = getattr(self, 'get_flags_' + hook_name)
                return hook()
        else:
            return hook_name()

    def can_ccompiler_link(self, ccompiler):
        """
        Check if the given C compiler can link objects produced by
        this compiler.
        """
        return True

    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.

        Parameters
        ----------
        objects : list
            List of object files to include.
        output_dir : str
            Output directory to place generated object files.
        extra_dll_dir : str
            Output directory to place extra DLL files that need to be
            included on Windows.

        Returns
        -------
        converted_objects : list of str
             List of converted object files.
             Note that the number of output files is not necessarily
             the same as inputs.

        """
        raise NotImplementedError()

    ## class FCompiler

_default_compilers = (
    # sys.platform mappings
    ('win32', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95',
               'intelvem', 'intelem', 'flang')),
    ('cygwin.*', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95')),
    ('linux.*', ('arm', 'gnu95', 'intel', 'lahey', 'pg', 'nv', 'absoft', 'nag',
                 'vast', 'compaq', 'intele', 'intelem', 'gnu', 'g95', 
                 'pathf95', 'nagfor', 'fujitsu')),
    ('darwin.*', ('gnu95', 'nag', 'nagfor', 'absoft', 'ibm', 'intel', 'gnu',
                 'g95', 'pg')),
    ('sunos.*', ('sun', 'gnu', 'gnu95', 'g95')),
    ('irix.*', ('mips', 'gnu', 'gnu95',)),
    ('aix.*', ('ibm', 'gnu', 'gnu95',)),
    # os.name mappings
    ('posix', ('gnu', 'gnu95',)),
    ('nt', ('gnu', 'gnu95',)),
    ('mac', ('gnu95', 'gnu', 'pg')),
    )

fcompiler_class = None
fcompiler_aliases = None

def load_all_fcompiler_classes():
    """Cache all the FCompiler classes found in modules in the
    numpy.distutils.fcompiler package.
    """
    from glob import glob
    global fcompiler_class, fcompiler_aliases
    if fcompiler_class is not None:
        return
    pys = os.path.join(os.path.dirname(__file__), '*.py')
    fcompiler_class = {}
    fcompiler_aliases = {}
    for fname in glob(pys):
        module_name, ext = os.path.splitext(os.path.basename(fname))
        module_name = 'numpy.distutils.fcompiler.' + module_name
        __import__ (module_name)
        module = sys.modules[module_name]
        if hasattr(module, 'compilers'):
            for cname in module.compilers:
                klass = getattr(module, cname)
                desc = (klass.compiler_type, klass, klass.description)
                fcompiler_class[klass.compiler_type] = desc
                for alias in klass.compiler_aliases:
                    if alias in fcompiler_aliases:
                        raise ValueError("alias %r defined for both %s and %s"
                                         % (alias, klass.__name__,
                                            fcompiler_aliases[alias][1].__name__))
                    fcompiler_aliases[alias] = desc

def _find_existing_fcompiler(compiler_types,
                             osname=None, platform=None,
                             requiref90=False,
                             c_compiler=None):
    from numpy.distutils.core import get_distribution
    dist = get_distribution(always=True)
    for compiler_type in compiler_types:
        v = None
        try:
            c = new_fcompiler(plat=platform, compiler=compiler_type,
                              c_compiler=c_compiler)
            c.customize(dist)
            v = c.get_version()
            if requiref90 and c.compiler_f90 is None:
                v = None
                new_compiler = c.suggested_f90_compiler
                if new_compiler:
                    log.warn('Trying %r compiler as suggested by %r '
                             'compiler for f90 support.' % (compiler_type,
                                                            new_compiler))
                    c = new_fcompiler(plat=platform, compiler=new_compiler,
                                      c_compiler=c_compiler)
                    c.customize(dist)
                    v = c.get_version()
                    if v is not None:
                        compiler_type = new_compiler
            if requiref90 and c.compiler_f90 is None:
                raise ValueError('%s does not support compiling f90 codes, '
                                 'skipping.' % (c.__class__.__name__))
        except DistutilsModuleError:
            log.debug("_find_existing_fcompiler: compiler_type='%s' raised DistutilsModuleError", compiler_type)
        except CompilerNotFound:
            log.debug("_find_existing_fcompiler: compiler_type='%s' not found", compiler_type)
        if v is not None:
            return compiler_type
    return None

def available_fcompilers_for_platform(osname=None, platform=None):
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    matching_compiler_types = []
    for pattern, compiler_type in _default_compilers:
        if re.match(pattern, platform) or re.match(pattern, osname):
            for ct in compiler_type:
                if ct not in matching_compiler_types:
                    matching_compiler_types.append(ct)
    if not matching_compiler_types:
        matching_compiler_types.append('gnu')
    return matching_compiler_types

def get_default_fcompiler(osname=None, platform=None, requiref90=False,
                          c_compiler=None):
    """Determine the default Fortran compiler to use for the given
    platform."""
    matching_compiler_types = available_fcompilers_for_platform(osname,
                                                                platform)
    log.info("get_default_fcompiler: matching types: '%s'",
             matching_compiler_types)
    compiler_type =  _find_existing_fcompiler(matching_compiler_types,
                                              osname=osname,
                                              platform=platform,
                                              requiref90=requiref90,
                                              c_compiler=c_compiler)
    return compiler_type

# Flag to avoid rechecking for Fortran compiler every time
failed_fcompilers = set()

def new_fcompiler(plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0,
                  requiref90=False,
                  c_compiler = None):
    """Generate an instance of some FCompiler subclass for the supplied
    platform/compiler combination.
    """
    global failed_fcompilers
    fcompiler_key = (plat, compiler)
    if fcompiler_key in failed_fcompilers:
        return None

    load_all_fcompiler_classes()
    if plat is None:
        plat = os.name
    if compiler is None:
        compiler = get_default_fcompiler(plat, requiref90=requiref90,
                                         c_compiler=c_compiler)
    if compiler in fcompiler_class:
        module_name, klass, long_description = fcompiler_class[compiler]
    elif compiler in fcompiler_aliases:
        module_name, klass, long_description = fcompiler_aliases[compiler]
    else:
        msg = "don't know how to compile Fortran code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler." % compiler
            msg = msg + " Supported compilers are: %s)" \
                  % (','.join(fcompiler_class.keys()))
        log.warn(msg)
        failed_fcompilers.add(fcompiler_key)
        return None

    compiler = klass(verbose=verbose, dry_run=dry_run, force=force)
    compiler.c_compiler = c_compiler
    return compiler

def show_fcompilers(dist=None):
    """Print list of available compilers (used by the "--help-fcompiler"
    option to "config_fc").
    """
    if dist is None:
        from distutils.dist import Distribution
        from numpy.distutils.command.config_compiler import config_fc
        dist = Distribution()
        dist.script_name = os.path.basename(sys.argv[0])
        dist.script_args = ['config_fc'] + sys.argv[1:]
        try:
            dist.script_args.remove('--help-fcompiler')
        except ValueError:
            pass
        dist.cmdclass['config_fc'] = config_fc
        dist.parse_config_files()
        dist.parse_command_line()
    compilers = []
    compilers_na = []
    compilers_ni = []
    if not fcompiler_class:
        load_all_fcompiler_classes()
    platform_compilers = available_fcompilers_for_platform()
    for compiler in platform_compilers:
        v = None
        log.set_verbosity(-2)
        try:
            c = new_fcompiler(compiler=compiler, verbose=dist.verbose)
            c.customize(dist)
            v = c.get_version()
        except (DistutilsModuleError, CompilerNotFound) as e:
            log.debug("show_fcompilers: %s not found" % (compiler,))
            log.debug(repr(e))

        if v is None:
            compilers_na.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2]))
        else:
            c.dump_properties()
            compilers.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2] + ' (%s)' % v))

    compilers_ni = list(set(fcompiler_class.keys()) - set(platform_compilers))
    compilers_ni = [("fcompiler="+fc, None, fcompiler_class[fc][2])
                    for fc in compilers_ni]

    compilers.sort()
    compilers_na.sort()
    compilers_ni.sort()
    pretty_printer = FancyGetopt(compilers)
    pretty_printer.print_help("Fortran compilers found:")
    pretty_printer = FancyGetopt(compilers_na)
    pretty_printer.print_help("Compilers available for this "
                              "platform, but not found:")
    if compilers_ni:
        pretty_printer = FancyGetopt(compilers_ni)
        pretty_printer.print_help("Compilers not available on this platform:")
    print("For compiler details, run 'config_fc --verbose' setup command.")


def dummy_fortran_file():
    fo, name = make_temp_file(suffix='.f')
    fo.write("      subroutine dummy()\n      end\n")
    fo.close()
    return name[:-2]


is_f_file = re.compile(r'.*\.(for|ftn|f77|f)\Z', re.I).match
_has_f_header = re.compile(r'-\*-\s*fortran\s*-\*-', re.I).search
_has_f90_header = re.compile(r'-\*-\s*f90\s*-\*-', re.I).search
_has_fix_header = re.compile(r'-\*-\s*fix\s*-\*-', re.I).search
_free_f90_start = re.compile(r'[^c*!]\s*[^\s\d\t]', re.I).match

def is_free_format(file):
    """Check if file is in free format Fortran."""
    # f90 allows both fixed and free format, assuming fixed unless
    # signs of free format are detected.
    result = 0
    with open(file, encoding='latin1') as f:
        line = f.readline()
        n = 10000 # the number of non-comment lines to scan for hints
        if _has_f_header(line) or _has_fix_header(line):
            n = 0
        elif _has_f90_header(line):
            n = 0
            result = 1
        while n>0 and line:
            line = line.rstrip()
            if line and line[0]!='!':
                n -= 1
                if (line[0]!='\t' and _free_f90_start(line[:5])) or line[-1:]=='&':
                    result = 1
                    break
            line = f.readline()
    return result

def has_f90_header(src):
    with open(src, encoding='latin1') as f:
        line = f.readline()
    return _has_f90_header(line) or _has_fix_header(line)

_f77flags_re = re.compile(r'(c|)f77flags\s*\(\s*(?P<fcname>\w+)\s*\)\s*=\s*(?P<fflags>.*)', re.I)
def get_f77flags(src):
    """
    Search the first 20 lines of fortran 77 code for line pattern
      `CF77FLAGS(<fcompiler type>)=<f77 flags>`
    Return a dictionary {<fcompiler type>:<f77 flags>}.
    """
    flags = {}
    with open(src, encoding='latin1') as f:
        i = 0
        for line in f:
            i += 1
            if i>20: break
            m = _f77flags_re.match(line)
            if not m: continue
            fcname = m.group('fcname').strip()
            fflags = m.group('fflags').strip()
            flags[fcname] = split_quoted(fflags)
    return flags

# TODO: implement get_f90flags and use it in _compile similarly to get_f77flags

if __name__ == '__main__':
    show_fcompilers()
