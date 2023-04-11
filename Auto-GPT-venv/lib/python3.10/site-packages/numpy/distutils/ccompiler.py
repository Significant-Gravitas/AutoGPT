import os
import re
import sys
import shlex
import time
import subprocess
from copy import copy
from distutils import ccompiler
from distutils.ccompiler import (
    compiler_class, gen_lib_options, get_default_compiler, new_compiler,
    CCompiler
)
from distutils.errors import (
    DistutilsExecError, DistutilsModuleError, DistutilsPlatformError,
    CompileError, UnknownFileError
)
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

from numpy.distutils import log
from numpy.distutils.exec_command import (
    filepath_from_subprocess_output, forward_bytes_to_stdout
)
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
                                      get_num_build_jobs, \
                                      _commandline_dep_string, \
                                      sanitize_cxx_flags

# globals for parallel build management
import threading

_job_semaphore = None
_global_lock = threading.Lock()
_processing_files = set()


def _needs_build(obj, cc_args, extra_postargs, pp_opts):
    """
    Check if an objects needs to be rebuild based on its dependencies

    Parameters
    ----------
    obj : str
        object file

    Returns
    -------
    bool
    """
    # defined in unixcompiler.py
    dep_file = obj + '.d'
    if not os.path.exists(dep_file):
        return True

    # dep_file is a makefile containing 'object: dependencies'
    # formatted like posix shell (spaces escaped, \ line continuations)
    # the last line contains the compiler commandline arguments as some
    # projects may compile an extension multiple times with different
    # arguments
    with open(dep_file, "r") as f:
        lines = f.readlines()

    cmdline =_commandline_dep_string(cc_args, extra_postargs, pp_opts)
    last_cmdline = lines[-1]
    if last_cmdline != cmdline:
        return True

    contents = ''.join(lines[:-1])
    deps = [x for x in shlex.split(contents, posix=True)
            if x != "\n" and not x.endswith(":")]

    try:
        t_obj = os.stat(obj).st_mtime

        # check if any of the dependencies is newer than the object
        # the dependencies includes the source used to create the object
        for f in deps:
            if os.stat(f).st_mtime > t_obj:
                return True
    except OSError:
        # no object counts as newer (shouldn't happen if dep_file exists)
        return True

    return False


def replace_method(klass, method_name, func):
    # Py3k does not have unbound method anymore, MethodType does not work
    m = lambda self, *args, **kw: func(self, *args, **kw)
    setattr(klass, method_name, m)


######################################################################
## Method that subclasses may redefine. But don't call this method,
## it i private to CCompiler class and may return unexpected
## results if used elsewhere. So, you have been warned..

def CCompiler_find_executables(self):
    """
    Does nothing here, but is called by the get_version method and can be
    overridden by subclasses. In particular it is redefined in the `FCompiler`
    class where more documentation can be found.

    """
    pass


replace_method(CCompiler, 'find_executables', CCompiler_find_executables)


# Using customized CCompiler.spawn.
def CCompiler_spawn(self, cmd, display=None, env=None):
    """
    Execute a command in a sub-process.

    Parameters
    ----------
    cmd : str
        The command to execute.
    display : str or sequence of str, optional
        The text to add to the log file kept by `numpy.distutils`.
        If not given, `display` is equal to `cmd`.
    env : a dictionary for environment variables, optional

    Returns
    -------
    None

    Raises
    ------
    DistutilsExecError
        If the command failed, i.e. the exit status was not 0.

    """
    env = env if env is not None else dict(os.environ)
    if display is None:
        display = cmd
        if is_sequence(display):
            display = ' '.join(list(display))
    log.info(display)
    try:
        if self.verbose:
            subprocess.check_output(cmd, env=env)
        else:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as exc:
        o = exc.output
        s = exc.returncode
    except OSError as e:
        # OSError doesn't have the same hooks for the exception
        # output, but exec_command() historically would use an
        # empty string for EnvironmentError (base class for
        # OSError)
        # o = b''
        # still that would make the end-user lost in translation!
        o = f"\n\n{e}\n\n\n"
        try:
            o = o.encode(sys.stdout.encoding)
        except AttributeError:
            o = o.encode('utf8')
        # status previously used by exec_command() for parent
        # of OSError
        s = 127
    else:
        # use a convenience return here so that any kind of
        # caught exception will execute the default code after the
        # try / except block, which handles various exceptions
        return None

    if is_sequence(cmd):
        cmd = ' '.join(list(cmd))

    if self.verbose:
        forward_bytes_to_stdout(o)

    if re.search(b'Too many open files', o):
        msg = '\nTry rerunning setup command until build succeeds.'
    else:
        msg = ''
    raise DistutilsExecError('Command "%s" failed with exit status %d%s' %
                            (cmd, s, msg))

replace_method(CCompiler, 'spawn', CCompiler_spawn)

def CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
    """
    Return the name of the object files for the given source files.

    Parameters
    ----------
    source_filenames : list of str
        The list of paths to source files. Paths can be either relative or
        absolute, this is handled transparently.
    strip_dir : bool, optional
        Whether to strip the directory from the returned paths. If True,
        the file name prepended by `output_dir` is returned. Default is False.
    output_dir : str, optional
        If given, this path is prepended to the returned paths to the
        object files.

    Returns
    -------
    obj_names : list of str
        The list of paths to the object files corresponding to the source
        files in `source_filenames`.

    """
    if output_dir is None:
        output_dir = ''
    obj_names = []
    for src_name in source_filenames:
        base, ext = os.path.splitext(os.path.normpath(src_name))
        base = os.path.splitdrive(base)[1] # Chop off the drive
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        if base.startswith('..'):
            # Resolve starting relative path components, middle ones
            # (if any) have been handled by os.path.normpath above.
            i = base.rfind('..')+2
            d = base[:i]
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        if ext not in self.src_extensions:
            raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
        if strip_dir:
            base = os.path.basename(base)
        obj_name = os.path.join(output_dir, base + self.obj_extension)
        obj_names.append(obj_name)
    return obj_names

replace_method(CCompiler, 'object_filenames', CCompiler_object_filenames)

def CCompiler_compile(self, sources, output_dir=None, macros=None,
                      include_dirs=None, debug=0, extra_preargs=None,
                      extra_postargs=None, depends=None):
    """
    Compile one or more source files.

    Please refer to the Python distutils API reference for more details.

    Parameters
    ----------
    sources : list of str
        A list of filenames
    output_dir : str, optional
        Path to the output directory.
    macros : list of tuples
        A list of macro definitions.
    include_dirs : list of str, optional
        The directories to add to the default include file search path for
        this compilation only.
    debug : bool, optional
        Whether or not to output debug symbols in or alongside the object
        file(s).
    extra_preargs, extra_postargs : ?
        Extra pre- and post-arguments.
    depends : list of str, optional
        A list of file names that all targets depend on.

    Returns
    -------
    objects : list of str
        A list of object file names, one per source file `sources`.

    Raises
    ------
    CompileError
        If compilation fails.

    """
    global _job_semaphore

    jobs = get_num_build_jobs()

    # setup semaphore to not exceed number of compile jobs when parallelized at
    # extension level (python >= 3.5)
    with _global_lock:
        if _job_semaphore is None:
            _job_semaphore = threading.Semaphore(jobs)

    if not sources:
        return []
    from numpy.distutils.fcompiler import (FCompiler, is_f_file,
                                           has_f90_header)
    if isinstance(self, FCompiler):
        display = []
        for fc in ['f77', 'f90', 'fix']:
            fcomp = getattr(self, 'compiler_'+fc)
            if fcomp is None:
                continue
            display.append("Fortran %s compiler: %s" % (fc, ' '.join(fcomp)))
        display = '\n'.join(display)
    else:
        ccomp = self.compiler_so
        display = "C compiler: %s\n" % (' '.join(ccomp),)
    log.info(display)
    macros, objects, extra_postargs, pp_opts, build = \
            self._setup_compile(output_dir, macros, include_dirs, sources,
                                depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    display = "compile options: '%s'" % (' '.join(cc_args))
    if extra_postargs:
        display += "\nextra options: '%s'" % (' '.join(extra_postargs))
    log.info(display)

    def single_compile(args):
        obj, (src, ext) = args
        if not _needs_build(obj, cc_args, extra_postargs, pp_opts):
            return

        # check if we are currently already processing the same object
        # happens when using the same source in multiple extensions
        while True:
            # need explicit lock as there is no atomic check and add with GIL
            with _global_lock:
                # file not being worked on, start working
                if obj not in _processing_files:
                    _processing_files.add(obj)
                    break
            # wait for the processing to end
            time.sleep(0.1)

        try:
            # retrieve slot from our #job semaphore and build
            with _job_semaphore:
                self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        finally:
            # register being done processing
            with _global_lock:
                _processing_files.remove(obj)


    if isinstance(self, FCompiler):
        objects_to_build = list(build.keys())
        f77_objects, other_objects = [], []
        for obj in objects:
            if obj in objects_to_build:
                src, ext = build[obj]
                if self.compiler_type=='absoft':
                    obj = cyg2win32(obj)
                    src = cyg2win32(src)
                if is_f_file(src) and not has_f90_header(src):
                    f77_objects.append((obj, (src, ext)))
                else:
                    other_objects.append((obj, (src, ext)))

        # f77 objects can be built in parallel
        build_items = f77_objects
        # build f90 modules serial, module files are generated during
        # compilation and may be used by files later in the list so the
        # ordering is important
        for o in other_objects:
            single_compile(o)
    else:
        build_items = build.items()

    if len(build) > 1 and jobs > 1:
        # build parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(jobs) as pool:
            res = pool.map(single_compile, build_items)
        list(res)  # access result to raise errors
    else:
        # build serial
        for o in build_items:
            single_compile(o)

    # Return *all* object filenames, not just the ones we just built.
    return objects

replace_method(CCompiler, 'compile', CCompiler_compile)

def CCompiler_customize_cmd(self, cmd, ignore=()):
    """
    Customize compiler using distutils command.

    Parameters
    ----------
    cmd : class instance
        An instance inheriting from `distutils.cmd.Command`.
    ignore : sequence of str, optional
        List of `CCompiler` commands (without ``'set_'``) that should not be
        altered. Strings that are checked for are:
        ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',
        'rpath', 'link_objects')``.

    Returns
    -------
    None

    """
    log.info('customize %s using %s' % (self.__class__.__name__,
                                        cmd.__class__.__name__))

    if hasattr(self, 'compiler') and 'clang' in self.compiler[0]:
        # clang defaults to a non-strict floating error point model.
        # Since NumPy and most Python libs give warnings for these, override:
        self.compiler.append('-ftrapping-math')
        self.compiler_so.append('-ftrapping-math')

    def allow(attr):
        return getattr(cmd, attr, None) is not None and attr not in ignore

    if allow('include_dirs'):
        self.set_include_dirs(cmd.include_dirs)
    if allow('define'):
        for (name, value) in cmd.define:
            self.define_macro(name, value)
    if allow('undef'):
        for macro in cmd.undef:
            self.undefine_macro(macro)
    if allow('libraries'):
        self.set_libraries(self.libraries + cmd.libraries)
    if allow('library_dirs'):
        self.set_library_dirs(self.library_dirs + cmd.library_dirs)
    if allow('rpath'):
        self.set_runtime_library_dirs(cmd.rpath)
    if allow('link_objects'):
        self.set_link_objects(cmd.link_objects)

replace_method(CCompiler, 'customize_cmd', CCompiler_customize_cmd)

def _compiler_to_string(compiler):
    props = []
    mx = 0
    keys = list(compiler.executables.keys())
    for key in ['version', 'libraries', 'library_dirs',
                'object_switch', 'compile_switch',
                'include_dirs', 'define', 'undef', 'rpath', 'link_objects']:
        if key not in keys:
            keys.append(key)
    for key in keys:
        if hasattr(compiler, key):
            v = getattr(compiler, key)
            mx = max(mx, len(key))
            props.append((key, repr(v)))
    fmt = '%-' + repr(mx+1) + 's = %s'
    lines = [fmt % prop for prop in props]
    return '\n'.join(lines)

def CCompiler_show_customization(self):
    """
    Print the compiler customizations to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Printing is only done if the distutils log threshold is < 2.

    """
    try:
        self.get_version()
    except Exception:
        pass
    if log._global_log.threshold<2:
        print('*'*80)
        print(self.__class__)
        print(_compiler_to_string(self))
        print('*'*80)

replace_method(CCompiler, 'show_customization', CCompiler_show_customization)

def CCompiler_customize(self, dist, need_cxx=0):
    """
    Do any platform-specific customization of a compiler instance.

    This method calls `distutils.sysconfig.customize_compiler` for
    platform-specific customization, as well as optionally remove a flag
    to suppress spurious warnings in case C++ code is being compiled.

    Parameters
    ----------
    dist : object
        This parameter is not used for anything.
    need_cxx : bool, optional
        Whether or not C++ has to be compiled. If so (True), the
        ``"-Wstrict-prototypes"`` option is removed to prevent spurious
        warnings. Default is False.

    Returns
    -------
    None

    Notes
    -----
    All the default options used by distutils can be extracted with::

      from distutils import sysconfig
      sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
                                'CCSHARED', 'LDSHARED', 'SO')

    """
    # See FCompiler.customize for suggested usage.
    log.info('customize %s' % (self.__class__.__name__))
    customize_compiler(self)
    if need_cxx:
        # In general, distutils uses -Wstrict-prototypes, but this option is
        # not valid for C++ code, only for C.  Remove it if it's there to
        # avoid a spurious warning on every compilation.
        try:
            self.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass

        if hasattr(self, 'compiler') and 'cc' in self.compiler[0]:
            if not self.compiler_cxx:
                if self.compiler[0].startswith('gcc'):
                    a, b = 'gcc', 'g++'
                else:
                    a, b = 'cc', 'c++'
                self.compiler_cxx = [self.compiler[0].replace(a, b)]\
                                    + self.compiler[1:]
        else:
            if hasattr(self, 'compiler'):
                log.warn("#### %s #######" % (self.compiler,))
            if not hasattr(self, 'compiler_cxx'):
                log.warn('Missing compiler_cxx fix for ' + self.__class__.__name__)


    # check if compiler supports gcc style automatic dependencies
    # run on every extension so skip for known good compilers
    if hasattr(self, 'compiler') and ('gcc' in self.compiler[0] or
                                      'g++' in self.compiler[0] or
                                      'clang' in self.compiler[0]):
        self._auto_depends = True
    elif os.name == 'posix':
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            fn = os.path.join(tmpdir, "file.c")
            with open(fn, "w") as f:
                f.write("int a;\n")
            self.compile([fn], output_dir=tmpdir,
                         extra_preargs=['-MMD', '-MF', fn + '.d'])
            self._auto_depends = True
        except CompileError:
            self._auto_depends = False
        finally:
            shutil.rmtree(tmpdir)

    return

replace_method(CCompiler, 'customize', CCompiler_customize)

def simple_version_match(pat=r'[-.\d]+', ignore='', start=''):
    """
    Simple matching of version numbers, for use in CCompiler and FCompiler.

    Parameters
    ----------
    pat : str, optional
        A regular expression matching version numbers.
        Default is ``r'[-.\\d]+'``.
    ignore : str, optional
        A regular expression matching patterns to skip.
        Default is ``''``, in which case nothing is skipped.
    start : str, optional
        A regular expression matching the start of where to start looking
        for version numbers.
        Default is ``''``, in which case searching is started at the
        beginning of the version string given to `matcher`.

    Returns
    -------
    matcher : callable
        A function that is appropriate to use as the ``.version_match``
        attribute of a `CCompiler` class. `matcher` takes a single parameter,
        a version string.

    """
    def matcher(self, version_string):
        # version string may appear in the second line, so getting rid
        # of new lines:
        version_string = version_string.replace('\n', ' ')
        pos = 0
        if start:
            m = re.match(start, version_string)
            if not m:
                return None
            pos = m.end()
        while True:
            m = re.search(pat, version_string[pos:])
            if not m:
                return None
            if ignore and re.match(ignore, m.group(0)):
                pos = m.end()
                continue
            break
        return m.group(0)
    return matcher

def CCompiler_get_version(self, force=False, ok_status=[0]):
    """
    Return compiler version, or None if compiler is not available.

    Parameters
    ----------
    force : bool, optional
        If True, force a new determination of the version, even if the
        compiler already has a version attribute. Default is False.
    ok_status : list of int, optional
        The list of status values returned by the version look-up process
        for which a version string is returned. If the status value is not
        in `ok_status`, None is returned. Default is ``[0]``.

    Returns
    -------
    version : str or None
        Version string, in the format of `distutils.version.LooseVersion`.

    """
    if not force and hasattr(self, 'version'):
        return self.version
    self.find_executables()
    try:
        version_cmd = self.version_cmd
    except AttributeError:
        return None
    if not version_cmd or not version_cmd[0]:
        return None
    try:
        matcher = self.version_match
    except AttributeError:
        try:
            pat = self.version_pattern
        except AttributeError:
            return None
        def matcher(version_string):
            m = re.match(pat, version_string)
            if not m:
                return None
            version = m.group('version')
            return version

    try:
        output = subprocess.check_output(version_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output
        status = exc.returncode
    except OSError:
        # match the historical returns for a parent
        # exception class caught by exec_command()
        status = 127
        output = b''
    else:
        # output isn't actually a filepath but we do this
        # for now to match previous distutils behavior
        output = filepath_from_subprocess_output(output)
        status = 0

    version = None
    if status in ok_status:
        version = matcher(output)
        if version:
            version = LooseVersion(version)
    self.version = version
    return version

replace_method(CCompiler, 'get_version', CCompiler_get_version)

def CCompiler_cxx_compiler(self):
    """
    Return the C++ compiler.

    Parameters
    ----------
    None

    Returns
    -------
    cxx : class instance
        The C++ compiler, as a `CCompiler` instance.

    """
    if self.compiler_type in ('msvc', 'intelw', 'intelemw'):
        return self

    cxx = copy(self)
    cxx.compiler_cxx = cxx.compiler_cxx
    cxx.compiler_so = [cxx.compiler_cxx[0]] + \
                      sanitize_cxx_flags(cxx.compiler_so[1:])
    if (sys.platform.startswith(('aix', 'os400')) and
            'ld_so_aix' in cxx.linker_so[0]):
        # AIX needs the ld_so_aix script included with Python
        cxx.linker_so = [cxx.linker_so[0], cxx.compiler_cxx[0]] \
                        + cxx.linker_so[2:]
    if sys.platform.startswith('os400'):
        #This is required by i 7.4 and prievous for PRId64 in printf() call.
        cxx.compiler_so.append('-D__STDC_FORMAT_MACROS')
        #This a bug of gcc10.3, which failed to handle the TLS init.
        cxx.compiler_so.append('-fno-extern-tls-init')
        cxx.linker_so.append('-fno-extern-tls-init')
    else:
        cxx.linker_so = [cxx.compiler_cxx[0]] + cxx.linker_so[1:]
    return cxx

replace_method(CCompiler, 'cxx_compiler', CCompiler_cxx_compiler)

compiler_class['intel'] = ('intelccompiler', 'IntelCCompiler',
                           "Intel C Compiler for 32-bit applications")
compiler_class['intele'] = ('intelccompiler', 'IntelItaniumCCompiler',
                            "Intel C Itanium Compiler for Itanium-based applications")
compiler_class['intelem'] = ('intelccompiler', 'IntelEM64TCCompiler',
                             "Intel C Compiler for 64-bit applications")
compiler_class['intelw'] = ('intelccompiler', 'IntelCCompilerW',
                            "Intel C Compiler for 32-bit applications on Windows")
compiler_class['intelemw'] = ('intelccompiler', 'IntelEM64TCCompilerW',
                              "Intel C Compiler for 64-bit applications on Windows")
compiler_class['pathcc'] = ('pathccompiler', 'PathScaleCCompiler',
                            "PathScale Compiler for SiCortex-based applications")
compiler_class['arm'] = ('armccompiler', 'ArmCCompiler',
                            "Arm C Compiler")

ccompiler._default_compilers += (('linux.*', 'intel'),
                                 ('linux.*', 'intele'),
                                 ('linux.*', 'intelem'),
                                 ('linux.*', 'pathcc'),
                                 ('nt', 'intelw'),
                                 ('nt', 'intelemw'))

if sys.platform == 'win32':
    compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler',
                                 "Mingw32 port of GNU C Compiler for Win32"\
                                 "(for MSC built Python)")
    if mingw32():
        # On windows platforms, we want to default to mingw32 (gcc)
        # because msvc can't build blitz stuff.
        log.info('Setting mingw32 as default compiler for nt.')
        ccompiler._default_compilers = (('nt', 'mingw32'),) \
                                       + ccompiler._default_compilers


_distutils_new_compiler = new_compiler
def new_compiler (plat=None,
                  compiler=None,
                  verbose=None,
                  dry_run=0,
                  force=0):
    # Try first C compilers from numpy.distutils.
    if verbose is None:
        verbose = log.get_threshold() <= log.INFO
    if plat is None:
        plat = os.name
    try:
        if compiler is None:
            compiler = get_default_compiler(plat)
        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError(msg)
    module_name = "numpy.distutils." + module_name
    try:
        __import__ (module_name)
    except ImportError as e:
        msg = str(e)
        log.info('%s in numpy.distutils; trying from distutils',
                 str(msg))
        module_name = module_name[6:]
        try:
            __import__(module_name)
        except ImportError as e:
            msg = str(e)
            raise DistutilsModuleError("can't compile C/C++ code: unable to load module '%s'" % \
                  module_name)
    try:
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except KeyError:
        raise DistutilsModuleError(("can't compile C/C++ code: unable to find class '%s' " +
               "in module '%s'") % (class_name, module_name))
    compiler = klass(None, dry_run, force)
    compiler.verbose = verbose
    log.debug('new_compiler returns %s' % (klass))
    return compiler

ccompiler.new_compiler = new_compiler

_distutils_gen_lib_options = gen_lib_options
def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
    # the version of this function provided by CPython allows the following
    # to return lists, which are unpacked automatically:
    # - compiler.runtime_library_dir_option
    # our version extends the behavior to:
    # - compiler.library_dir_option
    # - compiler.library_option
    # - compiler.find_library_file
    r = _distutils_gen_lib_options(compiler, library_dirs,
                                   runtime_library_dirs, libraries)
    lib_opts = []
    for i in r:
        if is_sequence(i):
            lib_opts.extend(list(i))
        else:
            lib_opts.append(i)
    return lib_opts
ccompiler.gen_lib_options = gen_lib_options

# Also fix up the various compiler modules, which do
# from distutils.ccompiler import gen_lib_options
# Don't bother with mwerks, as we don't support Classic Mac.
for _cc in ['msvc9', 'msvc', '_msvc', 'bcpp', 'cygwinc', 'emxc', 'unixc']:
    _m = sys.modules.get('distutils.' + _cc + 'compiler')
    if _m is not None:
        setattr(_m, 'gen_lib_options', gen_lib_options)

