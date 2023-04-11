""" Modified version of build_ext that handles fortran source files.

"""
import os
import subprocess
from glob import glob

from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError,\
    DistutilsError
from distutils.file_util import copy_file

from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import (
    filter_sources, get_ext_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)
from numpy.distutils.command.config_compiler import show_fortran_compilers
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt

class build_ext (old_build_ext):

    description = "build C/C++/F extensions (compile/link to build directory)"

    user_options = old_build_ext.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ('parallel=', 'j',
         "number of parallel jobs"),
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

    help_options = old_build_ext.help_options + [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
    ]

    boolean_options = old_build_ext.boolean_options + ['warn-error', 'disable-optimization']

    def initialize_options(self):
        old_build_ext.initialize_options(self)
        self.fcompiler = None
        self.parallel = None
        self.warn_error = None
        self.cpu_baseline = None
        self.cpu_dispatch = None
        self.disable_optimization = None
        self.simd_test = None

    def finalize_options(self):
        if self.parallel:
            try:
                self.parallel = int(self.parallel)
            except ValueError as e:
                raise ValueError("--parallel/-j argument must be an integer") from e

        # Ensure that self.include_dirs and self.distribution.include_dirs
        # refer to the same list object. finalize_options will modify
        # self.include_dirs, but self.distribution.include_dirs is used
        # during the actual build.
        # self.include_dirs is None unless paths are specified with
        # --include-dirs.
        # The include paths will be passed to the compiler in the order:
        # numpy paths, --include-dirs paths, Python include path.
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)
        incl_dirs = self.include_dirs or []
        if self.distribution.include_dirs is None:
            self.distribution.include_dirs = []
        self.include_dirs = self.distribution.include_dirs
        self.include_dirs.extend(incl_dirs)

        old_build_ext.finalize_options(self)
        self.set_undefined_options('build',
                                        ('parallel', 'parallel'),
                                        ('warn_error', 'warn_error'),
                                        ('cpu_baseline', 'cpu_baseline'),
                                        ('cpu_dispatch', 'cpu_dispatch'),
                                        ('disable_optimization', 'disable_optimization'),
                                        ('simd_test', 'simd_test')
                                  )
        CCompilerOpt.conf_target_groups["simd_test"] = self.simd_test

    def run(self):
        if not self.extensions:
            return

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        if self.distribution.has_c_libraries():
            if self.inplace:
                if self.distribution.have_run.get('build_clib'):
                    log.warn('build_clib already run, it is too late to '
                             'ensure in-place build of build_clib')
                    build_clib = self.distribution.get_command_obj(
                        'build_clib')
                else:
                    build_clib = self.distribution.get_command_obj(
                        'build_clib')
                    build_clib.inplace = 1
                    build_clib.ensure_finalized()
                    build_clib.run()
                    self.distribution.have_run['build_clib'] = 1

            else:
                self.run_command('build_clib')
                build_clib = self.get_finalized_command('build_clib')
            self.library_dirs.append(build_clib.build_clib)
        else:
            build_clib = None

        # Not including C libraries to the list of
        # extension libraries automatically to prevent
        # bogus linking commands. Extensions must
        # explicitly specify the C libraries that they use.

        from distutils.ccompiler import new_compiler
        from numpy.distutils.fcompiler import new_fcompiler

        compiler_type = self.compiler
        # Initialize C compiler:
        self.compiler = new_compiler(compiler=compiler_type,
                                     verbose=self.verbose,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution)
        self.compiler.customize_cmd(self)

        if self.warn_error:
            self.compiler.compiler.append('-Werror')
            self.compiler.compiler_so.append('-Werror')

        self.compiler.show_customization()

        if not self.disable_optimization:
            dispatch_hpath = os.path.join("numpy", "distutils", "include", "npy_cpu_dispatch_config.h")
            dispatch_hpath = os.path.join(self.get_finalized_command("build_src").build_src, dispatch_hpath)
            opt_cache_path = os.path.abspath(
                os.path.join(self.build_temp, 'ccompiler_opt_cache_ext.py')
            )
            if hasattr(self, "compiler_opt"):
                # By default `CCompilerOpt` update the cache at the exit of
                # the process, which may lead to duplicate building
                # (see build_extension()/force_rebuild) if run() called
                # multiple times within the same os process/thread without
                # giving the chance the previous instances of `CCompilerOpt`
                # to update the cache.
                self.compiler_opt.cache_flush()

            self.compiler_opt = new_ccompiler_opt(
                compiler=self.compiler, dispatch_hpath=dispatch_hpath,
                cpu_baseline=self.cpu_baseline, cpu_dispatch=self.cpu_dispatch,
                cache_path=opt_cache_path
            )
            def report(copt):
                log.info("\n########### EXT COMPILER OPTIMIZATION ###########")
                log.info(copt.report(full=True))

            import atexit
            atexit.register(report, self.compiler_opt)

        # Setup directory for storing generated extra DLL files on Windows
        self.extra_dll_dir = os.path.join(self.build_temp, '.libs')
        if not os.path.isdir(self.extra_dll_dir):
            os.makedirs(self.extra_dll_dir)

        # Create mapping of libraries built by build_clib:
        clibs = {}
        if build_clib is not None:
            for libname, build_info in build_clib.libraries or []:
                if libname in clibs and clibs[libname] != build_info:
                    log.warn('library %r defined more than once,'
                             ' overwriting build_info\n%s... \nwith\n%s...'
                             % (libname, repr(clibs[libname])[:300], repr(build_info)[:300]))
                clibs[libname] = build_info
        # .. and distribution libraries:
        for libname, build_info in self.distribution.libraries or []:
            if libname in clibs:
                # build_clib libraries have a precedence before distribution ones
                continue
            clibs[libname] = build_info

        # Determine if C++/Fortran 77/Fortran 90 compilers are needed.
        # Update extension libraries, library_dirs, and macros.
        all_languages = set()
        for ext in self.extensions:
            ext_languages = set()
            c_libs = []
            c_lib_dirs = []
            macros = []
            for libname in ext.libraries:
                if libname in clibs:
                    binfo = clibs[libname]
                    c_libs += binfo.get('libraries', [])
                    c_lib_dirs += binfo.get('library_dirs', [])
                    for m in binfo.get('macros', []):
                        if m not in macros:
                            macros.append(m)

                for l in clibs.get(libname, {}).get('source_languages', []):
                    ext_languages.add(l)
            if c_libs:
                new_c_libs = ext.libraries + c_libs
                log.info('updating extension %r libraries from %r to %r'
                         % (ext.name, ext.libraries, new_c_libs))
                ext.libraries = new_c_libs
                ext.library_dirs = ext.library_dirs + c_lib_dirs
            if macros:
                log.info('extending extension %r defined_macros with %r'
                         % (ext.name, macros))
                ext.define_macros = ext.define_macros + macros

            # determine extension languages
            if has_f_sources(ext.sources):
                ext_languages.add('f77')
            if has_cxx_sources(ext.sources):
                ext_languages.add('c++')
            l = ext.language or self.compiler.detect_language(ext.sources)
            if l:
                ext_languages.add(l)

            # reset language attribute for choosing proper linker
            #
            # When we build extensions with multiple languages, we have to
            # choose a linker. The rules here are:
            #   1. if there is Fortran code, always prefer the Fortran linker,
            #   2. otherwise prefer C++ over C,
            #   3. Users can force a particular linker by using
            #          `language='c'`  # or 'c++', 'f90', 'f77'
            #      in their config.add_extension() calls.
            if 'c++' in ext_languages:
                ext_language = 'c++'
            else:
                ext_language = 'c'  # default

            has_fortran = False
            if 'f90' in ext_languages:
                ext_language = 'f90'
                has_fortran = True
            elif 'f77' in ext_languages:
                ext_language = 'f77'
                has_fortran = True

            if not ext.language or has_fortran:
                if l and l != ext_language and ext.language:
                    log.warn('resetting extension %r language from %r to %r.' %
                             (ext.name, l, ext_language))

                ext.language = ext_language

            # global language
            all_languages.update(ext_languages)

        need_f90_compiler = 'f90' in all_languages
        need_f77_compiler = 'f77' in all_languages
        need_cxx_compiler = 'c++' in all_languages

        # Initialize C++ compiler:
        if need_cxx_compiler:
            self._cxx_compiler = new_compiler(compiler=compiler_type,
                                              verbose=self.verbose,
                                              dry_run=self.dry_run,
                                              force=self.force)
            compiler = self._cxx_compiler
            compiler.customize(self.distribution, need_cxx=need_cxx_compiler)
            compiler.customize_cmd(self)
            compiler.show_customization()
            self._cxx_compiler = compiler.cxx_compiler()
        else:
            self._cxx_compiler = None

        # Initialize Fortran 77 compiler:
        if need_f77_compiler:
            ctype = self.fcompiler
            self._f77_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90=False,
                                               c_compiler=self.compiler)
            fcompiler = self._f77_compiler
            if fcompiler:
                ctype = fcompiler.compiler_type
                fcompiler.customize(self.distribution)
            if fcompiler and fcompiler.get_version():
                fcompiler.customize_cmd(self)
                fcompiler.show_customization()
            else:
                self.warn('f77_compiler=%s is not available.' %
                          (ctype))
                self._f77_compiler = None
        else:
            self._f77_compiler = None

        # Initialize Fortran 90 compiler:
        if need_f90_compiler:
            ctype = self.fcompiler
            self._f90_compiler = new_fcompiler(compiler=self.fcompiler,
                                               verbose=self.verbose,
                                               dry_run=self.dry_run,
                                               force=self.force,
                                               requiref90=True,
                                               c_compiler=self.compiler)
            fcompiler = self._f90_compiler
            if fcompiler:
                ctype = fcompiler.compiler_type
                fcompiler.customize(self.distribution)
            if fcompiler and fcompiler.get_version():
                fcompiler.customize_cmd(self)
                fcompiler.show_customization()
            else:
                self.warn('f90_compiler=%s is not available.' %
                          (ctype))
                self._f90_compiler = None
        else:
            self._f90_compiler = None

        # Build extensions
        self.build_extensions()

        # Copy over any extra DLL files
        # FIXME: In the case where there are more than two packages,
        # we blindly assume that both packages need all of the libraries,
        # resulting in a larger wheel than is required. This should be fixed,
        # but it's so rare that I won't bother to handle it.
        pkg_roots = {
            self.get_ext_fullname(ext.name).split('.')[0]
            for ext in self.extensions
        }
        for pkg_root in pkg_roots:
            shared_lib_dir = os.path.join(pkg_root, '.libs')
            if not self.inplace:
                shared_lib_dir = os.path.join(self.build_lib, shared_lib_dir)
            for fn in os.listdir(self.extra_dll_dir):
                if not os.path.isdir(shared_lib_dir):
                    os.makedirs(shared_lib_dir)
                if not fn.lower().endswith('.dll'):
                    continue
                runtime_lib = os.path.join(self.extra_dll_dir, fn)
                copy_file(runtime_lib, shared_lib_dir)

    def swig_sources(self, sources, extensions=None):
        # Do nothing. Swig sources have been handled in build_src command.
        return sources

    def build_extension(self, ext):
        sources = ext.sources
        if sources is None or not is_sequence(sources):
            raise DistutilsSetupError(
                ("in 'ext_modules' option (extension '%s'), " +
                 "'sources' must be present and must be " +
                 "a list of source filenames") % ext.name)
        sources = list(sources)

        if not sources:
            return

        fullname = self.get_ext_fullname(ext.name)
        if self.inplace:
            modpath = fullname.split('.')
            package = '.'.join(modpath[0:-1])
            base = modpath[-1]
            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(package)
            ext_filename = os.path.join(package_dir,
                                        self.get_ext_filename(base))
        else:
            ext_filename = os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname))
        depends = sources + ext.depends

        force_rebuild = self.force
        if not self.disable_optimization and not self.compiler_opt.is_cached():
            log.debug("Detected changes on compiler optimizations")
            force_rebuild = True
        if not (force_rebuild or newer_group(depends, ext_filename, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        extra_args = ext.extra_compile_args or []
        extra_cflags = getattr(ext, 'extra_c_compile_args', None) or []
        extra_cxxflags = getattr(ext, 'extra_cxx_compile_args', None) or []

        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        c_sources, cxx_sources, f_sources, fmodule_sources = \
            filter_sources(ext.sources)

        if self.compiler.compiler_type == 'msvc':
            if cxx_sources:
                # Needed to compile kiva.agg._agg extension.
                extra_args.append('/Zm1000')
            # this hack works around the msvc compiler attributes
            # problem, msvc uses its own convention :(
            c_sources += cxx_sources
            cxx_sources = []

        # Set Fortran/C++ compilers for compilation and linking.
        if ext.language == 'f90':
            fcompiler = self._f90_compiler
        elif ext.language == 'f77':
            fcompiler = self._f77_compiler
        else:  # in case ext.language is c++, for instance
            fcompiler = self._f90_compiler or self._f77_compiler
        if fcompiler is not None:
            fcompiler.extra_f77_compile_args = (ext.extra_f77_compile_args or []) if hasattr(
                ext, 'extra_f77_compile_args') else []
            fcompiler.extra_f90_compile_args = (ext.extra_f90_compile_args or []) if hasattr(
                ext, 'extra_f90_compile_args') else []
        cxx_compiler = self._cxx_compiler

        # check for the availability of required compilers
        if cxx_sources and cxx_compiler is None:
            raise DistutilsError("extension %r has C++ sources"
                                 "but no C++ compiler found" % (ext.name))
        if (f_sources or fmodule_sources) and fcompiler is None:
            raise DistutilsError("extension %r has Fortran sources "
                                 "but no Fortran compiler found" % (ext.name))
        if ext.language in ['f77', 'f90'] and fcompiler is None:
            self.warn("extension %r has Fortran libraries "
                      "but no Fortran linker found, using default linker" % (ext.name))
        if ext.language == 'c++' and cxx_compiler is None:
            self.warn("extension %r has C++ libraries "
                      "but no C++ linker found, using default linker" % (ext.name))

        kws = {'depends': ext.depends}
        output_dir = self.build_temp

        include_dirs = ext.include_dirs + get_numpy_include_dirs()

        # filtering C dispatch-table sources when optimization is not disabled,
        # otherwise treated as normal sources.
        copt_c_sources = []
        copt_cxx_sources = []
        copt_baseline_flags = []
        copt_macros = []
        if not self.disable_optimization:
            bsrc_dir = self.get_finalized_command("build_src").build_src
            dispatch_hpath = os.path.join("numpy", "distutils", "include")
            dispatch_hpath = os.path.join(bsrc_dir, dispatch_hpath)
            include_dirs.append(dispatch_hpath)

            copt_build_src = None if self.inplace else bsrc_dir
            for _srcs, _dst, _ext in (
                ((c_sources,), copt_c_sources, ('.dispatch.c',)),
                ((c_sources, cxx_sources), copt_cxx_sources,
                    ('.dispatch.cpp', '.dispatch.cxx'))
            ):
                for _src in _srcs:
                    _dst += [
                        _src.pop(_src.index(s))
                        for s in _src[:] if s.endswith(_ext)
                    ]
            copt_baseline_flags = self.compiler_opt.cpu_baseline_flags()
        else:
            copt_macros.append(("NPY_DISABLE_OPTIMIZATION", 1))

        c_objects = []
        if copt_cxx_sources:
            log.info("compiling C++ dispatch-able sources")
            c_objects += self.compiler_opt.try_dispatch(
                copt_cxx_sources,
                output_dir=output_dir,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_args + extra_cxxflags,
                ccompiler=cxx_compiler,
                **kws
            )
        if copt_c_sources:
            log.info("compiling C dispatch-able sources")
            c_objects += self.compiler_opt.try_dispatch(
                copt_c_sources,
                output_dir=output_dir,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_args + extra_cflags,
                **kws)
        if c_sources:
            log.info("compiling C sources")
            c_objects += self.compiler.compile(
                c_sources,
                output_dir=output_dir,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_args + copt_baseline_flags +
                                extra_cflags),
                **kws)
        if cxx_sources:
            log.info("compiling C++ sources")
            c_objects += cxx_compiler.compile(
                cxx_sources,
                output_dir=output_dir,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_args + copt_baseline_flags +
                                extra_cxxflags),
                **kws)

        extra_postargs = []
        f_objects = []
        if fmodule_sources:
            log.info("compiling Fortran 90 module sources")
            module_dirs = ext.module_dirs[:]
            module_build_dir = os.path.join(
                self.build_temp, os.path.dirname(
                    self.get_ext_filename(fullname)))

            self.mkpath(module_build_dir)
            if fcompiler.module_dir_switch is None:
                existing_modules = glob('*.mod')
            extra_postargs += fcompiler.module_options(
                module_dirs, module_build_dir)
            f_objects += fcompiler.compile(fmodule_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

            if fcompiler.module_dir_switch is None:
                for f in glob('*.mod'):
                    if f in existing_modules:
                        continue
                    t = os.path.join(module_build_dir, f)
                    if os.path.abspath(f) == os.path.abspath(t):
                        continue
                    if os.path.isfile(t):
                        os.remove(t)
                    try:
                        self.move_file(f, module_build_dir)
                    except DistutilsFileError:
                        log.warn('failed to move %r to %r' %
                                 (f, module_build_dir))
        if f_sources:
            log.info("compiling Fortran sources")
            f_objects += fcompiler.compile(f_sources,
                                           output_dir=self.build_temp,
                                           macros=macros,
                                           include_dirs=include_dirs,
                                           debug=self.debug,
                                           extra_postargs=extra_postargs,
                                           depends=ext.depends)

        if f_objects and not fcompiler.can_ccompiler_link(self.compiler):
            unlinkable_fobjects = f_objects
            objects = c_objects
        else:
            unlinkable_fobjects = []
            objects = c_objects + f_objects

        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []
        libraries = self.get_libraries(ext)[:]
        library_dirs = ext.library_dirs[:]

        linker = self.compiler.link_shared_object
        # Always use system linker when using MSVC compiler.
        if self.compiler.compiler_type in ('msvc', 'intelw', 'intelemw'):
            # expand libraries with fcompiler libraries as we are
            # not using fcompiler linker
            self._libs_with_msvc_and_fortran(
                fcompiler, libraries, library_dirs)
            if ext.runtime_library_dirs:
                # gcc adds RPATH to the link. On windows, copy the dll into
                # self.extra_dll_dir instead.
                for d in ext.runtime_library_dirs:
                    for f in glob(d + '/*.dll'):
                        copy_file(f, self.extra_dll_dir)
                ext.runtime_library_dirs = []

        elif ext.language in ['f77', 'f90'] and fcompiler is not None:
            linker = fcompiler.link_shared_object
        if ext.language == 'c++' and cxx_compiler is not None:
            linker = cxx_compiler.link_shared_object

        if fcompiler is not None:
            objects, libraries = self._process_unlinkable_fobjects(
                    objects, libraries,
                    fcompiler, library_dirs,
                    unlinkable_fobjects)

        linker(objects, ext_filename,
               libraries=libraries,
               library_dirs=library_dirs,
               runtime_library_dirs=ext.runtime_library_dirs,
               extra_postargs=extra_args,
               export_symbols=self.get_export_symbols(ext),
               debug=self.debug,
               build_temp=self.build_temp,
               target_lang=ext.language)

    def _add_dummy_mingwex_sym(self, c_sources):
        build_src = self.get_finalized_command("build_src").build_src
        build_clib = self.get_finalized_command("build_clib").build_clib
        objects = self.compiler.compile([os.path.join(build_src,
                                                      "gfortran_vs2003_hack.c")],
                                        output_dir=self.build_temp)
        self.compiler.create_static_lib(
            objects, "_gfortran_workaround", output_dir=build_clib, debug=self.debug)

    def _process_unlinkable_fobjects(self, objects, libraries,
                                     fcompiler, library_dirs,
                                     unlinkable_fobjects):
        libraries = list(libraries)
        objects = list(objects)
        unlinkable_fobjects = list(unlinkable_fobjects)

        # Expand possible fake static libraries to objects;
        # make sure to iterate over a copy of the list as
        # "fake" libraries will be removed as they are
        # encountered
        for lib in libraries[:]:
            for libdir in library_dirs:
                fake_lib = os.path.join(libdir, lib + '.fobjects')
                if os.path.isfile(fake_lib):
                    # Replace fake static library
                    libraries.remove(lib)
                    with open(fake_lib, 'r') as f:
                        unlinkable_fobjects.extend(f.read().splitlines())

                    # Expand C objects
                    c_lib = os.path.join(libdir, lib + '.cobjects')
                    with open(c_lib, 'r') as f:
                        objects.extend(f.read().splitlines())

        # Wrap unlinkable objects to a linkable one
        if unlinkable_fobjects:
            fobjects = [os.path.abspath(obj) for obj in unlinkable_fobjects]
            wrapped = fcompiler.wrap_unlinkable_objects(
                    fobjects, output_dir=self.build_temp,
                    extra_dll_dir=self.extra_dll_dir)
            objects.extend(wrapped)

        return objects, libraries

    def _libs_with_msvc_and_fortran(self, fcompiler, c_libraries,
                                    c_library_dirs):
        if fcompiler is None:
            return

        for libname in c_libraries:
            if libname.startswith('msvc'):
                continue
            fileexists = False
            for libdir in c_library_dirs or []:
                libfile = os.path.join(libdir, '%s.lib' % (libname))
                if os.path.isfile(libfile):
                    fileexists = True
                    break
            if fileexists:
                continue
            # make g77-compiled static libs available to MSVC
            fileexists = False
            for libdir in c_library_dirs:
                libfile = os.path.join(libdir, 'lib%s.a' % (libname))
                if os.path.isfile(libfile):
                    # copy libname.a file to name.lib so that MSVC linker
                    # can find it
                    libfile2 = os.path.join(self.build_temp, libname + '.lib')
                    copy_file(libfile, libfile2)
                    if self.build_temp not in c_library_dirs:
                        c_library_dirs.append(self.build_temp)
                    fileexists = True
                    break
            if fileexists:
                continue
            log.warn('could not find library %r in directories %s'
                     % (libname, c_library_dirs))

        # Always use system linker when using MSVC compiler.
        f_lib_dirs = []
        for dir in fcompiler.library_dirs:
            # correct path when compiling in Cygwin but with normal Win
            # Python
            if dir.startswith('/usr/lib'):
                try:
                    dir = subprocess.check_output(['cygpath', '-w', dir])
                except (OSError, subprocess.CalledProcessError):
                    pass
                else:
                    dir = filepath_from_subprocess_output(dir)
            f_lib_dirs.append(dir)
        c_library_dirs.extend(f_lib_dirs)

        # make g77-compiled static libs available to MSVC
        for lib in fcompiler.libraries:
            if not lib.startswith('msvc'):
                c_libraries.append(lib)
                p = combine_paths(f_lib_dirs, 'lib' + lib + '.a')
                if p:
                    dst_name = os.path.join(self.build_temp, lib + '.lib')
                    if not os.path.isfile(dst_name):
                        copy_file(p[0], dst_name)
                    if self.build_temp not in c_library_dirs:
                        c_library_dirs.append(self.build_temp)

    def get_source_files(self):
        self.check_extensions_list(self.extensions)
        filenames = []
        for ext in self.extensions:
            filenames.extend(get_ext_source_files(ext))
        return filenames

    def get_outputs(self):
        self.check_extensions_list(self.extensions)

        outputs = []
        for ext in self.extensions:
            if not ext.sources:
                continue
            fullname = self.get_ext_fullname(ext.name)
            outputs.append(os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname)))
        return outputs
