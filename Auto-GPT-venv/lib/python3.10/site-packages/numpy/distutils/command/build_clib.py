""" Modified version of build_clib that handles fortran source files.
"""
import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
    DistutilsFileError

from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
    filter_sources, get_lib_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)
from numpy.distutils.ccompiler_opt import new_ccompiler_opt

# Fix Python distutils bug sf #1718574:
_l = old_build_clib.user_options
for _i in range(len(_l)):
    if _l[_i][0] in ['build-clib', 'build-temp']:
        _l[_i] = (_l[_i][0] + '=',) + _l[_i][1:]
#


class build_clib(old_build_clib):

    description = "build C/C++/F libraries used by Python extensions"

    user_options = old_build_clib.user_options + [
        ('fcompiler=', None,
         "specify the Fortran compiler type"),
        ('inplace', 'i', 'Build in-place'),
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
    ]

    boolean_options = old_build_clib.boolean_options + \
    ['inplace', 'warn-error', 'disable-optimization']

    def initialize_options(self):
        old_build_clib.initialize_options(self)
        self.fcompiler = None
        self.inplace = 0
        self.parallel = None
        self.warn_error = None
        self.cpu_baseline = None
        self.cpu_dispatch = None
        self.disable_optimization = None


    def finalize_options(self):
        if self.parallel:
            try:
                self.parallel = int(self.parallel)
            except ValueError as e:
                raise ValueError("--parallel/-j argument must be an integer") from e
        old_build_clib.finalize_options(self)
        self.set_undefined_options('build',
                                        ('parallel', 'parallel'),
                                        ('warn_error', 'warn_error'),
                                        ('cpu_baseline', 'cpu_baseline'),
                                        ('cpu_dispatch', 'cpu_dispatch'),
                                        ('disable_optimization', 'disable_optimization')
                                  )

    def have_f_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_f_sources(build_info.get('sources', [])):
                return True
        return False

    def have_cxx_sources(self):
        for (lib_name, build_info) in self.libraries:
            if has_cxx_sources(build_info.get('sources', [])):
                return True
        return False

    def run(self):
        if not self.libraries:
            return

        # Make sure that library sources are complete.
        languages = []

        # Make sure that extension sources are complete.
        self.run_command('build_src')

        for (lib_name, build_info) in self.libraries:
            l = build_info.get('language', None)
            if l and l not in languages:
                languages.append(l)

        from distutils.ccompiler import new_compiler
        self.compiler = new_compiler(compiler=self.compiler,
                                     dry_run=self.dry_run,
                                     force=self.force)
        self.compiler.customize(self.distribution,
                                need_cxx=self.have_cxx_sources())

        if self.warn_error:
            self.compiler.compiler.append('-Werror')
            self.compiler.compiler_so.append('-Werror')

        libraries = self.libraries
        self.libraries = None
        self.compiler.customize_cmd(self)
        self.libraries = libraries

        self.compiler.show_customization()

        if not self.disable_optimization:
            dispatch_hpath = os.path.join("numpy", "distutils", "include", "npy_cpu_dispatch_config.h")
            dispatch_hpath = os.path.join(self.get_finalized_command("build_src").build_src, dispatch_hpath)
            opt_cache_path = os.path.abspath(
                os.path.join(self.build_temp, 'ccompiler_opt_cache_clib.py')
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
                log.info("\n########### CLIB COMPILER OPTIMIZATION ###########")
                log.info(copt.report(full=True))

            import atexit
            atexit.register(report, self.compiler_opt)

        if self.have_f_sources():
            from numpy.distutils.fcompiler import new_fcompiler
            self._f_compiler = new_fcompiler(compiler=self.fcompiler,
                                             verbose=self.verbose,
                                             dry_run=self.dry_run,
                                             force=self.force,
                                             requiref90='f90' in languages,
                                             c_compiler=self.compiler)
            if self._f_compiler is not None:
                self._f_compiler.customize(self.distribution)

                libraries = self.libraries
                self.libraries = None
                self._f_compiler.customize_cmd(self)
                self.libraries = libraries

                self._f_compiler.show_customization()
        else:
            self._f_compiler = None

        self.build_libraries(self.libraries)

        if self.inplace:
            for l in self.distribution.installed_libraries:
                libname = self.compiler.library_filename(l.name)
                source = os.path.join(self.build_clib, libname)
                target = os.path.join(l.target_dir, libname)
                self.mkpath(l.target_dir)
                shutil.copy(source, target)

    def get_source_files(self):
        self.check_library_list(self.libraries)
        filenames = []
        for lib in self.libraries:
            filenames.extend(get_lib_source_files(lib))
        return filenames

    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            self.build_a_library(build_info, lib_name, libraries)

    def assemble_flags(self, in_flags):
        """ Assemble flags from flag list

        Parameters
        ----------
        in_flags : None or sequence
            None corresponds to empty list.  Sequence elements can be strings
            or callables that return lists of strings. Callable takes `self` as
            single parameter.

        Returns
        -------
        out_flags : list
        """
        if in_flags is None:
            return []
        out_flags = []
        for in_flag in in_flags:
            if callable(in_flag):
                out_flags += in_flag(self)
            else:
                out_flags.append(in_flag)
        return out_flags

    def build_a_library(self, build_info, lib_name, libraries):
        # default compilers
        compiler = self.compiler
        fcompiler = self._f_compiler

        sources = build_info.get('sources')
        if sources is None or not is_sequence(sources):
            raise DistutilsSetupError(("in 'libraries' option (library '%s'), " +
                                       "'sources' must be present and must be " +
                                       "a list of source filenames") % lib_name)
        sources = list(sources)

        c_sources, cxx_sources, f_sources, fmodule_sources \
            = filter_sources(sources)
        requiref90 = not not fmodule_sources or \
            build_info.get('language', 'c') == 'f90'

        # save source type information so that build_ext can use it.
        source_languages = []
        if c_sources:
            source_languages.append('c')
        if cxx_sources:
            source_languages.append('c++')
        if requiref90:
            source_languages.append('f90')
        elif f_sources:
            source_languages.append('f77')
        build_info['source_languages'] = source_languages

        lib_file = compiler.library_filename(lib_name,
                                             output_dir=self.build_clib)
        depends = sources + build_info.get('depends', [])

        force_rebuild = self.force
        if not self.disable_optimization and not self.compiler_opt.is_cached():
            log.debug("Detected changes on compiler optimizations")
            force_rebuild = True
        if not (force_rebuild or newer_group(depends, lib_file, 'newer')):
            log.debug("skipping '%s' library (up-to-date)", lib_name)
            return
        else:
            log.info("building '%s' library", lib_name)

        config_fc = build_info.get('config_fc', {})
        if fcompiler is not None and config_fc:
            log.info('using additional config_fc from setup script '
                     'for fortran compiler: %s'
                     % (config_fc,))
            from numpy.distutils.fcompiler import new_fcompiler
            fcompiler = new_fcompiler(compiler=fcompiler.compiler_type,
                                      verbose=self.verbose,
                                      dry_run=self.dry_run,
                                      force=self.force,
                                      requiref90=requiref90,
                                      c_compiler=self.compiler)
            if fcompiler is not None:
                dist = self.distribution
                base_config_fc = dist.get_option_dict('config_fc').copy()
                base_config_fc.update(config_fc)
                fcompiler.customize(base_config_fc)

        # check availability of Fortran compilers
        if (f_sources or fmodule_sources) and fcompiler is None:
            raise DistutilsError("library %s has Fortran sources"
                                 " but no Fortran compiler found" % (lib_name))

        if fcompiler is not None:
            fcompiler.extra_f77_compile_args = build_info.get(
                'extra_f77_compile_args') or []
            fcompiler.extra_f90_compile_args = build_info.get(
                'extra_f90_compile_args') or []

        macros = build_info.get('macros')
        if macros is None:
            macros = []
        include_dirs = build_info.get('include_dirs')
        if include_dirs is None:
            include_dirs = []
        # Flags can be strings, or callables that return a list of strings.
        extra_postargs = self.assemble_flags(
            build_info.get('extra_compiler_args'))
        extra_cflags = self.assemble_flags(
            build_info.get('extra_cflags'))
        extra_cxxflags = self.assemble_flags(
            build_info.get('extra_cxxflags'))

        include_dirs.extend(get_numpy_include_dirs())
        # where compiled F90 module files are:
        module_dirs = build_info.get('module_dirs') or []
        module_build_dir = os.path.dirname(lib_file)
        if requiref90:
            self.mkpath(module_build_dir)

        if compiler.compiler_type == 'msvc':
            # this hack works around the msvc compiler attributes
            # problem, msvc uses its own convention :(
            c_sources += cxx_sources
            cxx_sources = []

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

        objects = []
        if copt_cxx_sources:
            log.info("compiling C++ dispatch-able sources")
            objects += self.compiler_opt.try_dispatch(
                copt_c_sources,
                output_dir=self.build_temp,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_postargs + extra_cxxflags,
                ccompiler=cxx_compiler
            )

        if copt_c_sources:
            log.info("compiling C dispatch-able sources")
            objects += self.compiler_opt.try_dispatch(
                copt_c_sources,
                output_dir=self.build_temp,
                src_dir=copt_build_src,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=extra_postargs + extra_cflags)

        if c_sources:
            log.info("compiling C sources")
            objects += compiler.compile(
                c_sources,
                output_dir=self.build_temp,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_postargs +
                                copt_baseline_flags +
                                extra_cflags))

        if cxx_sources:
            log.info("compiling C++ sources")
            cxx_compiler = compiler.cxx_compiler()
            cxx_objects = cxx_compiler.compile(
                cxx_sources,
                output_dir=self.build_temp,
                macros=macros + copt_macros,
                include_dirs=include_dirs,
                debug=self.debug,
                extra_postargs=(extra_postargs +
                                copt_baseline_flags +
                                extra_cxxflags))
            objects.extend(cxx_objects)

        if f_sources or fmodule_sources:
            extra_postargs = []
            f_objects = []

            if requiref90:
                if fcompiler.module_dir_switch is None:
                    existing_modules = glob('*.mod')
                extra_postargs += fcompiler.module_options(
                    module_dirs, module_build_dir)

            if fmodule_sources:
                log.info("compiling Fortran 90 module sources")
                f_objects += fcompiler.compile(fmodule_sources,
                                               output_dir=self.build_temp,
                                               macros=macros,
                                               include_dirs=include_dirs,
                                               debug=self.debug,
                                               extra_postargs=extra_postargs)

            if requiref90 and self._f_compiler.module_dir_switch is None:
                # move new compiled F90 module files to module_build_dir
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
                        log.warn('failed to move %r to %r'
                                 % (f, module_build_dir))

            if f_sources:
                log.info("compiling Fortran sources")
                f_objects += fcompiler.compile(f_sources,
                                               output_dir=self.build_temp,
                                               macros=macros,
                                               include_dirs=include_dirs,
                                               debug=self.debug,
                                               extra_postargs=extra_postargs)
        else:
            f_objects = []

        if f_objects and not fcompiler.can_ccompiler_link(compiler):
            # Default linker cannot link Fortran object files, and results
            # need to be wrapped later. Instead of creating a real static
            # library, just keep track of the object files.
            listfn = os.path.join(self.build_clib,
                                  lib_name + '.fobjects')
            with open(listfn, 'w') as f:
                f.write("\n".join(os.path.abspath(obj) for obj in f_objects))

            listfn = os.path.join(self.build_clib,
                                  lib_name + '.cobjects')
            with open(listfn, 'w') as f:
                f.write("\n".join(os.path.abspath(obj) for obj in objects))

            # create empty "library" file for dependency tracking
            lib_fname = os.path.join(self.build_clib,
                                     lib_name + compiler.static_lib_extension)
            with open(lib_fname, 'wb') as f:
                pass
        else:
            # assume that default linker is suitable for
            # linking Fortran object files
            objects.extend(f_objects)
            compiler.create_static_lib(objects, lib_name,
                                       output_dir=self.build_clib,
                                       debug=self.debug)

        # fix library dependencies
        clib_libraries = build_info.get('libraries', [])
        for lname, binfo in libraries:
            if lname in clib_libraries:
                clib_libraries.extend(binfo.get('libraries', []))
        if clib_libraries:
            build_info['libraries'] = clib_libraries
