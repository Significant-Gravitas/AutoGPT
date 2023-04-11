"""
unixccompiler - can handle very long argument lists for ar.

"""
import os
import sys
import subprocess
import shlex

from distutils.errors import CompileError, DistutilsExecError, LibError
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.misc_util import _commandline_dep_string
from numpy.distutils import log

# Note that UnixCCompiler._compile appeared in Python 2.3
def UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    """Compile a single source files with a Unix-style compiler."""
    # HP ad-hoc fix, see ticket 1383
    ccomp = self.compiler_so
    if ccomp[0] == 'aCC':
        # remove flags that will trigger ANSI-C mode for aCC
        if '-Ae' in ccomp:
            ccomp.remove('-Ae')
        if '-Aa' in ccomp:
            ccomp.remove('-Aa')
        # add flags for (almost) sane C++ handling
        ccomp += ['-AA']
        self.compiler_so = ccomp
    # ensure OPT environment variable is read
    if 'OPT' in os.environ:
        # XXX who uses this?
        from sysconfig import get_config_vars
        opt = shlex.join(shlex.split(os.environ['OPT']))
        gcv_opt = shlex.join(shlex.split(get_config_vars('OPT')[0]))
        ccomp_s = shlex.join(self.compiler_so)
        if opt not in ccomp_s:
            ccomp_s = ccomp_s.replace(gcv_opt, opt)
            self.compiler_so = shlex.split(ccomp_s)
        llink_s = shlex.join(self.linker_so)
        if opt not in llink_s:
            self.linker_so = self.linker_so + shlex.split(opt)

    display = '%s: %s' % (os.path.basename(self.compiler_so[0]), src)

    # gcc style automatic dependencies, outputs a makefile (-MF) that lists
    # all headers needed by a c file as a side effect of compilation (-MMD)
    if getattr(self, '_auto_depends', False):
        deps = ['-MMD', '-MF', obj + '.d']
    else:
        deps = []

    try:
        self.spawn(self.compiler_so + cc_args + [src, '-o', obj] + deps +
                   extra_postargs, display = display)
    except DistutilsExecError as e:
        msg = str(e)
        raise CompileError(msg) from None

    # add commandline flags to dependency file
    if deps:
        # After running the compiler, the file created will be in EBCDIC
        # but will not be tagged as such. This tags it so the file does not
        # have multiple different encodings being written to it
        if sys.platform == 'zos':
            subprocess.check_output(['chtag', '-tc', 'IBM1047', obj + '.d'])
        with open(obj + '.d', 'a') as f:
            f.write(_commandline_dep_string(cc_args, extra_postargs, pp_opts))

replace_method(UnixCCompiler, '_compile', UnixCCompiler__compile)


def UnixCCompiler_create_static_lib(self, objects, output_libname,
                                    output_dir=None, debug=0, target_lang=None):
    """
    Build a static library in a separate sub-process.

    Parameters
    ----------
    objects : list or tuple of str
        List of paths to object files used to build the static library.
    output_libname : str
        The library name as an absolute or relative (if `output_dir` is used)
        path.
    output_dir : str, optional
        The path to the output directory. Default is None, in which case
        the ``output_dir`` attribute of the UnixCCompiler instance.
    debug : bool, optional
        This parameter is not used.
    target_lang : str, optional
        This parameter is not used.

    Returns
    -------
    None

    """
    objects, output_dir = self._fix_object_args(objects, output_dir)

    output_filename = \
                    self.library_filename(output_libname, output_dir=output_dir)

    if self._need_link(objects, output_filename):
        try:
            # previous .a may be screwed up; best to remove it first
            # and recreate.
            # Also, ar on OS X doesn't handle updating universal archives
            os.unlink(output_filename)
        except OSError:
            pass
        self.mkpath(os.path.dirname(output_filename))
        tmp_objects = objects + self.objects
        while tmp_objects:
            objects = tmp_objects[:50]
            tmp_objects = tmp_objects[50:]
            display = '%s: adding %d object files to %s' % (
                           os.path.basename(self.archiver[0]),
                           len(objects), output_filename)
            self.spawn(self.archiver + [output_filename] + objects,
                       display = display)

        # Not many Unices required ranlib anymore -- SunOS 4.x is, I
        # think the only major Unix that does.  Maybe we need some
        # platform intelligence here to skip ranlib if it's not
        # needed -- or maybe Python's configure script took care of
        # it for us, hence the check for leading colon.
        if self.ranlib:
            display = '%s:@ %s' % (os.path.basename(self.ranlib[0]),
                                   output_filename)
            try:
                self.spawn(self.ranlib + [output_filename],
                           display = display)
            except DistutilsExecError as e:
                msg = str(e)
                raise LibError(msg) from None
    else:
        log.debug("skipping %s (up-to-date)", output_filename)
    return

replace_method(UnixCCompiler, 'create_static_lib',
               UnixCCompiler_create_static_lib)
