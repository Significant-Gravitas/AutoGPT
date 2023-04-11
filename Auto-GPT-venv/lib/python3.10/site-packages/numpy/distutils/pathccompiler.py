from distutils.unixccompiler import UnixCCompiler

class PathScaleCCompiler(UnixCCompiler):

    """
    PathScale compiler compatible with an gcc built Python.
    """

    compiler_type = 'pathcc'
    cc_exe = 'pathcc'
    cxx_exe = 'pathCC'

    def __init__ (self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__ (self, verbose, dry_run, force)
        cc_compiler = self.cc_exe
        cxx_compiler = self.cxx_exe
        self.set_executables(compiler=cc_compiler,
                             compiler_so=cc_compiler,
                             compiler_cxx=cxx_compiler,
                             linker_exe=cc_compiler,
                             linker_so=cc_compiler + ' -shared')
