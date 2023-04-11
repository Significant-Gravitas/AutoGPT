from distutils.unixccompiler import UnixCCompiler                              

class ArmCCompiler(UnixCCompiler):

    """
    Arm compiler.
    """

    compiler_type = 'arm'
    cc_exe = 'armclang'
    cxx_exe = 'armclang++'

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        cc_compiler = self.cc_exe
        cxx_compiler = self.cxx_exe
        self.set_executables(compiler=cc_compiler +
                                      ' -O3 -fPIC',
                             compiler_so=cc_compiler +
                                         ' -O3 -fPIC',
                             compiler_cxx=cxx_compiler +
                                          ' -O3 -fPIC',
                             linker_exe=cc_compiler +
                                        ' -lamath',
                             linker_so=cc_compiler +
                                       ' -lamath -shared')
