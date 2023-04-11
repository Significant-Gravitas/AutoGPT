import os
from distutils.core import Command
from distutils.ccompiler import new_compiler
from numpy.distutils.misc_util import get_cmd

class install_clib(Command):
    description = "Command to install installable C libraries"

    user_options = []

    def initialize_options(self):
        self.install_dir = None
        self.outfiles = []

    def finalize_options(self):
        self.set_undefined_options('install', ('install_lib', 'install_dir'))

    def run (self):
        build_clib_cmd = get_cmd("build_clib")
        if not build_clib_cmd.build_clib:
            # can happen if the user specified `--skip-build`
            build_clib_cmd.finalize_options()
        build_dir = build_clib_cmd.build_clib

        # We need the compiler to get the library name -> filename association
        if not build_clib_cmd.compiler:
            compiler = new_compiler(compiler=None)
            compiler.customize(self.distribution)
        else:
            compiler = build_clib_cmd.compiler

        for l in self.distribution.installed_libraries:
            target_dir = os.path.join(self.install_dir, l.target_dir)
            name = compiler.library_filename(l.name)
            source = os.path.join(build_dir, name)
            self.mkpath(target_dir)
            self.outfiles.append(self.copy_file(source, target_dir)[0])

    def get_outputs(self):
        return self.outfiles
