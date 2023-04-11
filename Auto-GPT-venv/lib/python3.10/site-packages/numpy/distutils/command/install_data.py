import sys
have_setuptools = ('setuptools' in sys.modules)

from distutils.command.install_data import install_data as old_install_data

#data installer with improved intelligence over distutils
#data files are copied into the project directory instead
#of willy-nilly
class install_data (old_install_data):

    def run(self):
        old_install_data.run(self)

        if have_setuptools:
            # Run install_clib again, since setuptools does not run sub-commands
            # of install automatically
            self.run_command('install_clib')

    def finalize_options (self):
        self.set_undefined_options('install',
                                   ('install_lib', 'install_dir'),
                                   ('root', 'root'),
                                   ('force', 'force'),
                                  )
