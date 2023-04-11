import sys
if 'setuptools' in sys.modules:
    import setuptools.command.install as old_install_mod
    have_setuptools = True
else:
    import distutils.command.install as old_install_mod
    have_setuptools = False
from distutils.file_util import write_file

old_install = old_install_mod.install

class install(old_install):

    # Always run install_clib - the command is cheap, so no need to bypass it;
    # but it's not run by setuptools -- so it's run again in install_data
    sub_commands = old_install.sub_commands + [
        ('install_clib', lambda x: True)
    ]

    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase

    def setuptools_run(self):
        """ The setuptools version of the .run() method.

        We must pull in the entire code so we can override the level used in the
        _getframe() call since we wrap this call by one more level.
        """
        from distutils.command.install import install as distutils_install

        # Explicit request for old-style install?  Just do it
        if self.old_and_unmanageable or self.single_version_externally_managed:
            return distutils_install.run(self)

        # Attempt to detect whether we were called from setup() or by another
        # command.  If we were called by setup(), our caller will be the
        # 'run_command' method in 'distutils.dist', and *its* caller will be
        # the 'run_commands' method.  If we were called any other way, our
        # immediate caller *might* be 'run_command', but it won't have been
        # called by 'run_commands'.  This is slightly kludgy, but seems to
        # work.
        #
        caller = sys._getframe(3)
        caller_module = caller.f_globals.get('__name__', '')
        caller_name = caller.f_code.co_name

        if caller_module != 'distutils.dist' or caller_name!='run_commands':
            # We weren't called from the command line or setup(), so we
            # should run in backward-compatibility mode to support bdist_*
            # commands.
            distutils_install.run(self)
        else:
            self.do_egg_install()

    def run(self):
        if not have_setuptools:
            r = old_install.run(self)
        else:
            r = self.setuptools_run()
        if self.record:
            # bdist_rpm fails when INSTALLED_FILES contains
            # paths with spaces. Such paths must be enclosed
            # with double-quotes.
            with open(self.record, 'r') as f:
                lines = []
                need_rewrite = False
                for l in f:
                    l = l.rstrip()
                    if ' ' in l:
                        need_rewrite = True
                        l = '"%s"' % (l)
                    lines.append(l)
            if need_rewrite:
                self.execute(write_file,
                             (self.record, lines),
                             "re-writing list of installed files to '%s'" %
                             self.record)
        return r
