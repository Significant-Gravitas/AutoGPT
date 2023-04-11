import sys

from setuptools.command.egg_info import egg_info as _egg_info

class egg_info(_egg_info):
    def run(self):
        if 'sdist' in sys.argv:
            import warnings
            import textwrap
            msg = textwrap.dedent("""
                `build_src` is being run, this may lead to missing
                files in your sdist!  You want to use distutils.sdist
                instead of the setuptools version:

                    from distutils.command.sdist import sdist
                    cmdclass={'sdist': sdist}"

                See numpy's setup.py or gh-7131 for details.""")
            warnings.warn(msg, UserWarning, stacklevel=2)

        # We need to ensure that build_src has been executed in order to give
        # setuptools' egg_info command real filenames instead of functions which
        # generate files.
        self.run_command("build_src")
        _egg_info.run(self)
