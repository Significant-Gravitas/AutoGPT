""" Modified version of build_scripts that handles building scripts from functions.

"""
from distutils.command.build_scripts import build_scripts as old_build_scripts
from numpy.distutils import log
from numpy.distutils.misc_util import is_string

class build_scripts(old_build_scripts):

    def generate_scripts(self, scripts):
        new_scripts = []
        func_scripts = []
        for script in scripts:
            if is_string(script):
                new_scripts.append(script)
            else:
                func_scripts.append(script)
        if not func_scripts:
            return new_scripts

        build_dir = self.build_dir
        self.mkpath(build_dir)
        for func in func_scripts:
            script = func(build_dir)
            if not script:
                continue
            if is_string(script):
                log.info("  adding '%s' to scripts" % (script,))
                new_scripts.append(script)
            else:
                [log.info("  adding '%s' to scripts" % (s,)) for s in script]
                new_scripts.extend(list(script))
        return new_scripts

    def run (self):
        if not self.scripts:
            return

        self.scripts = self.generate_scripts(self.scripts)
        # Now make sure that the distribution object has this list of scripts.
        # setuptools' develop command requires that this be a list of filenames,
        # not functions.
        self.distribution.scripts = self.scripts

        return old_build_scripts.run(self)

    def get_source_files(self):
        from numpy.distutils.misc_util import get_script_files
        return get_script_files(self.scripts)
