# XXX: Handle setuptools ?
from distutils.core import Distribution

# This class is used because we add new files (sconscripts, and so on) with the
# scons command
class NumpyDistribution(Distribution):
    def __init__(self, attrs = None):
        # A list of (sconscripts, pre_hook, post_hook, src, parent_names)
        self.scons_data = []
        # A list of installable libraries
        self.installed_libraries = []
        # A dict of pkg_config files to generate/install
        self.installed_pkg_config = {}
        Distribution.__init__(self, attrs)

    def has_scons_scripts(self):
        return bool(self.scons_data)
