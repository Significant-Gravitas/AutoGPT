import os
import sys
if 'setuptools' in sys.modules:
    from setuptools.command.bdist_rpm import bdist_rpm as old_bdist_rpm
else:
    from distutils.command.bdist_rpm import bdist_rpm as old_bdist_rpm

class bdist_rpm(old_bdist_rpm):

    def _make_spec_file(self):
        spec_file = old_bdist_rpm._make_spec_file(self)

        # Replace hardcoded setup.py script name
        # with the real setup script name.
        setup_py = os.path.basename(sys.argv[0])
        if setup_py == 'setup.py':
            return spec_file
        new_spec_file = []
        for line in spec_file:
            line = line.replace('setup.py', setup_py)
            new_spec_file.append(line)
        return new_spec_file
