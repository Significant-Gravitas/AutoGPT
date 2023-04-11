#!/usr/bin/env python3
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('distutils', parent_package, top_path)
    config.add_subpackage('command')
    config.add_subpackage('fcompiler')
    config.add_subpackage('tests')
    config.add_data_files('site.cfg')
    config.add_data_files('mingw/gfortran_vs2003_hack.c')
    config.add_data_dir('checks')
    config.add_data_files('*.pyi')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core      import setup
    setup(configuration=configuration)
