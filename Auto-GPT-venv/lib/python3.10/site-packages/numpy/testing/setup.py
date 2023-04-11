#!/usr/bin/env python3

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('testing', parent_package, top_path)

    config.add_subpackage('_private')
    config.add_subpackage('tests')
    config.add_data_files('*.pyi')
    config.add_data_files('_private/*.pyi')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer="NumPy Developers",
          maintainer_email="numpy-dev@numpy.org",
          description="NumPy test module",
          url="https://www.numpy.org",
          license="NumPy License (BSD Style)",
          configuration=configuration,
          )
