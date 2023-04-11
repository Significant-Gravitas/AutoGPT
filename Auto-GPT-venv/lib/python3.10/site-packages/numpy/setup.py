#!/usr/bin/env python3

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('numpy', parent_package, top_path)

    config.add_subpackage('array_api')
    config.add_subpackage('compat')
    config.add_subpackage('core')
    config.add_subpackage('distutils')
    config.add_subpackage('doc')
    config.add_subpackage('f2py')
    config.add_subpackage('fft')
    config.add_subpackage('lib')
    config.add_subpackage('linalg')
    config.add_subpackage('ma')
    config.add_subpackage('matrixlib')
    config.add_subpackage('polynomial')
    config.add_subpackage('random')
    config.add_subpackage('testing')
    config.add_subpackage('typing')
    config.add_subpackage('_typing')
    config.add_data_dir('doc')
    config.add_data_files('py.typed')
    config.add_data_files('*.pyi')
    config.add_subpackage('tests')
    config.add_subpackage('_pyinstaller')
    config.make_config_py() # installs __config__.py
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
