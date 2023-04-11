def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('lib', parent_package, top_path)
    config.add_subpackage('tests')
    config.add_data_dir('tests/data')
    config.add_data_files('*.pyi')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
