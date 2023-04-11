import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('fft', parent_package, top_path)

    config.add_subpackage('tests')

    # AIX needs to be told to use large file support - at all times
    defs = [('_LARGE_FILES', None)] if sys.platform[:3] == "aix" else []
    # Configure pocketfft_internal
    config.add_extension('_pocketfft_internal',
                         sources=['_pocketfft.c'],
                         define_macros=defs,
                         )

    config.add_data_files('*.pyi')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
