import os
import sys
import sysconfig

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.ccompiler_opt import NPY_CXX_FLAGS
    from numpy.distutils.system_info import get_info, system_info
    config = Configuration('linalg', parent_package, top_path)

    config.add_subpackage('tests')

    # Configure lapack_lite

    src_dir = 'lapack_lite'
    lapack_lite_src = [
        os.path.join(src_dir, 'python_xerbla.c'),
        os.path.join(src_dir, 'f2c_z_lapack.c'),
        os.path.join(src_dir, 'f2c_c_lapack.c'),
        os.path.join(src_dir, 'f2c_d_lapack.c'),
        os.path.join(src_dir, 'f2c_s_lapack.c'),
        os.path.join(src_dir, 'f2c_lapack.c'),
        os.path.join(src_dir, 'f2c_blas.c'),
        os.path.join(src_dir, 'f2c_config.c'),
        os.path.join(src_dir, 'f2c.c'),
    ]
    all_sources = config.paths(lapack_lite_src)

    if os.environ.get('NPY_USE_BLAS_ILP64', "0") != "0":
        lapack_info = get_info('lapack_ilp64_opt', 2)
    else:
        lapack_info = get_info('lapack_opt', 0)  # and {}

    use_lapack_lite = not lapack_info

    if use_lapack_lite:
        # This makes numpy.distutils write the fact that lapack_lite
        # is being used to numpy.__config__
        class numpy_linalg_lapack_lite(system_info):
            def calc_info(self):
                info = {'language': 'c'}
                size_t_size = sysconfig.get_config_var("SIZEOF_SIZE_T")
                if size_t_size:
                    maxsize = 2**(size_t_size - 1) - 1
                else:
                    # We prefer using sysconfig as it allows cross-compilation
                    # but the information may be missing (e.g. on windows).
                    maxsize = sys.maxsize
                if maxsize > 2**32:
                    # Build lapack-lite in 64-bit integer mode.
                    # The suffix is arbitrary (lapack_lite symbols follow it),
                    # but use the "64_" convention here.
                    info['define_macros'] = [
                        ('HAVE_BLAS_ILP64', None),
                        ('BLAS_SYMBOL_SUFFIX', '64_')
                    ]
                self.set_info(**info)

        lapack_info = numpy_linalg_lapack_lite().get_info(2)

    def get_lapack_lite_sources(ext, build_dir):
        if use_lapack_lite:
            print("### Warning:  Using unoptimized lapack ###")
            return all_sources
        else:
            if sys.platform == 'win32':
                print("### Warning:  python_xerbla.c is disabled ###")
                return []
            return [all_sources[0]]

    config.add_extension(
        'lapack_lite',
        sources=['lapack_litemodule.c', get_lapack_lite_sources],
        depends=['lapack_lite/f2c.h'],
        extra_info=lapack_info,
    )

    # umath_linalg module
    config.add_extension(
        '_umath_linalg',
        sources=['umath_linalg.cpp', get_lapack_lite_sources],
        depends=['lapack_lite/f2c.h'],
        extra_info=lapack_info,
        extra_cxx_compile_args=NPY_CXX_FLAGS,
        libraries=['npymath'],
    )
    config.add_data_files('*.pyi')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
