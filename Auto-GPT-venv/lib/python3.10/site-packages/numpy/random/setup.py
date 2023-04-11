import os
import sys
from os.path import join

from numpy.distutils.system_info import platform_bits
from numpy.distutils.msvccompiler import lib_opts_if_msvc


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    config = Configuration('random', parent_package, top_path)

    def generate_libraries(ext, build_dir):
        config_cmd = config.get_config_cmd()
        libs = get_mathlibs()
        if sys.platform == 'win32':
            libs.extend(['Advapi32', 'Kernel32'])
        ext.libraries.extend(libs)
        return None

    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    if sys.platform[:3] == 'aix':
        defs = [('_LARGE_FILES', None)]
    else:
        defs = [('_FILE_OFFSET_BITS', '64'),
                ('_LARGEFILE_SOURCE', '1'),
                ('_LARGEFILE64_SOURCE', '1')]

    defs.append(('NPY_NO_DEPRECATED_API', 0))
    config.add_subpackage('tests')
    config.add_data_dir('tests/data')
    config.add_data_dir('_examples')

    EXTRA_LINK_ARGS = []
    EXTRA_LIBRARIES = ['npyrandom']
    if os.name != 'nt':
        # Math lib
        EXTRA_LIBRARIES.append('m')
    # Some bit generators exclude GCC inlining
    EXTRA_COMPILE_ARGS = ['-U__GNUC_GNU_INLINE__']

    if sys.platform == 'cygwin':
        # Export symbols without __declspec(dllexport) for using by cython.
        # Using __declspec(dllexport) does not export other necessary symbols
        # in Cygwin package's Cython environment, making it impossible to
        # import modules.
        EXTRA_LINK_ARGS += ['-Wl,--export-all-symbols']

    # Use legacy integer variable sizes
    LEGACY_DEFS = [('NP_RANDOM_LEGACY', '1')]
    PCG64_DEFS = []
    # One can force emulated 128-bit arithmetic if one wants.
    #PCG64_DEFS += [('PCG_FORCE_EMULATED_128BIT_MATH', '1')]
    depends = ['__init__.pxd', 'c_distributions.pxd', 'bit_generator.pxd']

    # npyrandom - a library like npymath
    npyrandom_sources = [
        'src/distributions/logfactorial.c',
        'src/distributions/distributions.c',
        'src/distributions/random_mvhg_count.c',
        'src/distributions/random_mvhg_marginals.c',
        'src/distributions/random_hypergeometric.c',
    ]

    def lib_opts(build_cmd):
        """ Add flags that depend on the compiler.

        We can't see which compiler we are using in our scope, because we have
        not initialized the distutils build command, so use this deferred
        calculation to run when we are building the library.
        """
        opts = lib_opts_if_msvc(build_cmd)
        if build_cmd.compiler.compiler_type != 'msvc':
            # Some bit generators require c99
            opts.append('-std=c99')
        return opts

    config.add_installed_library('npyrandom',
        sources=npyrandom_sources,
        install_dir='lib',
        build_info={
            'include_dirs' : [],  # empty list required for creating npyrandom.h
            'extra_compiler_args': [lib_opts],
        })

    for gen in ['mt19937']:
        # gen.pyx, src/gen/gen.c, src/gen/gen-jump.c
        config.add_extension(f'_{gen}',
                             sources=[f'_{gen}.c',
                                      f'src/{gen}/{gen}.c',
                                      f'src/{gen}/{gen}-jump.c'],
                             include_dirs=['.', 'src', join('src', gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=depends + [f'_{gen}.pyx'],
                             define_macros=defs,
                             )
    for gen in ['philox', 'pcg64', 'sfc64']:
        # gen.pyx, src/gen/gen.c
        _defs = defs + PCG64_DEFS if gen == 'pcg64' else defs
        config.add_extension(f'_{gen}',
                             sources=[f'_{gen}.c',
                                      f'src/{gen}/{gen}.c'],
                             include_dirs=['.', 'src', join('src', gen)],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=depends + [f'_{gen}.pyx',
                                   'bit_generator.pyx', 'bit_generator.pxd'],
                             define_macros=_defs,
                             )
    for gen in ['_common', 'bit_generator']:
        # gen.pyx
        config.add_extension(gen,
                             sources=[f'{gen}.c'],
                             libraries=EXTRA_LIBRARIES,
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             extra_link_args=EXTRA_LINK_ARGS,
                             include_dirs=['.', 'src'],
                             depends=depends + [f'{gen}.pyx', f'{gen}.pxd',],
                             define_macros=defs,
                             )
        config.add_data_files(f'{gen}.pxd')
    for gen in ['_generator', '_bounded_integers']:
        # gen.pyx, src/distributions/distributions.c
        config.add_extension(gen,
                             sources=[f'{gen}.c'],
                             libraries=EXTRA_LIBRARIES + ['npymath'],
                             extra_compile_args=EXTRA_COMPILE_ARGS,
                             include_dirs=['.', 'src'],
                             extra_link_args=EXTRA_LINK_ARGS,
                             depends=depends + [f'{gen}.pyx'],
                             define_macros=defs,
                             )
    config.add_data_files('_bounded_integers.pxd')
    mtrand_libs = ['m', 'npymath'] if os.name != 'nt' else ['npymath']
    config.add_extension('mtrand',
                         sources=['mtrand.c',
                                  'src/legacy/legacy-distributions.c',
                                  'src/distributions/distributions.c',
                                 ],
                         include_dirs=['.', 'src', 'src/legacy'],
                         libraries=mtrand_libs,
                         extra_compile_args=EXTRA_COMPILE_ARGS,
                         extra_link_args=EXTRA_LINK_ARGS,
                         depends=depends + ['mtrand.pyx'],
                         define_macros=defs + LEGACY_DEFS,
                         )
    config.add_data_files(*depends)
    config.add_data_files('*.pyi')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(configuration=configuration)
