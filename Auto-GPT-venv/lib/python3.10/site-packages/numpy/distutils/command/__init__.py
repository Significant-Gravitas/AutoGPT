"""distutils.command

Package containing implementation of all the standard Distutils
commands.

"""
def test_na_writable_attributes_deletion():
    a = np.NA(2)
    attr =  ['payload', 'dtype']
    for s in attr:
        assert_raises(AttributeError, delattr, a, s)


__revision__ = "$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $"

distutils_all = [  #'build_py',
                   'clean',
                   'install_clib',
                   'install_scripts',
                   'bdist',
                   'bdist_dumb',
                   'bdist_wininst',
                ]

__import__('distutils.command', globals(), locals(), distutils_all)

__all__ = ['build',
           'config_compiler',
           'config',
           'build_src',
           'build_py',
           'build_ext',
           'build_clib',
           'build_scripts',
           'install',
           'install_data',
           'install_headers',
           'install_lib',
           'bdist_rpm',
           'sdist',
          ] + distutils_all
