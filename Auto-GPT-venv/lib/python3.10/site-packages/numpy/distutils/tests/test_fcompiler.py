from numpy.testing import assert_
import numpy.distutils.fcompiler

customizable_flags = [
    ('f77', 'F77FLAGS'),
    ('f90', 'F90FLAGS'),
    ('free', 'FREEFLAGS'),
    ('arch', 'FARCH'),
    ('debug', 'FDEBUG'),
    ('flags', 'FFLAGS'),
    ('linker_so', 'LDFLAGS'),
]


def test_fcompiler_flags(monkeypatch):
    monkeypatch.setenv('NPY_DISTUTILS_APPEND_FLAGS', '0')
    fc = numpy.distutils.fcompiler.new_fcompiler(compiler='none')
    flag_vars = fc.flag_vars.clone(lambda *args, **kwargs: None)

    for opt, envvar in customizable_flags:
        new_flag = '-dummy-{}-flag'.format(opt)
        prev_flags = getattr(flag_vars, opt)

        monkeypatch.setenv(envvar, new_flag)
        new_flags = getattr(flag_vars, opt)

        monkeypatch.delenv(envvar)
        assert_(new_flags == [new_flag])

    monkeypatch.setenv('NPY_DISTUTILS_APPEND_FLAGS', '1')

    for opt, envvar in customizable_flags:
        new_flag = '-dummy-{}-flag'.format(opt)
        prev_flags = getattr(flag_vars, opt)
        monkeypatch.setenv(envvar, new_flag)
        new_flags = getattr(flag_vars, opt)

        monkeypatch.delenv(envvar)
        if prev_flags is None:
            assert_(new_flags == [new_flag])
        else:
            assert_(new_flags == prev_flags + [new_flag])

