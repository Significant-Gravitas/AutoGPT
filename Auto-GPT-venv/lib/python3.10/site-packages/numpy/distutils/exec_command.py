"""
exec_command

Implements exec_command function that is (almost) equivalent to
commands.getstatusoutput function but on NT, DOS systems the
returned status is actually correct (though, the returned status
values may be different by a factor). In addition, exec_command
takes keyword arguments for (re-)defining environment variables.

Provides functions:

  exec_command  --- execute command in a specified directory and
                    in the modified environment.
  find_executable --- locate a command using info from environment
                    variable PATH. Equivalent to posix `which`
                    command.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: 11 January 2003

Requires: Python 2.x

Successfully tested on:

========  ============  =================================================
os.name   sys.platform  comments
========  ============  =================================================
posix     linux2        Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
                        PyCrust 0.9.3, Idle 1.0.2
posix     linux2        Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
posix     sunos5        SunOS 5.9, Python 2.2, 2.3.2
posix     darwin        Darwin 7.2.0, Python 2.3
nt        win32         Windows Me
                        Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
                        Python 2.1.1 Idle 0.8
nt        win32         Windows 98, Python 2.1.1. Idle 0.8
nt        win32         Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
                        fail i.e. redefining environment variables may
                        not work. FIXED: don't use cygwin echo!
                        Comment: also `cmd /c echo` will not work
                        but redefining environment variables do work.
posix     cygwin        Cygwin 98-4.10, Python 2.3.3(cygming special)
nt        win32         Windows XP, Python 2.3.3
========  ============  =================================================

Known bugs:

* Tests, that send messages to stderr, fail when executed from MSYS prompt
  because the messages are lost at some point.

"""
__all__ = ['exec_command', 'find_executable']

import os
import sys
import subprocess
import locale
import warnings

from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log

def filepath_from_subprocess_output(output):
    """
    Convert `bytes` in the encoding used by a subprocess into a filesystem-appropriate `str`.

    Inherited from `exec_command`, and possibly incorrect.
    """
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    output = output.decode(mylocale, errors='replace')
    output = output.replace('\r\n', '\n')
    # Another historical oddity
    if output[-1:] == '\n':
        output = output[:-1]
    return output


def forward_bytes_to_stdout(val):
    """
    Forward bytes from a subprocess call to the console, without attempting to
    decode them.

    The assumption is that the subprocess call already returned bytes in
    a suitable encoding.
    """
    if hasattr(sys.stdout, 'buffer'):
        # use the underlying binary output if there is one
        sys.stdout.buffer.write(val)
    elif hasattr(sys.stdout, 'encoding'):
        # round-trip the encoding if necessary
        sys.stdout.write(val.decode(sys.stdout.encoding))
    else:
        # make a best-guess at the encoding
        sys.stdout.write(val.decode('utf8', errors='replace'))


def temp_file_name():
    # 2019-01-30, 1.17
    warnings.warn('temp_file_name is deprecated since NumPy v1.17, use '
                  'tempfile.mkstemp instead', DeprecationWarning, stacklevel=1)
    fo, name = make_temp_file()
    fo.close()
    return name

def get_pythonexe():
    pythonexe = sys.executable
    if os.name in ['nt', 'dos']:
        fdir, fn = os.path.split(pythonexe)
        fn = fn.upper().replace('PYTHONW', 'PYTHON')
        pythonexe = os.path.join(fdir, fn)
        assert os.path.isfile(pythonexe), '%r is not a file' % (pythonexe,)
    return pythonexe

def find_executable(exe, path=None, _cache={}):
    """Return full path of a executable or None.

    Symbolic links are not followed.
    """
    key = exe, path
    try:
        return _cache[key]
    except KeyError:
        pass
    log.debug('find_executable(%r)' % exe)
    orig_exe = exe

    if path is None:
        path = os.environ.get('PATH', os.defpath)
    if os.name=='posix':
        realpath = os.path.realpath
    else:
        realpath = lambda a:a

    if exe.startswith('"'):
        exe = exe[1:-1]

    suffixes = ['']
    if os.name in ['nt', 'dos', 'os2']:
        fn, ext = os.path.splitext(exe)
        extra_suffixes = ['.exe', '.com', '.bat']
        if ext.lower() not in extra_suffixes:
            suffixes = extra_suffixes

    if os.path.isabs(exe):
        paths = ['']
    else:
        paths = [ os.path.abspath(p) for p in path.split(os.pathsep) ]

    for path in paths:
        fn = os.path.join(path, exe)
        for s in suffixes:
            f_ext = fn+s
            if not os.path.islink(f_ext):
                f_ext = realpath(f_ext)
            if os.path.isfile(f_ext) and os.access(f_ext, os.X_OK):
                log.info('Found executable %s' % f_ext)
                _cache[key] = f_ext
                return f_ext

    log.warn('Could not locate executable %s' % orig_exe)
    return None

############################################################

def _preserve_environment( names ):
    log.debug('_preserve_environment(%r)' % (names))
    env = {name: os.environ.get(name) for name in names}
    return env

def _update_environment( **env ):
    log.debug('_update_environment(...)')
    for name, value in env.items():
        os.environ[name] = value or ''

def exec_command(command, execute_in='', use_shell=None, use_tee=None,
                 _with_python = 1, **env ):
    """
    Return (status,output) of executed command.

    .. deprecated:: 1.17
        Use subprocess.Popen instead

    Parameters
    ----------
    command : str
        A concatenated string of executable and arguments.
    execute_in : str
        Before running command ``cd execute_in`` and after ``cd -``.
    use_shell : {bool, None}, optional
        If True, execute ``sh -c command``. Default None (True)
    use_tee : {bool, None}, optional
        If True use tee. Default None (True)


    Returns
    -------
    res : str
        Both stdout and stderr messages.

    Notes
    -----
    On NT, DOS systems the returned status is correct for external commands.
    Wild cards will not work for non-posix systems or when use_shell=0.

    """
    # 2019-01-30, 1.17
    warnings.warn('exec_command is deprecated since NumPy v1.17, use '
                  'subprocess.Popen instead', DeprecationWarning, stacklevel=1)
    log.debug('exec_command(%r,%s)' % (command,
         ','.join(['%s=%r'%kv for kv in env.items()])))

    if use_tee is None:
        use_tee = os.name=='posix'
    if use_shell is None:
        use_shell = os.name=='posix'
    execute_in = os.path.abspath(execute_in)
    oldcwd = os.path.abspath(os.getcwd())

    if __name__[-12:] == 'exec_command':
        exec_dir = os.path.dirname(os.path.abspath(__file__))
    elif os.path.isfile('exec_command.py'):
        exec_dir = os.path.abspath('.')
    else:
        exec_dir = os.path.abspath(sys.argv[0])
        if os.path.isfile(exec_dir):
            exec_dir = os.path.dirname(exec_dir)

    if oldcwd!=execute_in:
        os.chdir(execute_in)
        log.debug('New cwd: %s' % execute_in)
    else:
        log.debug('Retaining cwd: %s' % oldcwd)

    oldenv = _preserve_environment( list(env.keys()) )
    _update_environment( **env )

    try:
        st = _exec_command(command,
                           use_shell=use_shell,
                           use_tee=use_tee,
                           **env)
    finally:
        if oldcwd!=execute_in:
            os.chdir(oldcwd)
            log.debug('Restored cwd to %s' % oldcwd)
        _update_environment(**oldenv)

    return st


def _exec_command(command, use_shell=None, use_tee = None, **env):
    """
    Internal workhorse for exec_command().
    """
    if use_shell is None:
        use_shell = os.name=='posix'
    if use_tee is None:
        use_tee = os.name=='posix'

    if os.name == 'posix' and use_shell:
        # On POSIX, subprocess always uses /bin/sh, override
        sh = os.environ.get('SHELL', '/bin/sh')
        if is_sequence(command):
            command = [sh, '-c', ' '.join(command)]
        else:
            command = [sh, '-c', command]
        use_shell = False

    elif os.name == 'nt' and is_sequence(command):
        # On Windows, join the string for CreateProcess() ourselves as
        # subprocess does it a bit differently
        command = ' '.join(_quote_arg(arg) for arg in command)

    # Inherit environment by default
    env = env or None
    try:
        # text is set to False so that communicate()
        # will return bytes. We need to decode the output ourselves
        # so that Python will not raise a UnicodeDecodeError when
        # it encounters an invalid character; rather, we simply replace it
        proc = subprocess.Popen(command, shell=use_shell, env=env, text=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    except OSError:
        # Return 127, as os.spawn*() and /bin/sh do
        return 127, ''

    text, err = proc.communicate()
    mylocale = locale.getpreferredencoding(False)
    if mylocale is None:
        mylocale = 'ascii'
    text = text.decode(mylocale, errors='replace')
    text = text.replace('\r\n', '\n')
    # Another historical oddity
    if text[-1:] == '\n':
        text = text[:-1]

    if use_tee and text:
        print(text)
    return proc.returncode, text


def _quote_arg(arg):
    """
    Quote the argument for safe use in a shell command line.
    """
    # If there is a quote in the string, assume relevants parts of the
    # string are already quoted (e.g. '-I"C:\\Program Files\\..."')
    if '"' not in arg and ' ' in arg:
        return '"%s"' % arg
    return arg

############################################################
