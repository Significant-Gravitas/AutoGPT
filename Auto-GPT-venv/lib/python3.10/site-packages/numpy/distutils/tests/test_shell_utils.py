import pytest
import subprocess
import json
import sys

from numpy.distutils import _shell_utils
from numpy.testing import IS_WASM

argv_cases = [
    [r'exe'],
    [r'path/exe'],
    [r'path\exe'],
    [r'\\server\path\exe'],
    [r'path to/exe'],
    [r'path to\exe'],

    [r'exe', '--flag'],
    [r'path/exe', '--flag'],
    [r'path\exe', '--flag'],
    [r'path to/exe', '--flag'],
    [r'path to\exe', '--flag'],

    # flags containing literal quotes in their name
    [r'path to/exe', '--flag-"quoted"'],
    [r'path to\exe', '--flag-"quoted"'],
    [r'path to/exe', '"--flag-quoted"'],
    [r'path to\exe', '"--flag-quoted"'],
]


@pytest.fixture(params=[
    _shell_utils.WindowsParser,
    _shell_utils.PosixParser
])
def Parser(request):
    return request.param


@pytest.fixture
def runner(Parser):
    if Parser != _shell_utils.NativeParser:
        pytest.skip('Unable to run with non-native parser')

    if Parser == _shell_utils.WindowsParser:
        return lambda cmd: subprocess.check_output(cmd)
    elif Parser == _shell_utils.PosixParser:
        # posix has no non-shell string parsing
        return lambda cmd: subprocess.check_output(cmd, shell=True)
    else:
        raise NotImplementedError


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.parametrize('argv', argv_cases)
def test_join_matches_subprocess(Parser, runner, argv):
    """
    Test that join produces strings understood by subprocess
    """
    # invoke python to return its arguments as json
    cmd = [
        sys.executable, '-c',
        'import json, sys; print(json.dumps(sys.argv[1:]))'
    ]
    joined = Parser.join(cmd + argv)
    json_out = runner(joined).decode()
    assert json.loads(json_out) == argv


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.mark.parametrize('argv', argv_cases)
def test_roundtrip(Parser, argv):
    """
    Test that split is the inverse operation of join
    """
    try:
        joined = Parser.join(argv)
        assert argv == Parser.split(joined)
    except NotImplementedError:
        pytest.skip("Not implemented")
