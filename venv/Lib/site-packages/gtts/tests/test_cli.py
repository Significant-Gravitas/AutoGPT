# -*- coding: utf-8 -*-
import pytest
import re
import os
from click.testing import CliRunner
from gtts.cli import tts_cli

# Need to look into gTTS' log output to test proper instantiation
# - Use testfixtures.LogCapture() b/c TestCase.assertLogs() needs py3.4+
# - Clear 'gtts' logger handlers (set in gtts.cli) to reduce test noise
import logging
from testfixtures import LogCapture

logger = logging.getLogger("gtts")
logger.handlers = []


"""Test options and arguments"""


def runner(args, input=None):
    return CliRunner().invoke(tts_cli, args, input)


def runner_debug(args, input=None):
    return CliRunner().invoke(tts_cli, args + ["--debug"], input)


# <text> tests
def test_text_no_text_or_file():
    """One of <test> (arg) and <file> <opt> should be set"""
    result = runner_debug([])

    assert "<file> required" in result.output
    assert result.exit_code != 0


def test_text_text_and_file(tmp_path):
    """<test> (arg) and <file> <opt> should not be set together"""
    filename = tmp_path / "test_and_file.txt"
    filename.touch()

    result = runner_debug(["--file", str(filename), "test"])

    assert "<file> can't be used together" in result.output
    assert result.exit_code != 0


def test_text_empty(tmp_path):
    """Exit on no text to speak (via <file>)"""
    filename = tmp_path / "text_empty.txt"
    filename.touch()

    result = runner_debug(["--file", str(filename)])

    assert "No text to speak" in result.output
    assert result.exit_code != 0


# <file> tests
def test_file_not_exists():
    """<file> should exist"""
    result = runner_debug(["--file", "notexist.txt", "test"])

    assert "No such file or directory" in result.output
    assert result.exit_code != 0


# <all> tests
@pytest.mark.net
def test_all():
    """Option <all> should return a list of languages"""
    result = runner(["--all"])

    # One or more of "  xy: name" (\n optional to match the last)
    # Ex. "<start>  xx: xxxxx\n  xx-yy: xxxxx\n  xx: xxxxx<end>"

    assert re.match(r"^(?:\s{2}(\w{2}|\w{2}-\w{2}): .+\n?)+$", result.output)
    assert result.exit_code == 0


# <lang> tests
@pytest.mark.net
def test_lang_not_valid():
    """Invalid <lang> should display an error"""
    result = runner(["--lang", "xx", "test"])

    assert "xx' not in list of supported languages" in result.output
    assert result.exit_code != 0


@pytest.mark.net
def test_lang_nocheck():
    """Invalid <lang> (with <nocheck>) should display an error message from gtts"""
    with LogCapture() as lc:
        result = runner_debug(["--lang", "xx", "--nocheck", "test"])

        log = str(lc)

    assert "lang: xx" in log
    assert "lang_check: False" in log
    assert "Unsupported language 'xx'" in result.output
    assert result.exit_code != 0


# Param set tests
@pytest.mark.net
def test_params_set():
    """Options should set gTTS instance arguments (read from debug log)"""
    with LogCapture() as lc:
        result = runner_debug(
            ["--lang", "fr", "--tld", "es", "--slow", "--nocheck", "test"]
        )

        log = str(lc)

    assert "lang: fr" in log
    assert "tld: es" in log
    assert "lang_check: False" in log
    assert "slow: True" in log
    assert "text: test" in log
    assert result.exit_code == 0


# Test all input methods
pwd = os.path.dirname(__file__)

# Text for stdin ('-' for <text> or <file>)
textstdin = """stdin
test
123"""

# Text for stdin ('-' for <text> or <file>) (Unicode)
textstdin_unicode = u"""你吃饭了吗？
你最喜欢哪部电影？
我饿了，我要去做饭了。"""

# Text for <text> and <file>
text = """Can you make pink a little more pinkish can you make pink a little more pinkish, nor can you make the font bigger?
How much will it cost the website doesn't have the theme i was going for."""

textfile_ascii = os.path.join(pwd, "input_files", "test_cli_test_ascii.txt")

# Text for <text> and <file> (Unicode)
text_unicode = u"""这是一个三岁的小孩
在讲述她从一系列照片里看到的东西。
对这个世界， 她也许还有很多要学的东西，
但在一个重要的任务上， 她已经是专家了：
去理解她所看到的东西。"""

textfile_utf8 = os.path.join(pwd, "input_files", "test_cli_test_utf8.txt")

"""
Method that mimics's LogCapture's __str__ method to make
the string in the comprehension a unicode literal for P2.7
https://github.com/Simplistix/testfixtures/blob/32c87902cb111b7ede5a6abca9b597db551c88ef/testfixtures/logcapture.py#L149
"""


def logcapture_str(lc):
    if not lc.records:
        return "No logging captured"

    return "\n".join([u"%s %s\n  %s" % r for r in lc.actual()])


@pytest.mark.net
def test_stdin_text():
    with LogCapture() as lc:
        result = runner_debug(["-"], textstdin)
        log = logcapture_str(lc)

    assert "text: %s" % textstdin in log
    assert result.exit_code == 0


@pytest.mark.net
def test_stdin_text_unicode():
    with LogCapture() as lc:
        result = runner_debug(["-"], textstdin_unicode)
        log = logcapture_str(lc)

    assert "text: %s" % textstdin_unicode in log
    assert result.exit_code == 0


@pytest.mark.net
def test_stdin_file():
    with LogCapture() as lc:
        result = runner_debug(["--file", "-"], textstdin)
        log = logcapture_str(lc)

    assert "text: %s" % textstdin in log
    assert result.exit_code == 0


@pytest.mark.net
def test_stdin_file_unicode():
    with LogCapture() as lc:
        result = runner_debug(["--file", "-"], textstdin_unicode)
        log = logcapture_str(lc)

    assert "text: %s" % textstdin_unicode in log
    assert result.exit_code == 0


@pytest.mark.net
def test_text():
    with LogCapture() as lc:
        result = runner_debug([text])
        log = logcapture_str(lc)

    assert "text: %s" % text in log
    assert result.exit_code == 0


@pytest.mark.net
def test_text_unicode():
    with LogCapture() as lc:
        result = runner_debug([text_unicode])
        log = logcapture_str(lc)

    assert "text: %s" % text_unicode in log
    assert result.exit_code == 0


@pytest.mark.net
def test_file_ascii():
    with LogCapture() as lc:
        result = runner_debug(["--file", textfile_ascii])
        log = logcapture_str(lc)

    assert "text: %s" % text in log
    assert result.exit_code == 0


@pytest.mark.net
def test_file_utf8():
    with LogCapture() as lc:
        result = runner_debug(["--file", textfile_utf8])
        log = logcapture_str(lc)

    assert "text: %s" % text_unicode in log
    assert result.exit_code == 0


@pytest.mark.net
def test_stdout():
    result = runner(["test"])

    # The MP3 encoding (LAME 3.99.5) used to leave a signature in the raw output
    # This no longer appears to be the case
    assert result.exit_code == 0


@pytest.mark.net
def test_file(tmp_path):
    filename = tmp_path / "out.mp3"

    result = runner(["test", "--output", str(filename)])

    # Check if files created is > 2k
    assert filename.stat().st_size > 2000
    assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main(["-x", __file__])
