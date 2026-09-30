"""The files a shell command runs directly (``bash x.sh``, ``python3 x.py``,
``source x``, ``./x``), so the supervisor judges a script by what it does rather
than by the line that starts it. A script run through ``make``, an npm script,
a pipe, ``eval`` or another script is not found.
"""

import posixpath
import re
import shlex

from backend.copilot.context import E2B_WORKDIR

_SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "source", "."}
_PYTHON = re.compile(r"python(\d+(\.\d+)?)?")
# Options whose argument is inline code or a module: nothing on disk is run.
_INLINE = {
    "python": {"-c", "-m"},
    "node": {"-e", "-p", "--eval", "--print"},
    "tsx": {"-e", "-p", "--eval", "--print"},
    "ts-node": {"-e", "-p", "--eval", "--print"},
    "ruby": {"-e"},
    "perl": {"-e", "-E"},
    "php": {"-r"},
}
_PREFIXES = {"sudo", "env", "nohup", "time", "exec", "command", "timeout"}
_OPERATORS = {"&&", "||", ";", "|", "&", ";;", "|&"}
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_DURATION = re.compile(r"\d+(\.\d+)?[smhd]?")


def executed_paths(command: str) -> list[str]:
    """Absolute paths of the files ``command`` runs directly."""
    paths: list[str] = []
    cwd: str | None = E2B_WORKDIR
    for words in _simple_commands(command):
        if words[0] == "cd":
            cwd = _resolve(cwd, words[1]) if len(words) > 1 else E2B_WORKDIR
            continue
        path = _run_target(words)
        resolved = _resolve(cwd, path) if path else None
        if resolved and resolved not in paths:
            paths.append(resolved)
    return paths


def _simple_commands(command: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in command.splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            # An unbalanced quote, often prose in a heredoc body: words are enough.
            tokens = line.split()
        current: list[str] = []
        for token in tokens:
            if token in _OPERATORS:
                commands.append(current)
                current = []
            else:
                current.append(token)
        commands.append(current)
    return [words for words in commands if words]


def _run_target(words: list[str]) -> str | None:
    """The script a simple command runs, or None when it runs none from disk."""
    while words and (
        _ASSIGNMENT.match(words[0])
        or words[0] in _PREFIXES
        or words[0].startswith("-")
        or _DURATION.fullmatch(words[0])
    ):
        words = words[1:]
    if not words:
        return None
    program = posixpath.basename(words[0])
    if _PYTHON.fullmatch(program):
        program = "python"
    if program in _SHELLS or program in _INLINE:
        for index, word in enumerate(words[1:], start=1):
            if word in _INLINE.get(program, ()) or _shell_inline(program, word):
                return None
            if word == "<":
                return words[index + 1] if index + 1 < len(words) else None
            if not word.startswith("-"):
                return word
        return None
    # A program named by path is itself the file that runs.
    return words[0] if "/" in words[0] else None


def _shell_inline(program: str, word: str) -> bool:
    # `bash -c`, and combined short flags such as `bash -ec`.
    return (
        program in _SHELLS and re.fullmatch(r"-[a-zA-Z]*c[a-zA-Z]*", word) is not None
    )


def _resolve(cwd: str | None, word: str) -> str | None:
    """None where the shell would read the word differently than a literal path."""
    word = re.sub(r"^(\$HOME|\$\{HOME\}|~)(?=/|$)", E2B_WORKDIR, word)
    if re.search(r"[$`*?\[{]", word) or word.startswith(("~", "-")):
        return None
    if not word.startswith("/") and cwd is None:
        return None
    return posixpath.normpath(posixpath.join(cwd or "/", word))
