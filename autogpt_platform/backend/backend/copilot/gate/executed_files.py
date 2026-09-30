"""The files a shell command runs directly (``bash x.sh``, ``python3 -W ignore x.py``,
``source x``, ``./x``), so the supervisor judges a script by what it does rather
than by the line that starts it. A run whose file cannot be told for certain is
reported as unclear, never guessed, and so is every run in a command with a
subshell, a ``{ }`` group or a command substitution, whose ``cd`` this does not
track. One through ``make``, a pipe, ``eval``, a wrapper not listed here or
another script is not found.
"""

import posixpath
import re
import shlex
from typing import Literal

from pydantic import BaseModel

from backend.copilot.context import E2B_WORKDIR

OptionKind = Literal["flag", "attached", "valued", "inline", "script"]


class RunTargets(BaseModel):
    paths: list[str] = []
    # Runs whose file could not be told: an unknown option or an unresolvable path.
    unclear: list[str] = []


class _Options(BaseModel):
    flags: frozenset[str] = frozenset()
    # Take the next word as their value.
    valued: frozenset[str] = frozenset()
    # The code is in the command itself, or on stdin: no file on disk runs.
    inline: frozenset[str] = frozenset()
    # Name the file that runs as their value (``php -f x.php``).
    script: frozenset[str] = frozenset()
    # Words before the file that are not it (``tsx watch x.ts``).
    subcommands: frozenset[str] = frozenset()


def _opts(
    flags: str = "",
    valued: str = "",
    inline: str = "",
    script: str = "",
    subcommands: str = "",
) -> _Options:
    return _Options(
        flags=frozenset(flags.split()),
        valued=frozenset(valued.split()),
        inline=frozenset(inline.split()),
        script=frozenset(script.split()),
        subcommands=frozenset(subcommands.split()),
    )


_SHELL = _opts(
    flags="-a -b -e -f -h -i -k -l -m -n -p -r -t -u -v -x -B -C -E -H -P -T "
    "--login --norc --noprofile --posix --restricted --verbose --noediting",
    valued="-o +o -O +O --rcfile --init-file",
    inline="-c -s",
)
_NODE = _opts(
    flags="--inspect --inspect-brk --no-warnings --no-deprecation --enable-source-maps "
    "--trace-warnings --trace-uncaught --throw-deprecation --preserve-symlinks "
    "--expose-gc --abort-on-uncaught-exception --watch --no-addons --check -c "
    "--transpile-only -T --files --swc --esm --skip-project --no-cache",
    valued="-r --require --import --loader --experimental-loader --inspect-port "
    "--title --env-file --conditions -C --input-type --tsconfig -P --project -O "
    "--compiler-options --dir --cwd",
    inline="-e --eval -p --print -i --interactive",
)
_INTERPRETERS = {
    **dict.fromkeys(("bash", "sh", "zsh", "dash", "ksh"), _SHELL),
    "node": _NODE.model_copy(update={"subcommands": frozenset({"inspect"})}),
    "tsx": _NODE.model_copy(update={"subcommands": frozenset({"watch"})}),
    "ts-node": _NODE,
    "python": _opts(
        flags="-b -B -d -E -I -i -O -OO -P -q -s -S -u -v -x",
        valued="-W -X --check-hash-based-pycs",
        inline="-c -m -",
    ),
    "ruby": _opts(
        flags="-a -c -d -l -n -p -s -v -w -W -y --verbose",
        valued="-r -I -C -E -F",
        inline="-e",
    ),
    "perl": _opts(
        flags="-a -c -i -l -n -p -s -t -T -u -U -v -w -W -X -0",
        valued="-I -M -m -D",
        inline="-e -E",
    ),
    "php": _opts(
        flags="-a -e -h -H -i -l -m -n -q -s -v",
        valued="-c -d -z",
        inline="-r -R -B -E",
        script="-f -F",
    ),
    # The file comes first, with no options.
    "source": _opts(),
    ".": _opts(),
}
# Words that run the rest of the command, with the options each takes.
_PREFIXES = {
    "sudo": _opts(
        flags="-E -H -n -S -b -i -s -P -k", valued="-u -g -C -D -h -p -r -t -U -T"
    ),
    "env": _opts(flags="-i -0 -v --ignore-environment", valued="-u --unset"),
    "nohup": _opts(),
    "time": _opts(flags="-p -v -a -q", valued="-o -f"),
    "exec": _opts(flags="-c -l", valued="-a"),
    # `command -v x` prints where x is and runs nothing.
    "command": _opts(flags="-p", inline="-v -V"),
    "nice": _opts(valued="-n --adjustment"),
    "stdbuf": _opts(valued="-i -o -e"),
    "setsid": _opts(flags="-f -w -c"),
    "timeout": _opts(
        flags="--preserve-status --foreground -v", valued="-s --signal -k --kill-after"
    ),
}
_PYTHON_NAME = re.compile(r"python(\d+(\.\d+)?)?")
_PUNCTUATION = set("();<>|&")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC = re.compile(r"<<(?P<strip>-?)\s*['\"]?(?P<tag>\w+)['\"]?")


class _Unclear(Exception):
    pass


def run_targets(command: str) -> RunTargets:
    """The absolute paths of the files ``command`` runs directly, and the runs it
    could not resolve."""
    targets = RunTargets()
    cwd: str | None = E2B_WORKDIR
    for words in _simple_commands(command):
        if words[0] == "cd":
            cwd = _resolve(cwd, words[1]) if len(words) > 1 else E2B_WORKDIR
            continue
        try:
            path = _run_target(words)
        except _Unclear:
            targets.unclear.append(shlex.join(words))
            continue
        if path is None:
            continue
        resolved = _resolve(cwd, path)
        if resolved is None:
            targets.unclear.append(shlex.join(words))
        elif resolved not in targets.paths:
            targets.paths.append(resolved)
    if targets.paths and any(_groups(line) for line in _command_lines(command)):
        return RunTargets(unclear=targets.unclear + targets.paths)
    return targets


def _command_lines(command: str) -> list[str]:
    """The lines the shell runs as commands: a heredoc's body is data."""
    lines: list[str] = []
    terminator: str | None = None
    strip_tabs = False
    for line in command.splitlines():
        if terminator is not None:
            if (line.lstrip("\t") if strip_tabs else line) == terminator:
                terminator = None
            continue
        lines.append(line)
        if heredoc := _HEREDOC.search(line):
            terminator, strip_tabs = heredoc["tag"], bool(heredoc["strip"])
    return lines


def _groups(line: str) -> bool:
    """An unquoted subshell, ``{ }`` group or command substitution, where a ``cd``
    may move the working directory or not."""
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote == "'":
            quote = None if char == "'" else quote
        elif char == "\\":
            index += 1
        elif char == "`" or line.startswith("$(", index):
            return True
        elif line.startswith("${", index):
            # A parameter expansion, not a group.
            end = line.find("}", index)
            index = len(line) if end < 0 else end
        elif quote == '"':
            quote = None if char == '"' else quote
        elif char in "'\"":
            quote = char
        elif char in "(){}":
            return True
        index += 1
    return False


def _simple_commands(command: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in _command_lines(command):
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError:
            # An unbalanced quote, often prose in a heredoc body: words are enough.
            tokens = line.split()
        current: list[str] = []
        for token in tokens:
            if _is_separator(token):
                commands.append(current)
                current = []
            else:
                current.append(token)
        commands.append(current)
    return [words for words in commands if words]


def _is_separator(token: str) -> bool:
    # shlex fuses adjacent punctuation (`);`, `)&&`), so match any such run; a
    # redirect such as `&>` or `>&` is not one.
    if not token or not set(token) <= _PUNCTUATION or _is_redirect(token):
        return False
    return bool(set(token) & set(";()|")) or token in ("&", "&&")


def _is_redirect(token: str) -> bool:
    return set(token) <= _PUNCTUATION and bool(set(token) & set("<>"))


def _run_target(words: list[str]) -> str | None:
    """The script a simple command runs, None when it runs none from disk; raises
    ``_Unclear`` when its options or redirects leave that uncertain."""
    words = _strip_prefixes(words)
    if not words:
        return None
    program = posixpath.basename(words[0])
    key = "python" if _PYTHON_NAME.fullmatch(program) else program
    options = _INTERPRETERS.get(key)
    if options is None:
        # A program named by path is itself the file that runs.
        return words[0] if "/" in words[0] else None
    script = _script_word(words[1:], options)
    if script is None and "<" in words:
        # No script named, so the code comes in on stdin (`bash < x.sh`).
        after = words.index("<") + 1
        return words[after] if after < len(words) else None
    return script


def _script_word(args: list[str], options: _Options) -> str | None:
    index = 0
    while index < len(args):
        word, following = args[index], args[index + 1 : index + 2]
        if word == "<" or word.startswith("<<"):
            # stdin feeds the code: the caller reads `<`; a heredoc is in the command.
            return None
        if _is_redirect(word) or (
            word.isdigit() and following and _is_redirect(following[0])
        ):
            raise _Unclear(word)
        if word == "--":
            return following[0] if following else None
        if word in options.inline:
            return None
        if word in options.subcommands:
            index += 1
            continue
        if not word.startswith(("-", "+")):
            return word
        kind = _option_kind(word, options)
        if kind == "inline":
            return None
        if kind == "script":
            return following[0] if following else None
        index += 2 if kind == "valued" else 1
    return None


def _strip_prefixes(words: list[str]) -> list[str]:
    while words:
        if _ASSIGNMENT.match(words[0]):
            words = words[1:]
            continue
        options = _PREFIXES.get(words[0])
        if options is None:
            return words
        prefix, words = words[0], words[1:]
        while words and words[0].startswith("-") and words[0] != "--":
            kind = _option_kind(words[0], options)
            if kind == "inline":
                return []
            words = words[2:] if kind == "valued" else words[1:]
        if words[:1] == ["--"]:
            words = words[1:]
        if prefix == "timeout":
            words = words[1:]  # the duration
    return words


def _option_kind(word: str, options: _Options) -> OptionKind:
    """How an option treats what follows it; raises ``_Unclear`` when it is unknown."""
    name, attached = word.split("=", 1)[0], "=" in word
    if name in options.inline:
        return "inline"
    if name in options.script:
        return "script"
    if name in options.valued:
        return "attached" if attached else "valued"
    if name in options.flags or (attached and word.startswith("--")):
        return "flag"
    if not word.startswith("--") and len(word) > 2:
        return _cluster_kind(word, options)
    raise _Unclear(word)


def _cluster_kind(word: str, options: _Options) -> OptionKind:
    """Combined short options (``-euo``) or one with its value attached (``-Wignore``)."""
    head = word[:2]
    if head in options.inline:
        return "inline"
    if head in options.valued:
        return "attached"
    if head in options.script:
        raise _Unclear(word)
    for index, letter in enumerate(word[1:], start=1):
        short = word[0] + letter
        if short in options.inline:
            return "inline"
        if short in options.valued or short in options.script:
            if index != len(word) - 1:
                raise _Unclear(word)
            return "script" if short in options.script else "valued"
        if short not in options.flags:
            raise _Unclear(word)
    return "flag"


def _resolve(cwd: str | None, word: str) -> str | None:
    """None where the shell would read the word differently than a literal path."""
    word = re.sub(r"^(\$HOME|\$\{HOME\}|~)(?=/|$)", E2B_WORKDIR, word)
    if re.search(r"[$`*?\[{]", word) or word.startswith(("~", "-")):
        return None
    if not word.startswith("/") and cwd is None:
        return None
    return posixpath.normpath(posixpath.join(cwd or "/", word))
