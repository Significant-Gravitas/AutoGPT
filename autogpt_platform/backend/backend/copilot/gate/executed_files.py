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

OptionKind = Literal["flag", "attached", "valued", "inline", "stdin", "script"]


class RunTargets(BaseModel):
    paths: list[str] = []
    # Runs whose file could not be told: an unknown option or an unresolvable path.
    unclear: list[str] = []


class _Options(BaseModel):
    flags: frozenset[str] = frozenset()
    # Take the next word as their value.
    valued: frozenset[str] = frozenset()
    # The code is in the command itself: no file on disk runs.
    inline: frozenset[str] = frozenset()
    # Read the code from stdin, so the words after the options are its arguments.
    stdin: frozenset[str] = frozenset()
    # Name the file that runs as their value (``php -f x.php``).
    script: frozenset[str] = frozenset()
    # Words before the file that are not it (``tsx watch x.ts``).
    subcommands: frozenset[str] = frozenset()


def _opts(
    flags: str = "",
    valued: str = "",
    inline: str = "",
    stdin: str = "",
    script: str = "",
    subcommands: str = "",
) -> _Options:
    return _Options(
        flags=frozenset(flags.split()),
        valued=frozenset(valued.split()),
        inline=frozenset(inline.split()),
        stdin=frozenset(stdin.split()),
        script=frozenset(script.split()),
        subcommands=frozenset(subcommands.split()),
    )


_SHELL = _opts(
    flags="-a -b -e -f -h -i -k -l -m -n -p -r -t -u -v -x -B -C -E -H -P -T "
    "--login --norc --noprofile --posix --restricted --verbose --noediting",
    valued="-o +o -O +O --rcfile --init-file",
    inline="-c",
    stdin="-s",
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
    stdin="-",
)
_INTERPRETERS = {
    **dict.fromkeys(("bash", "sh", "zsh", "dash", "ksh"), _SHELL),
    "node": _NODE.model_copy(update={"subcommands": frozenset({"inspect"})}),
    "tsx": _NODE.model_copy(update={"subcommands": frozenset({"watch"})}),
    "ts-node": _NODE,
    "python": _opts(
        flags="-b -B -d -E -I -i -O -OO -P -q -s -S -u -v -x",
        valued="-W -X --check-hash-based-pycs",
        inline="-c -m",
        stdin="-",
    ),
    "ruby": _opts(
        flags="-a -c -d -l -n -p -s -v -w -W -y --verbose",
        valued="-r -I -C -E -F",
        inline="-e",
        stdin="-",
    ),
    "perl": _opts(
        flags="-a -c -i -l -n -p -s -t -T -u -U -v -w -W -X -0",
        valued="-I -M -m -D",
        inline="-e -E",
        stdin="-",
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
_PYTHON_NAME = re.compile(r"python(\d+(\.\d+)*)?")
_PUNCTUATION = set("();<>|&")
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# Where a word ends, so a `#` after one starts a comment.
_BOUNDARY = set(" \t;&|()<>")


class _Unclear(Exception):
    pass


def run_targets(command: str) -> RunTargets:
    """The absolute paths of the files ``command`` runs directly, and the runs it
    could not resolve."""
    targets = RunTargets()
    cwd: str | None = E2B_WORKDIR
    scan = _scan(command)
    for words in _simple_commands(scan.lines):
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
    if targets.paths and scan.grouped:
        return RunTargets(unclear=targets.unclear + targets.paths)
    return targets


class _Frame(BaseModel):
    """An open ``$(``, ``$((`` or ``((``, inside which quoting starts afresh."""

    quote: str | None
    arithmetic: bool
    depth: int = 0


class _Scan(BaseModel):
    # The command lines, each joined across quotes and `\` continuations, with
    # comments and heredoc bodies removed.
    lines: list[str] = []
    # An unquoted subshell, `{ }` group or command substitution.
    grouped: bool = False
    # Where the shell is at the end of the row read so far.
    quote: str | None = None
    frames: list[_Frame] = []
    heredocs: list[tuple[str, bool]] = []
    word_start: bool = True


def _scan(command: str) -> _Scan:
    """Reads the command as the shell does: a ``<<`` in quotes, a comment or
    arithmetic opens no heredoc, and a body starts after the line that ends the
    command it is opened in."""
    scan, rows = _Scan(), command.split("\n")
    line, row = "", 0
    while row < len(rows):
        kept, continued = _scan_row(scan, rows[row])
        line, row = line + kept, row + 1
        if continued:
            continue
        if scan.quote is not None:
            line += "\n"
            continue
        row = _past_bodies(rows, row, scan.heredocs)
        scan.heredocs, scan.word_start = [], True
        scan.lines.append(line)
        line = ""
    if line:
        scan.lines.append(line)
    return scan


def _scan_row(scan: _Scan, text: str) -> tuple[str, bool]:
    """The part of a row the shell reads as command text, and whether a ``\\``
    joins the next row to it."""
    index = 0
    while index < len(text):
        char, boundary = text[index], False
        if scan.quote in ("'", "$'"):
            if char == "\\" and scan.quote == "$'":
                index += 1
            elif char == "'":
                scan.quote = None
        elif char == "\\":
            if index == len(text) - 1:
                return text[:index], True
            index += 1
        elif text.startswith("$((", index) or (
            scan.quote is None and scan.word_start and text.startswith("((", index)
        ):
            scan.grouped = True
            scan.frames.append(_Frame(quote=scan.quote, arithmetic=True))
            scan.quote = None
            index += 2 if char == "$" else 1
        elif text.startswith("$(", index):
            scan.grouped, boundary = True, True
            scan.frames.append(_Frame(quote=scan.quote, arithmetic=False))
            scan.quote = None
            index += 1
        elif char == "`":
            scan.grouped = True
        elif text.startswith("${", index):
            # A parameter expansion, not a group.
            close = text.find("}", index)
            index = len(text) if close < 0 else close
        elif scan.quote == '"':
            scan.quote = None if char == '"' else scan.quote
        elif text.startswith("$'", index):
            scan.quote = "$'"
            index += 1
        elif char in "'\"":
            scan.quote = char
        elif char == "#" and scan.word_start:
            return text[:index], False
        elif text.startswith("<<<", index):
            index += 2
        elif text.startswith("<<", index):
            if not (scan.frames and scan.frames[-1].arithmetic):
                if tag := _heredoc_tag(text[index + 2 :]):
                    scan.heredocs.append(tag)
            index += 1
        elif char == ")" and scan.frames:
            frame = scan.frames[-1]
            if frame.depth:
                frame.depth -= 1
            else:
                scan.quote = scan.frames.pop().quote
                if frame.arithmetic and text.startswith("))", index):
                    index += 1
        elif char in "(){}":
            scan.grouped = True
            if char == "(" and scan.frames:
                scan.frames[-1].depth += 1
        scan.word_start = boundary or (scan.quote is None and char in _BOUNDARY)
        index += 1
    return text, False


def _heredoc_tag(rest: str) -> tuple[str, bool] | None:
    """The delimiter after a ``<<``, unquoted, and whether ``<<-`` strips tabs; None
    where it is not a word, so no body is skipped on a guess."""
    strip_tabs = rest.startswith("-")
    lexer = shlex.shlex(
        rest[1:] if strip_tabs else rest, posix=True, punctuation_chars=True
    )
    lexer.whitespace_split, lexer.commenters = True, ""
    try:
        tag = lexer.get_token()
    except ValueError:
        return None
    if not tag or set(tag) <= _PUNCTUATION:
        return None
    return tag, strip_tabs


def _past_bodies(rows: list[str], row: int, heredocs: list[tuple[str, bool]]) -> int:
    """The row after the bodies of ``heredocs``, which start at ``row``."""
    for tag, strip_tabs in heredocs:
        while (
            row < len(rows)
            and (rows[row].lstrip("\t") if strip_tabs else rows[row]) != tag
        ):
            row += 1
        row += 1
    return row


def _simple_commands(lines: list[str]) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in lines:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        # The scan removed the comments; shlex would also cut at a `#` inside a word.
        lexer.whitespace_split, lexer.commenters = True, ""
        try:
            tokens = list(lexer)
        except ValueError:
            # An unbalanced quote: words are enough.
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
    return _script_word(words[1:], options)


def _script_word(args: list[str], options: _Options) -> str | None:
    """The file an interpreter runs: the one it names, else what feeds its stdin."""
    # None for a heredoc or here-string: that code is in the command itself.
    stdin: list[str | None] = []
    from_stdin = False
    index = 0
    while index < len(args):
        word, following = args[index], args[index + 1 : index + 2]
        if word in ("<", "<<", "<<<"):
            stdin.append(following[0] if word == "<" and following else None)
            index += 2
            continue
        if _is_redirect(word) or (
            word.isdigit() and following and _is_redirect(following[0])
        ):
            raise _Unclear(word)
        if from_stdin:
            index += 1  # An argument to the code on stdin.
            continue
        if word == "--":
            return following[0] if following else None
        if word in options.subcommands:
            index += 1
            continue
        if word == "-" or not word.startswith(("-", "+")):
            if word in options.stdin:
                from_stdin = True
            elif word != "-":
                return word
            index += 1
            continue
        kind = _option_kind(word, options)
        if kind == "inline":
            return None
        if kind == "script":
            return following[0] if following else None
        from_stdin = kind == "stdin"
        index += 2 if kind == "valued" else 1
    # Redirections apply left to right, so two feeding stdin leave it unclear.
    if len(stdin) > 1:
        raise _Unclear("<")
    return stdin[0] if stdin else None


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
    if name in options.stdin:
        return "stdin"
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
    kind: OptionKind = "flag"
    for index, letter in enumerate(word[1:], start=1):
        short = word[0] + letter
        if short in options.inline:
            return "inline"
        if short in options.stdin:
            kind = "stdin"
        elif short in options.valued or short in options.script:
            # One kind cannot carry both `-s` and a value.
            if index != len(word) - 1 or kind == "stdin":
                raise _Unclear(word)
            return "script" if short in options.script else "valued"
        elif short not in options.flags:
            raise _Unclear(word)
    return kind


def _resolve(cwd: str | None, word: str) -> str | None:
    """None where the shell would read the word differently than a literal path."""
    word = re.sub(r"^(\$HOME|\$\{HOME\}|~)(?=/|$)", E2B_WORKDIR, word)
    if re.search(r"[$`*?\[{]", word) or word.startswith(("~", "-")):
        return None
    if not word.startswith("/") and cwd is None:
        return None
    return posixpath.normpath(posixpath.join(cwd or "/", word))
