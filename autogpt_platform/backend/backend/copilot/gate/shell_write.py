"""A shell command that only writes one file into the workspace, told apart
from every other command so the gate can treat it as the file write it is.

Recognised, with nothing else in the command:

    [cd <dir> &&] cat > <path> << 'TAG'  (also >>, <<TAG, <<-TAG, "TAG")
    [cd <dir> &&] tee [-a] <path> << 'TAG'
    [cd <dir> &&] printf|echo '<text>' > <path>  (also >>)

The body must end at the first terminator line, and an unquoted terminator
only passes when the body has nothing the shell would expand. Anything else,
a path outside ``WORKSPACE_PATH``/``SHARED_PATH`` or through ``..`` included, is
not a write. Symlinks live on the sandbox's disk, so the caller checks those.
"""

import posixpath
import re

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.context import E2B_WORKDIR

# A literal path: quoted with nothing the quotes still expand, or bare with no
# glob, expansion, operator or quote character.
_WORD = r"""(?:'[^'\n]*'|"[^"$`\\\n]*"|[\w./~+,@%:=-]+)"""
# Bash splits words on a space or a tab only; a newline ends the command, and
# ``\s`` would also accept that and the characters bash keeps inside a word.
_GAP = r"[ \t]"
_CD = rf"(?:cd{_GAP}+(?P<dir>{_WORD}){_GAP}*&&{_GAP}*)?"
_HEREDOC = re.compile(
    rf"{_CD}(?:cat{_GAP}*>>?{_GAP}*(?P<cat>{_WORD})"
    rf"|tee{_GAP}+(?:-a{_GAP}+)?(?P<tee>{_WORD}))"
    rf"{_GAP}*<<(?P<strip>-?){_GAP}*(?P<q>['\"]?)(?P<tag>\w+)(?P=q)"
)
_PRINT = re.compile(
    rf"{_CD}(?:printf|echo){_GAP}+'[^']*'{_GAP}*>>?{_GAP}*(?P<path>{_WORD})"
)
_EXPANDS = re.compile(r"[$`\\]")
_ROOTS = (WORKSPACE_PATH + "/", SHARED_PATH + "/")


def workspace_write_target(command: str) -> str | None:
    """The absolute path ``command`` writes, or None if it does anything else."""
    # Only what bash itself skips: ``str.strip`` also drops word characters,
    # and trailing blanks on a heredoc's last line change its terminator.
    command = command.lstrip(" \t\n").rstrip("\n")
    head, _, body = command.partition("\n")
    heredoc = _HEREDOC.fullmatch(head.rstrip(" \t"))
    if heredoc is not None:
        if not _body_ends_at_terminator(body, heredoc):
            return None
        return _in_workspace(heredoc["dir"], heredoc["cat"] or heredoc["tee"])
    printed = _PRINT.fullmatch(command.rstrip(" \t"))
    if printed is not None:
        return _in_workspace(printed["dir"], printed["path"])
    return None


def _body_ends_at_terminator(body: str, heredoc: re.Match[str]) -> bool:
    """The shell runs whatever follows the first terminator line as commands."""
    lines = body.split("\n")
    strip = "\t" if heredoc["strip"] else ""
    ends = [i for i, line in enumerate(lines) if line.lstrip(strip) == heredoc["tag"]]
    if not ends or ends[0] != len(lines) - 1:
        return False
    return bool(heredoc["q"]) or not _EXPANDS.search("\n".join(lines[:-1]))


def _in_workspace(directory: str | None, path: str) -> str | None:
    base = _resolve(E2B_WORKDIR, directory) if directory else E2B_WORKDIR
    target = _resolve(base, path) if base else None
    return target if target and target.startswith(_ROOTS) else None


def _resolve(base: str, word: str) -> str | None:
    """None where the shell would read the word differently: ``~user``, ``cd -``,
    and ``..``, which the kernel takes from wherever a symlink before it points."""
    if word[:1] in "'\"":
        word = word[1:-1]
    elif word == "~" or word.startswith("~/"):
        word = E2B_WORKDIR + word[1:]
    elif word.startswith(("~", "-")):
        return None
    if ".." in word.split("/"):
        return None
    return posixpath.normpath(posixpath.join(base, word))
