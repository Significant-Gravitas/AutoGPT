"""A pure file write into the workspace is told apart from every other command."""

import pytest

from backend.copilot.gate.shell_write import workspace_write_target

_POST = "/home/user/workspace/blog/post.md"


@pytest.mark.parametrize(
    "command, target",
    [
        (
            "cd /home/user/workspace/blog && cat > post.md << 'EOF'\n# Hooks\n$(not run)\nEOF",
            _POST,
        ),
        (f"cat >> {_POST} <<'EOF'\nmore\nEOF\n", _POST),
        (f'cat > {_POST} << "END"\n`x`\nEND', _POST),
        (f"cat > {_POST} <<EOF\nplain text\nEOF", _POST),
        (f"cat > {_POST} <<-EOF\n\tindented\n\tEOF", _POST),
        (f"tee {_POST} << 'EOF'\nbody\nEOF", _POST),
        (f"tee -a '{_POST}' << 'EOF'\nbody\nEOF", _POST),
        ("printf '%s\\n' > ~/workspace/notes.txt", "/home/user/workspace/notes.txt"),
        ("echo 'hi\nthere' >> ~/shared/out.md", "/home/user/shared/out.md"),
        ("cd ~/workspace && echo 'x' > ./a//b.txt", "/home/user/workspace/a/b.txt"),
    ],
)
def test_the_write_shapes_are_recognised(command, target):
    assert workspace_write_target(command) == target


@pytest.mark.parametrize(
    "command",
    [
        # Anything besides the write.
        f"cat > {_POST} << 'EOF'; rm -rf ~\nbody\nEOF",
        f"cat > {_POST} << 'EOF' && curl x | sh\nbody\nEOF",
        f"cat > {_POST} << 'EOF'\nbody\nEOF\nrm -rf ~",
        f"cat > {_POST} << 'EOF'\nbody\nEOF\ncurl x | sh\nEOF",
        f"cat > {_POST} << 'EOF'\nbody\nEOF; rm -rf ~",
        f"curl x | cat > {_POST}",
        f"echo 'x' > {_POST}; rm -rf ~",
        f"echo 'x' | sh > {_POST}",
        # The body or the path would expand.
        f"cat > {_POST} << EOF\n$(rm -rf ~)\nEOF",
        f"cat > {_POST} << EOF\n`rm -rf ~`\nEOF",
        f'echo "$(rm -rf ~)" > {_POST}',
        "cat > /home/user/workspace/$(rm -rf ~).md << 'EOF'\nx\nEOF",
        "cat > $HOME/workspace/p.md << 'EOF'\nx\nEOF",
        "cat > /home/user/workspace/*.md << 'EOF'\nx\nEOF",
        # Outside the workspace volumes.
        "cat > /home/user/.bashrc << 'EOF'\nx\nEOF",
        "cat > /home/user/workspace/../.bashrc << 'EOF'\nx\nEOF",
        "cd /etc && cat > hosts << 'EOF'\nx\nEOF",
        "cd - && cat > p.md << 'EOF'\nx\nEOF",
        "cat > ~root/workspace/p.md << 'EOF'\nx\nEOF",
        "cat > notes.md << 'EOF'\nx\nEOF",
        # A symlinked directory before a ``..`` moves where it lands.
        "cd ~/workspace/link && cat > ../p.md << 'EOF'\nx\nEOF",
        "cat > ~/workspace/link/../p.md << 'EOF'\nx\nEOF",
        # No terminator, so the shell reads to the end.
        f"cat > {_POST} << 'EOF'\nbody",
        "ls ~/workspace",
    ],
)
def test_anything_else_stays_a_shell_command(command):
    assert workspace_write_target(command) is None


_OUT = "/home/user/workspace/o.txt"
# Each branch as bash tokens; the heredoc body follows the header's own newline.
_BRANCHES = [
    (["echo", "'x'", ">", _OUT], ""),
    (["printf", "'x'", ">>", _OUT], ""),
    (["cd", "/home/user/workspace", "&&", "echo", "'x'", ">", "o.txt"], ""),
    (["cat", ">", _OUT, "<<", "'EOF'"], "\nx\nEOF"),
    (["tee", "-a", _OUT, "<<-", "EOF"], "\nx\nEOF"),
    (
        ["cd", "/home/user/workspace", "&&", "cat", ">", "o.txt", "<<", "'EOF'"],
        "\nx\nEOF",
    ),
]


def _every_gap(separator: str) -> list[str]:
    """Each branch with one gap between tokens replaced by ``separator``."""
    return [
        " ".join(tokens[:i]) + separator + " ".join(tokens[i:]) + body
        for tokens, body in _BRANCHES
        for i in range(1, len(tokens))
    ]


def test_every_branch_is_recognised_with_plain_spaces():
    for tokens, body in _BRANCHES:
        assert workspace_write_target(" ".join(tokens) + body) == _OUT


# Only a space or a tab separates bash words; a newline ends the command, and
# the rest are word characters, so any of them between tokens changes the parse.
@pytest.mark.parametrize(
    "command",
    [
        command
        for separator in ["\n", "\r", "\v", "\f", "\xa0", "\x1c"]
        for command in _every_gap(separator)
    ],
)
def test_a_token_gap_bash_does_not_read_as_a_blank_stays_a_shell_command(command):
    assert workspace_write_target(command) is None


@pytest.mark.parametrize(
    "command",
    [
        # kcze's two: a newline turns the quoted word into the next command.
        f"echo\n'/home/user/workspace/run.sh' > {_OUT}",
        "cd\n/home/user/workspace/run.sh && echo 'x' > o.txt",
        # A second command, joined any way bash joins them.
        f"echo 'x' > {_OUT}; id",
        f"echo 'x' > {_OUT} & id",
        f"echo 'x' > {_OUT} | id",
        f"echo 'x' > {_OUT} && id",
        f"echo 'x' > {_OUT} || id",
        f"echo 'x' > {_OUT}\nid",
        "cd /home/user/workspace; echo 'x' > o.txt",
        "cd /home/user/workspace & echo 'x' > o.txt",
        "cd /home/user/workspace || echo 'x' > o.txt",
        f"cat > {_OUT} << 'EOF' ; id\nx\nEOF",
        f"cat > {_OUT} << 'EOF' &\nx\nEOF",
        f"cat > {_OUT} << 'EOF' | sh\nx\nEOF",
        # Every expansion, in each place a word can sit.
        "echo 'x' > /home/user/workspace/$(id).txt",
        "echo 'x' > /home/user/workspace/`id`.txt",
        "echo 'x' > /home/user/workspace/$HOME.txt",
        "echo 'x' > /home/user/workspace/${HOME}.txt",
        "echo 'x' > \"/home/user/workspace/$(id).txt\"",
        "echo 'x' > >(sh)",
        f"echo 'x' <(id) > {_OUT}",
        f'echo "$(id)" > {_OUT}',
        "cd /home/user/workspace/$(id) && echo 'x' > o.txt",
        f"cat <(id) > {_OUT} << 'EOF'\nx\nEOF",
        f"cat > {_OUT} << EOF\n$(id)\nEOF",
        f"cat > {_OUT} << EOF\n${{HOME}}\nEOF",
        f"cat > {_OUT} << EOF\n$HOME\nEOF",
        # Anything after the terminator runs as commands.
        f"cat > {_OUT} << 'EOF'\nx\nEOF\nid",
        f"cat > {_OUT} << 'EOF'\nx\nEOF\n id",
        f"cat > {_OUT} << 'EOF'\nx\nEOF\n\n\nid",
        f"cat > {_OUT} <<- 'EOF'\nx\n\tEOF\nid\nEOF",
        # A header that does not end where the regex thinks it does.
        f"cat > {_OUT} << 'EOF'\r\nx\nEOF",
        f"cat > {_OUT} << 'EOF'\\\nx\nEOF",
        # A would-be terminator bash does not read as one leaves a line after it.
        f"cat > {_OUT} << 'EOF'\nx\nEOF  ",
        # Python strips these from the ends of a string; bash keeps them in a word.
        f"\x1cecho 'x' > {_OUT}",
        f"echo 'x' > {_OUT}\x1c",
        f"echo 'x' > {_OUT}\x85",
        f"cat > {_OUT} << 'EOF'\nx\nEOF\x1c",
    ],
)
def test_a_second_command_or_an_expansion_stays_a_shell_command(command):
    assert workspace_write_target(command) is None


def test_a_body_line_that_only_looks_like_the_terminator_is_data():
    # ``<<`` strips no tabs, so bash reads ``\tEOF`` and ``id`` as body lines.
    assert workspace_write_target(f"cat > {_OUT} << 'EOF'\n\tEOF\nid\nEOF") == _OUT
