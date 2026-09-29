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
        ("cd ~/workspace && echo 'x' > a/../b.txt", "/home/user/workspace/b.txt"),
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
        # No terminator, so the shell reads to the end.
        f"cat > {_POST} << 'EOF'\nbody",
        "ls ~/workspace",
    ],
)
def test_anything_else_stays_a_shell_command(command):
    assert workspace_write_target(command) is None
