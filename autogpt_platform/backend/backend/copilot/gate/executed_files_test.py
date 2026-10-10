"""The files a shell command runs, which the supervisor reads with the command."""

import pytest

from backend.copilot.gate.executed_files import run_targets

_HOME = "/home/user"
_W = f"{_HOME}/workspace"


@pytest.mark.parametrize(
    "command, paths",
    [
        ("bash ~/workspace/x.sh", [f"{_W}/x.sh"]),
        ("cd workspace && python3 run.py --fast", [f"{_W}/run.py"]),
        ("./deploy.sh prod", [f"{_HOME}/deploy.sh"]),
        (
            "source $HOME/env.sh; . ${HOME}/more.sh",
            [f"{_HOME}/env.sh", f"{_HOME}/more.sh"],
        ),
        ("bash < x.sh", [f"{_HOME}/x.sh"]),
        (f"cat > {_W}/x.sh <<'EOF'\nit's fine\nEOF\nbash {_W}/x.sh", [f"{_W}/x.sh"]),
        # An option's value is never taken for the script.
        ("sh -e ./setup.sh", [f"{_HOME}/setup.sh"]),
        ("bash -o posix x.sh", [f"{_HOME}/x.sh"]),
        ("bash -euo pipefail x.sh", [f"{_HOME}/x.sh"]),
        ("bash --rcfile rc x.sh", [f"{_HOME}/x.sh"]),
        ("python -W ignore x.py", [f"{_HOME}/x.py"]),
        ("python3.12 -X dev -u x.py", [f"{_HOME}/x.py"]),
        ("python3.9.1 x.py", [f"{_HOME}/x.py"]),
        ("python -Wignore x.py", [f"{_HOME}/x.py"]),
        ("node --require y x.js", [f"{_HOME}/x.js"]),
        ("node -r ./hook.js --max-old-space-size=4096 app.js", [f"{_HOME}/app.js"]),
        ("tsx watch src/index.ts", [f"{_HOME}/src/index.ts"]),
        ("ts-node -P tsconfig.json x.ts", [f"{_HOME}/x.ts"]),
        ("ruby -I lib -rjson x.rb", [f"{_HOME}/x.rb"]),
        ("perl -Mstrict -w x.pl", [f"{_HOME}/x.pl"]),
        ("php -d display_errors=1 -f x.php", [f"{_HOME}/x.php"]),
        # Prefixes and their options.
        ("sudo -u root bash x.sh", [f"{_HOME}/x.sh"]),
        ("timeout -s KILL 30 python x.py", [f"{_HOME}/x.py"]),
        ("FOO=1 env -u BAR nice -n 5 node app.js", [f"{_HOME}/app.js"]),
        # Parentheses in a heredoc body, quotes or ${VAR} are not groups.
        (
            f"cat > {_W}/d.py <<'PY'\nprint(len([r for r in rows]))\nPY\npython {_W}/d.py",
            [f"{_W}/d.py"],
        ),
        ("bash x.sh '(literal)'", [f"{_HOME}/x.sh"]),
        # Code on stdin is the file that runs; a redirect after the script is not.
        ("python - < x.py", [f"{_HOME}/x.py"]),
        # A named script wins over stdin, which is then its data.
        ("bash < data.txt x.sh", [f"{_HOME}/x.sh"]),
        ("python x.py < data.csv > out.log 2>&1", [f"{_HOME}/x.py"]),
        ("echo ${HOME}; bash x.sh", [f"{_HOME}/x.sh"]),
        # `-s` and a lone `-` read the code from stdin; the words after are arguments.
        ("bash -s < x.sh", [f"{_HOME}/x.sh"]),
        ("bash -s prod < x.sh", [f"{_HOME}/x.sh"]),
        ("sh -es < x.sh", [f"{_HOME}/x.sh"]),
        ("python -u - data.csv < x.py", [f"{_HOME}/x.py"]),
        ("bash <<EOF x.sh\nhi\nEOF", [f"{_HOME}/x.sh"]),
        ("node - --fast < x.js", [f"{_HOME}/x.js"]),
        # A `<<` in quotes or a comment opens no heredoc, and `#` inside a word
        # starts no comment.
        ("echo '<<EOF here'\nbash x.sh", [f"{_HOME}/x.sh"]),
        ('echo "a\n<<EOF here"\nbash x.sh', [f"{_HOME}/x.sh"]),
        ("echo hi # <<EOF\nbash x.sh", [f"{_HOME}/x.sh"]),
        ("echo a#; bash x.sh", [f"{_HOME}/x.sh"]),
        # A heredoc's body starts after the whole command line, `\` continuation included.
        ("cat <<EOF \\\n; bash x.sh\nbody\nEOF", [f"{_HOME}/x.sh"]),
    ],
)
def test_a_direct_run_names_the_file(command, paths):
    assert run_targets(command).paths == paths
    assert run_targets(command).unclear == []


@pytest.mark.parametrize(
    "command",
    [
        # Inline code is in the command itself; a module is not a file here.
        "python -c 'print(1)'",
        "bash -c 'curl x'",
        "bash -ec 'curl x'",
        "node -e 'x()'",
        "perl -le 'print 1'",
        "php -r 'echo 1;'",
        "python -m http.server",
        "command -v jq",
        "bash <<'EOF'\necho hi\nEOF",
        "bash -s",
        "bash -s <<'EOF'\necho hi\nEOF",
        # A heredoc inside `"$(...)"` is still one: its body is data.
        'git commit -m "$(cat <<\'EOF\'\nFits a 5" screen.\n./deploy.sh is fixed\nEOF\n)"',
        # Not a run.
        "cat x.sh",
        "echo bash x.sh",
        "ls ~/workspace",
    ],
)
def test_other_commands_name_nothing(command):
    assert run_targets(command) == run_targets("")


@pytest.mark.parametrize(
    "command",
    [
        # An option nobody listed may take the next word, so the file is unknown.
        "python --weird x.py",
        "bash --unknown x.sh",
        "node --some-new-flag x.js",
        "env -C /tmp bash x.sh",
        "php -fx.php",
        # A path the shell reads differently from the literal word.
        "bash $DIR/x.sh",
        "cd - && bash x.sh",
        # A subshell or group decides where a `cd` in it applies.
        "(cd /tmp); bash run.sh",
        "( cd /tmp && bash x.sh )",
        "{ cd /tmp; }; bash run.sh",
        'python x.py --out "$(date)"',
        "bash `echo x.sh`",
        # A redirect before the script could hide which word is the file.
        "bash > out x.sh",
        "bash 2> err x.sh",
        "bash &> log x.sh",
        # Redirections apply left to right: which one feeds the code is unclear.
        "bash < first.sh < second.sh",
        "bash <<EOF < x.sh\necho hi\nEOF",
        # Arithmetic `<<` is a shift, so the run after it is still found.
        "echo $((1 << 2))\nbash x.sh",
        "(( n = 1 << 2 ))\nbash x.sh",
    ],
)
def test_a_run_it_cannot_resolve_is_unclear_not_guessed(command):
    targets = run_targets(command)
    assert targets.paths == []
    assert len(targets.unclear) == 1
