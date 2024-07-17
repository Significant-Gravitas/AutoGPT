import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    subprocess.run(["poetry", "run"] + list(command), cwd=directory, check=True)


def lint():
    run("ruff", "check", ".", "--exit-zero")
    run("autopep8", "--diff", "--exit-code", "--recursive", ".")
    run("pyright")


def format():
    run("autopep8", "--in-place", "--recursive", ".")
    run("ruff", "check", "--fix", ".")
    run("pyright", ".")
