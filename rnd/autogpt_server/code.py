import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    subprocess.run(["poetry", "run"] + list(command), cwd=directory, check=True)


def lint():
    run("ruff", "check", ".", "--exit-zero")
    run("isort", "--diff", "--check", "--profile", "black", ".")
    run("black", "--diff", "--check", ".")
    run("pyright")


def format():
    run("ruff", "check", "--fix", ".")
    run("isort", "--profile", "black", ".")
    run("black", ".")
    run("pyright", ".")
