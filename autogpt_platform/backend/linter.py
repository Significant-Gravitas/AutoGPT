import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))
target_dirs = ["../backend", "../autogpt_libs"]


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    subprocess.run(["poetry", "run"] + list(command), cwd=directory, check=True)


def lint():
    try:
        run("ruff", "check", *target_dirs, "--exit-zero")
        run("isort", "--diff", "--check", "--profile", "black", ".")
        run("black", "--diff", "--check", ".")
        run("pyright", *target_dirs)
    except subprocess.CalledProcessError as e:
        print("Lint failed, try running `poetry run format` to fix the issues: ", e)
        raise e


def format():
    run("ruff", "check", "--fix", *target_dirs)
    run("isort", "--profile", "black", ".")
    run("black", ".")
    run("pyright", *target_dirs)
