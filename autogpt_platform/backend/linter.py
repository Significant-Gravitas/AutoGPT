import os
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))

BACKEND_DIR = "."
LIBS_DIR = "../autogpt_libs"
TARGET_DIRS = [BACKEND_DIR, LIBS_DIR]


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    subprocess.run(["poetry", "run"] + list(command), cwd=directory, check=True)


def lint():
    try:
        run("ruff", "check", *TARGET_DIRS, "--exit-zero")
        run("ruff", "format", "--diff", "--check", LIBS_DIR)
        run("isort", "--diff", "--check", "--profile", "black", BACKEND_DIR)
        run("black", "--diff", "--check", BACKEND_DIR)
        run("pyright", *TARGET_DIRS)
    except subprocess.CalledProcessError as e:
        print("Lint failed, try running `poetry run format` to fix the issues: ", e)
        raise e


def format():
    run("ruff", "check", "--fix", *TARGET_DIRS)
    run("ruff", "format", LIBS_DIR)
    run("isort", "--profile", "black", BACKEND_DIR)
    run("black", BACKEND_DIR)
    run("pyright", *TARGET_DIRS)
