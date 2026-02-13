import os
import subprocess
import sys

directory = os.path.dirname(os.path.realpath(__file__))

BACKEND_DIR = "."
# autogpt_libs merged into backend (OPEN-2998)
TARGET_DIRS = [BACKEND_DIR]


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    try:
        subprocess.run(
            ["poetry", "run"] + list(command),
            cwd=directory,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8"), file=sys.stderr)
        raise


def lint():
    # Generate Prisma types stub before running pyright to prevent type budget exhaustion
    run("gen-prisma-stub")

    lint_step_args: list[list[str]] = [
        ["ruff", "check", *TARGET_DIRS, "--exit-zero"],
        # NOTE: ruff format check removed - ruff and black disagree on assert formatting
        # Black is the source of truth for formatting (runs last in format())
        ["isort", "--diff", "--check", "--profile", "black", BACKEND_DIR],
        ["black", "--diff", "--check", BACKEND_DIR],
        ["pyright", *TARGET_DIRS],
    ]
    lint_error = None
    for args in lint_step_args:
        try:
            run(*args)
        except subprocess.CalledProcessError as e:
            lint_error = e

    if lint_error:
        print("Lint failed, try running `poetry run format` to fix the issues")
        sys.exit(1)


def format():
    run("ruff", "check", "--fix", *TARGET_DIRS)
    run("ruff", "format", BACKEND_DIR)
    run("isort", "--profile", "black", BACKEND_DIR)
    run("black", BACKEND_DIR)
    # Generate Prisma types stub before running pyright to prevent type budget exhaustion
    run("gen-prisma-stub")
    run("pyright", *TARGET_DIRS)
