import os
import subprocess
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

BACKEND_DIR = "."
LIBS_DIR = "../autogpt_libs"
TARGET_DIRS = [BACKEND_DIR, LIBS_DIR]


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    try:
        subprocess.run(
            ["poetry", "run"] + list(command),
            cwd=backend_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8"), file=sys.stderr)
        raise


def lint():
    skip_pyright = "--skip-pyright" in sys.argv

    if not skip_pyright:
        # Generate Prisma types stub before running pyright to prevent type budget exhaustion
        run("gen-prisma-stub")

    lint_step_args: list[list[str]] = [
        ["ruff", "check", *TARGET_DIRS],
        ["ruff", "format", "--diff", "--check", LIBS_DIR],
        ["isort", "--diff", "--check", "--profile", "black", BACKEND_DIR],
        ["black", "--diff", "--check", BACKEND_DIR],
    ]
    if not skip_pyright:
        lint_step_args.append(["pyright", *TARGET_DIRS])
    lint_error = None
    for args in lint_step_args:
        try:
            run(*args)
        except subprocess.CalledProcessError as e:
            lint_error = e

    if lint_error:
        print(
            "Lint failed. `poetry run format` fixes most of it, but check"
            " `ruff check --fix --diff` first: an import that pytest only"
            " needs as a fixture reads as unused (F401) and the fix deletes"
            " it. Share fixtures through a conftest.py instead."
        )
        sys.exit(1)


def format():
    run("ruff", "check", "--fix", *TARGET_DIRS)
    run("ruff", "format", LIBS_DIR)
    run("isort", "--profile", "black", BACKEND_DIR)
    run("black", BACKEND_DIR)
    # Generate Prisma types stub before running pyright to prevent type budget exhaustion
    run("gen-prisma-stub")
    run("pyright", *TARGET_DIRS)
