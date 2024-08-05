import os
import re
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))


def run(*command: str) -> None:
    print(f">>>>> Running poetry run {' '.join(command)}")
    subprocess.run(["poetry", "run"] + list(command), cwd=directory, check=True)


def lint():
    try:
        run("ruff", "check", ".", "--exit-zero")
        run("isort", "--diff", "--check", "--profile", "black", ".")
        run("black", "--diff", "--check", ".")
        run("pyright")
    except subprocess.CalledProcessError as e:
        print("Lint failed, try running `poetry run format` to fix the issues: ", e)
        raise e

    try:
        run("schema_lint")
    except subprocess.CalledProcessError as e:
        print("Lint failed, try running `poetry run schema` to fix the issues: ", e)
        raise e


def format():
    run("ruff", "check", "--fix", ".")
    run("isort", "--profile", "black", ".")
    run("black", ".")
    run("pyright", ".")


def schema():
    file = os.path.join(directory, "schema.prisma")
    text = open(file, "r").read()
    text = re.sub(r'provider\s+=\s+".*"', 'provider = "postgresql"', text, 1)
    text = re.sub(r'url\s+=\s+".*"', 'url = env("DATABASE_URL")', text, 1)
    text = "// THIS FILE IS AUTO-GENERATED, RUN `poetry run schema` TO UPDATE\n" + text
    with open(os.path.join(directory, "postgres", "schema.prisma"), "w") as f:
        f.write(text)

    run("prisma", "format", "--schema", "schema.prisma")
    run("prisma", "format", "--schema", "postgres/schema.prisma")
    run("prisma", "migrate", "dev", "--schema", "schema.prisma")
    run("prisma", "migrate", "dev", "--schema", "postgres/schema.prisma")


def schema_lint():
    def read_schema(path: str) -> list[str]:
        with open(path, "r") as f:
            return [v for v in f.read().splitlines() if not v.startswith("//")]

    sqlite_schema = read_schema(os.path.join(directory, "schema.prisma"))
    postgres_schema = read_schema(os.path.join(directory, "postgres", "schema.prisma"))
    diff = [line.strip() for line in set(sqlite_schema) ^ set(postgres_schema)]

    if line := next((v for v in diff if not v.startswith(("provider", "url"))), None):
        raise Exception(f"schema.prisma & postgres/schema.prisma mismatch: {line}")
