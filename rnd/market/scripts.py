import os
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


def populate_database():
    import glob
    import json
    import pathlib

    import requests

    import market.model

    templates = (
        pathlib.Path(__file__).parent.parent / "autogpt_server" / "graph_templates"
    )

    all_files = glob.glob(str(templates / "*.json"))

    for file in all_files:
        with open(file, "r") as f:
            data = f.read()
            req = market.model.AddAgentRequest(
                graph=json.loads(data),
                author="Populate DB",
                categories=["Pre-Populated"],
                keywords=["test"],
            )
            response = requests.post(
                "http://localhost:8001/api/v1/market/admin/agent", json=req.model_dump()
            )
            print(response.text)


def format():
    run("ruff", "check", "--fix", ".")
    run("isort", "--profile", "black", ".")
    run("black", ".")
    run("pyright", ".")


def app():
    port = os.getenv("PORT", "8015")
    run("uvicorn", "market.app:app", "--reload", "--port", port, "--host", "0.0.0.0")


def setup():
    run("prisma", "generate")
    run("prisma", "migrate", "deploy")
