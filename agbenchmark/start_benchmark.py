import click
import pytest
import json
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option("--category", default=None, help="Specific category to run")
@click.option("--noreg", is_flag=True, help="Skip regression tests")
def start(category, noreg):
    """Start the benchmark tests. If a category flag is is provided, run the categories with that mark."""
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""
    config_file = "agbenchmark/config.json"

    # Check if configuration file exists and is not empty
    if not os.path.exists(config_file) or os.stat(config_file).st_size == 0:
        config = {}

        config["hostname"] = click.prompt(
            "\nPlease enter a new hostname", default="localhost"
        )
        config["port"] = click.prompt("Please enter a new port", default=8080)
        config["workspace"] = click.prompt(
            "Please enter a new workspace path", default="/path/to/workspace"
        )

        with open(config_file, "w") as f:
            json.dump(config, f)
    else:
        # If the configuration file exists and is not empty, load it
        with open(config_file, "r") as f:
            config = json.load(f)

    print("Current configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("Starting benchmark tests...", category)
    pytest_args = ["agbenchmark", "-vs"]
    if category:
        pytest_args.extend(
            ["-m", category]
        )  # run categorys that are of a specific marker
        if noreg:
            pytest_args.extend(
                ["-k", "not regression"]
            )  # run categorys that are of a specific marker but don't include regression categorys
        print(f"Running {'non-regression' + category if noreg else category} categorys")
    else:
        if noreg:
            print("Running all non-regression categorys")
            pytest_args.extend(
                ["-k", "not regression"]
            )  # run categorys that are not regression categorys
        else:
            print("Running all categorys")  # run all categorys

    # Run pytest with the constructed arguments
    pytest.main(pytest_args)


if __name__ == "__main__":
    start()
