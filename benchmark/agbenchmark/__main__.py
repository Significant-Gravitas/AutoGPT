import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
from click_default_group import DefaultGroup
from dotenv import load_dotenv

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.logging import configure_logging

load_dotenv()

# try:
#     if os.getenv("HELICONE_API_KEY"):
#         import helicone  # noqa

#         helicone_enabled = True
#     else:
#         helicone_enabled = False
# except ImportError:
#     helicone_enabled = False


class InvalidInvocationError(ValueError):
    pass


logger = logging.getLogger(__name__)

BENCHMARK_START_TIME_DT = datetime.now(timezone.utc)
BENCHMARK_START_TIME = BENCHMARK_START_TIME_DT.strftime("%Y-%m-%dT%H:%M:%S+00:00")


# if helicone_enabled:
#     from helicone.lock import HeliconeLockManager

#     HeliconeLockManager.write_custom_property(
#         "benchmark_start_time", BENCHMARK_START_TIME
#     )


@click.group(cls=DefaultGroup, default_if_no_args=True)
@click.option("--debug", is_flag=True, help="Enable debug output")
def cli(
    debug: bool,
) -> Any:
    configure_logging(logging.DEBUG if debug else logging.INFO)


@cli.command(hidden=True)
def start():
    raise DeprecationWarning(
        "`agbenchmark start` is deprecated. Use `agbenchmark run` instead."
    )


@cli.command(default=True)
@click.option(
    "-N", "--attempts", default=1, help="Number of times to run each challenge."
)
@click.option(
    "-c",
    "--category",
    multiple=True,
    help="(+) Select a category to run.",
)
@click.option(
    "-s",
    "--skip-category",
    multiple=True,
    help="(+) Exclude a category from running.",
)
@click.option("--test", multiple=True, help="(+) Select a test to run.")
@click.option("--maintain", is_flag=True, help="Run only regression tests.")
@click.option("--improve", is_flag=True, help="Run only non-regression tests.")
@click.option(
    "--explore",
    is_flag=True,
    help="Run only challenges that have never been beaten.",
)
@click.option(
    "--no-dep",
    is_flag=True,
    help="Run all (selected) challenges, regardless of dependency success/failure.",
)
@click.option("--cutoff", type=int, help="Override the challenge time limit (seconds).")
@click.option("--nc", is_flag=True, help="Disable the challenge time limit.")
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option("--keep-answers", is_flag=True, help="Keep answers")
@click.option(
    "--backend",
    is_flag=True,
    help="Write log output to a file instead of the terminal.",
)
# @click.argument(
#     "agent_path", type=click.Path(exists=True, file_okay=False), required=False
# )
def run(
    maintain: bool,
    improve: bool,
    explore: bool,
    mock: bool,
    no_dep: bool,
    nc: bool,
    keep_answers: bool,
    test: tuple[str],
    category: tuple[str],
    skip_category: tuple[str],
    attempts: int,
    cutoff: Optional[int] = None,
    backend: Optional[bool] = False,
    # agent_path: Optional[Path] = None,
) -> None:
    """
    Run the benchmark on the agent in the current directory.

    Options marked with (+) can be specified multiple times, to select multiple items.
    """
    from agbenchmark.main import run_benchmark, validate_args

    agbenchmark_config = AgentBenchmarkConfig.load()
    logger.debug(f"agbenchmark_config: {agbenchmark_config.agbenchmark_config_dir}")
    try:
        validate_args(
            maintain=maintain,
            improve=improve,
            explore=explore,
            tests=test,
            categories=category,
            skip_categories=skip_category,
            no_cutoff=nc,
            cutoff=cutoff,
        )
    except InvalidInvocationError as e:
        logger.error("Error: " + "\n".join(e.args))
        sys.exit(1)

    original_stdout = sys.stdout  # Save the original standard output
    exit_code = None

    if backend:
        with open("backend/backend_stdout.txt", "w") as f:
            sys.stdout = f
            exit_code = run_benchmark(
                config=agbenchmark_config,
                maintain=maintain,
                improve=improve,
                explore=explore,
                mock=mock,
                no_dep=no_dep,
                no_cutoff=nc,
                keep_answers=keep_answers,
                tests=test,
                categories=category,
                skip_categories=skip_category,
                attempts_per_challenge=attempts,
                cutoff=cutoff,
            )

        sys.stdout = original_stdout

    else:
        exit_code = run_benchmark(
            config=agbenchmark_config,
            maintain=maintain,
            improve=improve,
            explore=explore,
            mock=mock,
            no_dep=no_dep,
            no_cutoff=nc,
            keep_answers=keep_answers,
            tests=test,
            categories=category,
            skip_categories=skip_category,
            attempts_per_challenge=attempts,
            cutoff=cutoff,
        )

        sys.exit(exit_code)


@cli.command()
@click.option("--port", type=int, help="Port to run the API on.")
def serve(port: Optional[int] = None):
    """Serve the benchmark frontend and API on port 8080."""
    import uvicorn

    from agbenchmark.app import setup_fastapi_app

    config = AgentBenchmarkConfig.load()
    app = setup_fastapi_app(config)

    # Run the FastAPI application using uvicorn
    port = port or int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


@cli.command()
def config():
    """Displays info regarding the present AGBenchmark config."""
    try:
        config = AgentBenchmarkConfig.load()
    except FileNotFoundError as e:
        click.echo(e, err=True)
        return 1

    k_col_width = max(len(k) for k in config.dict().keys())
    for k, v in config.dict().items():
        click.echo(f"{k: <{k_col_width}} = {v}")


@cli.command()
def version():
    """Print version info for the AGBenchmark application."""
    import toml

    package_root = Path(__file__).resolve().parent.parent
    pyproject = toml.load(package_root / "pyproject.toml")
    version = pyproject["tool"]["poetry"]["version"]
    click.echo(f"AGBenchmark version {version}")


if __name__ == "__main__":
    cli()
