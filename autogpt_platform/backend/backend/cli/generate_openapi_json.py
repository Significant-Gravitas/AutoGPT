#!/usr/bin/env python3
"""
Script to generate OpenAPI JSON specification for the FastAPI app.

This script imports the FastAPI app from backend.api.rest_api and outputs
the OpenAPI specification as JSON to stdout or a specified file.

Usage:
  `poetry run python generate_openapi_json.py`
  `poetry run python generate_openapi_json.py --output openapi.json`
  `poetry run python generate_openapi_json.py --indent 4 --output openapi.json`
"""

import json
import os
from pathlib import Path

import click


@click.command()
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path (default: stdout)",
)
@click.option(
    "--pretty",
    type=click.BOOL,
    default=False,
    help="Pretty-print JSON output (indented 2 spaces)",
)
def main(output: Path, pretty: bool):
    """Generate and output the OpenAPI JSON specification."""
    openapi_schema = get_openapi_schema()

    json_output = json.dumps(openapi_schema, indent=2 if pretty else None)

    if output:
        output.write_text(json_output)
        click.echo(f"✅ OpenAPI specification written to {output}\n\nPreview:")
        click.echo(f"\n{json_output[:500]} ...")
    else:
        print(json_output)


def get_openapi_schema():
    """Get the OpenAPI schema from the FastAPI app"""
    from backend.api.rest_api import app

    return app.openapi()


if __name__ == "__main__":
    os.environ["LOG_LEVEL"] = "ERROR"  # disable stdout log output

    main()
