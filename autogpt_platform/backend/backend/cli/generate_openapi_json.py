#!/usr/bin/env python3
"""
Script to generate OpenAPI JSON specification for the FastAPI app.

This script imports the FastAPI app from backend.api.rest_api and outputs
the OpenAPI specification as JSON to stdout or a specified file.

Usage:
  `poetry run export-api-schema`
  `poetry run export-api-schema --output openapi.json`
  `poetry run export-api-schema --api v2 --output openapi.json`
"""

import json
import os
from pathlib import Path

import click

API_CHOICES = ["internal", "v1", "v2"]


@click.command()
@click.option(
    "--api",
    type=click.Choice(API_CHOICES),
    default="internal",
    help="Which API schema to export (default: internal)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path (default: stdout)",
)
@click.option(
    "--pretty",
    is_flag=True,
    help="Pretty-print JSON output (indented 2 spaces)",
)
def main(api: str, output: Path, pretty: bool):
    """Generate and output the OpenAPI JSON specification."""
    openapi_schema = get_openapi_schema(api)

    json_output = json.dumps(
        openapi_schema, indent=2 if pretty else None, ensure_ascii=False
    )

    if output:
        output.write_text(json_output, encoding="utf-8")
        click.echo(f"✅ OpenAPI specification written to {output}\n\nPreview:")
        click.echo(f"\n{json_output[:500]} ...")
    else:
        print(json_output)


def get_openapi_schema(api: str = "internal"):
    """Get the OpenAPI schema from the specified FastAPI app."""
    if api == "internal":
        from backend.api.rest_api import app

        return app.openapi()
    elif api == "v1":
        from backend.api.external.v1.app import v1_app

        return v1_app.openapi()
    elif api == "v2":
        from backend.api.external.v2.app import v2_app

        return v2_app.openapi()
    else:
        raise click.BadParameter(f"Unknown API: {api}. Choose from {API_CHOICES}")


if __name__ == "__main__":
    os.environ["LOG_LEVEL"] = "ERROR"  # disable stdout log output

    main()
