#!/usr/bin/env python3
"""
OAuth Application Credential Generator

Generates client IDs, client secrets, and SQL INSERT statements for OAuth applications.
Does NOT directly insert into the database - outputs SQL for manual execution.

Usage:
    # Generate credentials interactively (recommended)
    python -m backend.cli.oauth_admin generate-app

    # Generate credentials with all options provided
    python -m backend.cli.oauth_admin generate-app \\
        --name "My App" \\
        --description "My application description" \\
        --redirect-uris "https://app.example.com/callback,http://localhost:3000/callback" \\
        --scopes "EXECUTE_GRAPH,READ_GRAPH"

    # Mix of options and interactive prompts
    python -m backend.cli.oauth_admin generate-app --name "My App"

    # Hash an existing plaintext secret (for secret rotation)
    python -m backend.cli.oauth_admin hash-secret "my-plaintext-secret"

    # Validate a plaintext secret against a hash and salt
    python -m backend.cli.oauth_admin validate-secret "my-plaintext-secret" "hash" "salt"
"""

import secrets
import sys
import uuid
from datetime import datetime

import click
from autogpt_libs.api_key.keysmith import APIKeySmith
from prisma.enums import APIKeyPermission

keysmith = APIKeySmith()


def generate_client_id() -> str:
    """Generate a unique client ID"""
    return f"agpt_client_{secrets.token_urlsafe(16)}"


def generate_client_secret() -> tuple[str, str, str]:
    """
    Generate a client secret with its hash and salt.
    Returns (plaintext_secret, hashed_secret, salt)
    """
    # Generate a secure random secret (32 bytes = 256 bits of entropy)
    plaintext = f"agpt_secret_{secrets.token_urlsafe(32)}"

    # Hash using Scrypt (same as API keys)
    hashed, salt = keysmith.hash_key(plaintext)

    return plaintext, hashed, salt


def hash_secret(plaintext: str) -> tuple[str, str]:
    """Hash a plaintext secret using Scrypt. Returns (hash, salt)"""
    return keysmith.hash_key(plaintext)


def validate_secret(plaintext: str, hash_value: str, salt: str) -> bool:
    """Validate a plaintext secret against a stored hash and salt"""
    return keysmith.verify_key(plaintext, hash_value, salt)


def generate_app_credentials(
    name: str,
    redirect_uris: list[str],
    scopes: list[str],
    description: str | None = None,
    grant_types: list[str] | None = None,
) -> dict:
    """
    Generate complete credentials for an OAuth application.

    Returns dict with:
    - id: UUID for the application
    - name: Application name
    - description: Application description
    - client_id: Client identifier (plaintext)
    - client_secret_plaintext: Client secret (SENSITIVE - show only once)
    - client_secret_hash: Hashed client secret (for database)
    - redirect_uris: List of allowed redirect URIs
    - grant_types: List of allowed grant types
    - scopes: List of allowed scopes
    """
    if grant_types is None:
        grant_types = ["authorization_code", "refresh_token"]

    # Validate scopes
    try:
        validated_scopes = [APIKeyPermission(s.strip()) for s in scopes if s.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid scope: {e}")

    if not validated_scopes:
        raise ValueError("At least one scope is required")

    # Generate credentials
    app_id = str(uuid.uuid4())
    client_id = generate_client_id()
    client_secret_plaintext, client_secret_hash, client_secret_salt = (
        generate_client_secret()
    )

    return {
        "id": app_id,
        "name": name,
        "description": description,
        "client_id": client_id,
        "client_secret_plaintext": client_secret_plaintext,
        "client_secret_hash": client_secret_hash,
        "client_secret_salt": client_secret_salt,
        "redirect_uris": redirect_uris,
        "grant_types": grant_types,
        "scopes": [s.value for s in validated_scopes],
    }


def format_sql_insert(creds: dict) -> str:
    """
    Format credentials as a SQL INSERT statement.

    The statement includes placeholders that must be replaced:
    - YOUR_USER_ID_HERE: Replace with the owner's user ID
    """
    now_iso = datetime.utcnow().isoformat()

    # Format arrays for PostgreSQL
    redirect_uris_pg = (
        "{" + ",".join(f'"{uri}"' for uri in creds["redirect_uris"]) + "}"
    )
    grant_types_pg = "{" + ",".join(f'"{gt}"' for gt in creds["grant_types"]) + "}"
    scopes_pg = "{" + ",".join(creds["scopes"]) + "}"

    sql = f"""
-- ============================================================
-- OAuth Application: {creds['name']}
-- Generated: {now_iso} UTC
-- ============================================================

INSERT INTO "OAuthApplication" (
  id,
  "createdAt",
  "updatedAt",
  name,
  description,
  "clientId",
  "clientSecret",
  "clientSecretSalt",
  "redirectUris",
  "grantTypes",
  scopes,
  "ownerId",
  "isActive"
)
VALUES (
  '{creds['id']}',
  NOW(),
  NOW(),
  '{creds['name']}',
  {f"'{creds['description']}'" if creds['description'] else 'NULL'},
  '{creds['client_id']}',
  '{creds['client_secret_hash']}',
  '{creds['client_secret_salt']}',
  ARRAY{redirect_uris_pg}::TEXT[],
  ARRAY{grant_types_pg}::TEXT[],
  ARRAY{scopes_pg}::"APIKeyPermission"[],
  'YOUR_USER_ID_HERE',  -- ⚠️ REPLACE with actual owner user ID
  true
);

-- ============================================================
-- ⚠️ IMPORTANT: Save these credentials securely!
-- ============================================================
--
-- Client ID:     {creds['client_id']}
-- Client Secret: {creds['client_secret_plaintext']}
--
-- ⚠️ The client secret is shown ONLY ONCE!
-- ⚠️ Store it securely and share only with the application developer.
-- ⚠️ Never commit it to version control.
--
-- The client secret has been hashed in the database using Scrypt.
-- The plaintext secret above is needed by the application to authenticate.
-- ============================================================

-- To verify the application was created:
-- SELECT "clientId", name, scopes, "redirectUris", "isActive"
-- FROM "OAuthApplication"
-- WHERE "clientId" = '{creds['client_id']}';
"""
    return sql


@click.group()
def cli():
    """OAuth Application Credential Generator

    Generates client IDs, client secrets, and SQL INSERT statements for OAuth applications.
    Does NOT directly insert into the database - outputs SQL for manual execution.
    """
    pass


AVAILABLE_SCOPES = [
    "EXECUTE_GRAPH",
    "READ_GRAPH",
    "EXECUTE_BLOCK",
    "READ_BLOCK",
    "READ_STORE",
    "USE_TOOLS",
    "MANAGE_INTEGRATIONS",
    "READ_INTEGRATIONS",
    "DELETE_INTEGRATIONS",
]

DEFAULT_GRANT_TYPES = ["authorization_code", "refresh_token"]


def prompt_for_name() -> str:
    """Prompt for application name"""
    return click.prompt("Application name", type=str)


def prompt_for_description() -> str | None:
    """Prompt for application description"""
    description = click.prompt(
        "Application description (optional, press Enter to skip)",
        type=str,
        default="",
        show_default=False,
    )
    return description if description else None


def prompt_for_redirect_uris() -> list[str]:
    """Prompt for redirect URIs interactively"""
    click.echo("\nRedirect URIs (enter one per line, empty line to finish):")
    click.echo("  Example: https://app.example.com/callback")
    uris = []
    while True:
        uri = click.prompt("  URI", type=str, default="", show_default=False)
        if not uri:
            if not uris:
                click.echo("  At least one redirect URI is required.")
                continue
            break
        uris.append(uri.strip())
    return uris


def prompt_for_scopes() -> list[str]:
    """Prompt for scopes interactively with a menu"""
    click.echo("\nAvailable scopes:")
    for i, scope in enumerate(AVAILABLE_SCOPES, 1):
        click.echo(f"  {i}. {scope}")

    click.echo(
        "\nSelect scopes by number (comma-separated) or enter scope names directly:"
    )
    click.echo("  Example: 1,2 or EXECUTE_GRAPH,READ_GRAPH")

    while True:
        selection = click.prompt("Scopes", type=str)
        scopes = []

        for item in selection.split(","):
            item = item.strip()
            if not item:
                continue

            # Check if it's a number
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(AVAILABLE_SCOPES):
                    scopes.append(AVAILABLE_SCOPES[idx])
                else:
                    click.echo(f"  Invalid number: {item}")
                    scopes = []
                    break
            # Check if it's a valid scope name
            elif item.upper() in AVAILABLE_SCOPES:
                scopes.append(item.upper())
            else:
                click.echo(f"  Invalid scope: {item}")
                scopes = []
                break

        if scopes:
            return scopes
        click.echo("  Please enter valid scope numbers or names.")


def prompt_for_grant_types() -> list[str] | None:
    """Prompt for grant types interactively"""
    click.echo(f"\nGrant types (default: {', '.join(DEFAULT_GRANT_TYPES)})")
    grant_types_input = click.prompt(
        "Grant types (comma-separated, press Enter for default)",
        type=str,
        default="",
        show_default=False,
    )

    if not grant_types_input:
        return None  # Use default

    return [gt.strip() for gt in grant_types_input.split(",") if gt.strip()]


@cli.command(name="generate-app")
@click.option(
    "--name",
    default=None,
    help="Application name (e.g., 'My Cool App')",
)
@click.option(
    "--description",
    default=None,
    help="Application description",
)
@click.option(
    "--redirect-uris",
    default=None,
    help="Comma-separated list of redirect URIs (e.g., 'https://app.example.com/callback,http://localhost:3000/callback')",
)
@click.option(
    "--scopes",
    default=None,
    help="Comma-separated list of scopes (e.g., 'EXECUTE_GRAPH,READ_GRAPH')",
)
@click.option(
    "--grant-types",
    default=None,
    help="Comma-separated list of grant types (default: 'authorization_code,refresh_token')",
)
def generate_app(
    name: str | None,
    description: str | None,
    redirect_uris: str | None,
    scopes: str | None,
    grant_types: str | None,
):
    """Generate credentials for a new OAuth application

    All options are optional. If not provided, you will be prompted interactively.
    """
    # Interactive prompts for missing required values
    if name is None:
        name = prompt_for_name()

    if description is None:
        description = prompt_for_description()

    if redirect_uris is None:
        redirect_uris_list = prompt_for_redirect_uris()
    else:
        redirect_uris_list = [uri.strip() for uri in redirect_uris.split(",")]

    if scopes is None:
        scopes_list = prompt_for_scopes()
    else:
        scopes_list = [scope.strip() for scope in scopes.split(",")]

    if grant_types is None:
        grant_types_list = prompt_for_grant_types()
    else:
        grant_types_list = [gt.strip() for gt in grant_types.split(",")]

    try:
        creds = generate_app_credentials(
            name=name,
            description=description,
            redirect_uris=redirect_uris_list,
            scopes=scopes_list,
            grant_types=grant_types_list,
        )

        sql = format_sql_insert(creds)
        click.echo(sql)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command(name="hash-secret")
@click.argument("secret")
def hash_secret_command(secret):
    """Hash a plaintext secret using Scrypt"""
    hashed, salt = hash_secret(secret)
    click.echo(f"Hash: {hashed}")
    click.echo(f"Salt: {salt}")


@cli.command(name="validate-secret")
@click.argument("secret")
@click.argument("hash")
@click.argument("salt")
def validate_secret_command(secret, hash, salt):
    """Validate a plaintext secret against a hash and salt"""
    is_valid = validate_secret(secret, hash, salt)
    if is_valid:
        click.echo("✓ Secret is valid!")
        sys.exit(0)
    else:
        click.echo("✗ Secret is invalid!", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
