#!/usr/bin/env python3
"""
OAuth Application Credential Generator and Test Server

Generates client IDs, client secrets, and SQL INSERT statements for OAuth applications.
Also provides a test server to test the OAuth flows end-to-end.

Usage:
    # Generate credentials interactively (recommended)
    poetry run oauth-tool generate-app

    # Generate credentials with all options provided
    poetry run oauth-tool generate-app \\
        --name "My App" \\
        --description "My application description" \\
        --redirect-uris "https://app.example.com/callback,http://localhost:3000/callback" \\
        --scopes "EXECUTE_GRAPH,READ_GRAPH"

    # Mix of options and interactive prompts
    poetry run oauth-tool generate-app --name "My App"

    # Hash an existing plaintext secret (for secret rotation)
    poetry run oauth-tool hash-secret "my-plaintext-secret"

    # Validate a plaintext secret against a hash and salt
    poetry run oauth-tool validate-secret "my-plaintext-secret" "hash" "salt"

    # Run a test server to test OAuth flows
    poetry run oauth-tool test-server --owner-id YOUR_USER_ID
"""

import asyncio
import base64
import hashlib
import secrets
import sys
import uuid
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

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


# ============================================================================
# Test Server Command
# ============================================================================

TEST_APP_NAME = "OAuth Test App (CLI)"
TEST_APP_DESCRIPTION = "Temporary test application created by oauth_admin CLI"
TEST_SERVER_PORT = 9876


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)"""
    code_verifier = secrets.token_urlsafe(32)
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return code_verifier, code_challenge


def create_test_html(
    platform_url: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    backend_url: str,
) -> str:
    """Generate HTML page for test OAuth client"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OAuth Test Client</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{ color: #1a1a2e; margin-bottom: 8px; }}
        h2 {{ color: #333; margin-bottom: 16px; font-size: 1.25rem; }}
        p {{ color: #666; line-height: 1.6; margin-bottom: 16px; }}
        .subtitle {{ color: #888; font-size: 0.9rem; margin-bottom: 24px; }}
        .btn {{
            display: inline-block;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .btn-primary:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }}
        .btn-secondary {{
            background: #f0f0f0;
            color: #333;
        }}
        .btn-secondary:hover {{ background: #e0e0e0; }}
        .btn-group {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 24px; }}
        .info-box {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            font-family: monospace;
            font-size: 0.85rem;
            word-break: break-all;
        }}
        .info-box label {{ display: block; color: #888; font-size: 0.75rem; margin-bottom: 4px; font-family: sans-serif; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; }}
        .success h2 {{ color: #155724; }}
        .error {{ background: #f8d7da; border: 1px solid #f5c6cb; }}
        .error h2 {{ color: #721c24; }}
        .token-display {{
            background: #1a1a2e;
            color: #4ade80;
            padding: 16px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.8rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .section {{ margin-top: 32px; padding-top: 24px; border-top: 1px solid #eee; }}
        .hidden {{ display: none; }}
        #log {{ max-height: 300px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🔐 OAuth Test Client</h1>
            <p class="subtitle">Test the "Sign in with AutoGPT" and Integration Setup flows</p>

            <div class="info-box">
                <label>Client ID</label>
                {client_id}
            </div>

            <div class="btn-group">
                <button class="btn btn-primary" onclick="startOAuthFlow()">
                    Sign in with AutoGPT
                </button>
                <button class="btn btn-secondary" onclick="startIntegrationSetup()">
                    Test Integration Setup
                </button>
            </div>
        </div>

        <div id="result-card" class="card hidden">
            <h2 id="result-title">Result</h2>
            <div id="result-content"></div>
        </div>

        <div class="card">
            <h2>📋 Request Log</h2>
            <div id="log" class="token-display">Waiting for action...</div>
        </div>

        <div class="card">
            <h2>⚙️ Configuration</h2>
            <div class="info-box">
                <label>Platform URL</label>
                {platform_url}
            </div>
            <div class="info-box">
                <label>Backend URL</label>
                {backend_url}
            </div>
            <div class="info-box">
                <label>Redirect URI</label>
                {redirect_uri}
            </div>
        </div>
    </div>

    <script>
        const config = {{
            platformUrl: "{platform_url}",
            backendUrl: "{backend_url}",
            clientId: "{client_id}",
            clientSecret: "{client_secret}",
            redirectUri: "{redirect_uri}",
            scopes: ["EXECUTE_GRAPH", "READ_GRAPH", "READ_BLOCK"]
        }};

        let currentPkce = null;
        let currentState = null;

        function log(message) {{
            const logEl = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            logEl.textContent += `\\n[${{time}}] ${{message}}`;
            logEl.scrollTop = logEl.scrollHeight;
        }}

        function showResult(title, content, isError = false) {{
            const card = document.getElementById('result-card');
            const titleEl = document.getElementById('result-title');
            const contentEl = document.getElementById('result-content');

            card.classList.remove('hidden', 'success', 'error');
            card.classList.add(isError ? 'error' : 'success');
            titleEl.textContent = title;
            contentEl.innerHTML = content;
        }}

        function generatePkce() {{
            const verifier = Array.from(crypto.getRandomValues(new Uint8Array(32)))
                .map(b => b.toString(16).padStart(2, '0')).join('');
            return crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
                .then(hash => {{
                    const challenge = btoa(String.fromCharCode(...new Uint8Array(hash)))
                        .replace(/\\+/g, '-').replace(/\\//g, '_').replace(/=+$/, '');
                    return {{ verifier, challenge }};
                }});
        }}

        async function startOAuthFlow() {{
            log('Starting OAuth flow...');

            currentPkce = await generatePkce();
            currentState = crypto.randomUUID();

            const params = new URLSearchParams({{
                client_id: config.clientId,
                redirect_uri: config.redirectUri,
                scope: config.scopes.join(' '),
                state: currentState,
                code_challenge: currentPkce.challenge,
                code_challenge_method: 'S256',
                response_type: 'code'
            }});

            const authUrl = `${{config.platformUrl}}/auth/authorize?${{params}}`;
            log(`Redirecting to: ${{authUrl}}`);

            // Store PKCE in sessionStorage for callback
            sessionStorage.setItem('oauth_pkce_verifier', currentPkce.verifier);
            sessionStorage.setItem('oauth_state', currentState);

            window.location.href = authUrl;
        }}

        function startIntegrationSetup() {{
            log('Starting integration setup wizard...');

            currentState = crypto.randomUUID();

            // Example providers config with valid OAuth scopes
            // GitHub scopes: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps
            // Google scopes: https://developers.google.com/identity/protocols/oauth2/scopes
            const providers = [
                {{ provider: 'github', scopes: ['repo', 'read:user'] }},
                {{ provider: 'google' }}  // Google uses DEFAULT_SCOPES (email, profile, openid)
            ];

            const providersBase64 = btoa(JSON.stringify(providers));

            const params = new URLSearchParams({{
                client_id: config.clientId,
                providers: providersBase64,
                redirect_uri: config.redirectUri,
                state: currentState
            }});

            const wizardUrl = `${{config.platformUrl}}/auth/integrations/setup-wizard?${{params}}`;
            log(`Redirecting to: ${{wizardUrl}}`);

            sessionStorage.setItem('wizard_state', currentState);

            window.location.href = wizardUrl;
        }}

        async function exchangeCodeForTokens(code) {{
            log('Exchanging authorization code for tokens...');

            const verifier = sessionStorage.getItem('oauth_pkce_verifier');
            if (!verifier) {{
                throw new Error('PKCE verifier not found in session');
            }}

            // Use local proxy to avoid CORS issues
            // The proxy forwards the request to the backend
            const response = await fetch('/proxy/token', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    grant_type: 'authorization_code',
                    code: code,
                    redirect_uri: config.redirectUri,
                    client_id: config.clientId,
                    client_secret: config.clientSecret,
                    code_verifier: verifier
                }})
            }});

            if (!response.ok) {{
                const error = await response.json();
                throw new Error(error.detail || 'Token exchange failed');
            }}

            return response.json();
        }}

        // Handle callback on page load
        window.addEventListener('load', async () => {{
            const params = new URLSearchParams(window.location.search);

            // Check for OAuth callback
            if (params.has('code')) {{
                const code = params.get('code');
                const state = params.get('state');
                const savedState = sessionStorage.getItem('oauth_state');

                log(`Received authorization code: ${{code.substring(0, 20)}}...`);

                if (state !== savedState) {{
                    showResult('⚠️ State Mismatch', '<p>The state parameter does not match. This could indicate a CSRF attack.</p>', true);
                    log('ERROR: State mismatch!');
                    return;
                }}

                try {{
                    const tokens = await exchangeCodeForTokens(code);
                    log('Token exchange successful!');

                    showResult('✅ Authorization Successful', `
                        <p>Successfully obtained tokens!</p>
                        <div class="token-display">${{JSON.stringify(tokens, null, 2)}}</div>
                        <div class="btn-group">
                            <button class="btn btn-secondary" onclick="testAccessToken('${{tokens.access_token}}')">
                                Test Access Token
                            </button>
                        </div>
                    `);

                    // Clean up
                    sessionStorage.removeItem('oauth_pkce_verifier');
                    sessionStorage.removeItem('oauth_state');
                    window.history.replaceState({{}}, '', window.location.pathname);
                }} catch (error) {{
                    log(`ERROR: ${{error.message}}`);
                    showResult('❌ Token Exchange Failed', `<p>${{error.message}}</p>`, true);
                }}
            }}

            // Check for OAuth error
            else if (params.has('error')) {{
                const error = params.get('error');
                const description = params.get('error_description') || 'No description provided';
                log(`ERROR: ${{error}} - ${{description}}`);
                showResult('❌ Authorization Failed', `<p><strong>${{error}}</strong></p><p>${{description}}</p>`, true);
                window.history.replaceState({{}}, '', window.location.pathname);
            }}

            // Check for integration setup callback
            else if (params.has('success')) {{
                const success = params.get('success') === 'true';
                const state = params.get('state');
                const savedState = sessionStorage.getItem('wizard_state');

                if (state !== savedState) {{
                    showResult('⚠️ State Mismatch', '<p>The state parameter does not match.</p>', true);
                    return;
                }}

                if (success) {{
                    log('Integration setup completed successfully!');
                    showResult('✅ Integration Setup Complete', '<p>All requested integrations have been connected.</p>');
                }} else {{
                    log('Integration setup failed or was cancelled');
                    showResult('❌ Integration Setup Failed', '<p>The integration setup was not completed.</p>', true);
                }}

                sessionStorage.removeItem('wizard_state');
                window.history.replaceState({{}}, '', window.location.pathname);
            }}
        }});

        async function testAccessToken(token) {{
            log('Testing access token...');
            try {{
                // Use local proxy to call external API (OAuth tokens work with external-api, not internal api)
                const response = await fetch(`/proxy/external-api/v1/blocks?token=${{encodeURIComponent(token)}}`);

                if (response.ok) {{
                    const data = await response.json();
                    log('Access token is valid! API call successful.');
                    const blockCount = Array.isArray(data) ? data.length : 'unknown';
                    showResult('✅ Access Token Valid', `
                        <p>Successfully called /external-api/v1/blocks endpoint!</p>
                        <p>Found ${{blockCount}} blocks.</p>
                        <div class="token-display">Response (truncated): ${{JSON.stringify(data).substring(0, 500)}}...</div>
                    `);
                }} else {{
                    const error = await response.json();
                    log(`Access token test failed: ${{error.detail || response.statusText}}`);
                    showResult('❌ Access Token Invalid', `<p>${{error.detail || response.statusText}}</p>`, true);
                }}
            }} catch (error) {{
                log(`ERROR: ${{error.message}}`);
                showResult('❌ API Call Failed', `<p>${{error.message}}</p>`, true);
            }}
        }}
    </script>
</body>
</html>
"""


async def create_test_app_in_db(
    owner_id: str,
    redirect_uri: str,
) -> dict:
    """Create a temporary test OAuth application in the database"""
    from prisma.models import OAuthApplication

    from backend.data import db

    # Connect to database
    await db.connect()

    # Generate credentials
    creds = generate_app_credentials(
        name=TEST_APP_NAME,
        description=TEST_APP_DESCRIPTION,
        redirect_uris=[redirect_uri],
        scopes=AVAILABLE_SCOPES,  # All scopes for testing
    )

    # Insert into database
    app = await OAuthApplication.prisma().create(
        data={
            "id": creds["id"],
            "name": creds["name"],
            "description": creds["description"],
            "clientId": creds["client_id"],
            "clientSecret": creds["client_secret_hash"],
            "clientSecretSalt": creds["client_secret_salt"],
            "redirectUris": creds["redirect_uris"],
            "grantTypes": creds["grant_types"],
            "scopes": creds["scopes"],
            "ownerId": owner_id,
            "isActive": True,
        }
    )

    click.echo(f"✓ Created test OAuth application: {app.clientId}")

    return {
        "id": app.id,
        "client_id": app.clientId,
        "client_secret": creds["client_secret_plaintext"],
    }


async def cleanup_test_app(app_id: str) -> None:
    """Remove test application and all associated tokens from database"""
    from prisma.models import (
        OAuthAccessToken,
        OAuthApplication,
        OAuthAuthorizationCode,
        OAuthRefreshToken,
    )

    from backend.data import db

    if not db.is_connected():
        await db.connect()

    click.echo("\n🧹 Cleaning up test data...")

    # Delete authorization codes
    deleted_codes = await OAuthAuthorizationCode.prisma().delete_many(
        where={"applicationId": app_id}
    )
    if deleted_codes:
        click.echo(f"   Deleted {deleted_codes} authorization code(s)")

    # Delete access tokens
    deleted_access = await OAuthAccessToken.prisma().delete_many(
        where={"applicationId": app_id}
    )
    if deleted_access:
        click.echo(f"   Deleted {deleted_access} access token(s)")

    # Delete refresh tokens
    deleted_refresh = await OAuthRefreshToken.prisma().delete_many(
        where={"applicationId": app_id}
    )
    if deleted_refresh:
        click.echo(f"   Deleted {deleted_refresh} refresh token(s)")

    # Delete the application itself
    await OAuthApplication.prisma().delete(where={"id": app_id})
    click.echo("   Deleted test OAuth application")

    await db.disconnect()
    click.echo("✓ Cleanup complete!")


def run_test_server(
    port: int,
    platform_url: str,
    backend_url: str,
    client_id: str,
    client_secret: str,
) -> None:
    """Run a simple HTTP server for testing OAuth flows"""
    import json as json_module
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.request import Request, urlopen

    redirect_uri = f"http://localhost:{port}/callback"

    html_content = create_test_html(
        platform_url=platform_url,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        backend_url=backend_url,
    )

    class TestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib.parse import parse_qs

            # Parse the path
            parsed = urlparse(self.path)

            # Serve the test page for root and callback
            if parsed.path in ["/", "/callback"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_content.encode())

            # Proxy API calls to backend (avoids CORS issues)
            # Supports both /proxy/api/* and /proxy/external-api/*
            elif parsed.path.startswith("/proxy/"):
                try:
                    # Extract the API path and token from query params
                    api_path = parsed.path[len("/proxy") :]
                    query_params = parse_qs(parsed.query)
                    token = query_params.get("token", [None])[0]

                    headers = {}
                    if token:
                        headers["Authorization"] = f"Bearer {token}"

                    req = Request(
                        f"{backend_url}{api_path}",
                        headers=headers,
                        method="GET",
                    )

                    with urlopen(req) as response:
                        response_body = response.read()
                        self.send_response(response.status)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(response_body)

                except Exception as e:
                    error_msg = str(e)
                    status_code = 500
                    if hasattr(e, "code"):
                        status_code = e.code  # type: ignore
                    if hasattr(e, "read"):
                        try:
                            error_body = e.read().decode()  # type: ignore
                            error_data = json_module.loads(error_body)
                            error_msg = error_data.get("detail", error_msg)
                        except Exception:
                            pass

                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json_module.dumps({"detail": error_msg}).encode())

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            # Parse the path
            parsed = urlparse(self.path)

            # Proxy token exchange to backend (avoids CORS issues)
            if parsed.path == "/proxy/token":
                try:
                    # Read request body
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)

                    # Forward to backend
                    req = Request(
                        f"{backend_url}/api/oauth/token",
                        data=body,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )

                    with urlopen(req) as response:
                        response_body = response.read()
                        self.send_response(response.status)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(response_body)

                except Exception as e:
                    error_msg = str(e)
                    # Try to extract error detail from urllib error
                    if hasattr(e, "read"):
                        try:
                            error_body = e.read().decode()  # type: ignore
                            error_data = json_module.loads(error_body)
                            error_msg = error_data.get("detail", error_msg)
                        except Exception:
                            pass

                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json_module.dumps({"detail": error_msg}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress default logging
            pass

    server = HTTPServer(("localhost", port), TestHandler)
    click.echo(f"\n🚀 Test server running at http://localhost:{port}")
    click.echo("   Open this URL in your browser to test the OAuth flows\n")

    # Run server in a daemon thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Use a simple polling loop that can be interrupted
    try:
        while server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        pass

    click.echo("\n\n⏹️  Server stopped")
    server.shutdown()


async def setup_and_cleanup_test_app(
    owner_id: str,
    redirect_uri: str,
    port: int,
    platform_url: str,
    backend_url: str,
) -> None:
    """
    Async context manager that handles test app lifecycle.
    Creates the app, yields control to run the server, then cleans up.
    """
    app_info: Optional[dict] = None

    try:
        # Create test app in database
        click.echo("\n📝 Creating temporary OAuth application...")
        app_info = await create_test_app_in_db(owner_id, redirect_uri)

        click.echo(f"\n  Client ID:     {app_info['client_id']}")
        click.echo(f"  Client Secret: {app_info['client_secret'][:30]}...")

        # Run the test server (blocking, synchronous)
        click.echo("\n" + "-" * 60)
        click.echo("  Press Ctrl+C to stop the server and clean up")
        click.echo("-" * 60)

        run_test_server(
            port=port,
            platform_url=platform_url,
            backend_url=backend_url,
            client_id=app_info["client_id"],
            client_secret=app_info["client_secret"],
        )

    finally:
        # Always clean up - we're still in the same event loop
        if app_info:
            try:
                await cleanup_test_app(app_info["id"])
            except Exception as e:
                click.echo(f"\n⚠️  Cleanup error: {e}", err=True)
                click.echo(
                    f"   You may need to manually delete app with ID: {app_info['id']}"
                )


@cli.command(name="test-server")
@click.option(
    "--owner-id",
    required=True,
    help="User ID to own the temporary test OAuth application",
)
@click.option(
    "--port",
    default=TEST_SERVER_PORT,
    help=f"Port to run the test server on (default: {TEST_SERVER_PORT})",
)
@click.option(
    "--platform-url",
    default="http://localhost:3000",
    help="AutoGPT Platform frontend URL (default: http://localhost:3000)",
)
@click.option(
    "--backend-url",
    default="http://localhost:8006",
    help="AutoGPT Platform backend URL (default: http://localhost:8006)",
)
def test_server_command(
    owner_id: str,
    port: int,
    platform_url: str,
    backend_url: str,
):
    """Run a test server to test OAuth flows interactively

    This command:
    1. Creates a temporary OAuth application in the database
    2. Starts a minimal web server that acts as a third-party client
    3. Lets you test "Sign in with AutoGPT" and Integration Setup flows
    4. Cleans up all test data (app, tokens, codes) when you stop the server

    Example:
        poetry run oauth-tool test-server --owner-id YOUR_USER_ID

    The test server will be available at http://localhost:9876
    """
    redirect_uri = f"http://localhost:{port}/callback"

    click.echo("=" * 60)
    click.echo("  OAuth Test Server")
    click.echo("=" * 60)
    click.echo(f"\n  Owner ID:     {owner_id}")
    click.echo(f"  Platform URL: {platform_url}")
    click.echo(f"  Backend URL:  {backend_url}")
    click.echo(f"  Test Server:  http://localhost:{port}")
    click.echo(f"  Redirect URI: {redirect_uri}")
    click.echo("\n" + "=" * 60)

    try:
        # Run everything in a single event loop to keep Prisma client happy
        asyncio.run(
            setup_and_cleanup_test_app(
                owner_id=owner_id,
                redirect_uri=redirect_uri,
                port=port,
                platform_url=platform_url,
                backend_url=backend_url,
            )
        )
    except KeyboardInterrupt:
        # Already handled inside, just exit cleanly
        pass
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
