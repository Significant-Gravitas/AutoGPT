# OAuth Authorization Flow

This page implements the OAuth 2.0 authorization consent screen, where users approve or deny third-party applications access to their AutoGPT account.

## Flow Overview

1. Third-party app redirects user to `/auth/authorize` with OAuth parameters
2. User sees consent screen with app info and requested permissions
3. User clicks "Authorize" or "Deny"
4. User is redirected back to the app with an authorization code (or error)

## Testing the Flow

### Option 1: Test Server (Recommended)

The easiest way to test OAuth flows is using the built-in test server. It automatically:

- Creates a temporary OAuth application in the database
- Starts a web-based test client with "Sign in with AutoGPT" button
- Handles PKCE, token exchange, and access token testing
- Cleans up all test data when you stop the server

```bash
cd autogpt_platform/backend
poetry run python -m backend.cli.oauth_admin test-server --owner-id YOUR_USER_ID
```

Replace `YOUR_USER_ID` with your Supabase user ID (you can find this in the Supabase dashboard or by inspecting your JWT).

The test server starts at http://localhost:9876. Open it in your browser and click "Sign in with AutoGPT" to test the full flow.

Options:

- `--port`: Port for the test server (default: 9876)
- `--platform-url`: Frontend URL (default: http://localhost:3000)
- `--backend-url`: Backend URL (default: http://localhost:8006)

### Option 2: Manual Setup

If you need a persistent OAuth application or want to test from a real external client:

#### 1. Register an OAuth Application

Generate credentials using the CLI tool:

```bash
cd autogpt_platform/backend
poetry run python -m backend.cli.oauth_admin generate-app
```

Follow the interactive prompts:

- **Application name**: e.g., "My OAuth App"
- **Description**: e.g., "My application description"
- **Redirect URIs**: e.g., `http://localhost:8080/callback`
- **Scopes**: Select permissions (e.g., `1,2` for `EXECUTE_GRAPH` and `READ_GRAPH`)

The tool outputs SQL. Execute it in your database (replace `YOUR_USER_ID_HERE` with your user ID).

**Important**: Save the `client_id` and `client_secret` from the output - the secret is only shown once!

#### 2. Generate PKCE Parameters

OAuth requires PKCE (Proof Key for Code Exchange). Generate a code verifier and challenge:

```javascript
// Run in browser console or Node.js
const crypto = require("crypto");

// Generate code_verifier (random string)
const codeVerifier = crypto.randomBytes(32).toString("base64url");
console.log("code_verifier:", codeVerifier);

// Generate code_challenge (SHA256 hash of verifier, base64url encoded)
const codeChallenge = crypto
  .createHash("sha256")
  .update(codeVerifier)
  .digest("base64url");
console.log("code_challenge:", codeChallenge);
```

Or use an online PKCE generator like https://tonyxu-io.github.io/pkce-generator/

#### 3. Construct the Authorization URL

Build the URL with these parameters:

```
http://localhost:3000/auth/authorize?
  client_id=agpt_client_xxxx&
  redirect_uri=http://localhost:8080/callback&
  scope=EXECUTE_GRAPH%20READ_GRAPH&
  state=random-state-string&
  code_challenge=your-code-challenge&
  code_challenge_method=S256&
  response_type=code
```

Parameters:

- `client_id`: From the registered app
- `redirect_uri`: Must match one registered for the app
- `scope`: Space-separated list of permissions (URL-encoded)
- `state`: Random string for CSRF protection
- `code_challenge`: PKCE challenge (from step 2)
- `code_challenge_method`: `S256` (recommended) or `plain`
- `response_type`: Must be `code`

#### 4. Test the Flow

1. Open the authorization URL in your browser
2. Log in if not already authenticated
3. Review the consent screen showing:
   - Application name and description
   - Requested permissions
4. Click "Authorize" or "Deny"

**On Authorize**: You'll be redirected to:

```
http://localhost:8080/callback?code=AUTH_CODE&state=random-state-string
```

**On Deny**: You'll be redirected to:

```
http://localhost:8080/callback?error=access_denied&error_description=User%20denied%20access&state=random-state-string
```

#### 5. Exchange Code for Tokens

Exchange the authorization code for tokens:

```bash
curl -X POST http://localhost:8006/api/oauth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "authorization_code",
    "code": "AUTH_CODE_FROM_REDIRECT",
    "redirect_uri": "http://localhost:8080/callback",
    "client_id": "agpt_client_xxxx",
    "client_secret": "agpt_secret_xxxx",
    "code_verifier": "your-code-verifier-from-step-2"
  }'
```

## CLI Tool Commands

The `oauth_admin` CLI provides these commands:

| Command           | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| `generate-app`    | Generate credentials for a new OAuth application (outputs SQL) |
| `test-server`     | Run an interactive test server for OAuth flows                 |
| `hash-secret`     | Hash a plaintext secret using Scrypt                           |
| `validate-secret` | Validate a plaintext secret against a hash and salt            |

Run with `--help` for full options:

```bash
poetry run python -m backend.cli.oauth_admin --help
poetry run python -m backend.cli.oauth_admin generate-app --help
poetry run python -m backend.cli.oauth_admin test-server --help
```

## Available Scopes

| Scope                 | Description                                        |
| --------------------- | -------------------------------------------------- |
| `EXECUTE_GRAPH`       | Execute agents/graphs                              |
| `READ_GRAPH`          | Read agent/graph definitions and execution results |
| `EXECUTE_BLOCK`       | Execute individual blocks                          |
| `READ_BLOCK`          | Read block definitions                             |
| `READ_STORE`          | Access the agent store                             |
| `USE_TOOLS`           | Use platform tools                                 |
| `MANAGE_INTEGRATIONS` | Create and update user integrations                |
| `READ_INTEGRATIONS`   | Read user integration status                       |
| `DELETE_INTEGRATIONS` | Remove user integrations                           |

## Error Scenarios to Test

| Scenario                  | Expected Result                     |
| ------------------------- | ----------------------------------- |
| Missing `client_id`       | "Invalid Request" error page        |
| Invalid `client_id`       | "Application Not Found" error       |
| Disabled application      | "Application Not Found" error       |
| Invalid `redirect_uri`    | HTTP 400 error (not redirected)     |
| Invalid `scope`           | Redirect with `error=invalid_scope` |
| Scope not allowed for app | "Invalid Scopes" error page         |
| Missing `code_challenge`  | "Invalid Request" error page        |

## Related Files

- Backend OAuth endpoints: `backend/server/routers/oauth.py`
- OAuth data layer: `backend/data/auth/oauth.py`
- CLI tool: `backend/cli/oauth_admin.py`
- SSO integration guide: `autogpt_platform/SSO.md`

### API Endpoints

| Endpoint                     | Method | Description                                     |
| ---------------------------- | ------ | ----------------------------------------------- |
| `/api/oauth/app/{client_id}` | GET    | Get OAuth application info (for consent screen) |
| `/api/oauth/authorize`       | POST   | Create authorization code (after user consent)  |
| `/api/oauth/token`           | POST   | Exchange code for tokens / refresh tokens       |
| `/api/oauth/introspect`      | POST   | Check if a token is valid                       |
| `/api/oauth/revoke`          | POST   | Revoke an access or refresh token               |
