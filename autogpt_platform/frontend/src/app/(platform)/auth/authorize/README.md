# OAuth Authorization Flow

This page implements the OAuth 2.0 authorization consent screen, where users approve or deny third-party applications access to their AutoGPT account.

## Flow Overview

1. Third-party app redirects user to `/auth/authorize` with OAuth parameters
2. User sees consent screen with app info and requested permissions
3. User clicks "Authorize" or "Deny"
4. User is redirected back to the app with an authorization code (or error)

## Testing the Flow

### 1. Register an OAuth Application

First, generate credentials and register an app in the database using the CLI tool:

```bash
cd autogpt_platform/backend
poetry run python backend/cli/oauth_admin.py generate-app
```

Follow the interactive prompts:

- **Application name**: e.g., "Test OAuth App"
- **Description**: e.g., "Test application for OAuth flow"
- **Redirect URIs**: e.g., `http://localhost:8080/callback`
- **Scopes**: Select the permissions (e.g., `1,2` for `EXECUTE_GRAPH` and `READ_GRAPH`)

The tool outputs SQL. Execute it in your database:

```sql
-- Example output (your values will differ):
INSERT INTO "OAuthApplication" (
  id, "createdAt", "updatedAt", name, description,
  "clientId", "clientSecret", "clientSecretSalt",
  "redirectUris", "grantTypes", scopes, "ownerId", "isActive"
)
VALUES (
  'uuid-here',
  NOW(), NOW(),
  'Test OAuth App',
  'Test application for OAuth flow',
  'agpt_client_xxxx',  -- Save this client_id
  'hashed-secret',
  'salt',
  ARRAY['http://localhost:8080/callback']::TEXT[],
  ARRAY['authorization_code', 'refresh_token']::TEXT[],
  ARRAY['EXECUTE_GRAPH', 'READ_GRAPH']::"APIKeyPermission"[],
  'YOUR_USER_ID_HERE',  -- Replace with your user ID
  true
);
```

**Important**: Save the `client_id` and `client_secret` from the CLI output - the secret is only shown once!

### 2. Generate PKCE Parameters

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

### 3. Construct the Authorization URL

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

### 4. Test the Flow

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

### 5. Exchange Code for Tokens (Optional)

To complete the flow, exchange the authorization code for tokens:

```bash
curl -X POST http://localhost:8006/oauth/token \
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

- Backend endpoint: `backend/server/routers/oauth.py`
- App info endpoint: `GET /oauth/app/{client_id}`
- Authorization endpoint: `GET /oauth/authorize`
- Token endpoint: `POST /oauth/token`
- CLI tool: `backend/cli/oauth_admin.py`
