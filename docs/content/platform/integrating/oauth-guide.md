# AutoGPT Platform OAuth Integration Guide

This guide explains how to integrate your application with AutoGPT Platform using OAuth 2.0. OAuth can be used for API access, Single Sign-On (SSO), or both.

For general API information and endpoint documentation, see the [API Guide](api-guide.md) and the [Swagger documentation](https://backend.agpt.co/external-api/docs).

## Overview

AutoGPT Platform's OAuth implementation supports multiple use cases:

### OAuth for API Access

Use OAuth when your application needs to call AutoGPT APIs on behalf of users. This is the most common use case for third-party integrations.

**When to use:**

- Your app needs to run agents, access the store, or manage integrations for users
- You want user-specific permissions rather than a single API key
- Users should be able to revoke access to your app

### SSO: "Sign in with AutoGPT"

Use SSO when you want users to sign in to your app through their AutoGPT account. Request the `IDENTITY` scope to get user information.

**When to use:**

- You want to use AutoGPT as an identity provider
- Users already have AutoGPT accounts and you want seamless login
- You need to identify users without managing passwords

**Note:** SSO and API access can be combined. Request `IDENTITY` along with other scopes to both authenticate users and access APIs on their behalf.

### Integration Setup Wizard

A separate flow that guides users through connecting third-party services (GitHub, Google, etc.) to their AutoGPT account. See [Integration Setup Wizard](#integration-setup-wizard) below.

## Prerequisites

Before integrating, you need an OAuth application registered with AutoGPT Platform. Contact the platform administrator to obtain:

- **Client ID** - Public identifier for your application
- **Client Secret** - Secret key for authenticating your application (keep this secure!)
- **Registered Redirect URIs** - URLs where users will be redirected after authorization

## OAuth Flow

The OAuth flow is technically the same whether you're using it for API access, SSO, or both. The main difference is which scopes you request.

### Step 1: Redirect User to Authorization

Redirect the user to the AutoGPT authorization page with the required parameters:

```url
https://platform.agpt.co/auth/authorize?
  client_id={YOUR_CLIENT_ID}&
  redirect_uri=https://yourapp.com/callback&
  scope=EXECUTE_GRAPH READ_GRAPH&
  state={RANDOM_STATE_TOKEN}&
  code_challenge={PKCE_CHALLENGE}&
  code_challenge_method=S256&
  response_type=code
```

#### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `client_id` | Yes | Your OAuth application's client ID |
| `redirect_uri` | Yes | URL to redirect after authorization (must match registered URI) |
| `scope` | Yes | Space-separated list of permissions (see [Available Scopes](api-guide.md#available-scopes)) |
| `state` | Yes | Random string to prevent CSRF attacks (store and verify on callback) |
| `code_challenge` | Yes | PKCE code challenge (see [PKCE](#pkce-implementation)) |
| `code_challenge_method` | Yes | Must be `S256` |
| `response_type` | Yes | Must be `code` |

### Step 2: Handle the Callback

After the user approves (or denies) access, they'll be redirected to your `redirect_uri`:

**Success:**

```url
https://yourapp.com/callback?code=AUTHORIZATION_CODE&state=RANDOM_STATE_TOKEN
```

**Error:**

```url
https://yourapp.com/callback?error=access_denied&error_description=User%20denied%20access&state=RANDOM_STATE_TOKEN
```

Always verify the `state` parameter matches what you sent in Step 1.

### Step 3: Exchange Code for Tokens

Exchange the authorization code for access and refresh tokens:

```http
POST /api/oauth/token
Content-Type: application/json

{
  "grant_type": "authorization_code",
  "code": "{AUTHORIZATION_CODE}",
  "redirect_uri": "https://yourapp.com/callback",
  "client_id": "{YOUR_CLIENT_ID}",
  "client_secret": "{YOUR_CLIENT_SECRET}",
  "code_verifier": "{PKCE_VERIFIER}"
}
```

**Response:**

```json
{
  "token_type": "Bearer",
  "access_token": "agpt_xt_...",
  "access_token_expires_at": "2025-01-15T12:00:00Z",
  "refresh_token": "agpt_rt_...",
  "refresh_token_expires_at": "2025-02-14T12:00:00Z",
  "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"]
}
```

### Step 4: Use the Access Token

Include the access token in API requests:

```http
GET /external-api/v1/blocks
Authorization: Bearer agpt_xt_...
```

**For SSO:** If you requested the `IDENTITY` scope, fetch user info to identify the user:

```http
GET /external-api/v1/me
Authorization: Bearer agpt_xt_...
```

**Response:**

```json
{
  "id": "user-uuid",
  "name": "John Doe",
  "email": "john@example.com",
  "timezone": "Europe/Amsterdam"
}
```

See the [Swagger documentation](https://backend.agpt.co/external-api/docs) for all available endpoints.

### Step 5: Refresh Tokens

Access tokens expire after 1 hour. Use the refresh token to get new tokens:

```http
POST /api/oauth/token
Content-Type: application/json

{
  "grant_type": "refresh_token",
  "refresh_token": "agpt_rt_...",
  "client_id": "{YOUR_CLIENT_ID}",
  "client_secret": "{YOUR_CLIENT_SECRET}"
}
```

**Response:**

```json
{
  "token_type": "Bearer",
  "access_token": "agpt_xt_...",
  "access_token_expires_at": "2025-01-15T13:00:00Z",
  "refresh_token": "agpt_rt_...",
  "refresh_token_expires_at": "2025-02-14T12:00:00Z",
  "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"]
}
```

## Integration Setup Wizard

The Integration Setup Wizard guides users through connecting third-party services (like GitHub, Google, etc.) to their AutoGPT account. This is useful when your application needs users to have specific integrations configured.

### Redirect to the Wizard

```url
https://platform.agpt.co/auth/integrations/setup-wizard?
  client_id={YOUR_CLIENT_ID}&
  providers={BASE64_ENCODED_PROVIDERS}&
  redirect_uri=https://yourapp.com/callback&
  state={RANDOM_STATE_TOKEN}
```

#### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `client_id` | Yes | Your OAuth application's client ID |
| `providers` | Yes | Base64-encoded JSON array of provider configurations |
| `redirect_uri` | Yes | URL to redirect after setup completes |
| `state` | Yes | Random string to prevent CSRF attacks |

#### Provider Configuration

The `providers` parameter is a Base64-encoded JSON array:

```javascript
const providers = [
  { provider: 'github', scopes: ['repo', 'read:user'] },
  { provider: 'google', scopes: ['https://www.googleapis.com/auth/calendar'] },
  { provider: 'slack' }  // Uses default scopes
];

const providersBase64 = btoa(JSON.stringify(providers));
```

### Handle the Callback

After setup completes:

**Success:**

```url
https://yourapp.com/callback?success=true&state=RANDOM_STATE_TOKEN
```

**Failure/Cancelled:**

```url
https://yourapp.com/callback?success=false&state=RANDOM_STATE_TOKEN
```

## Provider Scopes Reference

When using the Integration Setup Wizard, you need to specify which scopes to request from each provider. Here are common providers and their scopes:

### GitHub

Documentation: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps

| Scope | Description |
|-------|-------------|
| `repo` | Full control of private repositories |
| `read:user` | Read user profile data |
| `user:email` | Access user email addresses |
| `gist` | Create and manage gists |
| `workflow` | Update GitHub Actions workflows |

**Example:**

```javascript
{ provider: 'github', scopes: ['repo', 'read:user'] }
```

### Google

Documentation: https://developers.google.com/identity/protocols/oauth2/scopes

| Scope | Description |
|-------|-------------|
| `email` | View email address (default) |
| `profile` | View basic profile info (default) |
| `openid` | OpenID Connect (default) |
| `https://www.googleapis.com/auth/calendar` | Google Calendar access |
| `https://www.googleapis.com/auth/drive` | Google Drive access |
| `https://www.googleapis.com/auth/gmail.readonly` | Read Gmail messages |

**Example:**

```javascript
{ provider: 'google', scopes: ['https://www.googleapis.com/auth/calendar'] }
// Or use defaults (email, profile, openid):
{ provider: 'google' }
```

### Notion

Documentation: https://developers.notion.com/reference/capabilities

Notion uses a single OAuth scope that grants access based on pages the user selects during authorization.

### Linear

Documentation: https://developers.linear.app/docs/oauth/authentication

| Scope | Description |
|-------|-------------|
| `read` | Read access to Linear data |
| `write` | Write access to Linear data |
| `issues:create` | Create issues |

## PKCE Implementation

PKCE (Proof Key for Code Exchange) is required for all authorization requests. Here's how to implement it:

### JavaScript Example

```javascript
async function generatePkce() {
  // Generate a random code verifier
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const verifier = Array.from(array, b => b.toString(16).padStart(2, '0')).join('');

  // Create SHA-256 hash and base64url encode it
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  const challenge = btoa(String.fromCharCode(...new Uint8Array(hash)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');

  return { verifier, challenge };
}

// Usage:
const pkce = await generatePkce();
// Store pkce.verifier securely (e.g., in session storage)
// Use pkce.challenge in the authorization URL
```

### Python Example

```python
import hashlib
import base64
import secrets

def generate_pkce():
    # Generate a random code verifier
    verifier = secrets.token_urlsafe(32)

    # Create SHA-256 hash and base64url encode it
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip('=')

    return verifier, challenge

# Usage:
verifier, challenge = generate_pkce()
# Store verifier securely in session
# Use challenge in the authorization URL
```

## Token Management

### Token Lifetimes

| Token Type | Lifetime |
|------------|----------|
| Access Token | 1 hour |
| Refresh Token | 30 days |
| Authorization Code | 10 minutes |

### Token Introspection

Check if a token is valid:

```http
POST /api/oauth/introspect
Content-Type: application/json

{
  "token": "agpt_xt_...",
  "token_type_hint": "access_token",
  "client_id": "{YOUR_CLIENT_ID}",
  "client_secret": "{YOUR_CLIENT_SECRET}"
}
```

**Response:**

```json
{
  "active": true,
  "scopes": ["EXECUTE_GRAPH", "READ_GRAPH"],
  "client_id": "agpt_client_...",
  "user_id": "user-uuid",
  "exp": 1705320000,
  "token_type": "access_token"
}
```

### Token Revocation

Revoke a token when the user logs out:

```http
POST /api/oauth/revoke
Content-Type: application/json

{
  "token": "agpt_xt_...",
  "token_type_hint": "access_token",
  "client_id": "{YOUR_CLIENT_ID}",
  "client_secret": "{YOUR_CLIENT_SECRET}"
}
```

## Security Best Practices

1. **Store client secrets securely** - Never expose them in client-side code or version control
2. **Always use PKCE** - Required for all authorization requests
3. **Validate state parameters** - Prevents CSRF attacks
4. **Use HTTPS** - All production redirect URIs must use HTTPS
5. **Request minimal scopes** - Only request the permissions your app needs
6. **Handle token expiration** - Implement automatic token refresh
7. **Revoke tokens on logout** - Clean up when users disconnect your app

## Error Handling

### Common OAuth Errors

| Error | Description | Solution |
|-------|-------------|----------|
| `invalid_client` | Client ID not found or inactive | Verify client ID is correct |
| `invalid_redirect_uri` | Redirect URI not registered | Register URI with platform admin |
| `invalid_scope` | Requested scope not allowed | Check allowed scopes for your app |
| `invalid_grant` | Code expired or already used | Authorization codes are single-use |
| `access_denied` | User denied authorization | Handle gracefully in your UI |

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 401 | Unauthorized (invalid/expired token) |
| 403 | Forbidden (insufficient scope) |
| 404 | Resource not found |

## Support

For issues or questions about OAuth integration:

- Open an issue on [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- See the [API Guide](api-guide.md) for general API information
- Check the [Swagger documentation](https://backend.agpt.co/external-api/docs) for endpoint details
