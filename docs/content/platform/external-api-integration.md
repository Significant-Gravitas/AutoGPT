# External API Integration Guide

This guide explains how third-party applications can integrate with AutoGPT Platform to execute agents on behalf of users using the OAuth Provider and Credential Broker system.

## Overview

The AutoGPT External API allows your application to:

- **Execute agents** - Run user-owned or marketplace agents with user-granted credentials
- **Access integrations** - Use third-party service credentials (Google, GitHub, etc.) that users have connected
- **Receive webhooks** - Get notified when agent executions complete

The integration uses standard OAuth 2.0 with PKCE for secure authentication and a popup-based "Connect" flow for credential access.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Your App       │────▶│  AutoGPT OAuth   │────▶│  External API   │
│                 │     │  Provider        │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Credential      │
                        │  Broker          │
                        └──────────────────┘
```

**Key concepts:**

1. **OAuth Client** - Your registered application with AutoGPT
2. **OAuth Tokens** - Access/refresh tokens for API authentication
3. **Credential Grants** - User permissions to use their connected integrations
4. **Integration Scopes** - Specific permissions for each integration (e.g., `google:gmail.readonly`)

## Getting Started

### 1. Register Your OAuth Client

Register your application to get a `client_id` and `client_secret`:

```bash
# Requires user authentication (JWT token)
curl -X POST https://platform.agpt.co/oauth/clients/ \
  -H "Authorization: Bearer <user_jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My App",
    "description": "Description of your app",
    "client_type": "confidential",
    "redirect_uris": ["https://myapp.com/oauth/callback"],
    "webhook_domains": ["myapp.com", "*.myapp.com"]
  }'
```

**Response:**
```json
{
  "client_id": "app_abc123xyz",
  "client_secret": "secret_xyz789..."
}
```

> **Important:** Store the `client_secret` securely - it's only shown once!

**Client types:**
- `confidential` - Server-side apps that can securely store secrets
- `public` - Browser/mobile apps (no client secret)

### 2. OAuth Authorization Flow

Use the standard OAuth 2.0 Authorization Code flow with PKCE to get user consent and tokens.

#### Generate PKCE Parameters

```javascript
// Generate code verifier and challenge
function generateCodeVerifier() {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return base64UrlEncode(array);
}

async function generateCodeChallenge(verifier) {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return base64UrlEncode(new Uint8Array(digest));
}
```

#### Redirect User to Authorization

```javascript
const state = crypto.randomUUID(); // Store this for validation
const codeVerifier = generateCodeVerifier(); // Store this securely
const codeChallenge = await generateCodeChallenge(codeVerifier);

const authUrl = new URL('https://platform.agpt.co/oauth/authorize');
authUrl.searchParams.set('response_type', 'code');
authUrl.searchParams.set('client_id', CLIENT_ID);
authUrl.searchParams.set('redirect_uri', REDIRECT_URI);
authUrl.searchParams.set('state', state);
authUrl.searchParams.set('code_challenge', codeChallenge);
authUrl.searchParams.set('code_challenge_method', 'S256');
authUrl.searchParams.set('scope', 'openid profile email agents:execute integrations:connect');

window.location.href = authUrl.toString();
```

**Available scopes:**

| Scope | Description |
|-------|-------------|
| `openid` | Required for OIDC |
| `profile` | Access user profile (name) |
| `email` | Access user email |
| `agents:execute` | Execute agents and check status |
| `integrations:list` | List user's credential grants |
| `integrations:connect` | Request new credential grants |
| `integrations:delete` | Delete credentials via grants |

#### Handle OAuth Callback

```javascript
// Your callback endpoint receives: ?code=xxx&state=xxx
app.get('/oauth/callback', async (req, res) => {
  const { code, state } = req.query;

  // Verify state matches what you stored
  if (state !== storedState) {
    return res.status(400).send('Invalid state');
  }

  // Exchange code for tokens
  const response = await fetch('https://platform.agpt.co/oauth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: REDIRECT_URI,
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      code_verifier: storedCodeVerifier,
    }),
  });

  const tokens = await response.json();
  // { access_token, refresh_token, token_type, expires_in }
});
```

### 3. Request Credential Grants (Connect Flow)

Before executing agents that require integrations (like Gmail, Google Sheets, GitHub), you need credential grants from the user.

#### Open Connect Popup

```javascript
function requestCredentialGrant(provider, scopes) {
  const nonce = crypto.randomUUID();

  // Store nonce to validate response
  sessionStorage.setItem('connect_nonce', nonce);

  const connectUrl = new URL(`https://platform.agpt.co/connect/${provider}`);
  connectUrl.searchParams.set('client_id', CLIENT_ID);
  connectUrl.searchParams.set('scopes', scopes.join(','));
  connectUrl.searchParams.set('nonce', nonce);
  connectUrl.searchParams.set('redirect_origin', window.location.origin);

  // Open popup (user must be logged into AutoGPT)
  const popup = window.open(
    connectUrl.toString(),
    'AutoGPT Connect',
    'width=500,height=600,popup=true'
  );

  // Listen for result
  window.addEventListener('message', handleConnectResult, { once: true });
}

function handleConnectResult(event) {
  // Verify origin
  if (event.origin !== 'https://platform.agpt.co') return;

  const data = event.data;
  if (data.type !== 'autogpt_connect_result') return;

  // Verify nonce
  if (data.nonce !== sessionStorage.getItem('connect_nonce')) return;

  if (data.success) {
    console.log('Grant created:', data.grant_id);
    console.log('Credential ID:', data.credential_id);
    console.log('Provider:', data.provider);
  } else {
    console.error('Connect failed:', data.error, data.error_description);
  }
}
```

**Integration scopes by provider:**

| Provider | Available Scopes |
|----------|------------------|
| Google | `google:gmail.readonly`, `google:gmail.send`, `google:sheets.read`, `google:sheets.write`, `google:calendar.read`, `google:calendar.write`, `google:drive.read`, `google:drive.write` |
| GitHub | `github:repo.read`, `github:repo.write`, `github:user.read` |
| Twitter/X | `twitter:tweet.read`, `twitter:tweet.write`, `twitter:user.read` |
| Linear | `linear:read`, `linear:write` |
| Notion | `notion:read`, `notion:write` |
| Slack | `slack:read`, `slack:write` |

### 4. Execute Agents

With an OAuth token and credential grants, you can execute agents:

```javascript
async function executeAgent(agentId, inputs, grantIds = null, webhookUrl = null) {
  const response = await fetch(
    `https://platform.agpt.co/api/external/v1/executions/agents/${agentId}/execute`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs,
        grant_ids: grantIds,  // Optional: specific grants to use
        webhook_url: webhookUrl,  // Optional: receive completion webhook
      }),
    }
  );

  const result = await response.json();
  // { execution_id, status: "queued", message: "..." }
  return result;
}
```

### 5. Check Execution Status

Poll for execution status or use webhooks:

```javascript
async function getExecutionStatus(executionId) {
  const response = await fetch(
    `https://platform.agpt.co/api/external/v1/executions/${executionId}`,
    {
      headers: { 'Authorization': `Bearer ${accessToken}` },
    }
  );

  return await response.json();
  // {
  //   execution_id,
  //   status: "queued" | "running" | "completed" | "failed",
  //   started_at,
  //   completed_at,
  //   outputs,  // Present when completed
  //   error,    // Present when failed
  // }
}
```

### 6. Handle Webhooks

If you provided a `webhook_url`, you'll receive POST requests with execution events:

```javascript
app.post('/webhooks/autogpt', (req, res) => {
  // Verify webhook signature (if configured)
  const signature = req.headers['x-webhook-signature'];
  const timestamp = req.headers['x-webhook-timestamp'];

  if (signature) {
    const expectedSignature = crypto
      .createHmac('sha256', WEBHOOK_SECRET)
      .update(JSON.stringify(req.body))
      .digest('hex');

    if (signature !== `sha256=${expectedSignature}`) {
      return res.status(401).send('Invalid signature');
    }
  }

  const { event, timestamp, data } = req.body;

  switch (event) {
    case 'execution.started':
      console.log(`Execution ${data.execution_id} started`);
      break;
    case 'execution.completed':
      console.log(`Execution ${data.execution_id} completed`, data.outputs);
      break;
    case 'execution.failed':
      console.error(`Execution ${data.execution_id} failed:`, data.error);
      break;
    case 'grant.revoked':
      console.log(`Grant ${data.grant_id} was revoked`);
      break;
  }

  res.status(200).send('OK');
});
```

> **Note:** Webhook URLs must match domains registered in your OAuth client's `webhook_domains`.

## API Reference

### External API Endpoints

Base URL: `https://platform.agpt.co/api/external/v1`

| Method | Endpoint | Scope Required | Description |
|--------|----------|----------------|-------------|
| GET | `/executions/capabilities` | `agents:execute` | Get available grants and scopes |
| POST | `/executions/agents/{agent_id}/execute` | `agents:execute` | Execute an agent |
| GET | `/executions/{execution_id}` | `agents:execute` | Get execution status |
| POST | `/executions/{execution_id}/cancel` | `agents:execute` | Cancel execution |
| GET | `/grants/` | `integrations:list` | List credential grants |
| GET | `/grants/{grant_id}` | `integrations:list` | Get grant details |
| DELETE | `/grants/{grant_id}/credential` | `integrations:delete` | Delete credential via grant |

### OAuth Endpoints

Base URL: `https://platform.agpt.co`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/oauth/authorize` | Authorization endpoint |
| POST | `/oauth/token` | Token endpoint |
| GET | `/oauth/userinfo` | OIDC UserInfo |
| POST | `/oauth/revoke` | Revoke tokens |
| GET | `/.well-known/openid-configuration` | OIDC Discovery |

### Client Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/oauth/clients/` | Register new client |
| GET | `/oauth/clients/` | List your clients |
| GET | `/oauth/clients/{client_id}` | Get client details |
| PATCH | `/oauth/clients/{client_id}` | Update client |
| DELETE | `/oauth/clients/{client_id}` | Delete client |
| POST | `/oauth/clients/{client_id}/rotate-secret` | Rotate client secret |

## Rate Limits

| Endpoint Type | Limit |
|--------------|-------|
| OAuth endpoints | 20-30 requests/minute |
| Agent execution | 10 requests/minute, 100/hour |
| Read endpoints | 60 requests/minute, 1000/hour |

Rate limit headers are included in responses:
- `X-RateLimit-Remaining` - Requests remaining in current window
- `X-RateLimit-Reset` - Unix timestamp when limit resets
- `Retry-After` - Seconds to wait (when rate limited)

## Error Handling

### OAuth Errors

```json
{
  "error": "invalid_grant",
  "error_description": "Authorization code has expired"
}
```

Common OAuth errors:
- `invalid_client` - Unknown or invalid client
- `invalid_grant` - Expired/invalid authorization code
- `access_denied` - User denied consent
- `invalid_scope` - Requested scope not allowed

### API Errors

```json
{
  "detail": "Grant validation failed: No valid grants found for requested integrations"
}
```

HTTP status codes:
- `400` - Bad request (invalid parameters)
- `401` - Unauthorized (invalid/expired token)
- `403` - Forbidden (insufficient scopes or grants)
- `404` - Resource not found
- `429` - Rate limited
- `500` - Internal server error

## Security Best Practices

1. **Store secrets securely** - Never expose `client_secret` in client-side code
2. **Validate state parameter** - Prevent CSRF attacks
3. **Use PKCE** - Required for all authorization flows
4. **Verify webhook signatures** - Prevent spoofed webhooks
5. **Request minimal scopes** - Only request what you need
6. **Handle token refresh** - Refresh tokens before they expire
7. **Validate redirect origins** - Only accept messages from expected origins

## Complete Integration Example

```javascript
class AutoGPTClient {
  constructor(clientId, clientSecret, redirectUri) {
    this.clientId = clientId;
    this.clientSecret = clientSecret;
    this.redirectUri = redirectUri;
    this.baseUrl = 'https://platform.agpt.co';
  }

  // Step 1: Generate authorization URL
  async getAuthorizationUrl(scopes) {
    const state = crypto.randomUUID();
    const codeVerifier = this.generateCodeVerifier();
    const codeChallenge = await this.generateCodeChallenge(codeVerifier);

    // Store for callback
    sessionStorage.setItem('oauth_state', state);
    sessionStorage.setItem('oauth_verifier', codeVerifier);

    const url = new URL(`${this.baseUrl}/oauth/authorize`);
    url.searchParams.set('response_type', 'code');
    url.searchParams.set('client_id', this.clientId);
    url.searchParams.set('redirect_uri', this.redirectUri);
    url.searchParams.set('state', state);
    url.searchParams.set('code_challenge', codeChallenge);
    url.searchParams.set('code_challenge_method', 'S256');
    url.searchParams.set('scope', scopes.join(' '));

    return url.toString();
  }

  // Step 2: Exchange code for tokens
  async exchangeCode(code, state) {
    if (state !== sessionStorage.getItem('oauth_state')) {
      throw new Error('Invalid state');
    }

    const response = await fetch(`${this.baseUrl}/oauth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code,
        redirect_uri: this.redirectUri,
        client_id: this.clientId,
        client_secret: this.clientSecret,
        code_verifier: sessionStorage.getItem('oauth_verifier'),
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    return response.json();
  }

  // Step 3: Request credential grant via popup
  requestGrant(provider, scopes) {
    return new Promise((resolve, reject) => {
      const nonce = crypto.randomUUID();

      const url = new URL(`${this.baseUrl}/connect/${provider}`);
      url.searchParams.set('client_id', this.clientId);
      url.searchParams.set('scopes', scopes.join(','));
      url.searchParams.set('nonce', nonce);
      url.searchParams.set('redirect_origin', window.location.origin);

      const popup = window.open(url.toString(), 'connect', 'width=500,height=600');

      const handler = (event) => {
        if (event.origin !== this.baseUrl) return;
        if (event.data?.type !== 'autogpt_connect_result') return;
        if (event.data?.nonce !== nonce) return;

        window.removeEventListener('message', handler);

        if (event.data.success) {
          resolve(event.data);
        } else {
          reject(new Error(event.data.error_description));
        }
      };

      window.addEventListener('message', handler);
    });
  }

  // Step 4: Execute agent
  async executeAgent(accessToken, agentId, inputs, options = {}) {
    const response = await fetch(
      `${this.baseUrl}/api/external/v1/executions/agents/${agentId}/execute`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs,
          grant_ids: options.grantIds,
          webhook_url: options.webhookUrl,
        }),
      }
    );

    if (!response.ok) {
      throw new Error(await response.text());
    }

    return response.json();
  }

  // Step 5: Poll for completion
  async waitForCompletion(accessToken, executionId, timeoutMs = 300000) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
      const response = await fetch(
        `${this.baseUrl}/api/external/v1/executions/${executionId}`,
        { headers: { 'Authorization': `Bearer ${accessToken}` } }
      );

      const status = await response.json();

      if (status.status === 'completed') {
        return status.outputs;
      }

      if (status.status === 'failed') {
        throw new Error(status.error || 'Execution failed');
      }

      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    throw new Error('Execution timeout');
  }

  // Helper methods
  generateCodeVerifier() {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return this.base64UrlEncode(array);
  }

  async generateCodeChallenge(verifier) {
    const encoder = new TextEncoder();
    const data = encoder.encode(verifier);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return this.base64UrlEncode(new Uint8Array(digest));
  }

  base64UrlEncode(buffer) {
    return btoa(String.fromCharCode(...buffer))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }
}

// Usage
const client = new AutoGPTClient(
  'app_abc123',
  'secret_xyz789',
  'https://myapp.com/oauth/callback'
);

// 1. Redirect to authorization
const authUrl = await client.getAuthorizationUrl([
  'openid', 'profile', 'agents:execute', 'integrations:connect'
]);
window.location.href = authUrl;

// 2. After callback, exchange code
const tokens = await client.exchangeCode(code, state);

// 3. Request Google credentials
const grant = await client.requestGrant('google', ['google:gmail.readonly']);

// 4. Execute an agent
const execution = await client.executeAgent(
  tokens.access_token,
  'agent-uuid-here',
  { query: 'Search my emails for invoices' },
  { grantIds: [grant.grant_id] }
);

// 5. Wait for results
const outputs = await client.waitForCompletion(
  tokens.access_token,
  execution.execution_id
);
console.log('Agent outputs:', outputs);
```

## Support

- [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues) - Bug reports and feature requests
- [Discord Community](https://discord.gg/autogpt) - Community support
