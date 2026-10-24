# AutoGPT Platform External API Guide

The AutoGPT Platform provides an External API that allows you to programmatically interact with agents, blocks, the store, and more.

## API Documentation

Full API documentation with interactive examples is available at:

**[https://backend.agpt.co/external-api/docs](https://backend.agpt.co/external-api/docs)**

This Swagger UI documentation includes all available endpoints, request/response schemas, and allows you to try out API calls directly.

{% hint style="info" %}
**Self-hosting?** Use your own server instead of `backend.agpt.co`. With Docker Compose the External API is at `http://localhost:8006/external-api/v1/...` and its Swagger UI at `http://localhost:8006/external-api/docs`. The single-container image serves the API under your public URL at `/_agpt/external-api/v1/...` and does not expose the Swagger UI.
{% endhint %}

## Authentication Methods

The External API supports three authentication methods:

### 1. API Keys

API keys are the simplest way to authenticate. Generate an API key from your AutoGPT Platform account settings and include it in your requests:

```http
GET /external-api/v1/blocks
X-API-Key: your_api_key_here
```

API keys are ideal for:
- Personal scripts and quick experiments
- Tools that run on your own machine

### 2. Keyless access (recommended for CI jobs, servers and websites)

Code running on GitHub Actions, Vercel, GitLab CI, Google Cloud or your own server can call the API without an API key. It proves where it runs with the identity token its platform gives it, and swaps that for an AutoGPT token that expires within the hour. There is no secret to store, leak or rotate:

```http
GET /external-api/v1/blocks
Authorization: Bearer agpt_wt_...
```

See the [Keyless API Access guide](keyless-access.md) for setup on each platform.

### 3. OAuth 2.0 (Single Sign-On)

For applications that need to act on behalf of users, use OAuth 2.0. This allows users to authorize your application to access their AutoGPT resources.

OAuth is ideal for:
- Third-party applications
- "Sign in with AutoGPT" (SSO, Single Sign-On) functionality
- Applications that need user-specific permissions

See the [OAuth Integration Guide](oauth-guide.md) for complete OAuth implementation details.

## Available Scopes

API keys, keyless workloads and OAuth applications all use the same scopes. Grant only the ones your integration needs:

| Scope | Description |
|-------|-------------|
| `IDENTITY` | Read user ID, e-mail, and timezone |
| `EXECUTE_GRAPH` | Run agents |
| `READ_GRAPH` | Read agent run results |
| `WRITE_GRAPH` | Create and update agent graphs |
| `EXECUTE_BLOCK` | Run individual blocks |
| `READ_BLOCK` | Read block definitions |
| `READ_STORE` | Access the agent store |
| `WRITE_LIBRARY` | Add agents to your library |
| `USE_TOOLS` | Use platform tools |
| `MANAGE_INTEGRATIONS` | Create and update user integrations |
| `READ_INTEGRATIONS` | Read user integration status |
| `DELETE_INTEGRATIONS` | Remove user integrations |

## Quick Start

### Using an API Key

```bash
# List available blocks
curl -H "X-API-Key: YOUR_API_KEY" \
  https://backend.agpt.co/external-api/v1/blocks
```

### Using OAuth

1. Register an OAuth application (contact platform administrator)
2. Implement the OAuth flow as described in the [OAuth Guide](oauth-guide.md)
3. Use the obtained access token:

```bash
curl -H "Authorization: Bearer agpt_xt_..." \
  https://backend.agpt.co/external-api/v1/blocks
```

## Support

For issues or questions about API integration:

- Open an issue on [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- Check the [Swagger documentation](https://backend.agpt.co/external-api/docs)
