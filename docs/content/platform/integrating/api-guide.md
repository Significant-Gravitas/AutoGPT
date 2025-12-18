# AutoGPT Platform External API Guide

The AutoGPT Platform provides an External API that allows you to programmatically interact with agents, blocks, the store, and more.

## API Documentation

Full API documentation with interactive examples is available at:

**[https://backend.agpt.co/external-api/docs](https://backend.agpt.co/external-api/docs)**

This Swagger UI documentation includes all available endpoints, request/response schemas, and allows you to try out API calls directly.

## Authentication Methods

The External API supports two authentication methods:

### 1. API Keys

API keys are the simplest way to authenticate. Generate an API key from your AutoGPT Platform account settings and include it in your requests:

```http
GET /external-api/v1/blocks
X-API-Key: your_api_key_here
```

API keys are ideal for:
- Server-to-server integrations
- Personal scripts and automation
- Backend services

### 2. OAuth 2.0 (Single Sign-On)

For applications that need to act on behalf of users, use OAuth 2.0. This allows users to authorize your application to access their AutoGPT resources.

OAuth is ideal for:
- Third-party applications
- "Sign in with AutoGPT" (SSO, Single Sign-On) functionality
- Applications that need user-specific permissions

See the [SSO Integration Guide](sso-guide.md) for complete OAuth implementation details.

## Available Scopes

When using OAuth, request only the scopes your application needs:

| Scope | Description |
|-------|-------------|
| `IDENTITY` | Read user ID, e-mail, and timezone |
| `EXECUTE_GRAPH` | Run agents |
| `READ_GRAPH` | Read agent run results |
| `EXECUTE_BLOCK` | Run individual blocks |
| `READ_BLOCK` | Read block definitions |
| `READ_STORE` | Access the agent store |
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
2. Implement the OAuth flow as described in the [SSO Guide](sso-guide.md)
3. Use the obtained access token:

```bash
curl -H "Authorization: Bearer agpt_xt_..." \
  https://backend.agpt.co/external-api/v1/blocks
```

## Support

For issues or questions about API integration:

- Open an issue on [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- Check the [Swagger documentation](https://backend.agpt.co/external-api/docs)
