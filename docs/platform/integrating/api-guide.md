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

#### Authentication attempt limits

API-key authentication allows 60 failed or in-flight attempts per client IP and
key prefix, and 120 per client IP across all prefixes, in a 60-second fixed
window. Successful validations refund their attempts, so normal sequential
requests with a valid key do not exhaust this budget. Large concurrent bursts
and clients sharing a NAT address share these limits. When throttled, wait for
the number of seconds in the HTTP 429 response's `Retry-After` header.

These checks run before expensive key verification. They are separate from any
quotas applied after authentication and never cache credentials or permissions.
The database filters candidates by the stored prefix and suffix; it still
verifies all matching active candidates as needed to preserve valid keys.

For self-hosting, configure Uvicorn's `FORWARDED_ALLOW_IPS` to trust only your
reverse proxy addresses, and prevent untrusted access through those addresses.
The limiter uses the ASGI client address and does not read raw
`X-Forwarded-For`. The single-container Nginx proxy replaces that header with
the connection's remote address and connects to Uvicorn over loopback. Additional
upstream proxies need their own trusted client-address configuration. Missing
client addresses, including invalid forwarded addresses from a misconfigured
trusted proxy, skip the IP limiter. Never trust all senders (`*`) on a directly
reachable backend. For a load balancer that appends multiple addresses, configure
the complete trusted proxy chain or rewrite the header at the trusted edge;
otherwise clients can share a proxy bucket or supply their own identity.

Redis admission has a 250ms deadline and fails open. Successful authentication
has a separate 250ms deadline for refunding its reservation; a failed refund
leaves that reservation charged until the window expires. These availability
fallbacks make working Redis and correct proxy configuration prerequisites for
enforcement.

### 2. OAuth 2.0 (Single Sign-On)

For applications that need to act on behalf of users, use OAuth 2.0. This allows users to authorize your application to access their AutoGPT resources.

OAuth is ideal for:
- Third-party applications
- "Sign in with AutoGPT" (SSO, Single Sign-On) functionality
- Applications that need user-specific permissions

See the [OAuth Integration Guide](oauth-guide.md) for complete OAuth implementation details.

## Available Scopes

When using OAuth, request only the scopes your application needs:

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
