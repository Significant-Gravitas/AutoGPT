# AutoGPT Platform External API Guide

The AutoGPT Platform provides an External API that allows you to programmatically interact with agents, blocks, the marketplace, and more.

## API Documentation

Full API documentation with interactive examples is available at:

- **Main**: [https://backend.agpt.co/external-api/docs](https://backend.agpt.co/external-api/docs)
- **v2 API**: [https://backend.agpt.co/external-api/v2/docs](https://backend.agpt.co/external-api/v2/docs)
- **v1 API**: [https://backend.agpt.co/external-api/v1/docs](https://backend.agpt.co/external-api/v1/docs)

The Swagger UI documentation includes all available endpoints, request/response schemas, and allows you to try out API calls directly.

**Recommendation**: New integrations should use the v2 API.

## Authentication Methods

The External API supports two authentication methods:

### 1. API Keys

API keys are the simplest way to authenticate. Generate an API key from your AutoGPT Platform account settings and include it in your requests using the `X-API-Key` header:

```bash
# List available blocks
curl -H "X-API-Key: YOUR_API_KEY" \
  https://backend.agpt.co/external-api/v1/blocks
```

API keys are ideal for:
- Server-to-server integrations
- Personal scripts and automation
- Backend services

### 2. OAuth 2.0 (Single Sign-On)

For applications that need to act on behalf of users, use OAuth 2.0. This allows users to authorize your application to access their AutoGPT resources.

To get started:

1. Register an OAuth application (contact platform administrator)
2. Implement the OAuth flow as described in the [OAuth Guide](oauth-guide.md)
3. Go through the OAuth flow to authorize your app and obtain an access token
4. Make API requests with the access token in the `Authorization: Bearer` header:

```bash
curl -H "Authorization: Bearer agpt_xt_..." \
  https://backend.agpt.co/external-api/v1/blocks
```

OAuth is ideal for:

- Third-party applications
- "Sign in with AutoGPT" (SSO, Single Sign-On) functionality
- Applications that need user-specific permissions

See the [OAuth Integration Guide](oauth-guide.md) for complete OAuth implementation details.

## Rate Limits

The v2 API enforces rate limits to ensure fair usage:

| Scope                        | Limit                            |
|------------------------------|----------------------------------|
| **Global (authenticated)**   | 200 requests per minute per user |
| **Global (unauthenticated)** | 5 requests per minute per IP     |

Some endpoints have additional per-endpoint limits (e.g. agent execution, file uploads, search). These are documented on each endpoint in the [v2 API docs](https://backend.agpt.co/external-api/v2/docs).

When a rate limit is exceeded, the API returns HTTP `429 Too Many Requests` with a JSON body:

```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded (200 requests per 60s). Try again shortly.",
    "details": null
  }
}
```

The numbers in the message are those of the limit that was hit.

## Errors

Every v2 response that is not 2xx has the same body:

```json
{
  "error": {
    "code": "not_found",
    "message": "Run #abc123 not found",
    "details": null
  }
}
```

`code` is a stable snake_case identifier — branch on it rather than on `message`,
which is written for humans and may be reworded. `details` carries structured
context when the failure has any (a `422` lists the fields that failed
validation) and is `null` otherwise.

| `code` | Status |
|--------|--------|
| `bad_request` | 400 |
| `unauthorized` | 401 |
| `payment_required` | 402 |
| `forbidden` | 403 |
| `not_found` | 404 |
| `conflict` | 409 |
| `validation_error` | 422 |
| `rate_limit_exceeded` | 429 |
| `internal_error` | 500 |
| `service_unavailable` | 503 |

## Pagination

Every list endpoint takes `limit` (1-100, default 20) and `cursor`, and returns:

```json
{
  "items": [],
  "next_cursor": "eyJwIjoyfQ"
}
```

Pass `next_cursor` back as `cursor` for the next page. `next_cursor` is `null`
on the last page. Cursors are opaque — do not parse or construct them.


## Available Scopes

When creating API keys or using OAuth, request only the scopes your application needs.

### v2 Scopes

| Scope | Description |
|-------|-------------|
| `READ_GRAPH` | Read graph definitions, versions, and the blocks a graph uses |
| `WRITE_GRAPH` | Create and update graphs, set the active version, change graph settings |
| `READ_BLOCK` | Read block definitions |
| `READ_STORE` | Read your own marketplace submissions |
| `WRITE_STORE` | Create, update, and delete marketplace submissions |
| `READ_LIBRARY` | List library agents, folders, and presets |
| `WRITE_LIBRARY` | Fork agents, add marketplace agents to your library, manage folders and presets |
| `RUN_AGENT` | Run agents and presets from your library |
| `READ_RUN` | List and get agent run details |
| `WRITE_RUN` | Stop and delete runs |
| `SHARE_RUN` | Share and unshare agent runs |
| `READ_RUN_REVIEW` | List human-in-the-loop reviews |
| `WRITE_RUN_REVIEW` | Submit human-in-the-loop review responses |
| `READ_SCHEDULE` | List execution schedules |
| `WRITE_SCHEDULE` | Create and delete schedules |
| `READ_CREDITS` | Get credit balance, transactions, invoices, and cost summaries |
| `READ_INTEGRATIONS` | List integration credentials |
| `MANAGE_INTEGRATIONS` | Create integration credentials |
| `DELETE_INTEGRATIONS` | Delete integration credentials |
| `READ_FILES` | List and download workspace files |
| `WRITE_FILES` | Upload and delete workspace files |

A few endpoints require two scopes at once:

| Endpoint | Scopes |
|----------|--------|
| `POST /graphs/{graph_id}/schedules` | `WRITE_SCHEDULE` + `RUN_AGENT` |
| `POST /library/presets/setup-trigger` | `WRITE_LIBRARY` + `RUN_AGENT` |
| `POST /runs/{run_id}/share` | `READ_RUN` + `SHARE_RUN` |

Public marketplace reads (`GET /marketplace/agents`, `/creators`, and their detail
routes) and `GET /search` require valid credentials but no particular scope.

### Legacy Scopes (v1 only)

| Scope | Description |
|-------|-------------|
| `IDENTITY` | Read user ID, e-mail, and timezone (v1 `GET /me`; v2 has no equivalent yet) |
| `EXECUTE_GRAPH` | Execute graphs directly (use `RUN_AGENT` in v2) |
| `EXECUTE_BLOCK` | Execute individual blocks |
| `USE_TOOLS` | Use chat tools via external API |

## Support

For issues or questions about API integration:

- Open an issue on [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- Check the [Swagger documentation](https://backend.agpt.co/external-api/docs)
