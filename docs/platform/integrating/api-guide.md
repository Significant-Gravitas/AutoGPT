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

### Organizations and teams

Every v2 request acts inside exactly one organization. An API key is created in
an organization and carries it for its whole life, so the key itself decides
which organization's agents, runs and credits you reach — there is no
per-request organization parameter. A key created before organizations existed
acts in your personal organization, which is what a personal account already
is.

`GET /external-api/v2/me` reports the organization and team a key acts in:

```bash
curl -H "X-API-Key: YOUR_API_KEY" https://backend.agpt.co/external-api/v2/me
```

To act inside one team of that organization, either pin the key to a team when
you create it, or send `X-Team-Id` on the request:

```bash
curl -H "X-API-Key: YOUR_API_KEY" -H "X-Team-Id: TEAM_ID" \
  https://backend.agpt.co/external-api/v2/library/agents
```

A key already pinned to a team rejects an `X-Team-Id` naming a different one.
Without either, the request acts organization-wide, and what it creates is
visible to every member of the organization.

Resources you create carry the organization and team you acted in, and runs are
billed to that organization's balance. To work across several organizations,
create one key per organization.

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

## Retrying a run

`POST /library/agents/{id}/runs` and `POST /library/presets/{id}/runs` take an
`Idempotency-Key` header. Repeating a request with the same value returns the run
the first one started instead of starting a second one and charging for it again.
A retry sent while the first request is still in flight gets `409`; retry once it
completes. Keys are scoped to the caller and expire after 24 hours.

```bash
curl -X POST https://backend.agpt.co/external-api/v2/library/agents/$ID/runs \
  -H "X-API-Key: $KEY" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {}}'
```

## Pagination

Every list endpoint takes the same two query parameters and returns the same
envelope:

| Parameter | |
|-----------|--|
| `limit` | Items per page. 1-100, default 20. |
| `cursor` | The previous response's `next_cursor`. Omit for the first page. |

```json
{
  "items": [],
  "next_cursor": "eyJ2IjoxLCJrIjoicCIsInAiOjJ9",
  "total_count": 137
}
```

Pass `next_cursor` back as `cursor` for the next page; it is `null` on the last
page. Cursors are opaque — do not parse or construct them, and do not carry one
from one endpoint to another: a cursor that did not come from this endpoint's
last response is rejected with `400 bad_request`.

`total_count` is the number of items matching the request across all pages. It
is present on every list endpoint, and `null` on the two where the source cannot
report one: `/credits/invoices` (Stripe does not return a total) and
`/credits/transactions` (the history groups raw rows, so a row count would not
match what paging yields).


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
| `IDENTITY` | Read who the credentials act as, and in which organization (`GET /me`) |
| `READ_FILES` | List and download workspace files |
| `WRITE_FILES` | Upload and delete workspace files |
| `USE_TOOLS` | Use MCP tools that spend platform resources: web search, web fetch, feature requests |

A few endpoints require two scopes at once:

| Endpoint | Scopes |
|----------|--------|
| `POST /schedules` | `WRITE_SCHEDULE` + `RUN_AGENT` |
| `POST /library/presets/setup-trigger` | `WRITE_LIBRARY` + `RUN_AGENT` |
| `POST /runs/{run_id}/share` | `READ_RUN` + `SHARE_RUN` |
| `DELETE /runs/{run_id}/share` | `READ_RUN` + `SHARE_RUN` |

Public marketplace reads (`GET /marketplace/agents`, `/creators`, and their detail
routes) require valid credentials but no particular scope, and so does `GET /search`
over public content. Searching your own content costs the scope that endpoint costs:
`content_types=LIBRARY_AGENT` needs `READ_LIBRARY`, `WORKSPACE_FILE` needs `READ_FILES`.

### Legacy Scopes (v1 only)

| Scope | Description |
|-------|-------------|
| `EXECUTE_GRAPH` | Execute graphs directly (use `RUN_AGENT` in v2) |
| `EXECUTE_BLOCK` | Execute individual blocks |

## Support

For issues or questions about API integration:

- Open an issue on [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- Check the [Swagger documentation](https://backend.agpt.co/external-api/docs)
