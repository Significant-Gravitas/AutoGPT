---
description: >-
  Point the same integration at AutoGPT Cloud or a self-hosted instance: base
  URLs, how to check them, and what differs when you host AutoGPT yourself.
icon: server
---

# Cloud and self-hosted

AutoGPT Cloud and self-hosted AutoGPT run the same API. An integration only needs to know two things: the instance's base URL and an API key created on that instance. Keep both in configuration, not in code:

```bash
AUTOGPT_API_URL=https://backend.agpt.co/external-api/v2
AUTOGPT_API_KEY=agpt_...
```

## Base URLs

| Where AutoGPT runs | `AUTOGPT_API_URL` |
| --- | --- |
| AutoGPT Cloud | `https://backend.agpt.co/external-api/v2` |
| Self-hosted with Docker Compose ([setup guide](../getting-started.md)) | `http://localhost:8006/external-api/v2` |
| Self-hosted single container ([setup guide](../single-container.md)) | `http://localhost:3000/_agpt/external-api/v2` |

For an instance on another machine or domain:

* **Docker Compose:** the API is served by the `rest_server` service on port `8006`. Use the address where that port is reachable, plus `/external-api/v2`, e.g. `https://autogpt-api.example.com/external-api/v2` behind a reverse proxy.
* **Single container:** use the container's `AUTOGPT_PUBLIC_URL` plus `/_agpt/external-api/v2`, e.g. `https://autogpt.example.com/_agpt/external-api/v2`.

The web app, where you create API keys, is on port `3000` in both self-hosted setups: `http://localhost:3000/settings/api-keys`.

## Check the URL and key

Run this before writing any code:

```bash
curl -s -w "\nHTTP %{http_code}\n" "$AUTOGPT_API_URL/me" -H "X-API-Key: $AUTOGPT_API_KEY"
```

| Result | Meaning |
| --- | --- |
| `HTTP 200` with your email | The URL and key are right. |
| `HTTP 401` | The URL is right, but the key is not valid **on this instance**. Keys from AutoGPT Cloud don't work on a self-hosted instance, and the reverse. |
| `HTTP 403` | The key works but lacks the **Identity** permission. Any other endpoint the key may use proves the URL too. |
| `HTTP 404` | Wrong path. Check the table above. If `$AUTOGPT_API_URL/openapi.json` also returns `404`, the instance runs a release from before API v2: upgrade it. |
| `HTTP 429` with `5 requests per 60s` | The request didn't carry a valid key, so it counted as anonymous. Fix the key and wait a minute. |
| No HTTP status | Nothing is listening at that address, or TLS failed. |

Every instance also serves its own reference for the API version it runs: the OpenAPI spec at `$AUTOGPT_API_URL/openapi.json` and an interactive explorer at `$AUTOGPT_API_URL/docs`.

## What's different on a self-hosted instance

The endpoints, requests and responses are the same. What differs is the account and the environment around them:

| Topic | AutoGPT Cloud | Self-hosted |
| --- | --- | --- |
| Accounts and API keys | Created at platform.agpt.co | Created in your instance's app. Keys only work on the instance that issued them. |
| Billing | Runs need an active plan and a positive balance, or they fail with `402 payment_required`. | No billing: runs are never refused or charged for, and `GET /credits` returns a fixed placeholder balance. |
| AI model blocks | Platform-provided model access is available. | Blocks that call model providers need provider keys configured on the instance, or credentials each user adds. See [AutoPilot on a self-hosted LLM](../copilot-local-llm.md) and [Advanced Setup](../advanced_setup.md). |
| Marketplace | The public AutoGPT marketplace. | The instance's own marketplace, separate from AutoGPT Cloud's. |
| Search (`GET /search`) | Indexed. | Uses the instance's own search index. A new instance may return few or no results until it's indexed. |
| OAuth apps | Registered by the AutoGPT team. | Register your own with `poetry run oauth-tool generate-app` in the backend. See [OAuth & SSO](oauth-guide.md). |
| Share links | `https://platform.agpt.co/share/...` | Built from the instance's public URL (`FRONTEND_BASE_URL` with Docker Compose, `AUTOGPT_PUBLIC_URL` in the single container), so set it to an address your readers can open. |
| Rate limits | As documented. | The same limits. |

### Behind a reverse proxy

Authenticated requests are rate-limited per user, so a proxy doesn't affect them. Unauthenticated requests are limited to 5 per minute **per client IP**, and behind a proxy every client can appear to come from the proxy's address and share that one allowance. If you rely on unauthenticated traffic, set `TRUSTED_PROXY_COUNT` in the backend's environment to the number of proxies in front of it, so the API reads the client address from `X-Forwarded-For`.

Serve the API over HTTPS whenever it is reachable from another machine. API keys travel in a request header and are as sensitive as a password.

## One integration for both

Read the base URL and key from configuration, and the same code runs against either:

{% tabs %}
{% tab title="Python" %}
```python
import os

API_URL = os.environ.get("AUTOGPT_API_URL", "https://backend.agpt.co/external-api/v2").rstrip("/")
HEADERS = {"X-API-Key": os.environ["AUTOGPT_API_KEY"]}
```
{% endtab %}

{% tab title="TypeScript" %}
```typescript
const API_URL = (process.env.AUTOGPT_API_URL ?? "https://backend.agpt.co/external-api/v2").replace(/\/$/, "");
const HEADERS = { "X-API-Key": process.env.AUTOGPT_API_KEY! };
```
{% endtab %}
{% endtabs %}

Things that legitimately vary by instance, and should be read at runtime rather than hard-coded: block IDs other than the core ones, agent and graph IDs, credential IDs, and marketplace agents. A graph JSON that uses a block missing from the instance fails with `400 bad_request` when you create it.
