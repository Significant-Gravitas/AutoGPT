---
description: >-
  The rules every AutoGPT API v2 endpoint follows: the error shape and codes,
  rate limits and retries, cursor pagination, idempotent runs and status codes.
icon: list-check
---

# Errors, rate limits, and pagination

Every v2 endpoint follows the conventions on this page. Write your client against them once and it handles every endpoint.

## Errors

Every response from the API that is not `2xx` has the same body:

```json
{
  "error": {
    "code": "not_found",
    "message": "Run #4b1f0c7e-... not found",
    "details": null
  }
}
```

* `code` is a stable, snake_case identifier. Branch on it.
* `message` is for people. It can be reworded at any time, so don't parse it.
* `details` carries structured context when there is any, and is `null` otherwise.
* Something in front of the API, such as a proxy or a wrong base URL, can answer with an HTML or empty body. Treat an error body that isn't this JSON as an unexpected error, and log its status.

### Error codes

| Status | `code` | What it means | Retry? |
| --- | --- | --- | --- |
| 400 | `bad_request` | The request is malformed: a bad cursor, an invalid graph, a value the operation can't accept. | No. Fix the request. |
| 401 | `unauthorized` | No credential, or an invalid or revoked one. | No. Fix the key. |
| 402 | `payment_required` | The account can't pay for this: no active plan, or a credit balance of zero. AutoGPT Cloud only. | No. Top up or subscribe, then retry. |
| 403 | `forbidden` | The credential lacks a permission, or isn't a member of the requested organization or team. | No. Grant the permission. |
| 404 | `not_found` | No such resource in the organization and team this request acts in. | No. |
| 405 | `method_not_allowed` | The path exists but not with this HTTP method. | No. |
| 409 | `conflict` | The request clashes with the current state, e.g. a file or folder name already in use, stopping a run that already finished, or a run with the same `Idempotency-Key` still being created. | Only the idempotency case: retry the same run start, with the same key, after a second. |
| 413 | `payload_too_large` | An upload is over the size limit. | No. |
| 422 | `validation_error` | The body or query parameters failed validation (`details.errors` lists each failing field), or an `Idempotency-Key` was reused for a different request. | No. Fix the request, or use a new key. |
| 428 | `precondition_required` | Something must be set up first; the message says what. | After doing it. |
| 429 | `rate_limit_exceeded` | A rate limit was hit. | Yes, after `Retry-After` seconds. |
| 500 | `internal_error` | Something failed on the server. The message names the operation but not the cause. | Yes, with backoff. Only retry a run start if you sent an `Idempotency-Key`. |
| 502 | `upstream_error` | A service AutoGPT depends on failed. | Yes, with backoff. |
| 503 | `service_unavailable` | A dependency is down or not configured on this instance, e.g. the store that holds `Idempotency-Key`s. | Yes, with backoff. A run start is safe to retry with the same key: it didn't start. |

### Validation errors

A `422` lists every field that failed in `details.errors`. `loc` is the path to the field:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid data for POST /external-api/v2/schedules",
    "details": {
      "errors": [
        {
          "type": "missing",
          "loc": ["body", "cron"],
          "msg": "Field required",
          "input": { "graph_id": "4b1f...", "name": "Daily" }
        }
      ]
    }
  }
}
```

## Rate limits

| Limit | Applies to | Window |
| --- | --- | --- |
| 200 requests | Every authenticated request, per user | 1 minute |
| 5 requests | Every unauthenticated request, per client IP, except reading the docs | 1 minute |
| 60 requests | Unauthenticated reads of `/openapi.json`, `/docs` and `/redoc`, per client IP | 1 minute |
| 30 failed sign-ins | Requests with an invalid key or token, per client IP. Past it, every request carrying a credential gets `429` until the window rolls over. | 1 minute |
| 60 requests | `POST /library/agents/{agent_id}/runs`, per user | 1 minute |
| 60 requests | `GET /credits/subscription`, per user | 1 minute |
| 30 requests | `GET /search`, per user | 1 minute |
| 20 requests | `POST /files/upload`, per user | 5 minutes |
| 10 requests | `POST /marketplace/submissions/media`, per user | 5 minutes |

The limits are per user, not per key: every key and token for the same account shares them. Each window is fixed and starts on a clock boundary, so a full window empties at the next minute (or five-minute) mark rather than one request at a time.

Every response carries your position in the window that applies to it: the endpoint's own limit for the endpoints in the table that have one, and the 200-per-minute window for everything else.

| Header | Meaning |
| --- | --- |
| `X-RateLimit-Limit` | Requests allowed in the current window. |
| `X-RateLimit-Remaining` | Requests left in the current window. |
| `X-RateLimit-Reset` | Seconds until the window resets. |

Header names are case-insensitive, and proxies may change their capitalisation (`X-Ratelimit-Limit`), so look them up case-insensitively. A `429` adds `Retry-After`, in seconds, and its `X-RateLimit-*` headers describe whichever limit was hit:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 23
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 23

{"error": {"code": "rate_limit_exceeded", "message": "Rate limit exceeded (60 requests per 60s). Try again shortly.", "details": null}}
```

### Retry safely

What you can retry depends on the request:

| Request | Retry on |
| --- | --- |
| Reads (`GET`) | `429`, `500`, `502`, `503`, and network errors. |
| Starting a run with an `Idempotency-Key` | The same, plus `409 conflict` (the first attempt is still starting the run). Always resend the same key. |
| Every other write (`POST`, `PUT`, `PATCH`, `DELETE`) | `429` only. The request was refused before it did anything. After a `5xx` or a network error the write may have happened, so check before you repeat it. |

Wait `Retry-After` seconds when it is present; otherwise back off exponentially with jitter. Give up after five attempts, or sooner if you're working to a deadline: inside a polling loop, pass a smaller `timeout` and fewer attempts so the retries fit in the time that's left. Don't retry any other `4xx`: it will fail the same way.

This helper takes the set of statuses to retry. Pass `READS` for reads, `RUN_STARTS` for a run start that carries an `Idempotency-Key`, and `OTHER_WRITES` for everything else:

{% tabs %}
{% tab title="Python" %}
```python
import random
import time

import requests

READS = frozenset({429, 500, 502, 503})  # network errors are retried too
RUN_STARTS = READS | {409}  # resend the same Idempotency-Key
OTHER_WRITES = frozenset({429})  # refused before it ran


def request_with_retries(method, url, retry_on, *, max_attempts=5, **kwargs):
    kwargs.setdefault("timeout", 30)
    for attempt in range(1, max_attempts + 1):
        last = attempt == max_attempts
        try:
            response = requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if last or retry_on is OTHER_WRITES:
                raise
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
            continue
        if response.status_code not in retry_on or last:
            return response
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(2**attempt, 30)
        time.sleep(delay + random.uniform(0, 1))
```
{% endtab %}

{% tab title="TypeScript" %}
```typescript
export const READS = new Set([429, 500, 502, 503]); // network errors are retried too
export const RUN_STARTS = new Set([...READS, 409]); // resend the same Idempotency-Key
export const OTHER_WRITES = new Set([429]); // refused before it ran

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function requestWithRetries(
  url: string,
  init: RequestInit,
  retryOn: Set<number>,
  maxAttempts = 5,
): Promise<Response> {
  for (let attempt = 1; ; attempt++) {
    const last = attempt === maxAttempts;
    let response: Response;
    try {
      response = await fetch(url, { ...init, signal: AbortSignal.timeout(30_000) });
    } catch (error) {
      if (last || retryOn === OTHER_WRITES) throw error;
      await sleep((Math.min(2 ** attempt, 30) + Math.random()) * 1000);
      continue;
    }
    if (!retryOn.has(response.status) || last) return response;
    const retryAfter = response.headers.get("Retry-After");
    const delay = retryAfter ? Number(retryAfter) : Math.min(2 ** attempt, 30);
    await sleep((delay + Math.random()) * 1000);
  }
}
```
{% endtab %}
{% endtabs %}

Retrying `POST /library/agents/{agent_id}/runs` after a timeout or `5xx` can start a second run, which you also pay for, unless you send an [`Idempotency-Key`](#idempotent-runs).

## Pagination

List endpoints take the same two query parameters and return the same envelope.

| Parameter | Meaning |
| --- | --- |
| `limit` | Items per page, from 1 to 100. Defaults to 20. |
| `cursor` | The `next_cursor` from the previous page. Leave it out for the first page. |

```json
{
  "items": [ { "id": "..." } ],
  "next_cursor": "eyJ2IjoxLCJrIjoicCIsInAiOjJ9",
  "total_count": 137
}
```

* Pass `next_cursor` back as `cursor` to get the next page. It is `null` on the last page.
* Cursors are opaque: don't decode, build or edit them. A malformed cursor is rejected with `400 bad_request`.
* Use a cursor only on the endpoint that returned it, with the same filters. Another endpoint may accept it and return the wrong page.
* `total_count` counts the matches across all pages. It is always present, and `null` where the source can't count: `GET /credits/transactions` (it groups the charges of one run into one item) and `GET /credits/invoices`.
* `GET /credits/invoices` is the one list that doesn't page: its `next_cursor` is always `null`. Raise `limit` to see further back. The helpers below read that one page and stop.

Read every page like this:

{% tabs %}
{% tab title="Python" %}
```python
def list_all(path, params=None):
    params = dict(params or {}, limit=100)
    while True:
        response = requests.get(
            f"{API_URL}{path}", headers=HEADERS, params=params, timeout=30
        )
        response.raise_for_status()
        page = response.json()
        yield from page["items"]
        if not page["next_cursor"]:
            return
        params["cursor"] = page["next_cursor"]


for agent in list_all("/library/agents"):
    print(agent["id"], agent["name"])
```
{% endtab %}

{% tab title="TypeScript" %}
```typescript
async function* listAll<T>(path: string, params: Record<string, string> = {}) {
  const query = new URLSearchParams({ ...params, limit: "100" });
  while (true) {
    const response = await fetch(`${API_URL}${path}?${query}`, {
      headers: HEADERS,
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
    const page = (await response.json()) as { items: T[]; next_cursor: string | null };
    yield* page.items;
    if (!page.next_cursor) return;
    query.set("cursor", page.next_cursor);
  }
}

for await (const agent of listAll<{ id: string; name: string }>("/library/agents")) {
  console.log(agent.id, agent.name);
}
```
{% endtab %}
{% endtabs %}

## Idempotent runs

Starting a run costs money, and a request that times out tells you nothing about whether the run started. Send an `Idempotency-Key` header with `POST /library/agents/{agent_id}/runs`, and a retry with the same key returns the run the first request started instead of starting another.

```bash
curl -s -X POST "$AUTOGPT_API_URL/library/agents/$AGENT_ID/runs" \
  -H "X-API-Key: $AUTOGPT_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: order-1234-summary" \
  -d '{"inputs": {"topic": "Q3 results"}}'
```

* Use a value that identifies the job, such as an order ID or a UUID you store with it. Up to 255 characters. A key made up fresh for each call protects only that call's own retries, so a library should let its callers pass the key.
* A key lasts 24 hours and is scoped to your user and organization.
* A key is bound to the request it first came with: the agent and the whole body. The first request with a key starts the run, and every later request with that key and the same agent and body returns `202` and the same run. The same key with another agent or a different body is `422 validation_error`. To start a different run, use a new key.
* If the first request is still starting the run, a duplicate gets `409 conflict`. Wait a moment and retry with the same key.
* If the key can't be checked (the store that holds keys is unreachable), the request is `503 service_unavailable` and no run starts. Retry with the same key.
* If `409` persists, the first request probably started a run but couldn't record it under the key, and the key stays blocked until it expires. Don't start the job again under a new key until you've looked for that run with `GET /runs?graph_id=...&started_after=...`.
* If the first request failed before a run started, the key is released and a retry starts the run.

## Status codes

| Status | Returned when |
| --- | --- |
| `200 OK` | A read or an update succeeded. |
| `201 Created` | A resource was created, e.g. `POST /graphs`, `POST /schedules`, `POST /files/upload`. |
| `202 Accepted` | Work was queued and continues after the response, e.g. starting or stopping a run, submitting reviews. |
| `204 No Content` | A delete succeeded. The body is empty. |

## Formats

* Requests and responses are JSON (`Content-Type: application/json`), except uploads (`multipart/form-data`) and file downloads (the file's own type).
* IDs are opaque strings. Today they are UUIDs; don't depend on that.
* Timestamps are ISO 8601 with a time zone, e.g. `2026-10-08T14:03:12.418000Z`.
* Money is in US cents in fields ending `_cents`, e.g. `balance_cents: 1250` is $12.50.
* Enum values are UPPER_CASE strings, e.g. `"status": "COMPLETED"`.
* Fields can be added to any response at any time. Ignore the ones you don't know.
