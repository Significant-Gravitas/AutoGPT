---
description: >-
  Move an integration from the deprecated v1 API to v2 before v1 stops working
  on 2026-12-31: endpoint mapping, permission changes and behaviour changes.
icon: arrow-right-arrow-left
---

# Migrate from v1

API v1 (`/external-api/v1/...`) is deprecated and **stops working on 2026-12-31**. Both versions run side by side until then, on the same keys, so you can move one call at a time.

## What changes

* **Base URL.** `/external-api/v1` becomes `/external-api/v2`.
* **Runs go through your library.** v1 ran a graph by graph ID and version. v2 runs a **library agent**: look it up once with `GET /graphs/{graph_id}/library-agent`, then run it with `POST /library/agents/{agent_id}/runs`.
* **Permissions.** `RUN_AGENT` replaces `EXECUTE_GRAPH`, reading runs needs `READ_RUN`, and most v2 areas have their own read and write permissions. A v1 key usually needs new permissions; see the table below.
* **Lists are paginated.** Every list endpoint takes `limit` and `cursor` and returns `{"items", "next_cursor", "total_count"}`. Code that expected a bare array must read `items` and follow `next_cursor`.
* **Errors have one shape**: `{"error": {"code", "message", "details"}}` instead of `{"detail": ...}`. See [Errors](api-conventions.md#errors).
* **Requests act inside an organization.** The key decides which one; see [Organizations and teams](authentication.md#organizations-and-teams).
* **Rate limits and retries.** v2 enforces [documented limits](api-conventions.md#rate-limits) and supports `Idempotency-Key` on run creation.

## Endpoint mapping

| v1 | v2 | v2 permission |
| --- | --- | --- |
| `GET /me` | `GET /me`. Returns `user_id` instead of `id`, plus `organization`, `team`, `scopes` and `credential_type`. | `IDENTITY` |
| `GET /blocks` | `GET /blocks` (paginated) | `READ_BLOCK` |
| `POST /blocks/{block_id}/execute` | No direct equivalent. Put the block in a one-block agent and run it. | — |
| `POST /graphs` | `POST /graphs` | `WRITE_GRAPH` |
| `POST /graphs/{graph_id}/execute/{graph_version}` | `GET /graphs/{graph_id}/library-agent`, then `POST /library/agents/{agent_id}/runs` | `READ_LIBRARY`, `RUN_AGENT` |
| `GET /graphs/{graph_id}/executions/{graph_exec_id}/results` | `GET /runs/{run_id}`. Status and outputs in one object. | `READ_RUN` |
| `GET /store/agents` | `GET /marketplace/agents` | any valid key |
| `GET /store/agents/{username}/{agent_name}` | `GET /marketplace/agents/{username}/{agent_name}` | any valid key |
| `GET /store/creators` | `GET /marketplace/creators` | any valid key |
| `GET /store/creators/{username}` | `GET /marketplace/creators/{username}` | any valid key |
| `GET /integrations/credentials`, `GET /integrations/{provider}/credentials` | `GET /integrations/credentials?provider=...` | `READ_INTEGRATIONS` |
| `POST /integrations/{provider}/credentials` | `POST /integrations/credentials`, with `provider` in the body | `MANAGE_INTEGRATIONS` |
| `DELETE /integrations/{provider}/credentials/{cred_id}` | `DELETE /integrations/credentials/{credential_id}` | `DELETE_INTEGRATIONS` |
| `GET /integrations/providers` | No equivalent. Read the providers an agent needs from `GET /library/agents/{agent_id}/credentials`. | — |
| `POST /integrations/{provider}/oauth/initiate`, `.../oauth/complete` | No equivalent. Users connect OAuth providers in the app, or through the [integration setup wizard](oauth-guide.md#integration-setup-wizard). | — |
| `POST /tools/find-agent`, `POST /tools/run-agent` | The `find_agent`, `find_library_agent` and `run_agent` tools on the [MCP server](mcp-server.md), or the REST endpoints above. | as listed there |

## Running a graph: before and after

{% tabs %}
{% tab title="v1" %}
```bash
curl -s -X POST "https://backend.agpt.co/external-api/v1/graphs/$GRAPH_ID/execute/$GRAPH_VERSION" \
  -H "X-API-Key: $AUTOGPT_API_KEY" -H "Content-Type: application/json" \
  -d '{"node_input": {"topic": "Q3 results"}}'

curl -s "https://backend.agpt.co/external-api/v1/graphs/$GRAPH_ID/executions/$EXEC_ID/results" \
  -H "X-API-Key: $AUTOGPT_API_KEY"
```
{% endtab %}

{% tab title="v2" %}
```bash
# Once: find the library agent for the graph.
AGENT_ID=$(curl -s "$AUTOGPT_API_URL/graphs/$GRAPH_ID/library-agent" \
  -H "X-API-Key: $AUTOGPT_API_KEY" | jq -r .id)

# Start the run. Inputs go under "inputs".
RUN_ID=$(curl -s -X POST "$AUTOGPT_API_URL/library/agents/$AGENT_ID/runs" \
  -H "X-API-Key: $AUTOGPT_API_KEY" -H "Content-Type: application/json" \
  -H "Idempotency-Key: q3-report" \
  -d '{"inputs": {"topic": "Q3 results"}}' | jq -r .id)

# Read status and outputs until the run is COMPLETED, FAILED or TERMINATED.
curl -s "$AUTOGPT_API_URL/runs/$RUN_ID" -H "X-API-Key: $AUTOGPT_API_KEY"
```
{% endtab %}
{% endtabs %}

The v2 run object returns everything at once: `status`, `outputs` keyed by output name, `node_executions`, `cost_cents` and timings. See [Run agents](running-agents.md).

## Checklist

1. Add the v2 permissions your calls need to the key, or create a new key. Check with `GET /me`.
2. Change the base URL to `/external-api/v2`.
3. Replace graph runs with library-agent runs, and send an `Idempotency-Key`.
4. Read `items` and follow `next_cursor` on every list.
5. Parse errors from `error.code`.
6. Test against your instance, then remove every `/external-api/v1` call before 2026-12-31.
