---
description: >-
  Run, build and manage AutoGPT agents from your own code. One REST API, the
  same on AutoGPT Cloud and on a self-hosted instance.
icon: code
---

# AutoGPT Platform API

The AutoGPT Platform API lets your code do what you do in the AutoGPT web app: run agents from your library, read their outputs, build new agents, schedule them, answer their human-in-the-loop reviews, and manage files and credentials. The same API ships with AutoGPT Cloud and with every self-hosted instance, so an integration written against one works against the other by changing a single URL.

This section documents **API v2**. v1 is deprecated and stops working on **2026-12-31**; see [Migrate from v1](migrate-from-v1.md).

## Start here

<table data-view="cards"><thead><tr><th></th><th></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td><strong>Quickstart</strong></td><td>Make your first call and run an agent in about five minutes.</td><td><a href="quickstart.md">quickstart.md</a></td></tr><tr><td><strong>Build with AI coding agents</strong></td><td>Copy-paste prompts, llms.txt and MCP set-up for Claude Code, Codex, Cursor and others.</td><td><a href="ai-coding-agents.md">ai-coding-agents.md</a></td></tr><tr><td><strong>Run agents</strong></td><td>Inputs, credentials, the run lifecycle, outputs, reviews, schedules and files.</td><td><a href="running-agents.md">running-agents.md</a></td></tr><tr><td><strong>Build agents</strong></td><td>Create and version agent graphs from code.</td><td><a href="building-agents.md">building-agents.md</a></td></tr><tr><td><strong>MCP server</strong></td><td>Give Claude, Cursor or any MCP client tools to find, build and run agents.</td><td><a href="mcp-server.md">mcp-server.md</a></td></tr><tr><td><strong>API reference</strong></td><td>Every endpoint, parameter and response, generated from the OpenAPI spec.</td><td><a href="api-reference.md">api-reference.md</a></td></tr></tbody></table>

## Base URL

Every endpoint in these docs is relative to the base URL of your instance. Put it in `AUTOGPT_API_URL` and every example on this site works unchanged.

| Where AutoGPT runs | Base URL (`AUTOGPT_API_URL`) | Create API keys at |
| --- | --- | --- |
| AutoGPT Cloud | `https://backend.agpt.co/external-api/v2` | [platform.agpt.co/settings/api-keys](https://platform.agpt.co/settings/api-keys) |
| Self-hosted with Docker Compose | `http://localhost:8006/external-api/v2` | `http://localhost:3000/settings/api-keys` |
| Self-hosted single container | `http://localhost:3000/_agpt/external-api/v2` | `http://localhost:3000/settings/api-keys` |

Self-hosted on another machine or domain? Replace `localhost` and the port with yours. [Cloud and self-hosted](environments.md) explains how to find and check the right URL.

## Your first request

Create an API key with the **Identity** permission, then:

```bash
export AUTOGPT_API_URL="https://backend.agpt.co/external-api/v2"
export AUTOGPT_API_KEY="agpt_..."

curl -s "$AUTOGPT_API_URL/me" -H "X-API-Key: $AUTOGPT_API_KEY"
```

The response names the account, organization and permissions the key acts with. The [Quickstart](quickstart.md) continues from here to a finished agent run.

## How the API works

* **Authentication.** Send an API key in the `X-API-Key` header, or an API key or OAuth access token as `Authorization: Bearer`. Each key carries only the permissions you grant it. See [Authentication and permissions](authentication.md).
* **Organizations.** Every request acts inside one organization, fixed by the key. Add `X-Team-Id` to act inside one of its teams.
* **Running agents.** You run agents that are in your library. A run is asynchronous: starting one returns `202` and a run ID, and you poll the run until it reaches a final status. See [Run agents](running-agents.md).
* **One list shape.** Every list endpoint takes `limit` and `cursor` and returns `{"items", "next_cursor", "total_count"}`.
* **One error shape.** Every non-2xx response is `{"error": {"code", "message", "details"}}`. Branch on `code`.
* **Limits.** 200 requests per minute per user, plus lower limits on a few expensive endpoints. Every response carries `X-RateLimit-*` headers.
* **Safe retries.** Send an `Idempotency-Key` when you start a run, and a retry returns the original run instead of starting (and paying for) a second one.

[Errors, rate limits, and pagination](api-conventions.md) covers these conventions in full.

## Machine-readable resources

For tools and AI agents, everything on this site is available as plain text:

| Resource | URL |
| --- | --- |
| OpenAPI 3.1 spec (served by every instance) | `$AUTOGPT_API_URL/openapi.json`, e.g. [backend.agpt.co/external-api/v2/openapi.json](https://backend.agpt.co/external-api/v2/openapi.json) |
| Interactive API explorer (served by every instance) | `$AUTOGPT_API_URL/docs` (Swagger UI) and `$AUTOGPT_API_URL/redoc` |
| Index of every docs page, for LLMs | [agpt.co/docs/llms.txt](https://agpt.co/docs/llms.txt) |
| Every docs page in one file | [agpt.co/docs/llms-full.txt](https://agpt.co/docs/llms-full.txt) |
| Any docs page as Markdown | Append `.md` to its URL |
| Docs search for AI agents (MCP) | `https://agpt.co/docs/~gitbook/mcp` |
| The platform's own MCP server | `$AUTOGPT_API_URL/mcp`, see [MCP server](mcp-server.md) |

## Versions

| Version | Status | Path |
| --- | --- | --- |
| **v2** | Current. Use it for everything new. | `/external-api/v2/...` |
| v1 | Deprecated. Stops working on 2026-12-31. | `/external-api/v1/...` |

Within v2, changes are additive: new endpoints, new optional parameters and new response fields can appear at any time, so ignore fields you don't recognise. Anything that would break a working integration ships as a new version.

## Get help

* Report bugs and request endpoints on [GitHub](https://github.com/Significant-Gravitas/AutoGPT/issues).
* Ask questions in the [AutoGPT Discord](https://discord.com/invite/autogpt).
* Building a product for other AutoGPT users? Use [OAuth](oauth-guide.md) instead of asking users for API keys.
