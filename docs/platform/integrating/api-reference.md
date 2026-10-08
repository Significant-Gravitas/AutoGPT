---
description: >-
  Every AutoGPT API v2 endpoint with its parameters, request body, responses
  and required permissions, generated from the OpenAPI specification.
icon: book
---

# API reference

The pages in this section are generated from the API's OpenAPI 3.1 specification, one page per endpoint, grouped by area. Each page lists the endpoint's parameters, request body, responses and required permission, and its Markdown version (append `.md` to the URL) contains a complete OpenAPI description of that endpoint, ready for code generators and AI agents.

## Base URL and authentication

All paths are relative to your instance's base URL:

| Where AutoGPT runs | Base URL |
| --- | --- |
| AutoGPT Cloud | `https://backend.agpt.co/external-api/v2` |
| Self-hosted with Docker Compose | `http://localhost:8006/external-api/v2` |
| Self-hosted single container | `http://localhost:3000/_agpt/external-api/v2` |

Authenticate every request with an API key in the `X-API-Key` header, or with `Authorization: Bearer` and an API key or OAuth access token. Each endpoint's page names the permission it needs. See [Authentication and permissions](authentication.md).

## The specification

Download the specification your code will talk to from the instance itself, since each instance serves the version it runs:

* `$AUTOGPT_API_URL/openapi.json`. AutoGPT Cloud: [backend.agpt.co/external-api/v2/openapi.json](https://backend.agpt.co/external-api/v2/openapi.json).
* `$AUTOGPT_API_URL/docs` is an interactive explorer (Swagger UI), and `$AUTOGPT_API_URL/redoc` a readable one.

Generate a typed client from it with any OpenAPI generator, for example:

{% tabs %}
{% tab title="TypeScript" %}
```bash
npx openapi-typescript "$AUTOGPT_API_URL/openapi.json" -o autogpt-api.d.ts
```
{% endtab %}

{% tab title="Python" %}
```bash
pipx run openapi-python-client generate --url "$AUTOGPT_API_URL/openapi.json"
```
{% endtab %}
{% endtabs %}

## Areas

| Area | What it covers |
| --- | --- |
| Graphs | Create, read and version agent graphs; graph settings. |
| Schedules | Run graphs on a cron schedule. |
| Blocks | The building blocks agents are made from. |
| Search | Search marketplace agents, blocks, docs, your library and your files. |
| Marketplace | Browse agents and creators; manage your submissions and profile. |
| Library | Your agents and folders; start runs. |
| Runs | Run status, outputs, stopping, sharing and human-in-the-loop reviews. |
| Credits | Balance, transactions, invoices, subscription and cost summaries. |
| Integrations | Third-party credentials your agents use. |
| Files | Upload, list, download and delete workspace files. |
| Identity | Who a credential acts as: `GET /me`. |

The conventions every endpoint shares, such as errors, pagination, rate limits and idempotency, are on [Errors, rate limits, and pagination](api-conventions.md).
