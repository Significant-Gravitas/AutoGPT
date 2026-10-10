---
description: >-
  Copy-paste prompts, llms.txt, Markdown docs, MCP servers, an AGENTS.md
  snippet and an Agent Skill that let Claude Code, Codex, Cursor or Copilot
  integrate AutoGPT for you.
icon: robot
---

# Build with AI coding agents

These docs are written to be read by coding agents as well as people. Give your agent one of the prompts below and it can wire AutoGPT into your project, check its own work against your instance, and tell you what it verified.

## Set up an integration

Paste this into Claude Code, Codex, Cursor, Copilot or any agent that can run commands in your project. Before you do:

1. [Create an API key](authentication.md#create-an-api-key) with **Identity, Read Library, Write Library, Write Graph, Run Agent, Read Run** and **Read Integrations**. That covers the prompt's checks, including creating and deleting a sample agent; your finished integration may need fewer.
2. Put `AUTOGPT_API_URL` and `AUTOGPT_API_KEY` in the environment the agent works in, or in the project's `.env`. The agent never needs to see the key.

{% prompt description="Integrate the AutoGPT Platform API into this project" icon="rectangle-terminal" openInAIProviders="true" defaultExpanded="full" %}
```markdown
Integrate the AutoGPT Platform API (v2) into this project.

Before writing code, read:
1. The docs index: https://agpt.co/docs/llms.txt. Read these "AutoGPT Platform API" pages
   (append .md to any docs URL for Markdown): Quickstart, Authentication and permissions,
   Run agents, and Errors, rate limits, and pagination.
2. The OpenAPI spec of the instance we use: $AUTOGPT_API_URL/openapi.json. Use only
   endpoints, fields and enum values that appear there. An agent's own input and output
   names, and a graph's node settings, aren't in the spec: take them from the agent's
   input_schema and output_schema and from the docs. The spec doesn't list every status
   code; the Errors page does. Don't use /external-api/v1.

Configuration:
- Read the base URL from AUTOGPT_API_URL and the key from AUTOGPT_API_KEY. If the project
  keeps them in a .env file, load it the way this project's stack normally does. If either
  is missing, stop and ask me for it. Don't print, log or commit the key. Keep both
  configurable: the same code has to work against AutoGPT Cloud
  (https://backend.agpt.co/external-api/v2) and a self-hosted instance.
- Send the key as the X-API-Key header on every request.
- The key needs Identity, Read Library, Write Library, Write Graph, Run Agent, Read Run
  and Read Integrations for the steps below. GET /me lists what it has under scopes.

Rules the integration must follow:
- The client's first call is GET /me, retried like any read. If it still isn't 200,
  stop and report it: 401 means a bad key (after five in a minute they turn into 429
  "5 requests per 60s"), 404 a bad base URL, 403 a missing permission or no access to
  the organization or team (error.message says which).
- Run agents with POST /library/agents/{agent_id}/runs, using the library agent ID, not
  the graph ID. Send an Idempotency-Key header on every run you start. Let callers pass
  their own key (such as a job ID) and generate one only when they don't.
- Build inputs from the agent's input_schema (properties and required). A run or
  schedule with an unknown input name, or a required input left out or null, is a 422
  whose error.details.errors names each one.
- A run is asynchronous. Poll GET /runs/{run_id} with backoff until status is COMPLETED,
  FAILED or TERMINATED, under one overall deadline that also bounds each request and each
  sleep. Treat REVIEW as waiting on a person, never as finished: by default, stop
  polling and report the run ID so someone can answer the review.
- COMPLETED doesn't guarantee success: check that the outputs you need exist, and collect
  `error` outputs from node_executions with status FAILED.
- Outputs map each output name to a list of values.
- Lists are paginated: follow next_cursor until it is null. GET /library/agents leaves
  input_schema empty; read it from GET /library/agents/{agent_id}.
- Errors look like {"error": {"code", "message", "details"}}. Branch on error.code.
  Retries: reads on 429 (after Retry-After), 500, 502, 503 and network errors; a run start
  on the same, plus 409, always resending the same Idempotency-Key; any other write on
  429 only.
- Before an agent's first run, read GET /library/agents/{agent_id}/credentials. An empty
  list means it needs none. Otherwise pass credentials_inputs keyed by field_name, and
  when several credentials match, let the caller choose instead of picking one.

Then:
1. Say which language, HTTP client and file layout you'll use, matching this project.
2. Implement a small client: identity check, list library agents, start a run, wait for it,
   return its outputs or raise with the node errors. If the project has no obvious place
   to call agents from yet, stop at the client and say where you'd wire it in.
3. Prove it works against the real instance: create the sample "Hello from the API" agent
   from the Quickstart, run it with {"name": "Ada"}, and check the output is
   {"greeting": ["Hello, Ada!"]}. Afterwards remove the library agent you created, by its
   ID because names aren't unique, with DELETE /library/agents/{agent_id}, even if the
   check fails. Its graph and run stay on the account; that's expected.
4. Add tests that don't touch the network. Record only response status codes and bodies,
   never request headers or the key, and mark hand-written fixtures as synthetic.
5. Report only what you verified: the GET /me result (email and permissions, not the key),
   the run ID and output from step 3, the files you changed, and anything you couldn't do.
```
{% endprompt %}

## Connect your coding agent to AutoGPT

To have your assistant run and build agents directly instead of writing code, connect it to the [MCP server](mcp-server.md). This prompt sets it up:

{% prompt description="Connect this coding agent to the AutoGPT MCP server" icon="plug" openInAIProviders="true" defaultExpanded="full" %}
```markdown
Connect yourself to the AutoGPT Platform MCP server.

- Server URL: $AUTOGPT_API_URL/mcp/ (keep the trailing slash). On AutoGPT Cloud that is
  https://backend.agpt.co/external-api/v2/mcp/.
- Transport: Streamable HTTP. Auth: the header "Authorization: Bearer <key>", with the key
  read from the AUTOGPT_API_KEY environment variable. Never write the key itself into a
  config file that is committed.
- Instructions for each client: https://agpt.co/docs/platform/api-and-integrations/api-guide/mcp-server.md

Before changing any MCP configuration file, read it and merge the "autogpt" entry without
removing other servers. Then confirm the connection: list the server's tools, call
find_library_agent with query "hello", and tell me how many tools you see and what the
call returned. If the tool list is shorter than you expect, the API key lacks
permissions: tell me which tools are missing.
```
{% endprompt %}

## Docs in machine-readable form

| Resource | Use it for |
| --- | --- |
| [agpt.co/docs/llms.txt](https://agpt.co/docs/llms.txt) | An index of every docs page with a one-line summary. Start here. |
| [agpt.co/docs/llms-full.txt](https://agpt.co/docs/llms-full.txt) | The full text of the docs, in parts of 100 pages (`/llms-full.txt/1`, `/llms-full.txt/2`, ...). |
| Any page + `.md` | That page as Markdown, e.g. [.../api-guide/quickstart.md](https://agpt.co/docs/platform/api-and-integrations/api-guide/quickstart.md). Sending `Accept: text/markdown` to the normal URL works too. |
| `$AUTOGPT_API_URL/openapi.json` | The exact API the instance runs, as OpenAPI 3.1, served without a key. Prefer it to anything remembered from training data. |
| The API reference pages | One page per endpoint, each with a self-contained OpenAPI description of that endpoint. See [API reference](api-reference.md). |
| `https://agpt.co/docs/~gitbook/mcp` | An MCP server for these docs, with search and page tools. |

Add the docs server to Claude Code with:

```bash
claude mcp add --transport http autogpt-docs https://agpt.co/docs/~gitbook/mcp
```

For Codex, add `[mcp_servers.autogpt-docs]` with `url = "https://agpt.co/docs/~gitbook/mcp"` to `~/.codex/config.toml`. Other clients take the same URL. The docs server only reads documentation. The [platform's MCP server](mcp-server.md) is the one that acts on your account.

## Add AutoGPT rules to your repository

Coding agents read standing instructions from `AGENTS.md` (Codex, Cursor, Copilot and most others) or `CLAUDE.md` (Claude Code). Paste this section into the one your project uses:

{% code title="AGENTS.md" %}
```markdown
## AutoGPT Platform API

- Use API v2 only. Base URL from AUTOGPT_API_URL (AutoGPT Cloud:
  https://backend.agpt.co/external-api/v2). Key from AUTOGPT_API_KEY, sent as X-API-Key.
  Never print, log or commit the key.
- Check endpoints and fields against $AUTOGPT_API_URL/openapi.json and the docs at
  https://agpt.co/docs/llms.txt before using them. Don't guess.
- Run agents by library agent ID: POST /library/agents/{agent_id}/runs, always with an
  Idempotency-Key header. Get a graph's library agent with GET /graphs/{graph_id}/library-agent.
  GET /library/agents leaves input_schema empty: read it from GET /library/agents/{agent_id}.
- Build inputs from input_schema (properties and required). An unknown input name, or a
  required input left out or null, is a 422 naming each in error.details.errors.
- Runs are asynchronous: poll GET /runs/{run_id} with backoff and a deadline until COMPLETED,
  FAILED or TERMINATED. REVIEW waits for a person. COMPLETED can still lack outputs: check
  them, and read `error` outputs from node_executions with status FAILED.
- outputs maps each output name to a list of values.
- Lists: limit (max 100) + cursor; follow next_cursor until null.
- Errors: {"error": {"code", "message", "details"}}. Branch on code. Retry reads on 429
  (honour Retry-After), 500, 502, 503 and network errors; run starts the same plus 409,
  with the same Idempotency-Key; other writes on 429 only.
- Limits: 200 requests/min per user; 60 run starts/min.
```
{% endcode %}

## Install the Agent Skill

An [Agent Skill](https://agentskills.io) gives an agent the same rules, but loads them only when a task involves AutoGPT. Save this as `.claude/skills/autogpt-api/SKILL.md` in your project for Claude Code, or in your agent's skills folder:

{% code title="SKILL.md" %}
```markdown
---
name: autogpt-api
description: Integrate with the AutoGPT Platform API v2 - run AutoGPT agents, read their outputs, build agent graphs, manage files, schedules and credentials, on AutoGPT Cloud or a self-hosted instance. Use when code calls AutoGPT, agpt.co, backend.agpt.co or /external-api/v2.
---

# AutoGPT Platform API v2

## Before writing code
1. Fetch https://agpt.co/docs/llms.txt and read the "AutoGPT Platform API" pages you need (append .md to a docs URL for Markdown).
2. Fetch $AUTOGPT_API_URL/openapi.json and use only the endpoints and fields in it.

## Configuration
- AUTOGPT_API_URL: base URL ending in /external-api/v2. Cloud: https://backend.agpt.co/external-api/v2. Self-hosted: http://localhost:8006/external-api/v2 (Docker Compose) or http://localhost:3000/_agpt/external-api/v2 (single container).
- AUTOGPT_API_KEY: starts with agpt_. Send as X-API-Key. Never print, log or commit it.
- Check both with GET /me before anything else (401 bad key, 404 bad URL, 403 missing permission).

## Running an agent
1. Find it: GET /library/agents (items[].id is the library agent ID). From a graph ID: GET /graphs/{graph_id}/library-agent.
2. Inputs: keys of input_schema.properties from GET /library/agents/{agent_id} (the list leaves input_schema empty). An unknown name, or a required input left out or null, is a 422 whose error.details.errors names each. File inputs (format "file") take a file_uri from POST /files/upload.
3. Credentials: GET /library/agents/{agent_id}/credentials. Pass credentials_inputs {field_name: {id, provider, type}} from matching_credentials.
4. Start: POST /library/agents/{agent_id}/runs with {"inputs": {...}, "credentials_inputs": {...}} and an Idempotency-Key header. Answers 202 with the run.
5. Wait: GET /runs/{run_id} with backoff and one overall deadline until COMPLETED, FAILED or TERMINATED. REVIEW means a person must answer (GET /runs/reviews, POST /runs/{run_id}/reviews): by default stop polling and report the run ID; stop the run with POST /runs/{run_id}/stop if nobody will answer.
6. Result: outputs maps each output name to a list. COMPLETED without the outputs you need is a failure: read `error` from node_executions with status FAILED, and if there is none, check the run's inputs.

## Conventions
- Lists: ?limit=1..100&cursor=; response {items, next_cursor, total_count}; follow next_cursor until null.
- Errors: {"error": {"code", "message", "details"}}; branch on code. Retry reads on 429 (Retry-After), 500, 502, 503 and network errors; run starts the same plus 409, resending the same Idempotency-Key; other writes on 429 only.
- Rate limits: 200 req/min per user; runs 60/min; uploads 20 per 5 min; search 30/min.
- 402 payment_required on AutoGPT Cloud: no active plan or a zero balance.

## Building agents
Graph = nodes (block_id + input_default) + links (source_id/source_name -> sink_id/sink_name). POST /graphs creates version 1 and adds it to the library; PUT /graphs/{graph_id} with the full graph (no id needed) saves the next version, even if nothing changed. Inputs: AgentInputBlock c0a8e994-ebf1-4a9c-a4d8-89d09c86741b. Outputs: AgentOutputBlock 363ae599-353e-4804-937e-b2ee3cef3da4; outputs record exactly what reaches its value pin. Shape text with FillTextTemplateBlock db7d8f02-2f44-4c55-ab7a-eae0941f0c30, whose format is a sandboxed Jinja2 template (filters such as upper work). Fill one key of a dict input with sink_name values_#_key. Discover blocks with GET /blocks (all pages; search can be empty on a new self-hosted instance).
```
{% endcode %}

## What agents get wrong, and the fix

| Mistake | Fix |
| --- | --- |
| Calling `/external-api/v1/...` endpoints remembered from training data | Use v2 and check the instance's `openapi.json`. |
| A base URL without `/external-api/v2`, or with `/api` | Use the URL from the [base URL table](environments.md#base-urls) exactly. |
| Running with the graph ID | Runs take the library agent ID. Look it up with `GET /graphs/{graph_id}/library-agent`. |
| Treating the `202` from starting a run as the result | Poll `GET /runs/{run_id}` until a final status. |
| Polling forever, or every 100 ms | Back off from about 1 s to 10 s and set a deadline. Every poll counts toward the 200-per-minute limit. |
| Reading `outputs["x"]` as a string | It's a list. Use `outputs["x"][0]` or the last element. |
| Trusting `COMPLETED` | Check the outputs exist, and read `error` outputs from failed nodes. |
| Retrying `POST .../runs` without an idempotency key | Send `Idempotency-Key`, or a retry starts and bills a second run. |
| Retrying other writes after a `5xx` | The write may have happened. Retry only on `429`, and check before repeating. |
| Sending an input name the agent doesn't have | A `422`. Take names from `input_schema.properties`. |
| Leaving out a required input, or sending it as `null` | A `422`. Send every name in `input_schema.required`. |
| Treating an MCP tool result as a success because `is_error` isn't set | Some failures arrive as an ordinary result whose JSON is `{"type": "error", "message": ...}`. Check `is_error` and `type`. |
| Getting `429` with `5 requests per 60s` | Requests aren't carrying a valid key, so they count as anonymous. Fix the header. |
