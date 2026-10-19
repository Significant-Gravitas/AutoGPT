---
description: >-
  Create an API key, add a sample agent, run it and read its output in about
  five minutes. Works the same on AutoGPT Cloud and self-hosted.
icon: rocket
---

# Quickstart

In about five minutes you will create an API key, add a small sample agent to your library through the API, run it, and read its result. The sample agent uses no AI model and no third-party account, so it runs the same on AutoGPT Cloud and on a brand-new self-hosted instance.

**You need**

* An AutoGPT account on [AutoGPT Cloud](https://platform.agpt.co) or on your self-hosted instance. New accounts finish a short onboarding in the app before the settings pages open.
* cURL, Python 3.9+ with `requests`, or Node.js 20+.
* On AutoGPT Cloud only: an active plan or trial and a credit balance above zero, or runs are refused with `402 payment_required`. Self-hosted instances don't bill.

{% hint style="info" %}
Want your coding agent to do this for you? Paste the set-up prompt from [Build with AI coding agents](ai-coding-agents.md) into Claude Code, Codex, Cursor or Copilot.
{% endhint %}

{% stepper %}
{% step %}
### Create an API key

1. In the AutoGPT app, open **Settings → AutoGPT API Keys**: [platform.agpt.co/settings/api-keys](https://platform.agpt.co/settings/api-keys) on Cloud, or `http://localhost:3000/settings/api-keys` on a self-hosted instance.
2. Click **Create Key** and name it `Quickstart`.
3. Tick these permissions: **Identity**, **Write Graph**, **Read Library**, **Write Library**, **Run Agent**, **Read Run**. (Write Library lets the full script at the end remove its sample agent again.)
4. Click **Create Key**, then copy the key. It starts with `agpt_` and is shown only once.

Put the key and your instance's base URL in environment variables. Every example in these docs reads them:

{% tabs %}
{% tab title="AutoGPT Cloud" %}
```bash
export AUTOGPT_API_URL="https://backend.agpt.co/external-api/v2"
export AUTOGPT_API_KEY="agpt_..."
```
{% endtab %}

{% tab title="Self-hosted (Docker Compose)" %}
```bash
export AUTOGPT_API_URL="http://localhost:8006/external-api/v2"
export AUTOGPT_API_KEY="agpt_..."
```
{% endtab %}

{% tab title="Self-hosted (single container)" %}
```bash
export AUTOGPT_API_URL="http://localhost:3000/_agpt/external-api/v2"
export AUTOGPT_API_KEY="agpt_..."
```
{% endtab %}
{% endtabs %}

If your instance runs on another host or port, change the URL to match. [Cloud and self-hosted](environments.md) explains how to work it out.

The cURL examples use Bash syntax; on Windows, run them in Git Bash or WSL. To use the Python or TypeScript scripts from PowerShell, set the variables with `$env:AUTOGPT_API_URL = "..."` and `$env:AUTOGPT_API_KEY = "agpt_..."`. In a project, keep both in a `.env` file that's in `.gitignore`:

```dotenv
AUTOGPT_API_URL=https://backend.agpt.co/external-api/v2
AUTOGPT_API_KEY=agpt_...
```
{% endstep %}

{% step %}
### Check the key

```bash
curl -s "$AUTOGPT_API_URL/me" -H "X-API-Key: $AUTOGPT_API_KEY"
```

You should get your account back:

```json
{
  "user_id": "2a9c2855-921d-421e-960b-dacb3a712a08",
  "email": "you@example.com",
  "name": "Ada",
  "timezone": "Europe/London",
  "organization": { "id": "5dad3c98-5c39-4782-b816-f9f6f2c53fe2", "name": "Ada", "is_personal": true },
  "team": null,
  "scopes": ["IDENTITY", "WRITE_GRAPH", "READ_LIBRARY", "WRITE_LIBRARY", "RUN_AGENT", "READ_RUN"],
  "credential_type": "api_key"
}
```

If you get something else, fix it before going on:

| You got | It means |
| --- | --- |
| `401 unauthorized` | The key is wrong or revoked, or the header name is not `X-API-Key`. |
| `403 forbidden` | The key lacks **Identity**, or you are no longer a member of the organization it was created in. The message says which. |
| `404` | `AUTOGPT_API_URL` is wrong. It must end in `/external-api/v2`. |
| `429` with `5 requests per 60s` | The API didn't recognise a key, so the request counted as anonymous: after five `401`s in a minute, you get this instead. Check the key and header, then wait a minute. |
| A connection error | The instance isn't running at that address. |
{% endstep %}

{% step %}
### Add a sample agent

An agent is a graph of blocks: `nodes` are the blocks, and `links` connect one block's output to another's input. This one takes a name, puts it into a greeting with a text template, and returns the greeting.

```bash
curl -s -X POST "$AUTOGPT_API_URL/graphs" \
  -H "X-API-Key: $AUTOGPT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hello from the API",
    "description": "Greets whoever you name. Created by the AutoGPT API quickstart.",
    "nodes": [
      {"id": "name-input", "block_id": "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
       "input_default": {"name": "name", "title": "Your name", "value": "World"}},
      {"id": "greeting-template", "block_id": "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
       "input_default": {"format": "Hello, {{ name }}!", "values": {}}},
      {"id": "greeting-output", "block_id": "363ae599-353e-4804-937e-b2ee3cef3da4",
       "input_default": {"name": "greeting", "title": "Greeting"}}
    ],
    "links": [
      {"id": "name-to-template", "source_id": "name-input", "source_name": "result",
       "sink_id": "greeting-template", "sink_name": "values_#_name"},
      {"id": "template-to-output", "source_id": "greeting-template", "source_name": "output",
       "sink_id": "greeting-output", "sink_name": "value"}
    ]
  }'
```

The response is the saved graph, with an `id`, `"version": 1`, and the `input_schema` and `output_schema` the API worked out from the input and output blocks. Save the ID:

```bash
export GRAPH_ID="992cb167-d6e3-4d7f-9dc3-002e34252de0"   # the "id" from the response
```

Creating a graph also adds it to your library. You run agents through their **library agent**, which has its own ID:

```bash
curl -s "$AUTOGPT_API_URL/graphs/$GRAPH_ID/library-agent" -H "X-API-Key: $AUTOGPT_API_KEY"
```

```json
{
  "id": "3acb5782-a2d9-4c3d-830a-6928317b280f",
  "graph_id": "992cb167-d6e3-4d7f-9dc3-002e34252de0",
  "graph_version": 1,
  "name": "Hello from the API",
  "input_schema": {
    "type": "object",
    "properties": { "name": { "title": "Your name", "default": "World" } },
    "required": []
  },
  "output_schema": {
    "type": "object",
    "properties": { "greeting": { "title": "Greeting" } },
    "required": ["greeting"]
  }
}
```

The response has more fields than shown here. Save the library agent ID:

```bash
export AGENT_ID="3acb5782-a2d9-4c3d-830a-6928317b280f"   # the "id" from the response
```
{% endstep %}

{% step %}
### Run it

```bash
curl -s -X POST "$AUTOGPT_API_URL/library/agents/$AGENT_ID/runs" \
  -H "X-API-Key: $AUTOGPT_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: quickstart-$(date +%s)" \
  -d '{"inputs": {"name": "Ada"}}'
```

`inputs` maps each property in the agent's `input_schema` to a value. The API answers `202 Accepted` straight away with the new run, before it starts:

```json
{
  "id": "34cc713f-7b02-4622-95c7-5c8ea1a8b213",
  "graph_id": "992cb167-d6e3-4d7f-9dc3-002e34252de0",
  "graph_version": 1,
  "status": "QUEUED",
  "started_at": null,
  "ended_at": null,
  "inputs": { "name": "Ada" },
  "cost_cents": 0,
  "duration_seconds": 0.0,
  "node_exec_count": 0
}
```

The `Idempotency-Key` makes the request safe to retry: sending it again with the same key returns this run instead of starting a second one.

```bash
export RUN_ID="34cc713f-7b02-4622-95c7-5c8ea1a8b213"   # the "id" from the response
```
{% endstep %}

{% step %}
### Get the result

Runs are asynchronous. Ask for the run until its `status` is `COMPLETED`, `FAILED` or `TERMINATED`:

```bash
curl -s "$AUTOGPT_API_URL/runs/$RUN_ID" -H "X-API-Key: $AUTOGPT_API_KEY"
```

This agent finishes in a second or two:

```json
{
  "id": "34cc713f-7b02-4622-95c7-5c8ea1a8b213",
  "status": "COMPLETED",
  "started_at": "2026-10-08T14:29:24.277000Z",
  "ended_at": "2026-10-08T14:29:25.098000Z",
  "inputs": { "name": "Ada" },
  "outputs": { "greeting": ["Hello, Ada!"] },
  "cost_cents": 0,
  "duration_seconds": 0.74,
  "node_exec_count": 3,
  "node_executions": [ "..." ]
}
```

`outputs` maps each output name to a **list** of values, because an output block can fire more than once in a run. The greeting is `outputs["greeting"][0]`.
{% endstep %}
{% endstepper %}

You have run an agent through the API.

## The whole flow as one script

Each script creates the sample agent, runs it with an idempotency key, waits for it with a deadline, fails unless the output is exactly `{"greeting": ["Hello, Ada!"]}`, and removes the sample agent from your library again. The graph and its run stay on your account. If a request fails before the script has the agent's ID, remove the agent in the app.

The scripts show the flow, not production code: they don't retry, and they don't branch on error codes. [Errors, rate limits, and pagination](api-conventions.md) covers both.

{% tabs %}
{% tab title="Python" %}
{% code title="quickstart.py" lineNumbers="true" %}
```python
"""AutoGPT API quickstart: create a sample agent, run it, print its output, clean up.

Needs: pip install requests (and python-dotenv to read a .env file)
Env:   AUTOGPT_API_URL, AUTOGPT_API_KEY (permissions: Identity, Write Graph,
       Read Library, Write Library, Run Agent, Read Run)
"""

import os
import time
import uuid

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_URL = os.environ["AUTOGPT_API_URL"].rstrip("/")
HEADERS = {"X-API-Key": os.environ["AUTOGPT_API_KEY"]}
FINAL_STATUSES = {"COMPLETED", "FAILED", "TERMINATED"}

HELLO_AGENT = {
    "name": "Hello from the API",
    "description": "Greets whoever you name. Created by the AutoGPT API quickstart.",
    "nodes": [
        {
            "id": "name-input",
            "block_id": "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",  # Agent Input
            "input_default": {"name": "name", "title": "Your name", "value": "World"},
        },
        {
            "id": "greeting-template",
            "block_id": "db7d8f02-2f44-4c55-ab7a-eae0941f0c30",  # Fill Text Template
            "input_default": {"format": "Hello, {{ name }}!", "values": {}},
        },
        {
            "id": "greeting-output",
            "block_id": "363ae599-353e-4804-937e-b2ee3cef3da4",  # Agent Output
            "input_default": {"name": "greeting", "title": "Greeting"},
        },
    ],
    "links": [
        {
            "id": "name-to-template",
            "source_id": "name-input",
            "source_name": "result",
            "sink_id": "greeting-template",
            "sink_name": "values_#_name",
        },
        {
            "id": "template-to-output",
            "source_id": "greeting-template",
            "source_name": "output",
            "sink_id": "greeting-output",
            "sink_name": "value",
        },
    ],
}


def api(method: str, path: str, timeout: float = 30, **kwargs):
    response = requests.request(
        method, f"{API_URL}{path}", headers=HEADERS | kwargs.pop("headers", {}),
        timeout=timeout, **kwargs,
    )
    if not response.ok:
        raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text}")
    return response.json() if response.content else None


def wait_for_run(run_id: str, timeout_s: float = 300) -> dict:
    deadline = time.monotonic() + timeout_s
    delay, status = 1.0, "unknown"
    while (remaining := deadline - time.monotonic()) > 0:
        run = api("GET", f"/runs/{run_id}", timeout=min(30, remaining))
        status = run["status"]
        if status in FINAL_STATUSES:
            return run
        if status == "REVIEW":
            raise RuntimeError(f"Run {run_id} is waiting for a human review")
        time.sleep(max(0, min(delay, deadline - time.monotonic())))
        delay = min(delay * 1.5, 10)
    raise TimeoutError(f"Run {run_id} still {status} after {timeout_s}s")


def node_errors(run: dict) -> list[str]:
    return [
        message
        for node in run.get("node_executions") or []
        if node["status"] == "FAILED"
        for message in node["outputs"].get("error", [])
    ]


me = api("GET", "/me")
print(f"Authenticated as {me['email']} in {me['organization']['name']}")

graph = api("POST", "/graphs", json=HELLO_AGENT)
agent = None
try:
    agent = api("GET", f"/graphs/{graph['id']}/library-agent")
    print(f"Created agent {agent['id']}")
    run = api(
        "POST",
        f"/library/agents/{agent['id']}/runs",
        json={"inputs": {"name": "Ada"}},
        headers={"Idempotency-Key": str(uuid.uuid4())},
    )
    print(f"Started run {run['id']} ({run['status']})")

    run = wait_for_run(run["id"])
    if run["status"] != "COMPLETED" or run["outputs"] != {"greeting": ["Hello, Ada!"]}:
        raise RuntimeError(f"Run {run['id']} {run['status']}: {node_errors(run) or run['outputs']}")
    print(f"Run {run['status']}: {run['outputs']['greeting'][0]}")
finally:
    if agent:
        api("DELETE", f"/library/agents/{agent['id']}")
        print("Removed the sample agent from the library")
```
{% endcode %}

Run it with `python quickstart.py`. With `python-dotenv` installed it also reads a `.env` file in the current folder. It prints:

```
Authenticated as you@example.com in Ada
Created agent 8f21cc9c-7345-4bc3-acea-125c4588fd2a
Started run d4436ae4-b15f-4dd3-aa9c-66996befd0fb (QUEUED)
Run COMPLETED: Hello, Ada!
Removed the sample agent from the library
```
{% endtab %}

{% tab title="TypeScript" %}
{% code title="quickstart.mts" lineNumbers="true" %}
```typescript
// AutoGPT API quickstart: create a sample agent, run it, print its output, clean up.
// Needs: Node.js 20+. Run with: npx tsx quickstart.mts
// Env:   AUTOGPT_API_URL, AUTOGPT_API_KEY (permissions: Identity, Write Graph,
//        Read Library, Write Library, Run Agent, Read Run)

const API_URL = process.env.AUTOGPT_API_URL!.replace(/\/$/, "");
const API_KEY = process.env.AUTOGPT_API_KEY!;
const FINAL_STATUSES = new Set(["COMPLETED", "FAILED", "TERMINATED"]);

const HELLO_AGENT = {
  name: "Hello from the API",
  description: "Greets whoever you name. Created by the AutoGPT API quickstart.",
  nodes: [
    {
      id: "name-input",
      block_id: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b", // Agent Input
      input_default: { name: "name", title: "Your name", value: "World" },
    },
    {
      id: "greeting-template",
      block_id: "db7d8f02-2f44-4c55-ab7a-eae0941f0c30", // Fill Text Template
      input_default: { format: "Hello, {{ name }}!", values: {} },
    },
    {
      id: "greeting-output",
      block_id: "363ae599-353e-4804-937e-b2ee3cef3da4", // Agent Output
      input_default: { name: "greeting", title: "Greeting" },
    },
  ],
  links: [
    {
      id: "name-to-template",
      source_id: "name-input",
      source_name: "result",
      sink_id: "greeting-template",
      sink_name: "values_#_name",
    },
    {
      id: "template-to-output",
      source_id: "greeting-template",
      source_name: "output",
      sink_id: "greeting-output",
      sink_name: "value",
    },
  ],
};

async function api(
  method: string,
  path: string,
  body?: unknown,
  headers: Record<string, string> = {},
  timeoutMs = 30_000,
) {
  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers: { "X-API-Key": API_KEY, "Content-Type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`${method} ${path} -> ${response.status}: ${await response.text()}`);
  }
  return response.status === 204 ? null : response.json();
}

async function waitForRun(runId: string, timeoutMs = 300_000) {
  const deadline = Date.now() + timeoutMs;
  let delay = 1000;
  let status = "unknown";
  for (let remaining = timeoutMs; remaining > 0; remaining = deadline - Date.now()) {
    const run = await api("GET", `/runs/${runId}`, undefined, {}, Math.min(30_000, remaining));
    status = run.status;
    if (FINAL_STATUSES.has(status)) return run;
    if (status === "REVIEW") throw new Error(`Run ${runId} is waiting for a human review`);
    const pause = Math.max(0, Math.min(delay, deadline - Date.now()));
    await new Promise((resolve) => setTimeout(resolve, pause));
    delay = Math.min(delay * 1.5, 10_000);
  }
  throw new Error(`Run ${runId} still ${status} after ${timeoutMs} ms`);
}

function nodeErrors(run: { node_executions?: { status: string; outputs: Record<string, unknown[]> }[] }) {
  return (run.node_executions ?? [])
    .filter((node) => node.status === "FAILED")
    .flatMap((node) => node.outputs.error ?? []);
}

const me = await api("GET", "/me");
console.log(`Authenticated as ${me.email} in ${me.organization.name}`);

const graph = await api("POST", "/graphs", HELLO_AGENT);
let agent: { id: string } | undefined;
try {
  agent = await api("GET", `/graphs/${graph.id}/library-agent`);
  console.log(`Created agent ${agent!.id}`);
  let run = await api(
    "POST",
    `/library/agents/${agent!.id}/runs`,
    { inputs: { name: "Ada" } },
    { "Idempotency-Key": crypto.randomUUID() },
  );
  console.log(`Started run ${run.id} (${run.status})`);

  run = await waitForRun(run.id);
  const expected = JSON.stringify({ greeting: ["Hello, Ada!"] });
  if (run.status !== "COMPLETED" || JSON.stringify(run.outputs) !== expected) {
    const errors = nodeErrors(run);
    throw new Error(`Run ${run.id} ${run.status}: ${JSON.stringify(errors.length ? errors : run.outputs)}`);
  }
  console.log(`Run ${run.status}: ${run.outputs.greeting[0]}`);
} finally {
  if (agent) {
    await api("DELETE", `/library/agents/${agent.id}`);
    console.log("Removed the sample agent from the library");
  }
}
```
{% endcode %}

Save it as `quickstart.mts` (the `.mts` extension lets it use top-level `await`) and run `npx tsx quickstart.mts`. On Node.js 23.6 or later you can run it directly, and load a `.env` file at the same time: `node --env-file=.env quickstart.mts`. It prints:

```
Authenticated as you@example.com in Ada
Created agent 01322f04-808c-40ba-af2f-a71ac8133d6e
Started run fd13103e-6caa-428b-9c1e-18196d0ff7f2 (QUEUED)
Run COMPLETED: Hello, Ada!
Removed the sample agent from the library
```
{% endtab %}
{% endtabs %}

The step-by-step version above leaves the sample agent in your library. Remove it in the app, or with `DELETE /library/agents/{agent_id}`.

## Next steps

* **Run your own agents.** List your library with `GET /library/agents` (needs **Read Library**), read an agent's `input_schema` from `GET /library/agents/{agent_id}`, and run it the same way. Add **Read Integrations** to the key to check which credentials an agent needs. [Run agents](running-agents.md) covers credentials, files, human reviews, schedules and failures.
* **Build agents in code.** [Build agents](building-agents.md) explains blocks, links and versions.
* **Handle errors and limits.** [Errors, rate limits, and pagination](api-conventions.md).
* **Connect an AI tool.** [MCP server](mcp-server.md) lets Claude, Cursor and other MCP clients do all of this conversationally.
