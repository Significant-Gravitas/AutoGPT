---
description: >-
  Connect Claude Code, Codex, Cursor, VS Code, Claude Desktop or any MCP client
  to AutoGPT, so an AI assistant can find, build and run your agents.
icon: plug
---

# MCP server

AutoGPT runs a [Model Context Protocol](https://modelcontextprotocol.io) server alongside the API. Connect an AI assistant to it and the assistant can search the marketplace and your library, build and edit agents, run them, read their results, and manage folders, schedules and files, all within the permissions of the key you give it.

| Where AutoGPT runs | MCP server URL |
| --- | --- |
| AutoGPT Cloud | `https://backend.agpt.co/external-api/v2/mcp/` |
| Self-hosted with Docker Compose | `http://localhost:8006/external-api/v2/mcp/` |
| Self-hosted single container | `http://localhost:3000/_agpt/external-api/v2/mcp/` |

It is `$AUTOGPT_API_URL/mcp/`. Keep the trailing slash: without it the server redirects, and some clients don't follow redirects.

* **Transport:** Streamable HTTP. Tool calls don't depend on each other: you pass the IDs from one result into the next call.
* **Authentication:** an [API key](authentication.md#create-an-api-key) sent as `Authorization: Bearer agpt_...`. OAuth access tokens work too.
* **Tools:** the server only lists the tools the key's permissions allow. Give the key exactly the powers you want the assistant to have.

{% hint style="info" %}
This server acts on your AutoGPT account. To let an assistant **read these docs** instead, use the docs server at `https://agpt.co/docs/~gitbook/mcp`. See [Build with AI coding agents](ai-coding-agents.md).
{% endhint %}

## Connect a client

Create an API key for the assistant first. For a coding assistant that should be able to do everything, tick **Identity, Read Library, Write Library, Read Graph, Write Graph, Read Block, Run Agent, Read Run, Read Files, Write Files, Read Schedule, Write Schedule**. Then set `AUTOGPT_API_URL` and `AUTOGPT_API_KEY` in the environment the client starts from.

{% tabs %}
{% tab title="Claude Code" %}
Add the server to a project in `.mcp.json`. Claude Code expands `${VAR}` from the environment, so the key stays out of the file:

{% code title=".mcp.json" %}
```json
{
  "mcpServers": {
    "autogpt": {
      "type": "http",
      "url": "${AUTOGPT_API_URL}/mcp/",
      "headers": {
        "Authorization": "Bearer ${AUTOGPT_API_KEY}"
      }
    }
  }
}
```
{% endcode %}

Or add it for your user from the command line. This stores the key in Claude Code's configuration:

```bash
claude mcp add --transport http --scope user autogpt "$AUTOGPT_API_URL/mcp/" \
  --header "Authorization: Bearer $AUTOGPT_API_KEY"
```

Check it with `claude mcp list`.
{% endtab %}

{% tab title="Codex" %}
Add the server to `~/.codex/config.toml`. Codex reads the key from the environment variable you name:

{% code title="~/.codex/config.toml" %}
```toml
[mcp_servers.autogpt]
url = "https://backend.agpt.co/external-api/v2/mcp/"
bearer_token_env_var = "AUTOGPT_API_KEY"
```
{% endcode %}

For a self-hosted instance, change `url` to your instance's MCP server URL.
{% endtab %}

{% tab title="Cursor" %}
Add the server to `.cursor/mcp.json` in your project, or `~/.cursor/mcp.json` for every project:

{% code title=".cursor/mcp.json" %}
```json
{
  "mcpServers": {
    "autogpt": {
      "url": "https://backend.agpt.co/external-api/v2/mcp/",
      "headers": {
        "Authorization": "Bearer ${env:AUTOGPT_API_KEY}"
      }
    }
  }
}
```
{% endcode %}
{% endtab %}

{% tab title="VS Code" %}
Add the server to `.vscode/mcp.json`. VS Code asks for the key the first time and stores it securely:

{% code title=".vscode/mcp.json" %}
```json
{
  "servers": {
    "autogpt": {
      "type": "http",
      "url": "https://backend.agpt.co/external-api/v2/mcp/",
      "headers": {
        "Authorization": "Bearer ${input:autogpt-api-key}"
      }
    }
  },
  "inputs": [
    {
      "type": "promptString",
      "id": "autogpt-api-key",
      "description": "AutoGPT API key",
      "password": true
    }
  ]
}
```
{% endcode %}
{% endtab %}

{% tab title="Claude Desktop" %}
Claude Desktop starts local servers, so bridge to the remote one with [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) (needs Node.js). In **Settings → Developer → Edit Config**:

{% code title="claude_desktop_config.json" %}
```json
{
  "mcpServers": {
    "autogpt": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://backend.agpt.co/external-api/v2/mcp/",
        "--header",
        "Authorization:${AUTOGPT_AUTH_HEADER}"
      ],
      "env": {
        "AUTOGPT_AUTH_HEADER": "Bearer agpt_..."
      }
    }
  }
}
```
{% endcode %}

Restart Claude Desktop after saving.
{% endtab %}

{% tab title="Python" %}
With the official [MCP Python SDK](https://pypi.org/project/mcp/) (`pip install mcp`; tested with 2.3):

```python
import asyncio
import os

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.environ["AUTOGPT_API_URL"].rstrip("/") + "/mcp/"
HEADERS = {"Authorization": f"Bearer {os.environ['AUTOGPT_API_KEY']}"}


async def main():
    async with httpx2.AsyncClient(headers=HEADERS, timeout=60) as http:
        async with streamable_http_client(URL, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print([tool.name for tool in tools.tools])


asyncio.run(main())
```

On version 1 of the SDK, import `httpx` instead of `httpx2`. [Call the tools from code](#call-the-tools-from-code) carries on from here.
{% endtab %}
{% endtabs %}

Any other client that supports Streamable HTTP works the same way: point it at the URL and send the `Authorization` header.

## Tools

| Tool | What it does | Permissions it needs |
| --- | --- | --- |
| `find_agent` | Search the public marketplace. | none |
| `find_library_agent` | Search your library, or look an agent up by ID. | Read Library |
| `run_agent` | Run or schedule a library agent, after checking its inputs and credentials. | Run Agent |
| `view_agent_output` | Read the outputs of an agent's runs. | Read Run |
| `get_agent_building_guide` | Return the guide the agent-building tools follow. | none |
| `create_agent` | Validate, fix and save a new agent from graph JSON. | Write Graph, Write Library, Read Library |
| `edit_agent` | Validate, fix and save a new version of an agent. | Write Graph, Write Library |
| `customize_agent` | Adapt a marketplace agent and save it to your library. | Write Graph, Write Library |
| `validate_agent_graph` | Check graph JSON for errors without saving it. | none |
| `fix_agent_graph` | Repair common graph JSON mistakes without saving. | none |
| `list_folders`, `create_folder`, `update_folder`, `move_folder`, `delete_folder`, `move_agents_to_folder` | Organize your library. | Read Library or Write Library |
| `list_schedules`, `delete_schedule` | See and remove scheduled runs. | Read Schedule, Write Schedule |
| `list_workspace_files`, `read_workspace_file`, `write_workspace_file`, `delete_workspace_file` | Work with files in your workspace. | Read Files or Write Files |
| `get_platform_info` | Your plan, credits and account details. | Read Credits |
| `search_docs`, `get_doc_page` | Search and read the AutoGPT documentation the instance has indexed. | none |
| `web_search`, `web_fetch` | Search the web and read a public page. These spend platform resources. | Use Tools |
| `search_feature_requests` | Find existing AutoGPT feature requests. | Use Tools |

The tools a key can't use don't appear in its tool list at all. If an assistant says a tool is missing, add the permission to a new key and reconnect.

## Call the tools from code

An assistant learns each tool's arguments from the server, so you only need this section to call the tools from your own code. A tool returns one JSON object, sent as text in the first content part of its result. The object's `type` says what kind of answer it is and its `message` describes it in words. Results also carry a `session_id`, which changes on every call; you never send it back.

Running an agent takes three calls:

| Tool | Arguments | What it returns |
| --- | --- | --- |
| `find_library_agent` | `query`: words from the agent's name or description | `{"type": "agents_found", "agents": [...]}`. Each agent's `id` is its library agent ID, and it comes with the agent's `input_schema` and `output_schema`. |
| `run_agent` | `library_agent_id`, `inputs` | `{"type": "execution_started", "execution_id": "..."}`, without outputs. If an input without a default is missing, nothing runs: you get `{"type": "agent_details"}` with a `message` that names the inputs to send. |
| `view_agent_output` | `library_agent_id` and `execution_id` | `{"execution": {"status": "COMPLETED", "outputs": {...}}}`. As in the REST API, each output is a [list of values](running-agents.md#read-the-outputs). The call returns at once; while the run is still going, call it again after a few seconds. |

This script runs the quickstart's sample agent by name and prints its outputs:

{% code title="mcp_run_agent.py" lineNumbers="true" %}
```python
import asyncio
import json
import os
import time

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = os.environ["AUTOGPT_API_URL"].rstrip("/") + "/mcp/"
HEADERS = {"Authorization": f"Bearer {os.environ['AUTOGPT_API_KEY']}"}


def payload(result) -> dict:
    text = result.content[0].text
    if result.is_error:  # isError on version 1 of the SDK
        raise RuntimeError(text)
    data = json.loads(text)
    if data["type"] == "error":
        raise RuntimeError(data["message"])
    return data


async def run_agent(session: ClientSession, name: str, inputs: dict) -> dict:
    found = payload(await session.call_tool("find_library_agent", {"query": name}))
    matches = [agent for agent in found.get("agents", []) if agent["name"] == name]
    if len(matches) != 1:
        raise LookupError(f"{len(matches)} library agents are named {name!r}")
    agent_id = matches[0]["id"]

    started = payload(
        await session.call_tool("run_agent", {"library_agent_id": agent_id, "inputs": inputs})
    )
    if started["type"] != "execution_started":
        raise RuntimeError(started["message"])

    deadline, delay = time.monotonic() + 600, 2.0
    while True:
        output = payload(
            await session.call_tool(
                "view_agent_output",
                {"library_agent_id": agent_id, "execution_id": started["execution_id"]},
            )
        )
        execution = output["execution"]
        if execution["status"] in ("COMPLETED", "FAILED", "TERMINATED", "REVIEW"):
            return execution
        if time.monotonic() > deadline:
            raise TimeoutError(f"run {started['execution_id']} is still {execution['status']}")
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 15)


async def main():
    async with httpx2.AsyncClient(headers=HEADERS, timeout=60) as http:
        async with streamable_http_client(URL, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                execution = await run_agent(session, "Hello from the API", {"name": "Ada"})
                print(execution["status"], execution["outputs"])


asyncio.run(main())
```
{% endcode %}

It prints `COMPLETED {'greeting': ['Hello, Ada!']}`. Change the name and inputs to run one of your own agents.

* `execution.status` is the status when the call returned. While the run is still going, call `view_agent_output` again after a few seconds, backing off as it runs longer.
* Without `execution_id`, `view_agent_output` reads the agent's latest run and also lists recent runs in `available_executions`. Their statuses can lag behind, so read the status from `execution`.
* Check for failure in two places. The result's `is_error` flag is set when a call fails or is refused: an unknown tool, a missing permission, an error inside the tool. A tool that ran but couldn't do what you asked, for example because an ID doesn't exist, returns an ordinary result whose JSON is `{"type": "error", "message": "..."}`. The `payload` helper above handles both.
* `find_library_agent` returns `{"type": "no_results"}` when nothing matches.

## Good to know

* `run_agent` checks an agent's inputs before it starts a run, which the REST API doesn't do. Calling it without `inputs` is a quick way to see what an agent needs.
* A call is held to the same rules as the REST API:
  * An argument the tool doesn't list is refused, so a typo fails loudly instead of being ignored.
  * Some arguments need a permission on top of the tool's own: scheduling with `run_agent` (`schedule_name` or `cron`) needs Write Schedule, running a marketplace agent by `username_agent_slug` needs Write Library, and reading or writing agent JSON through a workspace file (`agent_json_ref`, `write_to`, `write_graph_to`) needs Read Files or Write Files.
  * The key only reaches its own organization's agents, folders, runs and schedules, as in the REST API; an ID from another organization is not found.
  * Runs, uploads and searches count against the same [rate limits](api-conventions.md#rate-limits) as their REST endpoints, and `run_agent` refuses to start a run on a zero balance. `web_search` needs an active plan and counts against your AutoPilot usage allowance.
* Send the key once, in the `Authorization` header. A request that also sends `X-API-Key`, or two `Authorization` headers, is refused.
* Runs started over MCP are ordinary runs: they show up in the app and in `GET /runs`, and they cost the same.
* On a self-hosted instance, `search_docs` returns results only after the instance has indexed its documentation, and `web_search` needs OpenRouter credentials configured on the instance.
* The server doesn't support dynamic client registration, so a client's "sign in" or "authenticate" option won't work with it. Configure the API key header instead.
