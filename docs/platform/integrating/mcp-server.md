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

* **Transport:** Streamable HTTP. Each call is independent; there is no session state between tool calls.
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
With the official [MCP Python SDK](https://pypi.org/project/mcp/) (`pip install mcp`):

```python
import asyncio
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = os.environ["AUTOGPT_API_URL"].rstrip("/") + "/mcp/"
HEADERS = {"Authorization": f"Bearer {os.environ['AUTOGPT_API_KEY']}"}


async def main():
    async with streamablehttp_client(URL, headers=HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([tool.name for tool in tools.tools])
            result = await session.call_tool("find_library_agent", {"query": "report"})
            print(result.content[0].text)


asyncio.run(main())
```
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
| `search_feature_requests`, `create_feature_request` | Find or file AutoGPT feature requests. | Use Tools |

The tools a key can't use don't appear in its tool list at all. If an assistant says a tool is missing, add the permission to a new key and reconnect.

## Good to know

* Each tool call is independent, so the assistant passes IDs from one call to the next itself.
* Runs started over MCP are ordinary runs: they show up in the app and in `GET /runs`, and they cost the same.
* On a self-hosted instance, `search_docs` returns results only after the instance has indexed its documentation, and `web_search` needs OpenRouter credentials configured on the instance.
* The server publishes OAuth protected-resource metadata, but it doesn't support dynamic client registration. If a client offers to "sign in", configure the API key header instead.
