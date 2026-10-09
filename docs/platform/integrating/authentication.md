---
description: >-
  Create and use API keys, choose the permissions each integration needs, and
  control which organization and team a request acts in.
icon: key
---

# Authentication and permissions

Every request to the AutoGPT Platform API must carry a credential. There are two kinds:

| Credential | Use it when | How to send it |
| --- | --- | --- |
| **API key** | Your own scripts, servers and agents act on your account. | `X-API-Key: agpt_...` (or `Authorization: Bearer agpt_...`) |
| **OAuth access token** | Your app acts for *other* AutoGPT users, who sign in and approve it. | `Authorization: Bearer agpt_xt_...` |

API keys are the right choice for almost every integration. Use OAuth only when you are building a product that other AutoGPT users connect to; see [OAuth & SSO](oauth-guide.md).

## Create an API key

1. Sign in to AutoGPT and open **Settings → AutoGPT API Keys**:
   * AutoGPT Cloud: [platform.agpt.co/settings/api-keys](https://platform.agpt.co/settings/api-keys)
   * Self-hosted: `http://localhost:3000/settings/api-keys` (or your instance's address)
2. Click **Create key**. Give it a name you will recognise later, such as the name of the service that uses it.
3. Tick the [permissions](#permissions) the integration needs, and nothing else.
4. Copy the key and store it in a secret manager or an environment variable. Keys start with `agpt_` and **are shown only once**.

The key belongs to the organization you have selected in the app when you create it. To use another organization, switch to it first, then create the key.

To change what a key can do, create a new key with the permissions you want and delete the old one. Deleting a key revokes it: the next request that uses it fails with `401`.

## Send the key

Send the key in the `X-API-Key` header:

{% tabs %}
{% tab title="cURL" %}
```bash
curl -s "$AUTOGPT_API_URL/me" -H "X-API-Key: $AUTOGPT_API_KEY"
```
{% endtab %}

{% tab title="Python" %}
```python
import os
import requests

response = requests.get(
    f"{os.environ['AUTOGPT_API_URL']}/me",
    headers={"X-API-Key": os.environ["AUTOGPT_API_KEY"]},
    timeout=30,
)
response.raise_for_status()
print(response.json())
```
{% endtab %}

{% tab title="TypeScript" %}
```typescript
const response = await fetch(`${process.env.AUTOGPT_API_URL}/me`, {
  headers: { "X-API-Key": process.env.AUTOGPT_API_KEY! },
});
if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
console.log(await response.json());
```
{% endtab %}
{% endtabs %}

`Authorization: Bearer <key>` works too. Use it with tools that can only send a bearer token, such as most MCP clients.

Use the base URL exactly as given, with `https` for AutoGPT Cloud, so no redirect is involved: most HTTP clients send a custom header such as `X-API-Key` on to wherever a redirect points.

{% hint style="danger" %}
An API key is a password to your account. Keep it on servers and in secret stores. Never put it in browser code, a mobile app, a public repository or a log line. If a key leaks, revoke it on the API keys page and create a new one.
{% endhint %}

## Check what a key can do

`GET /me` needs the **Identity** permission and answers who the key acts as, in which organization and team, and with which permissions:

```json
{
  "user_id": "6f1c2d3e-...",
  "email": "you@example.com",
  "name": "Ada Lovelace",
  "timezone": "Europe/London",
  "organization": { "id": "b9a8...", "name": "Ada's workspace", "is_personal": true },
  "team": null,
  "scopes": ["IDENTITY", "READ_LIBRARY", "RUN_AGENT", "READ_RUN"],
  "credential_type": "api_key"
}
```

Call it first when you set up an integration. It proves the URL, the key and the header are all right before you do anything else.

## Permissions

A key can only call the endpoints its permissions cover. A request without the right permission fails with `403` and code `forbidden`, and the message names what is missing:

```json
{ "error": { "code": "forbidden", "message": "Missing required permission(s): RUN_AGENT", "details": null } }
```

Grant each key the smallest set that works. These sets cover the common integrations:

| Integration | Permissions to tick |
| --- | --- |
| Run existing agents and read their results | Identity, Read Library, Read Integrations, Run Agent, Read Run. Read Integrations lets you check which [credentials](running-agents.md#supply-credentials-it-needs) an agent needs. |
| …and answer human-in-the-loop reviews | add Read Run Review, Write Run Review |
| …and pass files to agents | add Read Files, Write Files |
| …and run agents on a schedule | add Read Schedule, Write Schedule |
| Build and update agents | Identity, Read Block, Read Graph, Write Graph, Read Library. To try them too, add Run Agent and Read Run, and Write Library to remove them from your library. |
| Supply third-party credentials an agent needs | Read Integrations, Manage Integrations |
| An AI coding agent working on your behalf over [MCP](mcp-server.md) | Identity, Read Library, Write Library, Read Graph, Write Graph, Read Block, Run Agent, Read Run, Read Files, Write Files, Read Schedule, Write Schedule |
| Billing dashboards | Read Credits |

The app shows each permission in title case (**Run Agent**); the API and OAuth use the constant (`RUN_AGENT`).

### What each permission allows

| Permission | Allows |
| --- | --- |
| `IDENTITY` | `GET /me`: the user, organization, team and permissions of the credentials. |
| `READ_BLOCK` | `GET /blocks`: list blocks with their input and output schemas and costs. |
| `READ_GRAPH` | `GET /graphs`, `GET /graphs/{graph_id}`, its versions and the blocks it uses. |
| `WRITE_GRAPH` | `POST /graphs`, `PUT /graphs/{graph_id}` (new version), set the active version, change graph settings. |
| `READ_LIBRARY` | List and get library agents and folders; `GET /graphs/{graph_id}/library-agent`. |
| `WRITE_LIBRARY` | Update, fork and remove library agents; add marketplace agents to the library; create, move and delete folders. |
| `RUN_AGENT` | `POST /library/agents/{agent_id}/runs`: start a run. |
| `READ_RUN` | `GET /runs`, `GET /runs/{run_id}`: list runs and read their status and outputs. |
| `WRITE_RUN` | `POST /runs/{run_id}/stop`, `DELETE /runs/{run_id}`. |
| `SHARE_RUN` | Turn a run's public share link on and off (also needs `READ_RUN`). |
| `READ_RUN_REVIEW` | `GET /runs/reviews`: list human-in-the-loop reviews. |
| `WRITE_RUN_REVIEW` | `POST /runs/{run_id}/reviews`: approve or reject them. |
| `READ_SCHEDULE` | `GET /schedules`. |
| `WRITE_SCHEDULE` | `POST /schedules` (also needs `RUN_AGENT`), `DELETE /schedules/{schedule_id}`. |
| `READ_FILES` | List workspace files, read their metadata, download them. |
| `WRITE_FILES` | Upload and delete workspace files. |
| `READ_INTEGRATIONS` | List your third-party credentials; list the credentials an agent needs. |
| `MANAGE_INTEGRATIONS` | Add API-key, username/password and host-scoped credentials. |
| `DELETE_INTEGRATIONS` | Delete credentials. |
| `READ_STORE` | Read your marketplace profile and submissions. |
| `WRITE_STORE` | Edit your marketplace profile; create, edit and delete submissions; upload their media. |
| `READ_CREDITS` | Read the credit balance, transactions, invoices, subscription and cost summary. |
| `USE_TOOLS` | MCP tools that act from platform infrastructure: web search (which counts against your AutoPilot usage allowance), web fetch and searching feature requests. |

Two endpoints need two permissions at once: `POST /schedules` needs `WRITE_SCHEDULE` and `RUN_AGENT`, and sharing a run needs `READ_RUN` and `SHARE_RUN`.

These need a valid credential but no particular permission: browsing the public marketplace (`GET /marketplace/agents`, `/marketplace/creators` and their detail pages) and `GET /search` over public content (marketplace agents, blocks, integrations and documentation). Searching your own content costs the matching permission: `content_types=LIBRARY_AGENT` needs `READ_LIBRARY`, and `WORKSPACE_FILE` needs `READ_FILES`.

`EXECUTE_GRAPH` and `EXECUTE_BLOCK` belong to the deprecated v1 API. v2 does not use them; `RUN_AGENT` replaces `EXECUTE_GRAPH`.

## Organizations and teams

Every request acts inside exactly one organization, and the key decides which: it keeps the organization it was created in for its whole life. There is no per-request organization parameter. A key created before organizations existed acts in your personal organization.

What a request creates belongs to that organization, and its runs are billed to that organization's balance. It can only see and change that organization's agents, graphs, folders, runs and schedules: anything in another organization answers `404`, as if it did not exist, and search leaves it out. To work in several organizations, create one key in each.

Two kinds of data belong to you rather than to an organization, so every key you create reaches them, whatever organization it acts in:

* **Workspace files** (`/files`): one workspace per account.
* **Integration credentials** (`/integrations/credentials`): stored per account.

Rows created before organizations existed carry no organization, and stay visible to their owner from every organization.

In an organization other than your personal one, reading the organization's balance, transactions and invoices (`GET /credits`, `/credits/transactions`, `/credits/invoices`) needs the owner or billing manager role, as it does in the app; other members get `403`.

### Act inside a team

An organization can contain teams. By default a request acts organization-wide, and what it creates is visible to every member of the organization. To act inside one team, send its ID in `X-Team-Id`:

```bash
curl -s "$AUTOGPT_API_URL/library/agents" \
  -H "X-API-Key: $AUTOGPT_API_KEY" \
  -H "X-Team-Id: $TEAM_ID"
```

`GET /me` with the same header confirms the team. Sending `X-Team-Id` for a team you are not an active member of, or one that belongs to another organization, fails with `403`.

Keys created on the API keys page act organization-wide until a request sends `X-Team-Id`. A key can also be restricted to a single team when it is created. A restricted key acts in its team on every request and rejects an `X-Team-Id` naming any other team with `403`; `GET /me` shows the team.

A team decides where what a request creates lives and whose balance its runs are billed to. It doesn't narrow what the request can read: a key acting in a team still sees everything in the organization that you can see in the app.

OAuth access tokens act in your personal organization. There is no way yet to choose another organization when you approve an app.

## Common authentication errors

| Status and code | Message | Fix |
| --- | --- | --- |
| `401 unauthorized` | `Missing authentication. Provide API key or access token.` | Send the `X-API-Key` header. Check the header name and that your HTTP client isn't dropping it on a redirect. |
| `401 unauthorized` | `Invalid API key` | The key is wrong, revoked, or from another instance: keys from AutoGPT Cloud don't work on a self-hosted instance, and the reverse. |
| `403 forbidden` | `Missing required permission(s): ...` | Keys can't be edited: create a key with the named permissions and delete the old one. |
| `403 forbidden` | `These credentials belong to an organization you are not an active member of` | You left the organization the key was created in. Create a new key. |
| `404 not_found` | `... not found` | The resource doesn't exist, or belongs to an organization or team this key can't see. |

## OAuth access tokens

OAuth lets other AutoGPT users connect your app to their accounts without handing you an API key. The user approves a set of scopes, which are the same permissions listed above, and your app receives an access token to send as `Authorization: Bearer agpt_xt_...`. Access tokens last one hour; use the refresh token to get a new one. The full flow is in [OAuth & SSO](oauth-guide.md).
