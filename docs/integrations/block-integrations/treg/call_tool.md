# Treg Call Tool
<!-- MANUAL: file_description -->
Call any tool in [Treg](https://treg.to)'s catalog, or one your Treg team registered, by id. Find ids, inputs and prices with Treg Search Catalog and Treg Get Tool. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Call Tool

### What it is
Call any of Treg's 4,000+ tools by id (SEO, social, ads, people and company data, scraping, generation), or a tool your Treg team registered. Find tool ids and prices with Treg Search Catalog.

### How it works
<!-- MANUAL: how_it_works -->
The block sends your request to `https://treg.to/call/<tool>` with your Treg key in the `X-Treg-Token` header; Treg injects the provider's credential on its side and relays the provider's answer unchanged. `tool` is a catalog id (e.g. `tikhub.tiktok.user.profile`) or a team tool and its path (e.g. `intercom/conversations`); anything that could leave Treg's call route, such as `..`, is refused before sending. On the platform's Treg key only catalog ids are accepted, because that key's Treg team is shared by every user; connect your own key to call your team's tools. Treg headers other than the route options (`X-Treg-Route-Verify`, `-Waterfall`, `-Prefer`, `-Exclude`, `-Confirm`, `-Strict-Filters`) are refused, so the spending cap can't be overridden. `method` AUTO sends POST when there is a body and GET otherwise. `max_cost_usd` is sent as `X-Treg-Route-Max-Cost`, so Treg refuses, without charge, a call whose estimated price is above it. A tool that starts a background task (a video or image generation) answers at once with `async_task`, the descriptor for checking on it with another call; `cost_usd` is then the amount reserved, settled when the task ends. A provider that reports a failure inside a successful answer raises an error. Failed calls are not retried. Because a team tool can write to an outside system, the block is treated as an outside action. Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tool | What to call: a catalog tool id, e.g. 'tikhub.tiktok.user.profile', as Treg Search Catalog lists it; or a tool your Treg team registered, with its path, e.g. 'intercom/conversations'. | str | Yes |
| query | Query parameters, e.g. {"uniqueId": "tiktok"}. A list value repeats the parameter. | Dict[str, Any] | No |
| body | A JSON body, for tools called with POST. Treg Get Tool lists the fields a tool takes. | Body | No |
| method | The HTTP method. AUTO sends POST when there is a body, GET otherwise. | "AUTO" \| "GET" \| "POST" \| "PUT" \| "PATCH" \| "DELETE" | No |
| headers | Extra headers, e.g. a provider option or Treg's 'X-Treg-Route-Verify: true'. Your Treg key is added for you. | Dict[str, str] | No |
| max_cost_usd | The most this call may cost, in USD. Treg refuses, without charge, a call it prices above this before sending it. | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| response | The provider's answer, exactly as it sent it. | Response |
| status | The provider's HTTP status. | int |
| cost_usd | What Treg charged, in USD. For a task still running (see async_task) this is the amount reserved, settled when it ends. | float |
| call_id | Treg's id for this call. | str |
| served_by | The tool that answered, for a routed tool. Emitted only when Treg names one. | str |
| cached | True when Treg answered from its archive of the same question, at a tenth of the price. | bool |
| async_task | For a tool that starts a background task (a video or image generation): how to check on it with another call. Emitted only then. | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Anything in the catalog**: Call a tool the other Treg blocks don't cover, such as TikTok profiles, keyword volumes or backlinks.

**Team tools**: Call an API your Treg team registered, without the key ever reaching the workflow.
<!-- END MANUAL -->

---
