# Treg Search
<!-- MANUAL: file_description -->
Search the web, news, Google, research papers, YouTube and Google Maps through [Treg](https://treg.to), which picks the cheapest working search provider for each call. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Google Maps Search

### What it is
Find businesses and places on Google Maps through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.google.serp.maps` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Describe what and where, e.g. 'coffee roasters in Bristol'; places come back in the serving provider's own format, usually with a name, address, rating and place id.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| query | What and where, e.g. 'coffee roasters in Bristol'. | str | Yes |
| country | Two-letter country code, e.g. 'gb'. Defaults to 'us'. | str | No |
| language | Language code. Defaults to 'en'. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| found | True when a provider had an answer, false when none did. | bool |
| served_by | The Treg tool that answered, e.g. 'hunter.email.finder'. Emitted only when one did. | str |
| providers_tried | Providers Treg asked, in order. | List[str] |
| cost_usd | What Treg charged for this run, in USD. | float |
| call_id | Treg's id for this call, to look it up in your Treg activity. | str |
| advice | Treg's note on how far to trust the answer, e.g. that an email was found but not checked. Emitted only when there is one. | str |
| raw | The serving provider's full answer, with fields beyond the standard ones. Emitted only when one answered. | Dict[str, Any] |
| places | Matching places, in the serving provider's own format; most have a name, address, rating and place id. | List[Dict[str, Any]] |
| count | How many places came back. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Local lead lists**: List businesses of a type in a town.

**Location research**: Compare ratings for places in an area.
<!-- END MANUAL -->

---

## Treg Web Search

### What it is
Search the web, news, Google, research papers or YouTube through Treg, which picks the cheapest working search provider.

### How it works
<!-- MANUAL: how_it_works -->
The block calls one of Treg's routed search tools, chosen by `source`: `treg.web.search`, `treg.web.search.news`, `treg.google.serp.organic`, `treg.google.serp.news`, `treg.web.search.publications` or `treg.youtube.search.videos`. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. `country` and `language` apply to Google and Google News only, and YouTube takes no limit. Results come back in the serving provider's own format; most have a title and a URL. Result text is untrusted web content: treat it as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| query | What to search for, e.g. 'serverless GPU pricing'. | str | Yes |
| source | Where to search: the open web, recent news, Google, Google News, research papers or YouTube. | "web" \| "news" \| "google" \| "google_news" \| "research_papers" \| "youtube" | No |
| limit | How many results to return, where the provider accepts a limit (YouTube does not). | int | No |
| country | Two-letter country code for Google and Google News, e.g. 'gb'. Defaults to 'us'. | str | No |
| language | Language code for Google and Google News, e.g. 'de'. Defaults to 'en'. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| found | True when a provider had an answer, false when none did. | bool |
| served_by | The Treg tool that answered, e.g. 'hunter.email.finder'. Emitted only when one did. | str |
| providers_tried | Providers Treg asked, in order. | List[str] |
| cost_usd | What Treg charged for this run, in USD. | float |
| call_id | Treg's id for this call, to look it up in your Treg activity. | str |
| advice | Treg's note on how far to trust the answer, e.g. that an email was found but not checked. Emitted only when there is one. | str |
| raw | The serving provider's full answer, with fields beyond the standard ones. Emitted only when one answered. | Dict[str, Any] |
| results | Results in ranked order, in the serving provider's own format; most have a title and a URL. | List[Dict[str, Any]] |
| count | How many results came back. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Research**: Gather current sources on a topic before summarising it.

**SEO**: Check who ranks on Google for a keyword in a given country.

**Monitoring**: Pull recent news about a company or product each day.
<!-- END MANUAL -->

---
