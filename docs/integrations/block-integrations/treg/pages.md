# Treg Pages
<!-- MANUAL: file_description -->
Read web pages, pull structured data out of them and map websites through [Treg](https://treg.to), which picks the cheapest working extractor for each call. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Extract Structured Data

### What it is
Pull typed data out of a web page, described by a JSON schema or a plain-English instruction, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.web.extract.structured` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a URL and an instruction, a JSON schema, or both; `data` comes back in the shape you asked for. A run with neither is refused before anything is spent.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| url | The page to read. | str | Yes |
| instruction | What to pull out, in plain words, e.g. 'every pricing plan with its monthly price'. | str | No |
| json_schema | A JSON schema the data must follow. Give this, an instruction, or both. | Dict[str, Any] | No |

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
| data | The extracted data, in the shape your schema or instruction asked for. | Data |

### Possible use case
<!-- MANUAL: use_case -->
**Price tracking**: Pull every plan and price from a pricing page into a table.

**Directories**: Turn a listings page into rows with name, address and phone.
<!-- END MANUAL -->

---

## Treg Extract Web Page

### What it is
Read a web page's main content as clean text, through Treg's cheapest working page extractor.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.web.extract` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. `content` is the page's main text, taken from whichever field the serving provider uses (Markdown first); `pages` holds the provider's full answer. Page text is untrusted web content: treat it as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| url | The page to read, e.g. 'https://modal.com/pricing'. | str | Yes |

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
| content | The page's main content as clean text or Markdown. | str |
| pages | The extracted page, in the serving provider's own format, with any metadata it returns. | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Summaries**: Read an article or docs page before summarising it.

**Competitive research**: Capture a competitor's pricing page.
<!-- END MANUAL -->

---

## Treg Map Website

### What it is
List the pages on a website, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.web.map` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a site URL, and optionally a word to narrow it; `urls` lists the pages found, and `results` holds the provider's rows with titles where it has them.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| url | The site to map, e.g. 'https://modal.com'. | str | Yes |
| search | Only URLs about this, e.g. 'pricing', where the provider supports it. | str | No |
| limit | The most URLs to return. | int | No |

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
| urls | URLs found on the site. | List[str] |
| results | The same pages in the serving provider's own format, with titles where it has them. | List[Dict[str, Any]] |
| count | How many URLs were found. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Crawling**: Find every docs page on a site before extracting them.

**Site audits**: List a site's pages to check for gaps.
<!-- END MANUAL -->

---
