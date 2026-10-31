# Treg Companies
<!-- MANUAL: file_description -->
Find and look up companies through [Treg](https://treg.to), which routes each request across company-data providers and bills per call with no provider signup. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Enrich Company

### What it is
Look up a company (what it does, size, industry, founding year, headquarters and LinkedIn) from its domain, name, LinkedIn page or an employee's email, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.companies.enrich` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a domain, a name, a LinkedIn company page or anyone's work email at the company; the block returns the standard fields any provider fills, and `raw` holds the rest.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| domain | The company's website domain, e.g. 'modal.com'. | str | No |
| name | The company's name. | str | No |
| linkedin_url | Its LinkedIn company page URL. | str | No |
| email | Anyone's work email at the company. | str | No |

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
| name | Company name. | str |
| domain | Website domain. | str |
| website | Website URL. | str |
| description | What the company does. | str |
| industry | Industry. | str |
| employees | Headcount or headcount range. | str |
| founded | Year founded. | int |
| location | Headquarters. | str |
| linkedin_url | LinkedIn company page URL. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Lead routing**: Size and categorise a company from a signup's email domain.

**Account research**: Get a one-line description, headcount and headquarters for each target account.
<!-- END MANUAL -->

---

## Treg Find Company Jobs

### What it is
List a company's open job listings, a strong signal of what it is investing in, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.companies.jobs.search` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a company's domain or name; the block returns open job listings in the serving provider's own format.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| domain | The company's website domain. | str | No |
| name | The company's name. | str | No |
| limit | How many results to return. Most providers bill per result, so this is the main price dial. | int | No |

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
| jobs | Open job listings, in the serving provider's own format. | List[Dict[str, Any]] |
| count | How many listings came back. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Buying signals**: Spot accounts hiring for roles that need your product.

**Recruiting intelligence**: Track what a competitor is hiring for.
<!-- END MANUAL -->

---

## Treg Find Similar Companies

### What it is
Find companies similar to one you name, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.companies.similar` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a company's domain; the block returns lookalike companies in the serving provider's own format.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| domain | The company to find lookalikes of, by website domain, e.g. 'modal.com'. | str | Yes |

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
| companies | Matching companies, in the serving provider's own format. | List[Dict[str, Any]] |
| count | How many companies came back. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Expansion**: Find more accounts like your best customers.

**Competitive research**: List a company's closest competitors.
<!-- END MANUAL -->

---

## Treg Search Companies

### What it is
Find companies by description, name, industry or the technology they use. Treg routes the search across many company-data providers.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.companies.search` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Describe the kind of company in plain words, or give a name, industry or technology they use, plus an optional country. `limit` is the main price dial. Rows come back in the serving provider's own format.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| description | What kind of company, in plain words, e.g. 'Series A developer tools startups in Berlin'. | str | No |
| name | Look for a company by name. | str | No |
| industry | Only companies in this industry. | str | No |
| technology | Only companies that use this technology, e.g. 'Snowflake'. | str | No |
| country | Two-letter country code, e.g. 'DE', where the provider supports it. | str | No |
| limit | How many results to return. Most providers bill per result, so this is the main price dial. | int | No |

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
| companies | Matching companies, in the serving provider's own format. | List[Dict[str, Any]] |
| count | How many companies came back. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Market maps**: List companies in a niche or region.

**Technographic targeting**: Find companies that use a technology your product integrates with.
<!-- END MANUAL -->

---
