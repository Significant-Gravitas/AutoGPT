# Treg People
<!-- MANUAL: file_description -->
Find people and look up who someone is through [Treg](https://treg.to), which routes each request across people-data providers (Apollo, People Data Labs, Crustdata and others) and bills per call with no provider signup. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Enrich Person

### What it is
Look up a person (name, job title, company, LinkedIn and location) from an email, a LinkedIn URL, or a name and company domain, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.enrich` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give an email, a LinkedIn URL, or a full name and company domain; the block returns the standard fields any provider fills (name, job title, company, LinkedIn, location), and `raw` holds everything else the provider sent.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| email | The person's email. | str | No |
| linkedin_url | Their LinkedIn profile URL. | str | No |
| full_name | Their full name, with a company domain. | str | No |
| company_domain | Their company's website domain. | str | No |

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
| full_name | Full name. | str |
| first_name | First name. | str |
| last_name | Last name. | str |
| title | Current job title. | str |
| company | Current company. | str |
| company_domain | Current company's domain. | str |
| linkedin_url | LinkedIn profile URL. | str |
| location | Where they are based. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inbound leads**: Turn a signup email into a name, title and company to route the lead.

**CRM enrichment**: Fill in job titles and companies for existing contacts.
<!-- END MANUAL -->

---

## Treg Find LinkedIn From Email

### What it is
Find the LinkedIn profile behind a work email, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.identity.resolve` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a work email; the block returns the LinkedIn profile URL behind it and the person's name when known.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| email | A work email address, e.g. 'jane@outerbounds.com'. | str | Yes |

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
| linkedin_url | Their LinkedIn profile URL. | str |
| full_name | Their full name, when known. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Research**: Go from an email in your inbox to the sender's LinkedIn profile.

**Enrichment chains**: Feed the LinkedIn URL into Treg Get LinkedIn Profile or Treg Find Phone Number.
<!-- END MANUAL -->

---

## Treg Search People

### What it is
Find people by description, company, job title or name. Treg routes the search across 26 people-data providers, cheapest first.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.search` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Describe who you want in plain words, or give a company domain, job title or name, plus optional keywords, country and location. `limit` is the main price dial, since most providers bill per row. Rows come back in the serving provider's own format. If that provider could not apply a filter you gave, the block lists it in `ignored_filters`, so the rows may be broader than asked. Emails in search rows are directory listings: verify them before sending.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| description | Who you are looking for, in plain words, e.g. 'heads of platform engineering at Series B AI infrastructure startups'. | str | No |
| company_domain | Only people at this company, by website domain, e.g. 'modal.com'. | str | No |
| job_title | Only people with this job title, e.g. 'CTO'. | str | No |
| full_name | Look for a person by name. | str | No |
| keywords | Skills or topics they must match, e.g. ['kubernetes']. | List[str] | No |
| country | Two-letter country code, e.g. 'US', where the provider supports it. | str | No |
| location | A place, e.g. 'London, United Kingdom', where the provider supports it. | str | No |
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
| people | Matching people, in the serving provider's own format. | List[Dict[str, Any]] |
| count | How many people came back. | int |
| ignored_filters | Filters the serving provider could not apply, so the results may be broader than asked. Emitted only when there are some. | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Account mapping**: List the platform engineering leaders at each target account.

**Recruiting**: Find engineers with specific skills in a city.

**Lead lists**: Build a list of people in a role across a set of companies, then find and verify their emails.
<!-- END MANUAL -->

---
