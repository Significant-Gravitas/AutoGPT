# Treg LinkedIn
<!-- MANUAL: file_description -->
Read public LinkedIn profiles and company pages through [Treg](https://treg.to), without a LinkedIn account or your own scraper. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Get LinkedIn Company

### What it is
Read a LinkedIn company page through Treg, without a LinkedIn account or scraper.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.linkedin.company.profile` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give the company page URL or just its handle.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| linkedin_url | The company page URL, e.g. 'https://www.linkedin.com/company/modal-labs', or just the handle, e.g. 'modal-labs'. | str | Yes |

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
| description | The page's About text. | str |
| website | Website URL. | str |
| followers | Follower count. | int |
| employees | Headcount or headcount range. | str |
| location | Headquarters. | str |
| linkedin_url | Canonical page URL. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Account research**: Read a company's LinkedIn description, size and follower count.

**Monitoring**: Track follower growth for a list of companies.
<!-- END MANUAL -->

---

## Treg Get LinkedIn Profile

### What it is
Read a LinkedIn member's public profile through Treg, without a LinkedIn account or scraper.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.linkedin.user.profile` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a profile URL or just the handle. Counts such as followers are returned as numbers when the provider gives one; a value like "500+" is left out of `connections` and stays in `raw`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| linkedin_url | The profile URL, e.g. 'https://www.linkedin.com/in/janedoe', or just the handle, e.g. 'janedoe'. | str | Yes |

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
| headline | Profile headline. | str |
| location | Location on the profile. | str |
| about | The About section. | str |
| followers | Follower count. | int |
| connections | Connection count. | int |
| linkedin_url | Canonical profile URL. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Meeting prep**: Pull a person's headline and About section before a call.

**Candidate screening**: Read profiles for a shortlist in bulk.
<!-- END MANUAL -->

---
