# Treg Emails
<!-- MANUAL: file_description -->
Find and check email addresses through [Treg](https://treg.to), which routes each lookup across dozens of contact-data providers (Hunter, Apollo, LeadMagic and others) and bills per call with no provider signup. Needs a Treg API key: a team agent token or dashboard API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Find Personal Email

### What it is
Find someone's personal email (Gmail, Outlook and so on) from their LinkedIn URL or work email, through Treg.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.email.find.personal` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a LinkedIn URL or a work email. A personal address is a provider's match to a profile; turn `verify` on to check the mailbox before you write to it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| linkedin_url | The person's LinkedIn profile URL. | str | No |
| work_email | Their work email, if you have it instead. | str | No |
| verify | Check that the address accepts mail in the same call (a fraction of a cent more). A found address is a provider's best match until it is checked. | bool | No |

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
| email | The email address found. | str |
| verified | True only when the mailbox was checked and accepts mail. | bool |
| verification | The check's verdict: valid, invalid, catch_all, risky or unknown. Emitted only when the address was checked. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Recruiting**: Reach a candidate at a personal address rather than at their current employer.

**Alumni outreach**: Reconnect with people who have left the company you knew them at.
<!-- END MANUAL -->

---

## Treg Find Work Email

### What it is
Find someone's work email from their name and company domain, or their LinkedIn URL. Treg tries 29 providers cheapest first and can check the address before you send to it.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.email.find` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a full name and company domain, or a LinkedIn URL. With `verify` on (the default) Treg also checks the mailbox in the same call for a small extra charge, and `verified` is true only when that check says `valid`; a catch-all domain comes back as `catch_all`, which is not confirmed. With `verify` off, `verified` reflects only the provider's own claim, and `advice` says the address was found but not checked.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| full_name | The person's full name, e.g. 'Jane Doe'. | str | No |
| company_domain | Their company's website domain, e.g. 'outerbounds.com'. | str | No |
| linkedin_url | Their LinkedIn profile URL. Enough on its own, without a name and domain. | str | No |
| verify | Check that the address accepts mail in the same call (a fraction of a cent more). A found address is a provider's best match until it is checked. | bool | No |

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
| email | The email address found. | str |
| verified | True only when the mailbox was checked and accepts mail. | bool |
| verification | The check's verdict: valid, invalid, catch_all, risky or unknown. Emitted only when the address was checked. | str |
| confidence | The provider's confidence in the match, from 0 to 1. | float |
| first_name | First name, as the provider has it. | str |
| last_name | Last name, as the provider has it. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Outbound lists**: Turn a list of names and company domains into checked work emails before a sequence starts.

**CRM hygiene**: Fill in missing emails on contacts that only have a LinkedIn URL.

**Event follow-up**: Find emails for speakers or attendees from their name and company.
<!-- END MANUAL -->

---

## Treg Verify Email

### What it is
Check whether an email address accepts mail before you send to it, through Treg's cheapest available verifier.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.email.verify` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. `valid` is the provider's verdict that the mailbox accepts mail; `status` is its own word for it, such as `valid`, `invalid`, `accept_all` (the domain accepts any address, so this one is unconfirmed) or `unknown`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| email | The email address to check. | str | Yes |

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
| valid | True when the mailbox accepts mail. | bool |
| status | The provider's verdict, e.g. valid, invalid, accept_all (the domain accepts any address, so this one is unconfirmed) or unknown. | str |
| score | The provider's deliverability score. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Before sending**: Drop dead addresses from a list so a campaign does not bounce.

**Signup checks**: Check an address a user typed before saving it.
<!-- END MANUAL -->

---
