# Treg Phone
<!-- MANUAL: file_description -->
Find phone numbers through [Treg](https://treg.to), which routes each lookup across many contact-data providers and bills per call. Needs a Treg API key from treg.to (`TREG_API_KEY` for a platform key). Every paid call is billed by Treg from your team's prepaid balance at the provider's price, and the block reports the charge as `cost_usd`. On a platform-provided key the run is charged in credits from that same amount; on your own Treg key it is free on the platform.
<!-- END MANUAL -->

## Treg Find Phone Number

### What it is
Find a person's phone number from their LinkedIn URL, email, or name and company domain. Treg tries providers cheapest first.

### How it works
<!-- MANUAL: how_it_works -->
The block sends one request to `POST https://treg.to/call/treg.people.phone.find` with your Treg key in the `X-Treg-Token` header. Treg tries the providers that do this job, your team's own keys first and then the cheapest per answer, and moves on when one fails or has no answer, until one answers or `max_cost_usd` (sent as `X-Treg-Route-Max-Cost`) would be passed. Turning off `try_more_providers` stops after the first provider. When nobody has an answer the block emits `found: false` rather than an error; some providers bill a miss, so `cost_usd` can be above zero. Failed calls raise a block error with Treg's reason (for example an empty balance, with the top-up link) and are not retried, because a repeated paid call can be billed again. Give a LinkedIn URL, an email, or a full name and company domain. A found number is unconfirmed; turn `verify` on to check that the line is live (numbers need a country code), and note that a live line does not prove the number is this person's.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| max_cost_usd | The most this run may spend, in USD, across every provider Treg tries. A call that would go over it is refused without charge. | float | No |
| try_more_providers | When a provider has no answer, try the next cheapest one, up to the spending cap. Some providers bill a miss. Turn off to stop after the first provider. | bool | No |
| linkedin_url | The person's LinkedIn profile URL. | str | No |
| email | Their email address. | str | No |
| full_name | Their full name, with a company domain. | str | No |
| company_domain | Their company's website domain. | str | No |
| verify | Check that the number is live on its network in the same call. Only numbers with a country code can be checked, and a live line does not prove it is this person's. | bool | No |

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
| phone | The phone number found. | str |
| line_type | mobile, landline, voip and so on. | str |
| country_code | The number's country code. | str |
| verification | The live check's verdict. Emitted only when it ran. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Sales calls**: Add direct dials to qualified leads before handing them to a rep.

**Recruiting**: Find a phone number for a shortlisted candidate.
<!-- END MANUAL -->

---
