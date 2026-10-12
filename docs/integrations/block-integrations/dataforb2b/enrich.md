# DataForB2B Enrich
<!-- MANUAL: file_description -->
Blocks for enriching companies and LinkedIn profiles using DataForB2B's B2B database — firmographics, headcount, work/personal email, phone, and GitHub discovery.
<!-- END MANUAL -->

## Company Enrichment

### What it is
Look up and enrich a company using DataForB2B's B2B database — firmographics, headcount/size, industry, domain and social profiles from a company domain, name or LinkedIn URL. Account enrichment for B2B sales and CRM.

### How it works
<!-- MANUAL: how_it_works -->
The block trims the provided `company_identifier` and rejects empty values before sending it to DataForB2B's company enrichment endpoint. The API resolves the identifier (domain, name, or LinkedIn company URL) against its B2B database and returns firmographic data such as industry, headcount, funding, and social profiles. Client and server errors from the API are caught and surfaced via the `error` output instead of raising an exception; identifiers that cannot be resolved may return an empty result.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| company_identifier | Company domain, name, or LinkedIn URL to enrich | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Full company enrichment response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Account Enrichment**: Enrich inbound leads or CRM records with firmographic data before routing them to sales.

**Target Account Research**: Pull company size, industry, and funding stage to qualify accounts for an ABM campaign.

**CRM Cleanup**: Resolve a company domain or LinkedIn URL into current firmographic data before deduplicating records.
<!-- END MANUAL -->

---

## Profile Enrichment

### What it is
Look up and enrich a professional profile from a LinkedIn URL using DataForB2B's B2B database — returns the full profile (current role, experience and skills) plus work email, personal email and GitHub. Works as an email finder for lead enrichment, contact enrichment, cold outreach and CRM. Disable enrich_profile if you only need the email/GitHub lookups.

### How it works
<!-- MANUAL: how_it_works -->
The block trims the provided `profile_identifier` (a LinkedIn profile URL or id) and rejects empty values before sending it to DataForB2B's profile enrichment endpoint along with the requested `enrich_*` flags. `enrich_profile` is enabled by default and returns the full profile; when every flag is false, the block enables `enrich_profile` so the API still receives a valid request. Client and server errors from the API are caught and surfaced via the `error` output instead of raising an exception; unavailable contact fields may be absent from the result.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| profile_identifier | LinkedIn profile URL (or profile id) to enrich | str | Yes |
| enrich_profile | Return the full LinkedIn profile (role, experience, skills) | bool | No |
| enrich_work_email | Find the professional / work email | bool | No |
| enrich_personal_email | Find the personal email | bool | No |
| enrich_github | Find the GitHub profile | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Full enrichment response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Email Finder**: Resolve a work or personal email address from a LinkedIn profile URL for cold outreach.

**Lead Enrichment**: Enrich a prospect list with current role, experience, and skills pulled straight from LinkedIn.

**Contact Verification**: Refresh a saved contact from a LinkedIn profile URL before routing it into an outreach sequence.
<!-- END MANUAL -->

---
