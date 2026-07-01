# Dataforb2B Enrich
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Company Enrichment

### What it is
Look up and enrich a company using DataForB2B's B2B database — firmographics, headcount/size, industry, domain and social profiles from a company domain, name or LinkedIn URL. Account enrichment for B2B sales and CRM.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| company_identifier | Company domain, name, or LinkedIn URL to enrich | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if enrichment failed | str |
| result | Full company enrichment response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Linkedin Profile Enrichment

### What it is
Look up and enrich a professional profile from a LinkedIn URL using DataForB2B's B2B database — returns the full profile (current role, experience, skills) plus work email, personal email and GitHub. Works as an email finder for lead enrichment, contact enrichment, cold outreach and CRM. Toggle the enrich_work_email flag to fetch only an email.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| error | Error message if enrichment failed | str |
| result | Full enrichment response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
