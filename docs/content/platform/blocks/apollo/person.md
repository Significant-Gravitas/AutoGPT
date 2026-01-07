# Get Person Detail

### What it is
Get detailed person data with Apollo API, including email reveal.

### What it does
Get detailed person data with Apollo API, including email reveal

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| person_id | Apollo person ID to enrich (most accurate method) | str | No |
| first_name | First name of the person to enrich | str | No |
| last_name | Last name of the person to enrich | str | No |
| name | Full name of the person to enrich (alternative to first_name + last_name) | str | No |
| email | Known email address of the person (helps with matching) | str | No |
| domain | Company domain of the person (e.g., 'google.com') | str | No |
| company | Company name of the person | str | No |
| linkedin_url | LinkedIn URL of the person | str | No |
| organization_id | Apollo organization ID of the person's company | str | No |
| title | Job title of the person to enrich | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if enrichment failed | str |
| contact | Enriched contact information | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
