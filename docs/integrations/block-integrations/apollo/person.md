# Apollo Person
<!-- MANUAL: file_description -->
Blocks for enriching individual person data including contact details and email discovery.
<!-- END MANUAL -->

## Get Person Detail

### What it is
Get detailed person data with Apollo API, including email reveal

### How it works
<!-- MANUAL: how_it_works -->
This block enriches person data using Apollo's API. You can look up by Apollo person ID for best accuracy, or match by name plus company information, LinkedIn URL, or email address.

Returns comprehensive contact details including email addresses (if available), job title, company information, and social profiles.
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
| contact | Enriched contact information | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Contact Enrichment**: Get full contact details from partial information like name and company.

**Email Discovery**: Find verified email addresses for outreach campaigns.

**Profile Completion**: Fill in missing contact details in your CRM or database.
<!-- END MANUAL -->

---
