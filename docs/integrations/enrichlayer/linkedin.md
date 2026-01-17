# Enrichlayer LinkedIn
<!-- MANUAL: file_description -->
Blocks for enriching LinkedIn profile data and looking up profiles using the Enrichlayer API.
<!-- END MANUAL -->

## Get Linkedin Profile

### What it is
Fetch LinkedIn profile data using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves comprehensive LinkedIn profile data using Enrichlayer's API. Provide a LinkedIn profile URL to fetch details including work history, education, skills, and contact information.

Configure caching options for performance and optionally include additional data like inferred salary, personal email, or social media links.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| linkedin_url | LinkedIn profile URL to fetch data from | str | Yes |
| fallback_to_cache | Cache usage if live fetch fails | "on-error" \| "never" | No |
| use_cache | Cache utilization strategy | "if-present" \| "never" | No |
| include_skills | Include skills data | bool | No |
| include_inferred_salary | Include inferred salary data | bool | No |
| include_personal_email | Include personal email | bool | No |
| include_personal_contact_number | Include personal contact number | bool | No |
| include_social_media | Include social media profiles | bool | No |
| include_extra | Include additional data | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| profile | LinkedIn profile data | PersonProfileResponse |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Enrichment**: Enrich sales leads with detailed professional background information.

**Recruitment Research**: Gather candidate information for hiring and outreach workflows.

**Contact Discovery**: Find contact details associated with LinkedIn profiles.
<!-- END MANUAL -->

---

## Get Linkedin Profile Picture

### What it is
Get LinkedIn profile pictures using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the profile picture URL for a LinkedIn profile using Enrichlayer's API. Provide the LinkedIn profile URL to get a direct link to the user's profile photo.

The returned URL can be used for display, download, or further image processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| linkedin_profile_url | LinkedIn profile URL | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| profile_picture_url | LinkedIn profile picture URL | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**CRM Enhancement**: Add profile photos to contact records for visual identification.

**Personalized Outreach**: Include profile pictures in personalized email or message templates.

**Identity Verification**: Retrieve profile photos for manual identity verification workflows.
<!-- END MANUAL -->

---

## Linkedin Person Lookup

### What it is
Look up LinkedIn profiles by person information using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
This block finds LinkedIn profiles by matching person details like name, company, and title using Enrichlayer's API. Provide first name and company domain as minimum inputs, with optional last name, location, and title for better matching.

Enable similarity checks and profile enrichment for more detailed results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| first_name | Person's first name | str | Yes |
| last_name | Person's last name | str | No |
| company_domain | Domain of the company they work for (optional) | str | Yes |
| location | Person's location (optional) | str | No |
| title | Person's job title (optional) | str | No |
| include_similarity_checks | Include similarity checks | bool | No |
| enrich_profile | Enrich the profile with additional data | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| lookup_result | LinkedIn profile lookup result | PersonLookupResponse |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Discovery**: Find LinkedIn profiles for leads when you only have name and company.

**Contact Matching**: Match CRM contacts to their LinkedIn profiles for enrichment.

**Prospecting**: Discover LinkedIn profiles of people at target companies.
<!-- END MANUAL -->

---

## Linkedin Role Lookup

### What it is
Look up LinkedIn profiles by role in a company using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
This block finds LinkedIn profiles by role title and company using Enrichlayer's API. Specify a role like CEO, CTO, or VP of Sales along with the company name to find matching profiles.

Enable enrich_profile to automatically fetch full profile data for the matched result.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| role | Role title (e.g., CEO, CTO) | str | Yes |
| company_name | Name of the company | str | Yes |
| enrich_profile | Enrich the profile with additional data | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| role_lookup_result | LinkedIn role lookup result | RoleLookupResponse |

### Possible use case
<!-- MANUAL: use_case -->
**Decision Maker Discovery**: Find key decision makers at target companies for sales outreach.

**Executive Research**: Look up C-suite executives for account-based marketing.

**Org Chart Building**: Map leadership at companies by looking up specific roles.
<!-- END MANUAL -->

---
