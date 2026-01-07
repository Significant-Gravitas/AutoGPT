# Get Linkedin Profile

### What it is
Fetch LinkedIn profile data using Enrichlayer.

### What it does
Fetch LinkedIn profile data using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| linkedin_url | LinkedIn profile URL to fetch data from | str | Yes |
| fallback_to_cache | Cache usage if live fetch fails | "on-error" | "never" | No |
| use_cache | Cache utilization strategy | "if-present" | "never" | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Get Linkedin Profile Picture

### What it is
Get LinkedIn profile pictures using Enrichlayer.

### What it does
Get LinkedIn profile pictures using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Linkedin Person Lookup

### What it is
Look up LinkedIn profiles by person information using Enrichlayer.

### What it does
Look up LinkedIn profiles by person information using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Linkedin Role Lookup

### What it is
Look up LinkedIn profiles by role in a company using Enrichlayer.

### What it does
Look up LinkedIn profiles by role in a company using Enrichlayer

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
