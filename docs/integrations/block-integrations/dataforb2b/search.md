# Dataforb2B Search
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Linkedin Company Search

### What it is
Search companies and accounts by structured filters — industry, headcount/size, location, funding, keywords — using DataForB2B's database. Build target-account lists for B2B sales and account-based marketing. Accepts LinkedIn URLs as identifiers.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| filters_json | Raw filter JSON {op, conditions:[{column,type,value,value2?}]}. Paste 'applied_filters' from Linkedin Smart Search and set 'offset' to paginate beyond the first page. Used alone, or merged (AND) with the filter slots above. | Dict[str, Any] | No |
| match | Combine slot conditions with 'and' or 'or' | str | No |
| count | Number of results to return | int | No |
| offset | Pagination offset — 0 for page 1, then 25, 50, … to page through results | int | No |
| enrich_live | Fetch fresh live data (uses more credits) | bool | No |
| filter_1_column | Filter 1 column | "name" \| "tagline" \| "description" \| "domain" \| "universal_name" \| "keyword" \| "industry" \| "employee_count" \| "country_iso_code" \| "city" \| "region" \| "office_country" \| "office_city" \| "office_region" \| "employee_growth_1m" \| "employee_growth_6m" \| "employee_growth_12m" \| "recent_hires_count" \| "founded_year" \| "company_type" \| "follower_count" \| "page_verified" \| "category" \| "last_funding_amount_usd" \| "last_funding_date" \| "funding_stage_normalized" \| "has_funding" | No |
| filter_1_operator | Filter 1 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_1_value | Filter 1 value | str | No |
| filter_2_column | Filter 2 column | "name" \| "tagline" \| "description" \| "domain" \| "universal_name" \| "keyword" \| "industry" \| "employee_count" \| "country_iso_code" \| "city" \| "region" \| "office_country" \| "office_city" \| "office_region" \| "employee_growth_1m" \| "employee_growth_6m" \| "employee_growth_12m" \| "recent_hires_count" \| "founded_year" \| "company_type" \| "follower_count" \| "page_verified" \| "category" \| "last_funding_amount_usd" \| "last_funding_date" \| "funding_stage_normalized" \| "has_funding" | No |
| filter_2_operator | Filter 2 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_2_value | Filter 2 value | str | No |
| filter_3_column | Filter 3 column | "name" \| "tagline" \| "description" \| "domain" \| "universal_name" \| "keyword" \| "industry" \| "employee_count" \| "country_iso_code" \| "city" \| "region" \| "office_country" \| "office_city" \| "office_region" \| "employee_growth_1m" \| "employee_growth_6m" \| "employee_growth_12m" \| "recent_hires_count" \| "founded_year" \| "company_type" \| "follower_count" \| "page_verified" \| "category" \| "last_funding_amount_usd" \| "last_funding_date" \| "funding_stage_normalized" \| "has_funding" | No |
| filter_3_operator | Filter 3 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_3_value | Filter 3 value | str | No |
| filter_4_column | Filter 4 column | "name" \| "tagline" \| "description" \| "domain" \| "universal_name" \| "keyword" \| "industry" \| "employee_count" \| "country_iso_code" \| "city" \| "region" \| "office_country" \| "office_city" \| "office_region" \| "employee_growth_1m" \| "employee_growth_6m" \| "employee_growth_12m" \| "recent_hires_count" \| "founded_year" \| "company_type" \| "follower_count" \| "page_verified" \| "category" \| "last_funding_amount_usd" \| "last_funding_date" \| "funding_stage_normalized" \| "has_funding" | No |
| filter_4_operator | Filter 4 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_4_value | Filter 4 value | str | No |
| filter_5_column | Filter 5 column | "name" \| "tagline" \| "description" \| "domain" \| "universal_name" \| "keyword" \| "industry" \| "employee_count" \| "country_iso_code" \| "city" \| "region" \| "office_country" \| "office_city" \| "office_region" \| "employee_growth_1m" \| "employee_growth_6m" \| "employee_growth_12m" \| "recent_hires_count" \| "founded_year" \| "company_type" \| "follower_count" \| "page_verified" \| "category" \| "last_funding_amount_usd" \| "last_funding_date" \| "funding_stage_normalized" \| "has_funding" | No |
| filter_5_operator | Filter 5 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_5_value | Filter 5 value | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| result | Full search response (total, count, results) | Dict[str, Any] |
| results | List of matching companies | List[Any] |
| total | Total number of matches | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Linkedin People Search

### What it is
Search people and B2B leads by structured filters — job title, company, location, industry, seniority, skills — using DataForB2B's database. Find employees at a company, people by job title, who works where, decision-makers and key contacts (owners, founders, C-suite, VPs, directors), and build a prospect or lead list. Accepts LinkedIn URLs as identifiers. The lead-sourcing step of a prospecting or outreach workflow.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| filters_json | Raw filter JSON {op, conditions:[{column,type,value,value2?}]}. Paste 'applied_filters' from Linkedin Smart Search and set 'offset' to paginate beyond the first page. Used alone, or merged (AND) with the filter slots above. | Dict[str, Any] | No |
| match | Combine slot conditions with 'and' or 'or' | str | No |
| count | Number of results to return | int | No |
| offset | Pagination offset — 0 for page 1, then 25, 50, … to page through results | int | No |
| enrich_live | Fetch fresh live data (uses more credits) | bool | No |
| filter_1_column | Filter 1 column | "first_name" \| "last_name" \| "profile_location" \| "profile_country" \| "profile_industry" \| "follower_count" \| "keyword" \| "current_company" \| "current_title" \| "current_job_location" \| "current_company_industry" \| "current_company_category" \| "current_company_size" \| "current_company_id" \| "current_employment_type" \| "years_in_current_position" \| "years_at_current_company" \| "current_company_has_funding" \| "current_company_funding_stage" \| "current_company_investor" \| "past_company" \| "past_title" \| "past_job_location" \| "past_company_industry" \| "past_company_size" \| "past_company_id" \| "past_employment_type" \| "years_at_past_company" \| "skill" \| "school" \| "degree" \| "degree_level" \| "field_of_study" \| "language" \| "language_iso" \| "language_proficiency" \| "certification" \| "certification_authority" \| "years_of_experience" \| "num_total_jobs" \| "is_currently_employed" | No |
| filter_1_operator | Filter 1 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_1_value | Filter 1 value | str | No |
| filter_2_column | Filter 2 column | "first_name" \| "last_name" \| "profile_location" \| "profile_country" \| "profile_industry" \| "follower_count" \| "keyword" \| "current_company" \| "current_title" \| "current_job_location" \| "current_company_industry" \| "current_company_category" \| "current_company_size" \| "current_company_id" \| "current_employment_type" \| "years_in_current_position" \| "years_at_current_company" \| "current_company_has_funding" \| "current_company_funding_stage" \| "current_company_investor" \| "past_company" \| "past_title" \| "past_job_location" \| "past_company_industry" \| "past_company_size" \| "past_company_id" \| "past_employment_type" \| "years_at_past_company" \| "skill" \| "school" \| "degree" \| "degree_level" \| "field_of_study" \| "language" \| "language_iso" \| "language_proficiency" \| "certification" \| "certification_authority" \| "years_of_experience" \| "num_total_jobs" \| "is_currently_employed" | No |
| filter_2_operator | Filter 2 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_2_value | Filter 2 value | str | No |
| filter_3_column | Filter 3 column | "first_name" \| "last_name" \| "profile_location" \| "profile_country" \| "profile_industry" \| "follower_count" \| "keyword" \| "current_company" \| "current_title" \| "current_job_location" \| "current_company_industry" \| "current_company_category" \| "current_company_size" \| "current_company_id" \| "current_employment_type" \| "years_in_current_position" \| "years_at_current_company" \| "current_company_has_funding" \| "current_company_funding_stage" \| "current_company_investor" \| "past_company" \| "past_title" \| "past_job_location" \| "past_company_industry" \| "past_company_size" \| "past_company_id" \| "past_employment_type" \| "years_at_past_company" \| "skill" \| "school" \| "degree" \| "degree_level" \| "field_of_study" \| "language" \| "language_iso" \| "language_proficiency" \| "certification" \| "certification_authority" \| "years_of_experience" \| "num_total_jobs" \| "is_currently_employed" | No |
| filter_3_operator | Filter 3 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_3_value | Filter 3 value | str | No |
| filter_4_column | Filter 4 column | "first_name" \| "last_name" \| "profile_location" \| "profile_country" \| "profile_industry" \| "follower_count" \| "keyword" \| "current_company" \| "current_title" \| "current_job_location" \| "current_company_industry" \| "current_company_category" \| "current_company_size" \| "current_company_id" \| "current_employment_type" \| "years_in_current_position" \| "years_at_current_company" \| "current_company_has_funding" \| "current_company_funding_stage" \| "current_company_investor" \| "past_company" \| "past_title" \| "past_job_location" \| "past_company_industry" \| "past_company_size" \| "past_company_id" \| "past_employment_type" \| "years_at_past_company" \| "skill" \| "school" \| "degree" \| "degree_level" \| "field_of_study" \| "language" \| "language_iso" \| "language_proficiency" \| "certification" \| "certification_authority" \| "years_of_experience" \| "num_total_jobs" \| "is_currently_employed" | No |
| filter_4_operator | Filter 4 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_4_value | Filter 4 value | str | No |
| filter_5_column | Filter 5 column | "first_name" \| "last_name" \| "profile_location" \| "profile_country" \| "profile_industry" \| "follower_count" \| "keyword" \| "current_company" \| "current_title" \| "current_job_location" \| "current_company_industry" \| "current_company_category" \| "current_company_size" \| "current_company_id" \| "current_employment_type" \| "years_in_current_position" \| "years_at_current_company" \| "current_company_has_funding" \| "current_company_funding_stage" \| "current_company_investor" \| "past_company" \| "past_title" \| "past_job_location" \| "past_company_industry" \| "past_company_size" \| "past_company_id" \| "past_employment_type" \| "years_at_past_company" \| "skill" \| "school" \| "degree" \| "degree_level" \| "field_of_study" \| "language" \| "language_iso" \| "language_proficiency" \| "certification" \| "certification_authority" \| "years_of_experience" \| "num_total_jobs" \| "is_currently_employed" | No |
| filter_5_operator | Filter 5 operator | "=" \| "!=" \| "like" \| "not_like" \| "in" \| "not_in" \| ">" \| ">=" \| "<" \| "<=" \| "between" | No |
| filter_5_value | Filter 5 value | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| result | Full search response (total, count, results) | Dict[str, Any] |
| results | List of matching LinkedIn people / leads | List[Any] |
| total | Total number of matches | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
