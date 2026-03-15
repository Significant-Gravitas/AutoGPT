# Apollo People
<!-- MANUAL: file_description -->
Blocks for searching people in Apollo's B2B contact database with various filters.
<!-- END MANUAL -->

## Search People

### What it is
Search for people in Apollo

### How it works
<!-- MANUAL: how_it_works -->
This block searches Apollo's database for people based on job titles, seniority, location, company, and other criteria. It's designed for finding prospects and contacts for sales and marketing.

Enable enrich_info to get detailed contact information including verified email addresses (costs more credits).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| person_titles | Job titles held by the people you want to find. For a person to be included in search results, they only need to match 1 of the job titles you add. Adding more job titles expands your search results.          Results also include job titles with the same terms, even if they are not exact matches. For example, searching for marketing manager might return people with the job title content marketing manager.          Use this parameter in combination with the person_seniorities[] parameter to find people based on specific job functions and seniority levels.          | List[str] | No |
| person_locations | The location where people live. You can search across cities, US states, and countries.          To find people based on the headquarters locations of their current employer, use the organization_locations parameter. | List[str] | No |
| person_seniorities | The job seniority that people hold within their current employer. This enables you to find people that currently hold positions at certain reporting levels, such as Director level or senior IC level.          For a person to be included in search results, they only need to match 1 of the seniorities you add. Adding more seniorities expands your search results.          Searches only return results based on their current job title, so searching for Director-level employees only returns people that currently hold a Director-level title. If someone was previously a Director, but is currently a VP, they would not be included in your search results.          Use this parameter in combination with the person_titles[] parameter to find people based on specific job functions and seniority levels. | List["owner" \| "founder" \| "c_suite" \| "partner" \| "vp" \| "head" \| "director" \| "manager" \| "senior" \| "entry" \| "intern"] | No |
| organization_locations | The location of the company headquarters for a person's current employer. You can search across cities, US states, and countries.          If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, people that work for the Boston-based company will not appear in your results, even if they match other parameters.          To find people based on their personal location, use the person_locations parameter. | List[str] | No |
| q_organization_domains | The domain name for the person's employer. This can be the current employer or a previous employer. Do not include www., the @ symbol, or similar.          You can add multiple domains to search across companies.            Examples: apollo.io and microsoft.com | List[str] | No |
| contact_email_statuses | The email statuses for the people you want to find. You can add multiple statuses to expand your search. | List["verified" \| "unverified" \| "likely_to_engage" \| "unavailable"] | No |
| organization_ids | The Apollo IDs for the companies (employers) you want to include in your search results. Each company in the Apollo database is assigned a unique ID.          To find IDs, call the Organization Search endpoint and identify the values for organization_id. | List[str] | No |
| organization_num_employees_range | The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.          Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma. | List[int] | No |
| q_keywords | A string of words over which we want to filter the results | str | No |
| max_results | The maximum number of results to return. If you don't specify this parameter, the default is 25. Limited to 500 to prevent overspending. | int | No |
| enrich_info | Whether to enrich contacts with detailed information including real email addresses. This will double the search cost. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| people | List of people found | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Prospecting**: Find decision-makers at target companies for outbound sales.

**Recruiting**: Search for candidates with specific titles and experience.

**ABM Campaigns**: Build contact lists at specific accounts for account-based marketing.
<!-- END MANUAL -->

---
