# Apollo Organization
<!-- MANUAL: file_description -->
Blocks for searching and retrieving organization data from Apollo's B2B database.
<!-- END MANUAL -->

## Search Organizations

### What it is
Search for organizations in Apollo

### How it works
<!-- MANUAL: how_it_works -->
This block searches the Apollo database for organizations using various filters like employee count, location, and keywords. Apollo maintains a comprehensive database of company information for sales and marketing purposes.

Results can be filtered by headquarters location, excluded locations, industry keywords, and specific Apollo organization IDs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| organization_num_employees_range | The number range of employees working for the company. This enables you to find companies based on headcount. You can add multiple ranges to expand your search results.  Each range you add needs to be a string, with the upper and lower numbers of the range separated only by a comma. | List[int] | No |
| organization_locations | The location of the company headquarters. You can search across cities, US states, and countries.  If a company has several office locations, results are still based on the headquarters location. For example, if you search chicago but a company's HQ location is in boston, any Boston-based companies will not appear in your search results, even if they match other parameters.  To exclude companies based on location, use the organization_not_locations parameter.  | List[str] | No |
| organizations_not_locations | Exclude companies from search results based on the location of the company headquarters. You can use cities, US states, and countries as locations to exclude.  This parameter is useful for ensuring you do not prospect in an undesirable territory. For example, if you use ireland as a value, no Ireland-based companies will appear in your search results.  | List[str] | No |
| q_organization_keyword_tags | Filter search results based on keywords associated with companies. For example, you can enter mining as a value to return only companies that have an association with the mining industry. | List[str] | No |
| q_organization_name | Filter search results to include a specific company name.  If the value you enter for this parameter does not match with a company's name, the company will not appear in search results, even if it matches other parameters. Partial matches are accepted. For example, if you filter by the value marketing, a company called NY Marketing Unlimited would still be eligible as a search result, but NY Market Analysis would not be eligible. | str | No |
| organization_ids | The Apollo IDs for the companies you want to include in your search results. Each company in the Apollo database is assigned a unique ID.  To find IDs, identify the values for organization_id when you call this endpoint. | List[str] | No |
| max_results | The maximum number of results to return. If you don't specify this parameter, the default is 100. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| organizations | List of organizations found | List[Dict[str, Any]] |
| organization | Each found organization, one at a time | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Market Research**: Find companies matching specific criteria for market analysis.

**Lead List Building**: Build targeted lists of companies for outbound sales campaigns.

**Competitive Intelligence**: Research competitors and similar companies in your market.
<!-- END MANUAL -->

---
