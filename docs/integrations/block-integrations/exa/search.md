# Exa Search
<!-- MANUAL: file_description -->
Blocks for searching the web with Exa's search API, from quick lookups to deep research with structured answers.
<!-- END MANUAL -->

## Exa Search

### What it is
Searches the web with Exa and returns ranked pages with their text or highlights. Choose 'lite' for the cheapest, fastest search or 'deep' for multi-step research, and set an Output Schema to get a synthesized, cited answer.

### How it works
<!-- MANUAL: how_it_works -->
This block sends one request to Exa's `/search` endpoint and returns the ranked results, each with up to 10,000 characters of page text unless you change Contents. The type sets speed, depth and price: 'auto' (default), 'lite' (cheapest and fastest), 'deep-lite' or 'deep' (multi-step research); the legacy types 'fast', 'instant', 'deep-reasoning', 'neural' and 'keyword' are still sent unchanged, but Exa recommends 'lite', 'deep' or 'auto' instead. The categories 'people' and 'publication' are new, and the legacy 'research paper' and 'linkedin profile' are sent as 'publication' and 'people'.

Set Output Schema (and optionally System Prompt) to have Exa synthesize an answer from the results into Output Content, with sources in Output Grounding. Before sending, the block rejects options 'lite' does not support (such as Output Schema, System Prompt, Summary or Max Age Hours) by name, and Additional Queries unless the type is a deep one. The run is billed from the dollar cost Exa reports for the request, at 100 credits per US dollar.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | What to search for, in natural language. Describe the pages you want rather than listing keywords. | str | Yes |
| type | How Exa searches. 'auto' (default): balanced quality and speed, ~1 s, $7 per 1k requests. 'lite': cheapest and fastest, ~450 ms, $4 per 1k; cached content only and accepts only the query, result count, text/highlights, snapshot, category, domain and published-date filters, location, moderation, objective and user identifier. 'deep-lite': lightweight research with synthesis, ~4 s, $12 per 1k. 'deep': multi-step research with synthesis, 4-15 s, $12 per 1k; best with Output Schema. Legacy: 'fast' and 'instant' (use 'lite'), 'deep-reasoning' (use 'deep'; 12-40 s, $15 per 1k), 'neural' and 'keyword' (no longer documented by Exa; use 'auto'). | "auto" \| "lite" \| "deep-lite" \| "deep" \| "fast" \| "instant" \| "deep-reasoning" \| "neural" \| "keyword" | No |
| category | Focus the search on one kind of page: company, people, news, publication (scholarly papers, preprints and journal articles), personal site or financial report. 'company' and 'people' do not support published-date filters, and 'people' does not support Exclude Domains. Legacy: 'research paper' is sent as 'publication' and 'linkedin profile' as 'people', the replacements Exa documents; 'pdf', 'github' and 'tweet' are being retired by Exa. | "company" \| "people" \| "news" \| "publication" \| "personal site" \| "financial report" \| "research paper" \| "linkedin profile" \| "pdf" \| "github" \| "tweet" | No |
| number_of_results | Number of results to return (1 to 100). Each result above 10 is billed extra. | int | No |
| include_domains | Only return results from these domains or paths, e.g. 'arxiv.org', 'anthropic.com/news' or '*.substack.com' | List[str] | No |
| exclude_domains | Never return results from these domains or paths | List[str] | No |
| start_published_date | Only return pages published after this date | str (date-time) | No |
| end_published_date | Only return pages published before this date | str (date-time) | No |
| contents | What to return for each result: text, highlights, summary, freshness. Left empty, each result gets up to 10,000 characters of page text. | ContentSettings | No |
| output_schema | Ask Exa to synthesize an answer from the results. Use {"type": "text", "description": "..."} for prose or a JSON schema with "type": "object" (at most 10 properties in total and 2 levels of nesting) for structured data. The result arrives in Output Content, with sources in Output Grounding. Adds ~2 s; not supported by 'lite'; 'deep' types work best. | Dict[str, Any] | No |
| system_prompt | Instructions for how Exa researches and writes the synthesized output, e.g. source preferences or what to exclude. Not supported by 'lite'. | str | No |
| additional_queries | Up to 10 extra query variations for deep search types only ('deep-lite', 'deep', 'deep-reasoning'). Add ones that cover a genuinely different angle, not rephrasings. | List[str] | No |
| objective | The broader goal this search serves: which documents should rank first, which to exclude, and which facts or figures to pull. Most useful when an AI writes the query as one step of a larger task. Up to 4096 characters. | str | No |
| user_location | The end user's two-letter ISO country code (e.g. 'US') to localize results | str | No |
| user_identifier | Opaque ID of the end user this search is for (1-64 letters, digits, '_' or '-', e.g. a UUID or hash). Lets Exa attribute usage per user. Never put an email or name here. | str | No |
| moderation | Filter unsafe content out of the results | bool | No |
| include_text | Deprecated by Exa: put the terms in the query instead. Strings that must appear in the page text (approximate, word-level matching). Not supported by 'lite'. | List[str] | No |
| exclude_text | Deprecated by Exa: put the terms in the query instead. Strings that must not appear in the page text. Not supported by 'lite'. | List[str] | No |
| start_crawl_date | Deprecated by Exa and ignored by the API, so this block no longer sends it. Use Start Published Date instead. | str (date-time) | No |
| end_crawl_date | Deprecated by Exa and ignored by the API, so this block no longer sends it. Use End Published Date instead. | str (date-time) | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of search results | List[ExaSearchResults] |
| result | Single search result | ExaSearchResults |
| output_content | The synthesized answer when Output Schema is set: text, or an object matching the schema | str \| Dict[str, Any] |
| output_grounding | Sources and confidence for each field of Output Content | List[OutputGrounding] |
| context | Deprecated by Exa: the results as one string for LLMs, returned only when Contents > Context is enabled. | str |
| search_type | Legacy: no longer returned by Exa. | str |
| resolved_search_type | Deprecated by Exa: the search type Exa used; may be empty. | str |
| notices | Notes on how the results were produced (e.g. a 'site:' filter in the query was relaxed). Pass these to an LLM with the results. | List[SearchNotice] |
| warnings | Notes about the request itself, such as a field with no effect. Meant for whoever builds the workflow; do not pass to an LLM. | List[SearchWarning] |
| request_id | Exa's ID for this request | str |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
**Competitive Research**: Run a 'deep' search with an Output Schema to get a cited, structured comparison of companies in a market.

**Content Curation**: Find recent news or publications on a topic with the category and published-date filters for a newsletter.

**Fast Lookups**: Use the 'lite' type with highlights to ground an LLM step with relevant passages at the lowest cost.
<!-- END MANUAL -->

---
