# Dataforseo AI Mentions
<!-- MANUAL: file_description -->
Blocks for seeing where a site or term appears in AI search answers, from DataForSEO AI Optimization.
<!-- END MANUAL -->

## Data For Seo Ai Mentions

### What it is
Find AI search answers (Google AI Overviews or ChatGPT) that cite a website or mention a term, with the question, answer and cited sources, from DataForSEO AI Optimization

### How it works
<!-- MANUAL: how_it_works -->
This block calls the DataForSEO LLM Mentions API, which holds answers collected from Google AI Overviews and ChatGPT. Give a domain to find answers citing that site, a keyword to find answers using that term, or both. Each answer comes with the question asked, the answer as markdown, the pages it cites and DataForSEO's monthly search estimate for the question, most-searched first. ChatGPT data is US English only.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| domain | Find answers that cite this site, without https:// or www. (e.g. 'example.com'). Leave empty to search by keyword only | str | No |
| keyword | Find answers whose question or text uses this term, e.g. a brand or product name. Leave empty to search by domain only | str | No |
| platform | google for AI Overviews, chat_gpt for ChatGPT (US English only) | "google" \| "chat_gpt" | No |
| location_code | Location code for targeting (e.g., 2840 for USA) | int | No |
| language_code | Language code (e.g., 'en' for English) | str | No |
| limit | Maximum answers to return, most-searched questions first | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| mentions | AI answers matching the domain and/or keyword, with the question asked, the answer as markdown and the pages it cites. ai_search_volume is DataForSEO's monthly estimate for the question | List[AiMention] |
| total_count | How many matching answers there are in total | int |

### Possible use case
<!-- MANUAL: use_case -->
**AI visibility**: See which questions AI answers cite your site for, and which they don't.

**Competitor citations**: Find the pages AI answers cite for your category instead of yours.

**Content gaps**: Learn the questions buyers ask AI about your product category.
<!-- END MANUAL -->

---
