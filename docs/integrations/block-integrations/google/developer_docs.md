# Google Developer Docs
<!-- MANUAL: file_description -->
Blocks for searching and reading Google's official developer documentation (Android, Firebase, Google Cloud, Maps, Chrome, Flutter, Go, Google AI and more) through Google's Developer Knowledge API. They need a Google Cloud API key with the Developer Knowledge API enabled, added as a Google Developer Docs credential. Search and Ask return document names that Get Google Developer Docs turns into whole pages.
<!-- END MANUAL -->

## Ask Google Developer Docs

### What it is
Answer a question about Google developer products with an answer Google writes from its official documentation, plus the passages it used. Covers Android, Firebase, Google Cloud, Maps and more. Answers have a small daily quota; use Search Google Developer Docs for many questions.

### How it works
<!-- MANUAL: how_it_works -->
Sends the question to the Developer Knowledge API's `answerQuery` method, which retrieves matching passages from Google's documentation and has a Gemini model write an answer grounded in them. The block returns the answer, the passages it drew on (each with its page title, link and document name) and the cited pages as a list of document names for Get Google Developer Docs. Sites and the optional custom filter limit which documentation the answer can draw on.

Google allows 50 answers per Google Cloud project per day by default. When that quota runs out the block fails straight away with a quota message instead of retrying; Search Google Developer Docs has a much larger quota (100 requests a minute).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| question | The question, e.g. 'How do I create a BigQuery dataset?' | str | Yes |
| sites | Only answer from these documentation sites (e.g. firebase.google.com, developer.android.com or docs.cloud.google.com) | List[str] | No |
| custom_filter | Extra filter in Google's filter syntax, ANDed with the sites, e.g. 'update_time >= "2026-01-01T00:00:00Z"' | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| answer | The answer, written from Google's documentation | str |
| sources | The passages the answer is based on | List[DeveloperDocPassage] |
| source | Each passage the answer is based on | DeveloperDocPassage |
| document_names | The pages the answer cites, for Get Google Developer Docs | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Developer Q&A Bot**: Answer questions about Firebase or Google Cloud in a support channel, with links to the official pages.

**Grounded Coding Help**: Give an AI coding agent a current answer from Google's docs before it writes Android or Cloud code.

**Cited Explanations**: Explain how a Google API behaves today and cite the pages the explanation comes from.
<!-- END MANUAL -->

---

## Get Google Developer Docs

### What it is
Get whole pages of Google's developer documentation as Markdown, up to 20 at a time. Takes document names from Search Google Developer Docs, or links to the pages.

### How it works
<!-- MANUAL: how_it_works -->
Turns each input into a document name (a link such as `https://firebase.google.com/docs/auth?hl=en` becomes `documents/firebase.google.com/docs/auth`), drops duplicates and fetches up to 20 pages in one `documents.batchGet` call. Each page comes back as Markdown with its title, link, description, site and last-update time, in the order you asked for them.

The whole call fails if any page isn't in Google's index, so use names from Search Google Developer Docs or links to pages on the indexed sites. The Markdown is generated from Google's HTML and can contain small formatting glitches.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| document_names | Up to 20 pages: document names from Search or Ask Google Developer Docs (documents/...), or links to the pages | List[str] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| documents | The pages, in the order asked for | List[DeveloperDoc] |
| document | Each page | DeveloperDoc |

### Possible use case
<!-- MANUAL: use_case -->
**Deep Dive After Search**: Pull the full pages behind the top search results so an AI block can read them end to end.

**Guide Summaries**: Fetch a setup guide by its link and have an AI block turn it into a checklist for your team.

**Docs Snapshots**: Save the current Markdown of a set of API reference pages to compare against a later run.
<!-- END MANUAL -->

---

## Search Google Developer Docs

### What it is
Search Google's developer documentation and return the best-matching passages with links to their pages. Covers Android, Firebase, Google Cloud, Maps, Chrome, Flutter, Go, Google AI and more.

### How it works
<!-- MANUAL: how_it_works -->
Calls the Developer Knowledge API's `documents.searchDocumentChunks` method, which searches the sites in Google's corpus reference and returns passages ranked by relevance, each with a score from 0 to 1. Sites become a `data_source` filter, and the custom filter is ANDed with them in Google's filter syntax (for example `update_time >= "2026-01-01T00:00:00Z"`). Queries can be up to 500 characters.

Several passages can come from the same page, so the block also lists each page's document name once, in rank order; pass that list to Get Google Developer Docs for the full pages. Use `next_page_token` to get more results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | What to look for, e.g. 'How do I create a Cloud Storage bucket?' | str | Yes |
| sites | Only search these documentation sites (e.g. firebase.google.com, developer.android.com or docs.cloud.google.com) | List[str] | No |
| custom_filter | Extra filter in Google's filter syntax, ANDed with the sites, e.g. 'update_time >= "2026-01-01T00:00:00Z"' | str | No |
| max_results | Maximum number of passages to return | int | No |
| page_token | Page token from a previous search, to get the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | Matching passages, most relevant first | List[DeveloperDocPassage] |
| result | Each matching passage | DeveloperDocPassage |
| document_names | The pages the passages come from, for Get Google Developer Docs | List[str] |
| next_page_token | Token for the next page, when there are more results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Coding Agent Grounding**: Find current Android or Firebase guidance before an agent generates code.

**Support Answers**: Match a developer's question to the relevant Google Cloud documentation passages and links.

**Docs Monitoring**: Search one documentation site for recently changed pages about an API you depend on.
<!-- END MANUAL -->

---
