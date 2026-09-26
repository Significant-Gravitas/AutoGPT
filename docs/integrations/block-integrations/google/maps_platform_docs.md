# Google Maps Platform Docs
<!-- MANUAL: file_description -->
Blocks for Google Maps Platform documentation, sample code and coding guidance, backed by Google's Maps Code Assist MCP server. Maps Code Assist is experimental: it needs no credentials and costs nothing for now, but Google may change it. Its terms require showing results with their source links, and using them only with AI models that don't train on your inputs.
<!-- END MANUAL -->

## Get Google Maps Platform Coding Instructions

### What it is
Get Google's system prompt for AI assistants that write Google Maps Platform code. It covers how to plan answers and ground them with Search Google Maps Platform Docs, which terms apply (including the EEA terms) and how to cite sources.

### How it works
<!-- MANUAL: how_it_works -->
Calls the `retrieve-instructions` tool of the Maps Code Assist MCP server with one JSON-RPC `tools/call` request and joins the instruction sections it returns. The text is Google's rulebook for AI coding assistants: how to plan an answer, when to search the Maps Platform docs, how to handle the European Economic Area terms and how to cite links.

Google asks AI clients to load these instructions before searching its Maps Platform docs. The text is long (about 29,000 characters), so load it once and use it as an AI block's system prompt rather than adding it to every prompt.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| instructions | Google's instructions for an AI assistant that writes Google Maps Platform code, ready to use as a system prompt | str |

### Possible use case
<!-- MANUAL: use_case -->
**Maps Coding Assistant**: Use the instructions as the system prompt of an AI block that answers Maps Platform coding questions.

**Terms-Aware Answers**: Give an agent Google's terms and EEA guidance before it recommends Maps Platform APIs.

**Consistent Citations**: Have an AI block follow Google's rules for linking to the Maps Platform pages it quotes.
<!-- END MANUAL -->

---

## Search Google Maps Platform Docs

### What it is
Search Google Maps Platform documentation and code samples and return the best-matching passages with their source links. Covers the Maps, Routes and Places APIs and SDKs, architecture guides and Google's official GitHub samples.

### How it works
<!-- MANUAL: how_it_works -->
Calls the `retrieve-google-maps-platform-docs` tool of the Maps Code Assist MCP server with one JSON-RPC `tools/call` request. Google publishes no REST API for it, and it needs no credentials. The server searches Maps Platform documentation, code samples, the architecture center, the trust center, the terms of service and Google's official GitHub repositories, and returns a handful of passages ranked by relevance. The optional product filter narrows the search, for example to `Places API`.

Each passage comes with its source link and the API's state as Google labels it (for example NEW, CURRENT or LEGACY). Documentation passages arrive with their line breaks escaped, so the block restores them to keep code samples usable. Show the source links with anything built from the results, as Google's terms require.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | What you want to know, e.g. 'How do I add a marker with the Maps JavaScript API?' | str | Yes |
| product_filter | Narrow the search to an API or product area, e.g. 'Places API' | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | Matching passages, most relevant first | List[MapsPlatformDocPassage] |
| result | Each matching passage | MapsPlatformDocPassage |

### Possible use case
<!-- MANUAL: use_case -->
**Maps Coding Assistant**: Ground an AI block's Maps JavaScript or Places code in current documentation and official samples.

**Migration Checks**: Find out whether an API you use is labelled LEGACY and what Google recommends instead.

**Architecture Research**: Pull Google's architecture guidance for a store locator or route planner before designing one.
<!-- END MANUAL -->

---
