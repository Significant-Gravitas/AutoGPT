# Exa Code Context
<!-- MANUAL: file_description -->
Blocks for searching code repositories and documentation using Exa's code context API.
<!-- END MANUAL -->

## Exa Code Context

### What it is
Legacy: Exa no longer documents its code context API; for new workflows use the Exa Search block, which Exa's Code & Docs guide recommends (e.g. type 'lite' with highlights). Searches GitHub repos, docs and Stack Overflow for relevant code examples.

### How it works
<!-- MANUAL: how_it_works -->
This block sends your query to Exa's `/context` endpoint, which Exa no longer documents: it is missing from Exa's API spec and its reference page redirects to the Code & Docs search guide. The block is kept so saved workflows keep running; for new workflows, use the Exa Search block (for example type 'lite' with highlights), as that guide recommends.

The endpoint returns one formatted string of code snippets and examples with their sources from GitHub repositories, documentation and Stack Overflow, plus the result count, search time, output token count and cost. Tokens Num sets the response size: 'dynamic' (default) or a token count such as 5000 or 10000. The run is billed from the dollar cost Exa reports, at 100 credits per US dollar.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query to find relevant code snippets. Describe what you're trying to do or what code you're looking for. | str | Yes |
| tokens_num | Token limit for response. Use 'dynamic' for automatic sizing, 5000 for standard queries, or 10000 for comprehensive examples. (In Exa Search the equivalent is text max characters.) | str \| int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| request_id | Unique identifier for this request | str |
| query | The search query used | str |
| response | Formatted code snippets and contextual examples with sources | str |
| results_count | Number of code sources found and included | int |
| cost_dollars | Cost of this request in dollars | str |
| search_time | Time taken to search in milliseconds | float |
| output_tokens | Number of tokens in the response | int |

### Possible use case
<!-- MANUAL: use_case -->
**API Integration Examples**: Find real-world code examples showing how to integrate with specific APIs or libraries.

**Debugging Assistance**: Search for code patterns related to error messages or specific programming challenges.

**Learning New Technologies**: Discover implementation examples when learning a new framework or programming language.
<!-- END MANUAL -->

---
