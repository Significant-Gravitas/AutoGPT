# Exa Code Context
<!-- MANUAL: file_description -->
Blocks for searching code repositories and documentation using Exa's code context API.
<!-- END MANUAL -->

## Exa Code Context

### What it is
Search billions of GitHub repos, docs, and Stack Overflow for relevant code examples

### How it works
<!-- MANUAL: how_it_works -->
This block uses Exa's specialized code search API to find relevant code examples from GitHub repositories, official documentation, and Stack Overflow. The search is optimized for code context, returning formatted snippets with source references.

The block returns code snippets along with metadata including the source URL, search time, and token counts. You can control response size with the tokens_num parameter to balance comprehensiveness with cost.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query to find relevant code snippets. Describe what you're trying to do or what code you're looking for. | str | Yes |
| tokens_num | Token limit for response. Use 'dynamic' for automatic sizing, 5000 for standard queries, or 10000 for comprehensive examples. | str \| int | No |

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
