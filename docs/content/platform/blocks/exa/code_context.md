# Exa Code Context

### What it is
Search billions of GitHub repos, docs, and Stack Overflow for relevant code examples.

### What it does
Search billions of GitHub repos, docs, and Stack Overflow for relevant code examples

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query to find relevant code snippets. Describe what you're trying to do or what code you're looking for. | str | Yes |
| tokens_num | Token limit for response. Use 'dynamic' for automatic sizing, 5000 for standard queries, or 10000 for comprehensive examples. | str | int | No |

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
_Add practical use case examples here._
<!-- END MANUAL -->

---
