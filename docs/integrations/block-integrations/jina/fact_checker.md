# Jina Fact Checker
<!-- MANUAL: file_description -->
Blocks for verifying statement factuality using Jina AI's Grounding API.
<!-- END MANUAL -->

## Fact Checker

### What it is
This block checks the factuality of a given statement using Jina AI's Grounding API.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Jina AI's Grounding API to verify the factuality of statements. It analyzes the statement against reliable sources and returns a factuality score, result, reasoning, and supporting references.

The API searches for evidence and determines whether the statement is supported, contradicted, or uncertain based on available information.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| statement | The statement to check for factuality | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| factuality | The factuality score of the statement | float |
| result | The result of the factuality check | bool |
| reason | The reason for the factuality result | str |
| references | List of references supporting or contradicting the statement | List[Reference] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Verification**: Verify claims in articles or social media posts before publishing.

**AI Output Validation**: Check factuality of AI-generated content to ensure accuracy.

**Research Support**: Validate statements in research or journalism with supporting references.
<!-- END MANUAL -->

---
