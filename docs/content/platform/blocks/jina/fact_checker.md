# Fact Checker

### What it is
This block checks the factuality of a given statement using Jina AI's Grounding API.

### What it does
This block checks the factuality of a given statement using Jina AI's Grounding API.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
