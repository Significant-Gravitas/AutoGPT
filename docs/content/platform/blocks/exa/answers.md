# Exa Answer

### What it is
Get an LLM answer to a question informed by Exa search results.

### What it does
Get an LLM answer to a question informed by Exa search results

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The question or query to answer | str | Yes |
| text | Include full text content in the search results used for the answer | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| answer | The generated answer based on search results | str |
| citations | Search results used to generate the answer | List[AnswerCitation] |
| citation | Individual citation from the answer | AnswerCitation |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
