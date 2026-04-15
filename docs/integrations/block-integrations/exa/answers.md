# Exa Answers
<!-- MANUAL: file_description -->
Blocks for getting AI-generated answers to questions using Exa's search-informed answer API.
<!-- END MANUAL -->

## Exa Answer

### What it is
Get an LLM answer to a question informed by Exa search results

### How it works
<!-- MANUAL: how_it_works -->
This block sends your question to the Exa Answer API, which performs a semantic search across billions of web pages to find relevant information. The API then uses an LLM to synthesize the search results into a coherent answer with citations.

The block returns both the generated answer and the source citations that informed it. You can optionally include full text content from the search results for more comprehensive answers.
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
**Research Assistance**: Get quick, sourced answers to complex questions without manually searching multiple websites.

**Fact Verification**: Verify claims or statements by getting answers backed by real web sources with citations.

**Content Creation**: Generate research-backed content by asking questions about topics and using the cited sources.
<!-- END MANUAL -->

---
