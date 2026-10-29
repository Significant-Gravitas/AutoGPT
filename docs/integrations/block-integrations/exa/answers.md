# Exa Answers
<!-- MANUAL: file_description -->
Blocks for getting AI-generated answers to questions using Exa's search-informed answer API.
<!-- END MANUAL -->

## Exa Answer

### What it is
Get an LLM answer to a question informed by Exa search results, with citations; optionally as structured JSON

### How it works
<!-- MANUAL: how_it_works -->
This block sends your question to Exa's `/answer` endpoint in one non-streaming request: Exa searches the web and an LLM writes an answer from the results, returned with the citations it used (with their full text when Text is on). Model picks the Exa model that writes the answer: 'exa' (default, Exa's own default so it is not sent), 'exa-pro', 'exa-fast' or 'exa-research'; Exa does not document how they differ. System Prompt adds instructions such as tone, length or source preferences.

Set Output Schema to a JSON Schema to get a structured answer: the object arrives in Answer Structured and as JSON text in Answer. User Location must be a two-letter country code and User Identifier 1-64 letters, digits, '_' or '-', or the block fails before calling Exa. The run is billed from the dollar cost Exa reports, at 100 credits per US dollar.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The question or query to answer | str | Yes |
| text | Include full text content in the search results used for the answer | bool | No |
| model | The Exa model that writes the answer: 'exa' (default), 'exa-pro', 'exa-fast' or 'exa-research'. Exa does not document how they differ; all are billed at the /answer rate. | "exa" \| "exa-pro" \| "exa-fast" \| "exa-research" | No |
| output_schema | A JSON Schema (draft 7) for a structured answer, e.g. {"type": "object", "properties": {"valuation": {"type": "string"}}}. When set, the answer arrives as an object in Answer Structured (and as JSON text in Answer). | Dict[str, Any] | No |
| system_prompt | Extra instructions for the answer, e.g. source preferences, tone or length | str | No |
| user_location | The end user's two-letter ISO country code (e.g. 'US') to localize the search | str | No |
| user_identifier | Opaque ID of the end user this request is for (1-64 letters, digits, '_' or '-', e.g. a UUID or hash). Never an email or name. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| answer | The generated answer. With an Output Schema, the structured answer as JSON text. | str |
| answer_structured | The answer as an object, when Exa returns one (normally only when Output Schema is set) | Dict[str, Any] |
| citations | Search results used to generate the answer | List[AnswerCitation] |
| citation | Individual citation from the answer | AnswerCitation |
| request_id | Exa's ID for this request | str |
| cost_dollars | Cost breakdown for the request | Cost |

### Possible use case
<!-- MANUAL: use_case -->
**Research Assistance**: Get quick, sourced answers to complex questions without manually searching multiple websites.

**Fact Verification**: Verify claims or statements by getting answers backed by real web sources with citations.

**Structured Data Extraction**: Ask for a company's latest funding round with an Output Schema and pass the resulting fields straight to the next block.
<!-- END MANUAL -->

---
