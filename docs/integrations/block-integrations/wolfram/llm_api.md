# Wolfram LLM API
<!-- MANUAL: file_description -->
Blocks for querying Wolfram Alpha's computational knowledge engine.
<!-- END MANUAL -->

## Ask Wolfram

### What it is
Ask Wolfram Alpha a question

### How it works
<!-- MANUAL: how_it_works -->
This block queries Wolfram Alpha's LLM API with a natural language question and returns a computed answer. It leverages Wolfram's computational knowledge engine for math, science, geography, and factual queries.

The answer is returned as a string suitable for display or further processing in your workflow.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| question | The question to ask | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| answer | The answer to the question | str |

### Possible use case
<!-- MANUAL: use_case -->
**Factual Queries**: Get accurate answers to math, science, or geography questions.

**Calculations**: Perform complex mathematical or unit conversion calculations.

**Data Lookups**: Retrieve factual data like population statistics, chemical properties, or historical dates.
<!-- END MANUAL -->

---
