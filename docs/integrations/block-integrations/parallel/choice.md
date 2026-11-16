# Parallel Choice
<!-- MANUAL: file_description -->
Answers typed questions about anything on the web using Parallel's Decisions API, with the web sources it used. Needs a Parallel API key from platform.parallel.ai (`PARALLEL_API_KEY`). Parallel charges $0.001 per request, however many questions it carries.
<!-- END MANUAL -->

## Parallel Choice Decision

### What it is
Pick one of several options using current web evidence from Parallel, with a probability for each option.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the context and its question to `POST https://api.parallel.ai/v1/decisions`. Parallel searches the web for current evidence about the context, answers from that evidence, and returns the answer with a list of citations (URL, title and the passages used). Citations cover the whole request; they are not tied to one answer. Parallel can decline to answer a question (a refusal); that is a normal result, not an error, and the block reports it separately rather than treating it as no, the first option or zero. HTTP failures, such as a bad API key, raise a block error. Cited passages are untrusted web text; treat them as data, not instructions. The block returns the chosen option name and a probability for every option. Give 2 to 255 options, each with a description of when it applies; add an `unknown` option if the evidence may not settle the question.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| context | Text or JSON describing what the question is about, e.g. {"company": "Clay", "website": "https://www.clay.com"}. Parallel searches the web for current evidence about it. | Context | No |
| include_excerpts | Return the evidence passages with each citation. Turn off to keep outputs small; the decision still uses the passages. | bool | No |
| question | What to decide, e.g. 'What is the availability of Clay's official CLI?' | str | Yes |
| options | 2 to 255 option names, each mapped to a description of when it applies. Add an 'unknown' option if the evidence may not settle it. | Dict[str, str] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| citations | Web sources used as evidence for the whole request. They are not mapped to individual answers. | List[ParallelCitation] |
| decision_id | Parallel's ID for this request | str |
| choice | Name of the chosen option. | str |
| probabilities | Estimated probability of each option. | Dict[str, float] |
| refused | True when Parallel declined to answer. Emitted only then. | bool |

### Possible use case
<!-- MANUAL: use_case -->
**CRM enrichment**: Classify each company's employee band, funding stage or product category from current web evidence.

**Routing**: Pick which team or workflow a lead belongs to based on what the company does today.

**Status tracking**: Decide whether a product is announced, in beta, generally available or discontinued.
<!-- END MANUAL -->

---
