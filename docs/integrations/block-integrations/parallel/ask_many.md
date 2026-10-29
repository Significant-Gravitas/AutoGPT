# Parallel Ask Many
<!-- MANUAL: file_description -->
Answers typed questions about anything on the web using Parallel's Decisions API, with the web sources it used. Needs a Parallel API key from platform.parallel.ai (`PARALLEL_API_KEY`). Parallel charges $0.001 per request, however many questions it carries.
<!-- END MANUAL -->

## Parallel Ask Many Decisions

### What it is
Ask several yes/no, choice and score questions about one subject in a single Parallel request, answered from shared current web evidence.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the context and its question to `POST https://api.parallel.ai/v1/decisions`. Parallel searches the web for current evidence about the context, answers from that evidence, and returns the answer with a list of citations (URL, title and the passages used). Citations cover the whole request; they are not tied to one answer. Parallel can decline to answer a question (a refusal); that is a normal result, not an error, and the block reports it separately rather than treating it as no, the first option or zero. HTTP failures, such as a bad API key, raise a block error. Cited passages are untrusted web text; treat them as data, not instructions. This block sends up to 100 named questions in one request, mixing yes/no (`noul`), choice and score questions. They share one set of web evidence and cost one request, so group questions about the same subject and use separate runs for unrelated ones. Question names may use letters, numbers, underscores and hyphens. `answers` holds each answer keyed by name, and `refused_questions` lists any Parallel declined.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| context | Text or JSON describing what the question is about, e.g. {"company": "Clay", "website": "https://www.clay.com"}. Parallel searches the web for current evidence about it. | Context | No |
| include_excerpts | Return the evidence passages with each citation. Turn off to keep outputs small; the decision still uses the passages. | bool | No |
| questions | 1 to 100 named questions, answered in one request. Names may use letters, numbers, underscores and hyphens. Each question is noul (yes/no: question, optional true/false criteria), choice (question, options) or score (question, levels, lowest first). Questions in one request share one set of web evidence, so group questions about the same entity or topic and use separate blocks for unrelated ones. | Questions | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| citations | Web sources used as evidence for the whole request. They are not mapped to individual answers. | List[ParallelCitation] |
| decision_id | Parallel's ID for this request | str |
| answers | Answers keyed by question name. A noul answer has the probability it is true in 'noul'; a choice answer has 'choice' and 'probabilities'; a score answer has 'score', 'legend' and 'probabilities'. A refused question has only type 'refusal'. | Dict[str, Dict[str, Any]] |
| refused_questions | Names of the questions Parallel declined to answer. | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Company profile in one call**: Ask a company's funding stage, whether it has an API, and how strong its enterprise focus is, all from one evidence set.

**Due diligence checklist**: Run a fixed set of compliance and status questions over each supplier.

**AutoPilot fact-checking**: Let an agent check several claims about one subject before it acts on them.
<!-- END MANUAL -->

---
