# Parallel Yes No
<!-- MANUAL: file_description -->
Answers typed questions about anything on the web using Parallel's Decisions API, with the web sources it used. Needs a Parallel API key from platform.parallel.ai (`PARALLEL_API_KEY`). Parallel charges $0.001 per request, however many questions it carries.
<!-- END MANUAL -->

## Parallel Yes No Decision

### What it is
Answer a yes/no question using current web evidence from Parallel, and forward data to the yes, no or unsure output.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the context and its question to `POST https://api.parallel.ai/v1/decisions`. Parallel searches the web for current evidence about the context, answers from that evidence, and returns the answer with a list of citations (URL, title and the passages used). Citations cover the whole request; they are not tied to one answer. Parallel can decline to answer a question (a refusal); that is a normal result, not an error, and the block reports it separately rather than treating it as no, the first option or zero. HTTP failures, such as a bad API key, raise a block error. Cited passages are untrusted web text; treat them as data, not instructions. The answer is the probability that the statement is true. If it is at least `min_probability` the data goes to `yes`; if the probability of false is at least `min_probability` it goes to `no`; otherwise, or on a refusal, it goes to `unsure`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| context | Text or JSON describing what the question is about, e.g. {"company": "Clay", "website": "https://www.clay.com"}. Parallel searches the web for current evidence about it. | Context | No |
| include_excerpts | Return the evidence passages with each citation. Turn off to keep outputs small; the decision still uses the passages. | bool | No |
| question | A statement or question that is either true or false, e.g. 'Does Clay offer an official command-line interface?' | str | Yes |
| data | Value forwarded unchanged to the yes, no or unsure output. | Data | No |
| min_probability | How sure the answer must be before data goes to yes or no. Anything less certain goes to unsure. 0.5 never gives unsure. | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| citations | Web sources used as evidence for the whole request. They are not mapped to individual answers. | List[ParallelCitation] |
| decision_id | Parallel's ID for this request | str |
| probability | Estimated probability, from 0 to 1, that the answer is yes. | float |
| yes | Data when the answer is yes. | Yes |
| no | Data when the answer is no. | No |
| unsure | Data when the answer is not certain enough, or Parallel declined to answer. | Unsure |
| refused | True when Parallel declined to answer. Emitted only then. | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Check a claim before acting**: Before an agent emails a prospect about a funding round, check that the round really happened and route only confirmed leads onward.

**Eligibility screening**: Ask whether a company currently meets a programme's published rules and send unsure cases to a human with the citations attached.

**List cleaning**: Run each record through 'Is this company still operating?' and drop the ones that come back no.
<!-- END MANUAL -->

---
