# Graphiti Luna extraction prompt evaluation

Evaluated October 19, 2026, following #15656 and #15622. The scoped prompt change
improves durable-memory answer coverage over the Luna configuration on `dev`.
It does **not** establish that Luna never loses an answer compared with mini.

## Problem and change

Graphiti 0.30.2's entity system prompt forbids abstract concepts, while our custom
extraction instructions require specific Concept, Preference, and Rule entities.
Appending more instructions leaves the higher-priority prohibition in place.
The original Luna extraction also drops concrete details embedded in requests,
leaving recent episode text as their only retrievable copy.

The adapter replaces the system policy only for Luna's known entity and edge
extraction contracts. It permits subject-qualified anchors, retains concrete
values and qualifications, and distinguishes requested actions from completed
actions. Source messages, schemas, provenance, language instructions, retries,
model selection, and request parameters remain intact. Attribute extraction,
deduplication, other models, and the reranker retain their existing behavior.

This follows OpenAI's advice to [audit conflicting instructions and evaluate
prompts on the chosen model](https://developers.openai.com/api/docs/guides/latest-model#prompting-best-practices).
The existing non-reasoning request settings follow the [migration guidance](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart).
The application keeps [source data separated from instructions](https://developers.openai.com/api/docs/guides/prompt-engineering#message-formatting-with-markdown-and-xml),
without parsing or rewriting user-controlled delimiters.

## Method

- Base application: `dace3cbfba00fd247a030a03ce8596163d2f3311`.
- Dataset: [AgentProbe multi-session memory scenarios](https://github.com/Significant-Gravitas/agentprobe/blob/185e443aeecd0be3d5abdea201193513d8db62f8/data/multi-session-memory.yaml),
  31 scenarios, 60 ingested user messages, 35 answer probes, 141 atomic checks.
- Each configuration used fresh, isolated FalkorDB 4.18.9 graphs, actual
  production ingestion and warm-context retrieval, and live model calls.
- Mini used the pinned upstream OpenAI client with `gpt-4.1-mini` for both model
  sizes and the old reranker from `d772be3095b7888c65e12bcca6ed6c4b36acc4ee`.
  Other application code stayed on the same base; this is a model/prompt
  comparison, not a replay of the entire historical application.
- Original Luna used the current client and reranker with the new policy
  disabled. Migrated Luna used this policy. Both used `gpt-6-luna`, no reasoning,
  and unchanged extraction temperature/token settings. OpenRouter routed the
  model requests; embeddings used OpenAI `text-embedding-3-small`.
- A fixed Luna answerer used temperature zero and the same questions and answer
  instructions for all configurations. Expected answers were never sent to it.
  Objective per-field checks and manual inspection of misses supplied the scores.
- Normal memory included facts and up to five recent episodes. Fact-only memory
  suppressed recent episodes to expose missing durable facts. Memory-off supplied
  neither. These are three different tests, not interchangeable quality scores.
- Session gaps were replayed cumulatively from April 6, 2026. The reader was
  told the replay date. User turns were ingested directly; tool execution, the
  full chat loop, Dream, and production load were not part of this run.
- The final candidate was run in two disjoint concurrent scenario subsets.
  This is one sample per case, with correlated checks. It is not a statistical
  non-inferiority test, and prompt development used some of these cases.

## Results

The raw scorer counts all 141 checks, including known scorer/scope problems:

| Configuration | Normal memory | Facts only | Memory off |
| --- | ---: | ---: | ---: |
| Mini | 136/141 | 105/141 | 14/141 |
| Luna, original prompts | 133/141 | 81/141 | 14/141 |
| Luna, migrated prompts | 134/141 | 126/141 | 15/141 |

Adjudication excludes the two forgetting checks affected by a shared retrieval
defect reproduced in these runs, and two checks whose questions did not ask for those
fields (Notion subscription tier and the introspection company's name). The
company regex also accepted a substring of an email address. These exclusions
apply identically to every configuration; they leave 137 checks in 34 probes.

| Configuration | Normal memory | Facts only | Memory off | Entire normal probes | Entire fact-only probes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mini | 132/137 | 104/137 | 12/137 | 31/34 | 19/34 |
| Luna, original prompts | 129/137 | 79/137 | 12/137 | 30/34 | 16/34 |
| Luna, migrated prompts | 131/137 | 125/137 | 13/137 | 31/34 | 28/34 |

No normal or fact-only check that passed with original Luna failed with migrated
Luna in this comparison. Improvements include incidental details, taught
procedures, project totals, business identifiers, and multi-session recall.

Ten additional synthetic controls covered JSON and text input, subject
separation, units, one-off exceptions, unknown preferences, corrections,
conditional rules, and uncompleted requests. Mini scored 61/61 normal and 58/61
fact-only; migrated Luna scored 61/61 in both. Both scored 6/61 without memory.
Their raw scores were respectively 58/61 and 56/61 for mini, and 59/61 and 59/61
for Luna. Adjudication accepted equivalent CC/unknown-time/euro wording and
rejected "unknown whether placed" as evidence that an order was unplaced.
These controls include development cases and should not be treated as a fully
untouched holdout set.

[Per-probe scores and failed check names](graphiti-luna-prompt-evaluation.csv)
retain both raw and adjudicated results, including the excluded forgetting probe.

## Remaining differences and limitations

Two checks passed with mini but failed with the candidate:

1. **Pricing, normal memory:** mini returned the $24.99 wholesale rate but
   abstained on the discount and final price; Luna returned `UNKNOWN`. Both failed
   the complete probe. Luna's retrieved fact and recent episode contain $24.99,
   the 50-unit threshold, and the 10% first-order discount, but the fact bundles
   the general policy into an UrbanFit quote. Applying it to NorthStar remains
   unreliable.
2. **Shipping, facts only:** both retrieved the Reno warehouse and Tuesday/
   Thursday shipping rules. Mini correctly answered Friday–Monday for 3–4
   business-day transit after Tuesday dispatch; Luna answered Thursday–Friday.
   This is a reader calculation failure with the necessary memory present.

The fixed reader's instruction to use only supplied memory can cause excessive
abstention when an arithmetic input is in the question. A separate sensitivity
run allowing question-provided inputs improved some answers but remained
inconsistent. It does not replace the primary scores or prove the gaps fixed.
The interrupted-task continuation probe still fails in both configurations.
The shared fact-only forgetting failure remains in all three configurations.
It is distinct from the correction and briefing-recall examples tracked in #15655.

The opt-in supplier regression initially failed the message case on its minimum
order check. The old assertion did not retain the raw facts, so omission versus
equivalent number wording cannot be established. After accepting both `18` and
`eighteen` and printing synthetic facts on failure, a message repeat and the
complete three-source run passed. This remains a small stochastic sample.

A later wording experiment and two new scope controls hit the provider key's
daily limit (HTTP 402). That experiment was reverted; none of those incomplete
results are counted here. The shipped policy matches the fully evaluated
candidate hash below. The evidence supports improvement over current Luna,
but further representative repeats/full-chat evaluation are needed for a
blanket no-regression claim against mini.

## Cost, latency, and request validity

| Configuration | Ingestion median / max | Warm context median / max | Recorded model cost |
| --- | ---: | ---: | ---: |
| Mini | 7.00 / 13.73 s | 1.11 / 2.64 s | $0.26446 |
| Luna, original prompts | 8.63 / 15.50 s | 1.32 / 3.94 s | $0.10326 |
| Luna, migrated prompts | 10.89 / 17.94 s | 1.56 / 3.82 s | $0.11417 |

Costs include the common answerer and exclude direct embedding charges.
These concurrent local runs are not production percentiles. The prompt change
increases ingestion work; observed warm-context calls stayed inside the existing
eight-second budget. The largest natural ranking batch had 15 candidates.

A separate fixed 40-candidate ranking test used four relevant facts and 36
distractors. Luna's existing reranker put all four relevant facts first in all
three runs (3.53, 4.72, 3.10 seconds). Mini's old reranker retrieved two of four
in its top four (2.31, 2.04, 2.49 seconds). All runs returned 40 results within
eight seconds. This component result is not a load test or a reranker change.

The three primary runs completed with no HTTP or scenario errors. Their 347,
471, and 576 structured responses respectively validated against the requested
schemas. This validates output shape, not semantic completeness.

## Regression checks and reproduction

Initial evaluation checks:

- Graphiti non-integration suite: **330 passed, one pre-existing xfail**.
- Focused client/prompt suite: **65 passed**, 100% statement coverage of the
  prompt dispatcher and adapter; three paid tests skipped by default.
- Paid regression: **three passed**, covering message, JSON, and text ingestion
  into a fresh graph. It inspects stored relationships for six concrete details.
- Black, isort, Ruff, and scoped Pyright passed. Full application E2E was not run.

The first review follow-up passed **339 unit tests** with one pre-existing xfail.
New tests invoke graphiti-core's real node/edge extraction entry points and reject
an inactive dispatcher. Four assertion regressions were reproduced before fixing
row-order dependence and cross-fact matches. Detail checks now match individual
facts and require each dispatch day in a dispatch-related fact. CI's complete
Python 3.11/3.12/3.13 reports validate after registering the three intentional
paid-test skips. The paid run above predates the stricter assertions; it was not
repeated after the provider's daily quota was exhausted.

The assertion follow-ups passed **398 unit tests** with one pre-existing xfail.
Thirty false passes were reproduced before adding exact payment-value checks
and scenario-specific checks for invented signing claims and explicit company
misattribution. Payment values must directly follow a payment term, and explicit
counterparty detail claims include the "has" form. Signing instructions cannot
hide a later affirmative claim across "but" or "and"; punctuation-boundary
controls also pass. Two false rejections of negated instructions with coordinated
subjects were also reproduced and fixed. Positive controls allow negated signing
statements, source instructions, and supplier briefs naming the counterparty.
The phrase "for Elm Studio" alone can describe the requested brief's context and is
not treated as an explicit claim about that company's own terms. These pattern
checks cover the demonstrated wording; they are not a general semantic judge. Offline
replay of four saved migrated-Luna runs of the incidental-detail scenario passed
the tightened checks. This replay made no new model calls and does not replace
a fresh three-source paid integration run.

The full pre-commit invocation encountered local Windows/WSL tooling failures:
Bash could not resolve the Windows worktree metadata or find Poetry/Node, and the
isort hook's Poetry option was unsupported locally. Native scoped equivalents
passed. The hook's unscoped typecheck also scanned Classic and reported 2,980
errors, including unresolved Classic imports; it is not a passing backend-wide
typecheck. CI must validate the repository-wide hooks.

From `autogpt_platform/backend`, with an isolated FalkorDB and configured test
model/embedding credentials, set `RUN_GRAPHITI_LLM_EVAL=1`, select
`GRAPHITI_LLM_MODEL=gpt-6-luna`, and run:

```sh
poetry run pytest backend/copilot/graphiti/extraction_quality_integration_test.py -q
```

`GRAPHITI_FALKORDB_HOST`/`PORT`/`PASSWORD` select the test database service. The
fixture creates a unique graph, cleans its data, and closes clients. This command
makes paid calls. It reproduces the durable-detail regression, not the complete
adapted benchmark. The full request captures and local harness are not committed;
the CSV is a score export, not an executable benchmark. Temporary test services
were removed after validation.

## Evidence fingerprints

SHA-256 identifies the evaluated artifacts; source hashes use LF bytes. The
review follow-up changes only comments in the prompt module: its syntax tree and
prompt strings match the evaluated revision.

| Artifact | SHA-256 |
| --- | --- |
| Evaluated `extraction_prompts.py` at `51cdfdd` | `34b04d8c8d12475d77d9c1b90414a78ddf5ac86434c0f2c6d853f9f63475c5cd` |
| AgentProbe source YAML | `faf01e2ec24135acc3fb558bee02ad1a31faf999919b7a854fb77245e563412b` |
| Adapted probe/check definitions | `413edbfbd714941a9e578fb52c31d1593c2886de5b245c27d054cb0ff25f503f` |
| Mini primary capture | `91415816d889790d21d525d5bb6e70f168ad375701c63040293db5fc3f16722a` |
| Original Luna primary capture | `e41139bfef13d900fb6bbd927b4eea395097d962cc5d4c7005c2e19163adb1dc` |
| Migrated Luna primary capture | `fafe5822dba724ad0eb6872d022caa8ecad0fb61c6e3381be1559fba02e2ca02` |
