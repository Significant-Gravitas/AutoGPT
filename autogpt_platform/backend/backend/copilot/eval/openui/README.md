# OpenUI task journeys

`journeys.jsonl` contains 24 realistic user journeys with 39 turns, frozen UI expectations, and outcome criteria. It adds constraint-changing follow-ups and natural negative controls to the earlier component-oriented collection.

There are 16 development cases and eight final-check cases; 12 positive, nine negative, and three boundary cases overall. The San Francisco journey tests a Wednesday museum closure and a later wheelchair constraint. Other cases cover apartment costs, workshop timing, dietary catering, pantry inventory, incomplete sales periods, geography, deadlines, moving progress, and simple requests that should stay in text.

Run two fresh authenticated Copilot sessions per case and version, retaining the session across a journey's turns. Judge every frozen criterion against visible replies, generated views, and tool evidence. A complete journey passes only if every turn satisfies its task criteria, makes the expected OpenUI choice, completes, and passes the canonical frontend parser/schema. Built-in clarification questions remain appropriate for missing information; the negative UI expectation concerns unnecessary OpenUI views.

Keep task correctness separate from UI selection and parser validity. Preserve failed samples and all attempted candidates. Select the candidate using development results before reviewing final-check responses. These authored cases and implementing-agent judgments are exploratory, not an independent production benchmark.

The October 9 Desktop collection `OpenUI-hill-climb-2026-10-09` includes the live runner, repeated outputs, explicit reviews, all candidate snapshots, comparison scripts, browser evidence, and a portable HTML report. Its `evaluation-design.md` and `protocol-notes.md` document the method and deviations. No account files or tokens belong in the repository or collection.

Recorded sources can be checked from `autogpt_platform/frontend` using `scripts/evaluate-openui.ts`; the Desktop collection's `prepare_parser_inputs.py` adapts journey turns to that existing scorer. The frozen fixture is useful for regression work, but does not itself execute a paid model call during the unit suite.

`validation-journeys.jsonl` adds 16 fresh scenarios (eight positive, six negative, two boundary) and reuses the known SF regression. Its 28 turns cover broad equipment comparisons, revised schedules, dietary overlap, travel buffers, changed commute assumptions, stock allocation, supplied map locations, work-hour arithmetic, missing data, invalid locations, explicit plain-text preferences, and instructions quoted as data. Run both versions twice; report the 16 new cases separately from the reused regression. The separate Desktop collection `OpenUI-validation-2026-10-09` preserves the frozen protocol, raw outputs, judgments, technical repair probes and code checks. Technical probes deliberately submit invalid source and are not ordinary-user success samples.

`connected-journeys.jsonl` adds sixteen October 10 cases for mobile A/B choices, editable costs, compound calculations, richer input types, and appropriate omission of UI. Eight positive, six negative and two boundary cases include supplied SF guided-package choices, catering arithmetic, optional/multiple preferences, unknown venues, missing prices and preserving three alternatives. Four cases are held out. The Desktop collection `OpenUI-connected-2026-10-10` preserves runs, startup failures, component regressions, browser evidence and review.

`reliability-journeys.jsonl` expands the October 10 regression work to 32 scenarios and 50 turns per pass: 20 development cases and 12 held out. Known bike-warranty, SF access, changed-commute and scheduling failures are explicitly development cases. Additional variants exercise shared headcount across charges, refundable deposits, capacity versus accessibility, unknown prices, retained map/checklist progress, multiple alternatives, quoted instructions and plain-text requests. These extend existing task families; repeated runs and newly worded variants are not new independent use cases.

The frozen design runs the baseline once, each development candidate twice and the selected candidate's held-out cases twice. Review all published views and prose against every criterion, test both A/B values and change shared numeric inputs by one. Default totals alone cannot establish that a form is connected correctly. Keep completion, UI choice, canonical validity and task correctness separate; retain failed attempts and collection errors. The Desktop collection `OpenUI-reliability-2026-10-10` contains the protocol, case lineage, raw sessions, explicit judgments, candidate fingerprints, formula checks, browser evidence and final report.


`correctness-journeys.jsonl` retains all 32 prior reliability scenarios as development regressions and adds twelve final-check scenarios, for **44 scenarios and 65 turns**. `prior_split` and `origin` preserve lineage; prior held-out cases are now development cases, not fresh evidence. New cases cover unknown recurring costs, fractional shared quantities, interrupted work, opening windows, deadlines, unsupported warranty claims, missing data and simple text-only requests.

The frozen design uses two fresh sessions per case and candidate. Select using the 32 development cases, then run the twelve final-check cases after freezing the selected candidate. Preserve every accepted turn, including failures; reconnect an accepted session instead of resubmitting it. Review component choice, completion, canonical validity, correct calculations and all task criteria separately. The implementing agent authored and judges these cases; the final check is held apart from tuning, not independent or blinded. The Desktop collection `OpenUI-correctness-2026-10-10` retains frozen criteria, source fingerprints, raw visible outputs, explicit reviews, scripts and browser evidence. Previous-round baseline results are historical, not a simultaneous matched control.

Recorded correctness-round results: three repeated development candidates scored
58/64, 57/64 and 58/64 strict journeys. The selected generation policy plus a
deterministic schedule-summary fix passed 20/24 reserved journeys, with 26/30
appropriate UI choices and 23/23 canonically valid views. The four strict failures
were unnecessary UI in two repeated feasibility scenarios; arithmetic was correct.
All 14 positive final episodes passed. A separate targeted museum regression passed
one of two full journeys despite both showing the corrected capacity distinction;
the other generated an inaccurate rounded alternative. The complete collection has
218 recorded episodes and 334 user turns. These are exploratory results, with
explicit remaining model failures and no demonstrated overall development gain.

`capability-journeys.jsonl` adds 36 scenarios and 44 turns for the expanded inline
catalog: 24 development cases and 12 reserved checks. It refines the previous
capability audit's proposed workflows, rather than claiming every variant as a
new independent use case. Journeys cover a dated SF itinerary and calendar,
multi-option shortlists, contact repair, numeric ledger sorting, 65-record
selection, connected dashboards, missing chart values, conditional registration,
typed event briefs, prose editing, workspace image outputs and execution status.
Negative cases cover missing locations/data/run IDs, ambiguous local times,
misleading charts, infeasible plans, credential collection and false claims that
selecting something executes an action.

The Desktop collection `OpenUI-capabilities-2026-10-11` freezes criteria and splits
before generation, retains two sessions per scenario/candidate, and separates
model output quality from canonical validation and actual UI interactions. The
implementing agent authors and reviews the criteria; results are not independent
or blinded. Require stable view IDs from the first response and compare identities
across revisions. A source can parse successfully yet fail durable editing.
`frontend/scripts/inspect-openui-journeys.ts` validates recorded sources and
exports their parsed component properties for review; it is not a semantic judge.
