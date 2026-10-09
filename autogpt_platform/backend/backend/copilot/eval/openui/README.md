# OpenUI task journeys

`journeys.jsonl` contains 24 realistic user journeys with 39 turns, frozen UI expectations, and outcome criteria. It adds constraint-changing follow-ups and natural negative controls to the earlier component-oriented collection.

There are 16 development cases and eight final-check cases; 12 positive, nine negative, and three boundary cases overall. The San Francisco journey tests a Wednesday museum closure and a later wheelchair constraint. Other cases cover apartment costs, workshop timing, dietary catering, pantry inventory, incomplete sales periods, geography, deadlines, moving progress, and simple requests that should stay in text.

Run two fresh authenticated Copilot sessions per case and version, retaining the session across a journey's turns. Judge every frozen criterion against visible replies, generated views, and tool evidence. A complete journey passes only if every turn satisfies its task criteria, makes the expected OpenUI choice, completes, and passes the canonical frontend parser/schema. Built-in clarification questions remain appropriate for missing information; the negative UI expectation concerns unnecessary OpenUI views.

Keep task correctness separate from UI selection and parser validity. Preserve failed samples and all attempted candidates. Select the candidate using development results before reviewing final-check responses. These authored cases and implementing-agent judgments are exploratory, not an independent production benchmark.

The October 9 Desktop collection `OpenUI-hill-climb-2026-10-09` includes the live runner, repeated outputs, explicit reviews, all candidate snapshots, comparison scripts, browser evidence, and a portable HTML report. Its `evaluation-design.md` and `protocol-notes.md` document the method and deviations. No account files or tokens belong in the repository or collection.

Recorded sources can be checked from `autogpt_platform/frontend` using `scripts/evaluate-openui.ts`; the Desktop collection's `prepare_parser_inputs.py` adapts journey turns to that existing scorer. The frozen fixture is useful for regression work, but does not itself execute a paid model call during the unit suite.

`validation-journeys.jsonl` adds 16 fresh scenarios (eight positive, six negative, two boundary) and reuses the known SF regression. Its 28 turns cover broad equipment comparisons, revised schedules, dietary overlap, travel buffers, changed commute assumptions, stock allocation, supplied map locations, work-hour arithmetic, missing data, invalid locations, explicit plain-text preferences, and instructions quoted as data. Run both versions twice; report the 16 new cases separately from the reused regression. The separate Desktop collection `OpenUI-validation-2026-10-09` preserves the frozen protocol, raw outputs, judgments, technical repair probes and code checks. Technical probes deliberately submit invalid source and are not ordinary-user success samples.
