---
name: review-feature-plan
description: Review an AutoGPT Platform implementation plan before code is written. Architectural gate for backend, frontend, block, data-model, and cross-cutting plans. Reject plans that are missing dimensions, superficial, or contradicted by code evidence.
user-invocable: true
args: "[plan] — the full plan text, or a path to it. Also provide the original task description."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Review Feature Plan

Review the plan as an architectural gate. Reject the plan if any required dimension is missing, superficial, or contradicted by code evidence. Verify claims against the actual codebase — a plan that cites files is only as good as the files it cites.

## Required Checks

1. **Class vs instance**
   - How many concrete cases does the plan cover? It must name at least three in the class, or justify single-case scope.
   - Reject point fixes when the same bug/gap exists in sibling code paths the plan ignores.

2. **Best fix, not merely plausible**
   - The plan must name alternatives considered and why the chosen approach wins.
   - Reject plans whose justification is "this works" without weighing at least one alternative.

3. **Analogous trace verification**
   - The plan must cite an analogous existing feature with a file-path trace. Spot-check that the cited files exist and actually follow the claimed pattern.
   - Reject plans with no trace, or whose trace misrepresents the cited code.

4. **Building-block reuse**
   - Reject duplicated logic already covered by `backend/util/request.py`, `backend/util/file.py`, `backend/sdk/`, `backend/data/`, generated frontend API hooks, or the design system.
   - Every new helper needs a justification naming what was checked and why it didn't fit.

5. **Layer correctness**
   - Routes thin; business logic in services; data access (and user-ID scoping) only in `backend/data/`.
   - Executor semantics stay in `backend/executor/`. No business rules enforced only in the frontend.
   - Frontend follows `.tsx` / `use*.ts` / `helpers.ts` separation; generated hooks, not `BackendAPI`; design system, not `__legacy__`.

6. **Auth & security dimension**
   - Changes touching `backend/data/*.py` must state the user-ID check or why none is needed.
   - Broadened schemas must show the auth path still holds end to end.
   - User-influenced URLs go through the SSRF-safe `Requests` wrapper. Secrets never logged. Input validated at boundaries.

7. **Data model & migration safety**
   - Prisma changes must address backward compatibility with running deployments and indexes for new query patterns.
   - Redis usage must be cluster-safe: locks and multi-step primitives operate on a single key (`SET NX`, single-key Lua) — no multi-key transactions.

8. **Test plan adequacy**
   - TDD: a failing test is named before implementation (`@pytest.mark.xfail` backend; integration test frontend).
   - Tests exercise the failure path and the reusable building block, not just one happy-path case.
   - Frontend plans account for required codecov patch coverage. Snapshot changes are called out for review.

9. **Convention compliance**
   - Backend: Pydantic models over dicts, no duck typing, no linter suppressors, top-level absolute imports, file/function length limits.
   - Frontend: function declarations, no `any`, Phosphor icons, Tailwind tokens.

10. **Scope discipline**
    - One concern per PR; out-of-scope changes under 20%. If the plan bundles separable concerns, require a split.
    - Multi-PR features name the feature flag that guards incomplete surface area.

## Output

Report findings only — numbered, each with the check it violates and the code evidence. If every check passes, return exactly: **"Plan review clean — no findings."**

Do not rewrite the plan. Do not implement anything.
