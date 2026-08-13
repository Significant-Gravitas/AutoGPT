# Flag-off smoke checklist

**Contract:** with the experts/onboarding feature flags **OFF**, the platform must
behave **byte-identically to pre-experts production**. A new feature ships behind a
flag; flipping that flag off must return the exact prior UX and the exact prior
copilot system prompt.

Automated regression suites already enforce this in CI — they run in the normal
Vitest/pytest pipeline and are discoverable with `grep -rn flag-off`. This page is
the manual pre-release smoke pass **and** the map of what each surface guarantees,
so a reviewer can tell at a glance which test fails if a flag-off promise breaks.

## Flags covered

| Flag value | Enum | Default |
| --- | --- | --- |
| `hire-experts` | `Flag.HIRE_EXPERTS` | `false` (fail-closed) |
| `onboarding-brain-dump` | `Flag.ONBOARDING_BRAIN_DUMP` | `false` (fail-closed) |

Both resolve to `false` whenever LaunchDarkly is unconfigured or silent (local dev,
CI, Playwright), so the default environment is already the flag-off environment. To
force a value explicitly, set `NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS=false` /
`NEXT_PUBLIC_FORCE_FLAG_ONBOARDING_BRAIN_DUMP=false`.

## Smoke steps — `hire-experts` OFF

| Surface | Expected flag-off behavior | Automated guard |
| --- | --- | --- |
| **Copilot** empty state | Pre-experts pulse strip renders; no briefing recap; no `/briefing` request | `copilot/components/CopilotHome/__tests__/flag-off.test.tsx` |
| **Marketplace** | `ExpertsSection` not mounted; pre-experts workflows subtitle ("Ready-made automations from the community.") | `marketplace/components/MainMarketplacePage/__tests__/main.test.tsx` |
| **Marketplace agent page** | No "Install on Expert…" action and no `/api/experts` request, even signed-in with hired experts | `marketplace/components/InstallOnExpertButton/__tests__/main.test.tsx` |
| **Library / sidebar** | Sidebar keeps the **Agents** entry; no **Home** entry and no **Team** row — including when other flags (e.g. brain dump) are on | `components/layout/AppSidebar/__tests__/AppSidebar.test.tsx` |
| `/home` | `notFound()` (404) | `app/(platform)/home/__tests__/main.test.tsx` |
| `/team` and `/team/[expertId]` | `notFound()` (404); no expert or schedule requests fire | `app/(platform)/team/__tests__/main.test.tsx`, `app/(platform)/team/[expertId]/__tests__/main.test.tsx` |
| **Copilot system prompt** (plain session, no expert) | Byte-identical to the pre-experts prompt for both engines (SDK and baseline) in every deterministic config (`use_e2b` × Graphiti); pinned SHA-256 digests of the shared `compose_system_prompt` output | `backend/copilot/flag_off_prompt_test.py`, `backend/copilot/expert_context_test.py` |
| Flag default | `HIRE_EXPERTS` resolves to `false` when LaunchDarkly is silent | `services/feature-flags/__tests__/flag-defaults.test.ts` |

## Smoke steps — `onboarding-brain-dump` OFF

| Surface | Expected flag-off behavior | Automated guard |
| --- | --- | --- |
| Onboarding pain-points step | Legacy `PainPointsStep` renders (not the brain-dump step); step indices/URL unchanged | `app/(no-navbar)/onboarding/__tests__/brain-dump.test.tsx` |
| Preparing step copy | Generic checklist copy on the flag-off path | `app/(no-navbar)/onboarding/steps/__tests__/usePreparingStep.test.ts` |
| Flag default | `ONBOARDING_BRAIN_DUMP` resolves to `false` when LaunchDarkly is silent | `services/feature-flags/__tests__/flag-defaults.test.ts` |

## When a guard fails

The backend prompt-hash guard fails with:

> `flag-off prompt changed; if intentional, update hash + call out in PR description`

If the change to the plain-session system prompt was **intentional**, update the
pinned SHA-256 digests in `backend/copilot/flag_off_prompt_test.py` (and the
base-constant snapshot in `backend/copilot/expert_context_test.py`) and call it
out in the PR description so reviewers know the cacheable prompt moved. If it was
**not** intentional, the flag-off path has drifted from production — fix the code,
not the test.

Both engines assemble their system prompt through `compose_system_prompt` in
`backend/copilot/prompting.py`; add any new prompt component there so the pinned
digests catch it.

Scope note: production may source the base prompt from Langfuse at runtime. That
content lives outside the repo, so CI pins the in-repo fallback composition
(`CACHEABLE_SYSTEM_PROMPT`); flag-off parity of a Langfuse-hosted base must be
maintained in the Langfuse template itself.
