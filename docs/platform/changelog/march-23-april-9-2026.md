# Smarter Starts, Faster Feedback

*23 March – 9 April 2026*

**Platform version:** `v0.6.54`

## Themed Prompt Categories

When you start a new AutoPilot conversation, you now see **themed prompt categories** — Learn, Create, Automate, and Organize. Each theme button opens a popover with **5 contextual prompts** tailored to the category. Personalized prompts are distributed across themes automatically. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12515)

<figure><img src="../.gitbook/assets/v0654-prompt-categories-hero.png" alt="Themed prompt categories in the AutoPilot empty session"><figcaption><p>Themed prompt categories on the new conversation screen</p></figcaption></figure>

## Live Timer Stats

A **live elapsed timer** now appears in the AutoPilot thinking indicator while the AI is processing — showing "23s", "1m 5s" and so on after a 20-second threshold so quick responses aren't cluttered. Once the response completes, a **"Thought for Xm Ys" badge** freezes in place below the message. The duration **stays with the conversation**, so you can always see how long a response took — even after refreshing the page or revisiting an older chat. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12583)

<figure><img src="../.gitbook/assets/v0654-live-timer-hero.png" alt="Live timer stats showing elapsed thinking time"><figcaption><p>Live elapsed timer and persisted thinking duration</p></figcaption></figure>

## Redesigned Onboarding

The onboarding wizard has been **completely redesigned** with an Autopilot-first flow. A polished 4-step experience — Welcome → Role selection → Pain points → Preparing workspace — collects your role and pain points to **personalize Autopilot suggestions** from the start. Pain points are **reordered based on your selected role** (e.g. Sales sees "Finding leads" first), and steps sync with URL params so browser back/forward just works. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12640)

<figure><img src="../.gitbook/assets/v0654-onboarding-hero.png" alt="Redesigned onboarding wizard with role selection"><figcaption><p>New 4-step onboarding wizard with role-based personalization</p></figcaption></figure>

## Copy Your Prompts

You can now **copy your own prompt messages** in AutoPilot with a single click. A copy button appears on hover — right-aligned below the message — using the same `CopyButton` component already available on assistant responses. No more manual text selection when you want to **reuse or share a prompt** you've written. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12571)

<figure><img src="../.gitbook/assets/v0654-copy-prompts-hero.png" alt="Copy button on user prompt messages"><figcaption><p>One-click copy button on your own prompt messages</p></figcaption></figure>

<details>

<summary>✨ Improvements</summary>

- **Spend credits to reset AutoPilot daily rate limit** — Hit your daily message cap? Spend credits to keep going without waiting for the reset. ([#12526](https://github.com/Significant-Gravitas/AutoGPT/pull/12526))
- **Rate-limit tiering system** — AutoPilot now supports configurable rate-limit tiers so different user plans get different message caps. ([#12581](https://github.com/Significant-Gravitas/AutoGPT/pull/12581))
- **Extended thinking execution mode** — OrchestratorBlock gains an extended thinking mode for deeper, multi-step reasoning. ([#12512](https://github.com/Significant-Gravitas/AutoGPT/pull/12512))
- **Automatic AgentMail setup** — AgentMail now works out of the box. Your email address is created automatically the first time you need it — no API keys or manual setup required. ([#12537](https://github.com/Significant-Gravitas/AutoGPT/pull/12537))
- **Session-level dry-run flag** — Autopilot sessions can now be flagged as dry-run at the session level. ([#12582](https://github.com/Significant-Gravitas/AutoGPT/pull/12582))
- **Generic ask_question AutoPilot tool** — A new tool lets AutoPilot ask structured clarifying questions mid-conversation. ([#12647](https://github.com/Significant-Gravitas/AutoGPT/pull/12647))
- **Git committer identity from GitHub profile** — E2B sandbox commits now use the user's real GitHub name and email. ([#12650](https://github.com/Significant-Gravitas/AutoGPT/pull/12650))
- **Create → dry-run → fix agent generation loop** — Agent generation now follows an iterative create, dry-run, and fix cycle for better results. ([#12578](https://github.com/Significant-Gravitas/AutoGPT/pull/12578))
- **All 12 Z.ai GLM models via OpenRouter** — Access every Z.ai GLM model through OpenRouter integration. ([#12672](https://github.com/Significant-Gravitas/AutoGPT/pull/12672))
- **include_graph option for find_library_agent** — Fetch the full graph structure when searching library agents for debugging or editing. ([#12622](https://github.com/Significant-Gravitas/AutoGPT/pull/12622))
- **Cursor-based message pagination** — AutoPilot messages load newest-first with cursor-based pagination for faster load times. ([#12328](https://github.com/Significant-Gravitas/AutoGPT/pull/12328))
- **Codecov coverage reporting** — Coverage reporting set up across platform and classic with Playwright E2E included. ([#12655](https://github.com/Significant-Gravitas/AutoGPT/pull/12655), [#12665](https://github.com/Significant-Gravitas/AutoGPT/pull/12665))
- **React integration testing** — Vitest + React Testing Library + MSW test infrastructure added. ([#12667](https://github.com/Significant-Gravitas/AutoGPT/pull/12667))
- **Cost tracking for system credentials** — Platform now tracks costs incurred by system-managed credentials. ([#12696](https://github.com/Significant-Gravitas/AutoGPT/pull/12696))

</details>

<details>

<summary>🎨 UI/UX Improvements</summary>

- **Auto-reconnect after device sleep** — AutoPilot chat reconnects automatically after your device wakes from sleep. ([#12519](https://github.com/Significant-Gravitas/AutoGPT/pull/12519))
- **Marketplace card descriptions** — Cards now show 3-line descriptions with a fallback color when no image exists. ([#12557](https://github.com/Significant-Gravitas/AutoGPT/pull/12557))
- **Hide placeholder during voice recording** — Placeholder text hides when AutoPilot voice recording is active. ([#12534](https://github.com/Significant-Gravitas/AutoGPT/pull/12534))
- **Array field layout fix** — Fixed array field item layout and added FormRenderer Storybook stories. ([#12532](https://github.com/Significant-Gravitas/AutoGPT/pull/12532))
- **Realistic AutoPilot suggestions** — Replaced unrealistic AutoPilot suggestion prompts with practical ones. ([#12564](https://github.com/Significant-Gravitas/AutoGPT/pull/12564))
- **Show all agent outputs** — Agent results now display all outputs, not just the last one. ([#12504](https://github.com/Significant-Gravitas/AutoGPT/pull/12504))
- **Notification follow-ups** — AutoPilot notifications gain branding, UX improvements, persistence, and cross-tab sync. ([#12428](https://github.com/Significant-Gravitas/AutoGPT/pull/12428))
- **Horizontal scroll for JSON output** — Builder JSON output data now scrolls horizontally instead of overflowing. ([#12638](https://github.com/Significant-Gravitas/AutoGPT/pull/12638))
- **Onboarding polish** — Refined AutoPilot onboarding with branding, auto-advance, soft cap, and visual polish. ([#12686](https://github.com/Significant-Gravitas/AutoGPT/pull/12686))

</details>

<details>

<summary>🐛 Bug Fixes</summary>

- **Prevent graph execution stuck** — Fixed stuck graph executions and steered SDK away from bash_exec. ([#12548](https://github.com/Significant-Gravitas/AutoGPT/pull/12548))
- **Sentry error reduction** — Multiple rounds of prod Sentry error fixes to reduce on-call alert noise. ([#12560](https://github.com/Significant-Gravitas/AutoGPT/pull/12560), [#12565](https://github.com/Significant-Gravitas/AutoGPT/pull/12565))
- **Sink input validation** — AgentValidator now validates sink inputs to catch wiring errors early. ([#12514](https://github.com/Significant-Gravitas/AutoGPT/pull/12514))
- **Docker Node.js upgrade** — Upgraded from EOL Node.js v21 to v22 LTS in Docker images. ([#12561](https://github.com/Significant-Gravitas/AutoGPT/pull/12561))
- **Thinking block preservation** — Thinking blocks are now preserved during transcript compaction. ([#12574](https://github.com/Significant-Gravitas/AutoGPT/pull/12574))
- **AIConditionBlock error handling** — Errors are now raised instead of silently swallowed. ([#12593](https://github.com/Significant-Gravitas/AutoGPT/pull/12593))
- **Host-scoped credentials** — Fixed credential resolution for authenticated web requests. ([#12579](https://github.com/Significant-Gravitas/AutoGPT/pull/12579))
- **Duplicate tool name disambiguation** — OrchestratorBlock now disambiguates duplicate tool names. ([#12555](https://github.com/Significant-Gravitas/AutoGPT/pull/12555))
- **Gmail recipient validation** — Gmail blocks validate email recipients before making API calls. ([#12546](https://github.com/Significant-Gravitas/AutoGPT/pull/12546))
- **Tool call circuit breakers** — Added circuit breakers and intermediate persistence in AutoPilot. ([#12604](https://github.com/Significant-Gravitas/AutoGPT/pull/12604))
- **Prompt-too-long retry** — Automatic compaction, model-aware compression, and truncated tool call recovery. ([#12625](https://github.com/Significant-Gravitas/AutoGPT/pull/12625))
- **Credential loading fix** — Fixed AutoPilot credential loading across event loops. ([#12628](https://github.com/Significant-Gravitas/AutoGPT/pull/12628))
- **Duplicate block execution prevention** — Prevented duplicate execution from pre-launch argument mismatch. ([#12632](https://github.com/Significant-Gravitas/AutoGPT/pull/12632))
- **Tool output file reading** — Fixed tool output file reading between E2B and host. ([#12646](https://github.com/Significant-Gravitas/AutoGPT/pull/12646))
- **Dry-run mode propagation** — Dry-run mode now propagates to special blocks with LLM-powered simulation. ([#12575](https://github.com/Significant-Gravitas/AutoGPT/pull/12575))
- **Double-submit prevention** — Prevented duplicate side effects from double-submit and stale-cache races. ([#12660](https://github.com/Significant-Gravitas/AutoGPT/pull/12660))
- **Silent DataError fix** — Wrapped PlatformCostLog metadata in SafeJson to fix silent DataError. ([#12713](https://github.com/Significant-Gravitas/AutoGPT/pull/12713))

</details>
