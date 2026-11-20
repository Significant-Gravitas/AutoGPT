# AutoGPT Platform Contribution Guide

This guide provides context for coding agents when updating the **autogpt_platform** folder.

## Linear tickets (core team)

For the AutoGPT core team, whose agents reach our Linear workspace through the Linear MCP server (`https://mcp.linear.app/mcp`). Outside contributors: skip this section and link the GitHub issue your PR addresses instead.

### Before you start any work

Run these steps before any piece of work, even when your user asks you to start right away. If the Linear MCP is not connected, say so and ask your user for the ticket ID.

1. **Find the ticket.** Search Linear for the work (title keywords, error text, PR number) and check GitHub for open PRs or branches on the same thing. When you continue work on a PR that already links a ticket, that is the ticket.
2. **Assigned to someone else:** stop. Tell your user it is already being worked on, by whom, and link the ticket. Continue only if your user says so.
3. **No assignee:** assign it to your user (`me` in the Linear MCP), then start.
4. **No ticket:** create one in the right project (see below), assign it to your user, then start.
5. **Get context before writing code:** the ticket and its project, the linked spec, related PRs on GitHub, and the project's Discord thread if you can read Discord. If you can't, ask your user for the thread.
6. **Net-new features need an explicit ask.** We are polishing what we already ship. If the work adds something new and nothing on the ticket shows John or Toran asked for it, flag it to your user before building. Bugs, fixes and polish need no ask.

### Creating a ticket

- **Project**, first match wins:
  1. An active feature or area project the work clearly belongs to, in the current quarter initiative (`Quarter N - YYYY`).
  2. A project inside an `Exploration: ...` initiative, when the work is part of that exploration.
  3. `<Month> <YYYY> Updates` for bugs, fixes and small polish on the live app.
  4. `Tech Debt & Improvements (QN YYYY)` for refactors, cleanup, tests, dev experience and infra.
  5. `Feature Suggestions (QN YYYY)` for ideas and requests nobody has committed to.
  6. Nothing fits: leave it in Triage with a comment saying where you think it goes.
- **Team:** Internal (`SECRT`). Use AutoGPT Open Source (`OPEN`) only for work the open source community should see.
- **Title:** verb first, specific, under 80 characters. **Description:** what happens, what should happen, the done condition and links. End it with `Filed by <agent> for <person>`.
- **Labels:** one of Bug, Feature or Improvement, plus an Area label when one applies (Data & Tracking covers events, dashboards, pipelines, attribution, consent and privacy).
- Create at most 5 top-level tickets in one pass. Put the rest under a parent ticket as sub-issues.
- Do not create, rename, close or move initiatives or projects, and do not move a ticket someone else filed, unless your user asked for that exact change.

### Branches and pull requests

- Branch from the ticket: use its `gitBranchName` from the Linear MCP, for example `name/secrt-123-short-title`.
- Put `Resolves SECRT-123` in the PR description, or `Part of SECRT-123` when the PR does not finish the ticket. Linear links the PR and closes the ticket when it merges.

The full system (projects, quarters, areas) lives in the team's Notion page "Linear System: how we organize initiatives, projects, tickets and PRs": https://app.notion.com/p/3f400ef5bece8178b5d3c7efabeed911

## Directory overview

- `autogpt_platform/backend` – FastAPI based backend service.
- `autogpt_platform/autogpt_libs` – Shared Python libraries.
- `autogpt_platform/frontend` – Next.js + Typescript frontend.
- `autogpt_platform/docker-compose.yml` – development stack.

See `docs/platform/getting-started.md` for setup instructions.

## Documentation

- `docs/home/`, `docs/platform/` and `docs/integrations/` are the public docs site (agpt.co/docs). Only pages written for people outside the team go there: people using, self-hosting or contributing to AutoGPT.
- The team's working documents go in `docs/engineering/`, which is not on the docs site: analytics and tracking plans, rollout plans, runbooks, internal references, and design and architecture notes. It is still public, like the rest of the repository, so no secrets or confidential material.

See `docs/AGENTS.md` before adding a document.

## Code style

- Format Python code with `poetry run format`.
- Format frontend code using `pnpm format`.

## Frontend guidelines:

See `/frontend/CONTRIBUTING.md` for complete patterns. Quick reference:

1. **Pages**: Create in `src/app/(platform)/feature-name/page.tsx`
   - Add `usePageName.ts` hook for logic
   - Put sub-components in local `components/` folder
2. **Components**: Structure as `ComponentName/ComponentName.tsx` + `useComponentName.ts` + `helpers.ts`
   - Use design system components from `src/components/` (atoms, molecules, organisms)
   - Never use `src/components/__legacy__/*`
3. **Data fetching**: Use generated API hooks from `@/app/api/__generated__/endpoints/`
   - Regenerate with `pnpm generate:api`
   - Pattern: `use{Method}{Version}{OperationName}`
4. **Styling**: Tailwind CSS only, use design tokens, Hugeicons only (through the `Icon` atom)
5. **Testing**: Integration tests (Vitest + RTL + MSW) are the default (~90%, page-level). Playwright for E2E critical flows. Storybook for design system components. See `autogpt_platform/frontend/TESTING.md`
6. **Code conventions**: Function declarations (not arrow functions) for components/handlers
7. **Keyboard handling**: Use `isKey(e, "Enter")` (or `isKey(e, "Enter", " ")`) from `@/lib/keyboard` instead of comparing `e.key`. It returns false while an IME is composing (Japanese, Chinese, Korean input), when Enter/Space/arrows belong to the input method, not the app. The `Input` atom drops composing keydowns before calling `onKeyDown` as a safety net; every handler on a raw `<input>`/`<textarea>`, a container, or `document` must use `isKey` itself. That atom-level guard is deliberately unconditional, so a modifier chord wired through `<Input onKeyDown>` is dropped mid-composition too — handle chords outside the atom. For focus traps and other containment handlers, which must keep holding a key even while composing, use `isKeyIgnoringComposition`. ESLint (`no-restricted-syntax`) flags direct `.key` comparisons and switches against the IME key names only; `// eslint-disable-next-line no-restricted-syntax` is the escape hatch for a domain object that merely has a `.key` field (e.g. `column.key === "Delete"`). Modifier chords like Cmd+K, `.key.toLowerCase()` and `[...].includes(e.key)` are not checked and are out of scope, since an IME never owns them. Passing `e.key` on as a function argument (e.g. into a roving-focus helper) is also invisible to the rule — guard those handlers with `isComposingEvent(e)` at the top.
8. **Text field values**: a controlled `<input>`/`<textarea>` must show exactly what was typed, in the same render as the keystroke. If React writes any other value back (the old one, because the new one arrives a render later through a `useEffect`, a context `register` or a store; a reformatted one, such as pretty-printed JSON; or a rebuilt one, after an edit that didn't parse was dropped), the caret jumps to the end and every IME composition is cancelled (Japanese `nihongo` comes out as `nいhおnngお`). Lint can't see this, so: keep the user's text in state the field reads directly, send it up from the change handler, and replace it from outside only when the stored value really differs (`isJsonTextFor` in `@/lib/json-text` does that check for JSON). Test any field that transforms or round-trips its value with `expectTypingKept` from `@/tests/integrations/typing`.

- Component props should be `interface Props { ... }` (not exported) unless the interface needs to be used outside the component
- Separate render logic from business logic (component.tsx + useComponent.ts + helpers.ts)
- Colocate state when possible and avoid creating large components, use sub-components ( local `/components` folder next to the parent component ) when sensible
- Avoid large hooks, abstract logic into `helpers.ts` files when sensible
- Use function declarations for components, arrow functions only for callbacks
- No barrel files or `index.ts` re-exports
- Avoid comments at all times unless the code is very complex
- Do not use `useCallback` or `useMemo` unless asked to optimise a given function
- Do not type hook returns, let Typescript infer as much as possible
- Never type with `any`, if not types available use `unknown`

## User-facing copy

In text people read in the product, never write "agent" or "graph" in any form ("AI agent", "agent builder", "Graph ID"). What the code calls an agent or graph is a **workflow**. The AI doing the work is **Otto** (not AutoPilot or Copilot), an **expert** or **your team** ("experts and workflows" when a sentence covers both), and Otto's helper runs are **subtasks**. Choose the noun by meaning — never search-and-replace, and never call a workflow an expert. See PR #15251.

- Frontend: every string in `autogpt_platform/frontend` a person can see, `.ts` helpers and `__legacy__` components included: toasts and the `Error` messages they show, page titles, default names, `aria-label`/`title`/alt text, and messages the app sends to Otto for the user.
- Backend: strings a person reads follow the same nouns when you add or edit them: Otto tool-response `message`/`title`/`missing_message`, approval-card schema `title`/`description`, error `detail`s, emails and notifications, and Home attention items. Fix them in the backend; never patch backend text with a frontend replace. Keep tool names, arguments and descriptions Otto reads, prompts, log lines, OpenAPI `summary`/`description`, the "Scheduled agent run" analytics title, the "You have no credits left to run an agent." error that analytics queries match, and `pipeline.py::_GENERIC_NAMES`.
- Change copy only, because renaming breaks things. Keep agent, graph, copilot and autopilot in every identifier and every value code compares or stores, e.g. routes (`/library/agents/[id]`), API fields and values (`"agent_run"`), tool names and arguments even inside messages to Otto (`find_library_agent`, `agent_id`), `TOOL_AGENT = "Agent"`, `data-testid`/`data-id`, and analytics, storage and flag keys. A value that is both shown and sent as a tag (ErrorCard `context`) is copy. Never hand-edit generated files (`src/app/api/__generated__`, `openapi.json`, the approval-card `realCards.json`/`referenceCards.json`); regenerate them instead.
- A label built from a key or tool name that would show agent or graph gets an entry in a key-to-label map (e.g. `TOOL_DISPLAY_NAMES`); never rename the key or show it humanized (`humanizeKey` turns `agent_ids` into "Agent ids").
- Before rewording, search for a distinctive fragment of the string from the repo root (`git grep -inF '<fragment>' -- autogpt_platform`). If code or SQL matches it, reword only outside the matched part and keep the matcher, since saved messages keep the old text (e.g. the `**Here are my answers:**` frame `experts/onboarding.py` reads from stored chats). Update tests and Playwright page objects asserting the string in the same PR; never pin old copy in a new test or story.
- Do not take wording from identifiers, your task or PR text. Recheck any string you move into a helper, rewrite in a new conditional, copy from a neighbouring message or carry through a merge conflict.
- Leave as is: the `agent.json` file name, the "Your agents have moved" notice, hidden search keywords (`GlobalSearchModal/navigation.tsx`), admin and dev pages (`copilot/styleguide`), anything users name or type, third-party names (AgentMail, GitHub Copilot, Microsoft 365 Copilot, including a bare "Copilot" that means Microsoft's), titles of third-party pages we link to (the n8n examples in `ExternalWorkflowTab`) and other senses (user-agent, knowledge graph, charts).
- Awaiting a product decision, so leave them as they appear, without a display map, and do not copy their wording: SEO metadata and og:image alt text (`SITE_DESCRIPTION`, `public/manifest.webmanifest`, the marketplace home and tour pages), `TourHero`, `SignupMarketingPanel`, the `plans.ts` pricing bullets, block names, categories and descriptions however the builder shows them ("Agent Input"), and the `humanizePermission` API-key labels. New text added beside any of these exceptions follows the rule.
- Out of scope: `classic/`, `docs/`, code comments, test names, and commit and PR text.

## Testing

- Backend: `poetry run test` (runs pytest with a docker based postgres + prisma).
- Frontend integration tests: `pnpm test:unit` (Vitest + RTL + MSW, primary testing approach).
- Frontend E2E tests: `pnpm test` or `pnpm test-ui` for Playwright tests.
- See `autogpt_platform/frontend/TESTING.md` for the full testing strategy.

Always run the relevant linters and tests before committing.
Use conventional commit messages for all commits (e.g. `feat(backend): add API`).
Types: - feat - fix - refactor - ci - dx (developer experience)
Scopes: - platform - platform/library - platform/marketplace - backend - backend/executor - frontend - frontend/library - frontend/marketplace - blocks

For a change that is complex always add an e2e test that tests the principle input and output and never the inner logic. 
For example: if we add a agent billing per turn of a session we should have a test that is 3 turns each at 1.2 cents and the final is 3.6 cents actually billed

## Commit attribution

For every commit you create in this repository, ensure the commit message includes a `Co-authored-by:` trailer identifying the large language model and agent platform:

```text
Co-authored-by: MODEL NAME/VERSION (AGENT PLATFORM) <COAUTHOR EMAIL>
```

- Your harness may add this trailer automatically. Do not add a duplicate; ensure the resulting trailer identifies both the model and platform.
- Replace the placeholders with the model name/version reported by your runtime and the agent platform (e.g. Codex, Claude Code, or AutoGPT). Write `unknown` for unavailable model details; do not guess.
- Use the platform's configured co-author email, or `agent@example.invalid` if none is available.
- Include exactly one trailer per distinct platform/model pair that contributed to the commit, after a blank line at the end of the commit message. Preserve existing human co-author trailers.

## Pull requests

- Use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Rely on the pre-commit checks for linting and formatting
- Fill out the **Changes**, **Agents and large language models used**, and checklist sections. List each platform with its model name/version, or `None` if no agents were used.
- Use conventional commit titles with a scope (e.g. `feat(frontend): add feature`).
- Keep out-of-scope changes under 20% of the PR.
- Ensure PR descriptions are complete.
- For changes touching `data/*.py`, validate user ID checks or explain why not needed.
- If adding protected frontend routes, update `autogpt_platform/frontend/src/middleware.ts`
  (matcher config) and `autogpt_platform/frontend/src/lib/auth/middleware.ts`
  (better-auth protection logic).
- Core team: branch from the Linear ticket and put `Resolves SECRT-123` in the description (see **Linear tickets (core team)** above).
