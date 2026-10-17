# Fix the frontend design system

People using the product see a generic kit, because color is chosen in three different ways and the shared tokens still hold stock values. The team cannot change that look in one file. This plan puts color, type, and radius on one role-token layer, deletes the extra copies, and keeps a count that fails when raw values come back.

The count is `pnpm audit:tokens` in `autogpt_platform/frontend`. The PRs are ds-budget, ds-contract, ds-shell, ds-values, ds-button-text, ds-palette, ds-dark-prefix, and ds-legacy.

## How to read this

One box is one unit of work. Every box names the evidence that checks it. A nested box is a sub-step of the box above it. Check a box only when its evidence exists, a file, a log line, a screenshot, a test run, or a SHA. The body is a how-to. The appendices explain and record.

The program runs `pstack/skills/poteto-mode/playbooks/autopilot-stack.md`. The operator lands every PR. Owners stop at stack-ready. No owner merges.

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

## Program checklist

### Arm the program

- [ ] State the protocol and this plan to the operator, then stop. Start execution only on the operator's explicit go.
- [ ] Read these from trunk at program start. Re-read them at every tick. This repo does not contain those plugin files. Read them from the local pstack plugin when `git show` cannot.
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/autopilot-stack.md`
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/autopilot-full.md`
  - [ ] `git show origin/main:pstack/skills/swarm/SKILL.md`
  - [ ] `git show origin/main:pstack/skills/poteto-mode/playbooks/opening-a-pr.md`
  - [ ] `git show origin/main:pstack/skills/prove-it-works/SKILL.md`
- [ ] On the operator's go, arm the audit tick as `/loop 1h` with the tick prompt below. Never leave the cadence to memory.
- [ ] Use this tick prompt, verbatim.

```
Re-read the execution playbook from trunk. Audit the operation against it and fix drift in this tick. Probe every active lane and judge progress by side effects only. Stand down a stuck lane and dispatch its replacement now. Then post a short status message to the operator in chat only when the audit found a tracked change that no earlier status message reported, such as a PR opened, a code-ready head, a round launched or closed, a verdict, a merge, a stuck agent and the action taken, a blocker added or cleared, or a decision only the operator can make. Name every such change and nothing else. Do not repeat a table, the merged list, or an unchanged blocker. If the audit found none, end the turn with no reply text. Either way, log this tick's row in your decision trail. The row names the items reported, or none.
```

- [ ] On the operator's hold or stand-down, send every owner a zero-writes order at once.

### Spawn owners

- [ ] Spawn one owner per PR with the full lifecycle in `playbooks/autopilot-stack.md`.
- [ ] Follow this dependency graph. Start dependent work only after its parent merges, or base it on the parent branch when the stack is being built.
  - [ ] ds-budget and ds-contract are independent and first. Both branch from `dev` after the audit PR is on `dev`.
  - [ ] ds-shell after ds-budget.
  - [ ] ds-values after ds-shell.
  - [ ] ds-button-text after ds-values.
  - [ ] ds-palette after ds-button-text.
  - [ ] ds-dark-prefix after ds-budget. It does not wait on ds-values.
  - [ ] ds-legacy after ds-button-text.
- [ ] Hold the file boundaries. ds-budget touches only the audit script, its report, and the frontend CI script entry. ds-values touches only `src/app/globals.css` token blocks.
- [ ] Hold the review gate. ds-shell, ds-values, ds-button-text, ds-palette, and ds-legacy change an interaction. They wait for the operator's review in chat with screenshots and a video before merge.

### PR mechanics, for every PR

- [ ] Resolve the forge once. Default to `gh`. If `command -v origin` succeeds and Origin can resolve the repository, use `origin pr` for every PR operation. Record any fallback to `gh`. Never require `gt`.
- [ ] Open the PR ready, never draft, per Opening a PR. Use the run's built-in PR tool when it has one, else `origin pr create --status open --base <base-branch>` or `gh pr create --base <base-branch>` according to the resolved forge. A stack child targets its parent branch.
- [ ] Run the repo's lint and typecheck once before the PR-facing push. Push with hooks on.
- [ ] Run deslop before each commit and no-comments before review.
- [ ] Triage every Bugbot and security-reviewer comment per `pstack/skills/poteto-mode/references/bugbot-triage.md`.
- [ ] Rebase onto current trunk before the code-ready report and babysit. Keep that merge base in fix rounds. Rebase again only at merge prep, on a `git merge-tree` conflict with trunk, or on a CI failure that comes from a change on trunk.

### Verdict and merge, for every PR

- [ ] A full swarm runs once, at the code-ready head. Another full swarm runs only for new behavior, a redesign, or a large new diff. A rebase, a merge-conflict resolution, or a minor fix gets a single verifier at the new head. Run each full swarm per `pstack/skills/swarm/SKILL.md`. One gates lane. The ten live lanes from the PR's Verify, live block. The perf lane from its Verify, perf block. Two or more audit lanes, each with its own focus, that read the diff and the receipts and distrust the PR body. The root audits the receipts in the merge-ready report before the verdict.
- [ ] Clean only when every lane is `PASS`. Findings go back to the owner, including a defect that a lane filed as a note. A new head gets a fresh verdict, except for results that stay valid under the patch-id rule in `playbooks/shipping.md`.
- [ ] The root appends the PR to the `dev` stack and the operator lands it bottom-up. Apply the patch-id rule from `playbooks/shipping.md` before each append.

### Boot recipe, for every live lane

Each live lane runs on its own cloud VM at the PR head. `control-ui` is not in this environment. Drive the frontend with Playwright from `autogpt_platform/frontend`, signed in as a local test user, and save screenshots from that run.

- [ ] `git fetch origin <head-branch> && git checkout <head SHA>`.
- [ ] Start the frontend with `pnpm dev` in `autogpt_platform/frontend` and the backend the login flow needs. Wait until `/login` returns a page.
- [ ] Drive input with Playwright commands only. Read the audit script output as the read-only diagnostic.
- [ ] Save every screenshot to `/tmp/swarm-<pr-id>/worker-<n>/<slug>.png` and return the paths with the report.

## Add a count that only tightens (ds-budget)

**Depends on.** The audit PR that added `scripts/audit-design-tokens.mjs`, merged to `dev`.

**Files.**

- [ ] Edit `autogpt_platform/frontend/scripts/audit-design-tokens.mjs`.
- [ ] Edit `autogpt_platform/frontend/scripts/audit-design-tokens.report.json`.
- [ ] Edit `autogpt_platform/frontend/package.json`.

**Build.**

- [ ] Add `--check` to `audit-design-tokens.mjs`. It reads the committed report and exits 1 when chrome outside-both, `darkPrefix`, or `legacyImport` is higher than that report.
- [ ] Point `pnpm audit:tokens` at `node scripts/audit-design-tokens.mjs --check` so a normal run does not rewrite the baseline.
- [ ] Treat `--write` as an override that refreshes the baseline even when the npm script also passes `--check`.
- [ ] Keep `--write` as the only way to lower the baseline, and say that in the script's `--help` text.

**You see.**

- [ ] `pnpm audit:tokens` exits 0 on a clean tree.
- [ ] Adding one `bg-neutral-100` class in a scratch file makes `pnpm audit:tokens` exit 1, and removing it makes the command exit 0 again.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] The classifier self-test still passes. Run `node scripts/audit-design-tokens.mjs --self-test`.
- [ ] A fixture with one extra default-palette class fails `--check`. Run `node scripts/audit-design-tokens.mjs --check` in that fixture and expect exit 1.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when the signed-in library shows the same cards and the same background color at both SHAs.
- [ ] Lane 2. Open `/home`. Save `home.png`. Pass when the home heading is visible and no new error banner appears.
- [ ] Lane 3. Open `/build`. Save `build.png`. Pass when the workflow canvas is visible.
- [ ] Lane 4. Open `/copilot`. Save `copilot.png`. Pass when the composer accepts a typed character.
- [ ] Lane 5. Open `/login`. Save `login.png`. Pass when the email field is visible.
- [ ] Lane 6. Open `/signup`. Save `signup.png`. Pass when the signup form is visible.
- [ ] Lane 7. Open `/marketplace`. Save `marketplace.png`. Pass when at least one card or empty state is visible.
- [ ] Lane 8. Open `/profile`. Save `profile.png`. Pass when the profile page renders for the test user.
- [ ] Lane 9. Open `/team`. Save `team.png`. Pass when the team page renders.
- [ ] Lane 10. Open `/artifacts` at a 390 pixel width. Save `artifacts-narrow.png`. Pass when the page does not overflow sideways.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Wall-clock seconds for `node scripts/audit-design-tokens.mjs --check`, and the time until `/library` shows its first card.
- [ ] Probe. Run the script once at trunk and once at head, interleaved with one cold `/library` load on each. Trunk has no `--check` flag, so time `node scripts/audit-design-tokens.mjs --no-write` there and `--check` at head.
- [ ] Baseline. Record the trunk script seconds and the trunk `/library` time first.
- [ ] Rule. Fail if the head script takes 10 seconds or more. Fail if the head `/library` time is more than 20 percent above the trunk time. Do not compare the script time with the page time.

**Review gate.** None. ds-budget is not review-gated.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-budget to the `dev` stack and the operator lands it.

## Make the written rule match the tokens (ds-contract)

**Depends on.** None. Branch from `dev` after the audit PR is merged. Do not wait for ds-budget.

**Files.**

- [ ] Edit `autogpt_platform/frontend/src/components/tokens/colors.stories.tsx`.
- [ ] Edit `autogpt_platform/frontend/components.json`.
- [ ] Edit `autogpt_platform/frontend/CONTRIBUTING.md` only where the styling section names the approved color API.

**Build.**

- [ ] Rewrite the colors story so the approved API is the role tokens in `globals.css`, and the ramps in `colors.ts` are the primitive layer feature code does not use.
- [ ] Remove `border-gray-200`, `text-blue-500`, and the other default-palette classes from that story.
- [ ] Set `tailwind.cssVariables` to `true` and `iconLibrary` to the library the icon atom actually uses, so a future shadcn add does not emit `neutral` utilities and `dark:` variants.
- [ ] In that story, record the ramp defects beside the primitive layer. `red.900` in `colors.ts` is the string `#641D1D` plus a trailing space. `slate.700` and `slate.800` are both `#475263`. The slate ramp has no `950` step, so `text-slate-950` on login and signup is Tailwind's default slate. `customGray` in `tailwind.config.ts` has no class use under `src`. The old swatch class `bg-light-grey` does not match the key `bgLightGrey`, whose utility is `bg-bgLightGrey`. Do not change those hex values in this PR.

**You see.**

- [ ] Storybook Tokens / Colors shows role swatches, and it contains no `gray-`, `blue-`, or `neutral-` class.
- [ ] `components.json` has `"cssVariables": true`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `pnpm audit:tokens` still exits 0. Run `pnpm audit:tokens`.
- [ ] A search of `colors.stories.tsx` for `gray-`, `blue-`, and `neutral-` returns no class. Run `rg -n "gray-|blue-|neutral-" src/components/tokens/colors.stories.tsx` and expect no matches.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when the library matches trunk, because this PR does not change app CSS.
- [ ] Lane 2. Open the Storybook colors story. Save `colors-story.png`. Pass when the page says to use role tokens and shows no default-palette swatch.
- [ ] Lane 3. Open `/home`. Save `home.png`. Pass when it matches the trunk screenshot.
- [ ] Lane 4. Open `/build`. Save `build.png`. Pass when the canvas is visible.
- [ ] Lane 5. Open `/copilot`. Save `copilot.png`. Pass when the composer is visible.
- [ ] Lane 6. Open `/login`. Save `login.png`. Pass when the form matches trunk.
- [ ] Lane 7. Open `/marketplace`. Save `marketplace.png`. Pass when the page matches trunk.
- [ ] Lane 8. Open `/profile`. Save `profile.png`. Pass when the page matches trunk.
- [ ] Lane 9. Open `/team`. Save `team.png`. Pass when the page matches trunk.
- [ ] Lane 10. Open `/artifacts` at 390 pixels. Save `artifacts-narrow.png`. Pass when the page matches trunk.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/library` shows its first card, at trunk and at head.
- [ ] Probe. One cold load of `/library` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 2 into `docs/engineering/media/ds-contract-review-colors.png`.
- [ ] Record a 30 to 60 second video of the colors story on a lane VM. Save it as `docs/engineering/media/ds-contract-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-contract to the `dev` stack and the operator lands it.

## Point the page shell at role tokens (ds-shell)

**Depends on.** ds-budget.

**Files.**

- [ ] Edit `autogpt_platform/frontend/src/app/globals.css`.

**Build.**

- [ ] Change the `body` rule from `bg-[#F6F7F8]` to `bg-background`.
- [ ] Replace `gray`, `zinc`, `white`, and `violet` utilities inside the `agpt-*` classes with role tokens such as `border-border`, `bg-card`, and `bg-accent`.
- [ ] Leave the token values in `:root` and `.dark` alone. ds-values owns those numbers.

**You see.**

- [ ] The page background comes from `--background`.
- [ ] `globals.css` has no `#F6F7F8`, no `bg-white`, and no `border-zinc-300` in the `agpt-*` classes.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `pnpm audit:tokens` exits 0, and the new baseline is lower or equal on hex and default-palette counts. Run `pnpm audit:tokens` after `--write` only if the count fell.
- [ ] `rg -n "F6F7F8|border-zinc-300|bg-white" src/app/globals.css` returns no matches.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when the page still lists the same workflows and the background is the token color rather than a second hardcoded gray.
- [ ] Lane 2. Open `/home`. Save `home.png`. Pass when text on the page background stays readable.
- [ ] Lane 3. Open `/build`. Save `build.png`. Pass when node borders are still visible.
- [ ] Lane 4. Open `/copilot`. Save `copilot.png`. Pass when the composer border is visible.
- [ ] Lane 5. Open `/login`. Save `login.png`. Pass when the form fields are visible on the page background.
- [ ] Lane 6. Open `/signup`. Save `signup.png`. Pass when the form matches the login treatment.
- [ ] Lane 7. Open `/marketplace`. Save `marketplace.png`. Pass when cards are distinguishable from the page.
- [ ] Lane 8. Open `/profile`. Save `profile.png`. Pass when section cards still have a border.
- [ ] Lane 9. Open `/team`. Save `team.png`. Pass when the page heading is readable.
- [ ] Lane 10. Open `/library` at 390 pixels. Save `library-narrow.png`. Pass when the background color matches the wide screenshot.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/library` shows its first card, at trunk and at head.
- [ ] Probe. One cold load of `/library` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1 and lane 5 into `docs/engineering/media/ds-shell-review-library.png` and `docs/engineering/media/ds-shell-review-login.png`.
- [ ] Record a 30 to 60 second video that opens library, copilot, and login. Save it as `docs/engineering/media/ds-shell-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-shell to the `dev` stack and the operator lands it.

## Give the role tokens a non-stock value (ds-values)

**Depends on.** ds-shell.

**Files.**

- [ ] Edit the `:root` and `.dark` blocks in `autogpt_platform/frontend/src/app/globals.css`.

**Build.**

- [ ] Keep the accent hue at `262 83% 58%` in both `:root` and `.dark`, unless the operator has named a different accent before this PR starts.
- [ ] Set `.dark` `--accent` and `--accent-foreground` from that same hue. Do not copy the zinc gray accent.
- [ ] Change `--background` and `--primary` so they are not the shadcn zinc template values cited in Appendix F. Keep contrast of text on background and text on primary high enough to read.
- [ ] Do not edit feature components in this PR.

**You see.**

- [ ] `pnpm audit:tokens` reports light-theme differences on accent, background, and primary, and it no longer reports `.dark` as 24 of 24 identical to the zinc template.
- [ ] A selected control and a primary button use the same accent hue on a light page and on a `.dark` page.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] The theme overlap section of the audit JSON shows `.dark` accent different from `240 3.7% 15.9%`. Run `node scripts/audit-design-tokens.mjs --write` only to refresh the overlap record, then inspect `themeOverlap.dark.different`.
- [ ] Storybook still builds. Run `pnpm build-storybook`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when the same workflows are listed and the head background is the new token, not the trunk gray.
- [ ] Lane 2. Open `/home`. Save `home.png`. Pass when the heading contrast is readable against the new background.
- [ ] Lane 3. Open `/build`. Save `build.png`. Pass when a selected node uses the accent hue.
- [ ] Lane 4. Open `/copilot`. Save `copilot.png`. Pass when the send control uses the accent hue.
- [ ] Lane 5. Open `/login`. Save `login.png`. Pass when the primary action uses the accent hue.
- [ ] Lane 6. Open `/signup`. Save `signup.png`. Pass when the primary action matches login.
- [ ] Lane 7. Open `/marketplace`. Save `marketplace.png`. Pass when cards are still separate from the page.
- [ ] Lane 8. Open `/profile`. Save `profile.png`. Pass when body text is readable.
- [ ] Lane 9. Force the `.dark` class on `/library` from the console. Save `library-dark.png`. Pass when the accent hue matches the light screenshot and is not the stock gray.
- [ ] Lane 10. Open `/library` at 390 pixels. Save `library-narrow.png`. Pass when the background matches the wide head screenshot.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/library` shows its first card, at trunk and at head.
- [ ] Probe. One cold load of `/library` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time. CSS variable changes do not add a script. There is no second metric.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1, lane 5, and lane 9 into `docs/engineering/media/ds-values-review-library.png`, `docs/engineering/media/ds-values-review-login.png`, and `docs/engineering/media/ds-values-review-dark.png`.
- [ ] Record a 30 to 60 second video of library, login, and the forced dark library. Save it as `docs/engineering/media/ds-values-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-values to the `dev` stack and the operator lands it.

## Teach Button and Text to use roles (ds-button-text)

**Depends on.** ds-values.

**Files.**

- [ ] Edit `autogpt_platform/frontend/src/components/atoms/Button/helpers.ts`.
- [ ] Edit `autogpt_platform/frontend/src/components/atoms/Text/helpers.ts`.
- [ ] Edit `autogpt_platform/frontend/src/components/ui/button.tsx` only to stop new use. Delete it if the import count of `@/components/ui/button` reaches 0 in this PR.
- [ ] Edit call sites that import `@/components/ui/button` so they import the atom.

**Build.**

- [ ] Map Button variants to role tokens. Primary uses `bg-primary text-primary-foreground`. Destructive uses `bg-destructive`. Outline uses `border-border` and drops `#a6a6a6`.
- [ ] Map Text tones to role tokens. `primary`, `secondary`, `muted`, and `danger` stop using `text-zinc-*` and `text-red-600`.
- [ ] Leave the legacy button for ds-legacy.

**You see.**

- [ ] A primary atom button and a destructive atom button use the role tokens from ds-values.
- [ ] `rg -n "neutral-|#[0-9a-fA-F]" src/components/atoms/Button/helpers.ts` returns no matches.
- [ ] Feature code no longer imports `@/components/ui/button`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Button tests cover primary, destructive, and outline. Run `pnpm test:unit src/components/atoms/Button`.
- [ ] Text tests cover the four tones. Run `pnpm test:unit src/components/atoms/Text`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when primary actions use the ds-values accent and disabled buttons are still visibly disabled.
- [ ] Lane 2. Open `/home` and press the main action. Save `home-action.png`. Pass when the action shows a pressed or hover state that is not a different gray family.
- [ ] Lane 3. Open `/build` and select a node. Save `build.png`. Pass when the canvas still accepts a click.
- [ ] Lane 4. Open `/copilot` and focus the composer. Save `copilot.png`. Pass when focus is visible without `ring-neutral-950`.
- [ ] Lane 5. Open `/login` and submit an empty form. Save `login-error.png`. Pass when the error text uses the danger tone and stays readable.
- [ ] Lane 6. Open `/signup`. Save `signup.png`. Pass when the primary button matches login.
- [ ] Lane 7. Open `/marketplace`. Save `marketplace.png`. Pass when card actions still have a visible border.
- [ ] Lane 8. Open `/profile`. Save `profile.png`. Pass when secondary text is distinguishable from primary text.
- [ ] Lane 9. Open `/settings/billing`. Save `billing.png`. Pass when a destructive or primary billing action is readable.
- [ ] Lane 10. Open `/login` at 390 pixels. Save `login-narrow.png`. Pass when the primary button stays inside the viewport.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/library` shows its first card, at trunk and at head.
- [ ] Probe. One cold load of `/library` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1, lane 5, and lane 10 into `docs/engineering/media/ds-button-text-review-library.png`, `docs/engineering/media/ds-button-text-review-login.png`, and `docs/engineering/media/ds-button-text-review-narrow.png`.
- [ ] Record a 30 to 60 second video of a primary click, a disabled button, and the login error. Save it as `docs/engineering/media/ds-button-text-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-button-text to the `dev` stack and the operator lands it.

## Remove default palette colors from the design system (ds-palette)

**Depends on.** ds-button-text.

**Files.**

- [ ] Edit files under `autogpt_platform/frontend/src/components` except `__legacy__`, `styles/colors.ts`, and artwork paths the audit script already skips.
- [ ] Do not edit `src/app/(platform)/raise/components/ColorStep/helpers.ts` in this PR. That file is a catalog of expert colors and needs its own role.

**Build.**

- [ ] Replace `neutral`, `gray`, `amber`, `violet`, and `blue` utilities in `src/components` with role tokens or, where a status color has no role yet, add one role in `globals.css` and use it.
- [ ] Add `success` and `warning` roles if a component needs them. Do not leave `emerald` and `amber` in the component as a stand-in.
- [ ] Lower the baseline with `--write` only after the count of default-palette hits in the design-system area drops.

**You see.**

- [ ] `src/components/ui/button.tsx` is gone or has no `neutral` class.
- [ ] The audit report's design-system `defaultPalette` count is lower than the audit baseline of 182.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `pnpm audit:tokens` exits 0 on the new baseline. Run `pnpm audit:tokens`.
- [ ] Component tests for Button, Text, and DropdownMenu pass. Run `pnpm test:unit src/components/atoms src/components/molecules/DropdownMenu`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when menus, badges, and buttons still separate primary text from secondary text.
- [ ] Lane 2. Open a dropdown on `/library`. Save `library-menu.png`. Pass when the hover row is visible.
- [ ] Lane 3. Open `/build`. Save `build.png`. Pass when node chrome is visible.
- [ ] Lane 4. Open `/copilot`. Save `copilot.png`. Pass when a message and the composer are both readable.
- [ ] Lane 5. Open `/login`. Save `login.png`. Pass when focus rings are visible.
- [ ] Lane 6. Open `/marketplace`. Save `marketplace.png`. Pass when cards keep a border.
- [ ] Lane 7. Open `/profile`. Save `profile.png`. Pass when status colors still read as success or danger, not as an unlabeled gray.
- [ ] Lane 8. Open `/team`. Save `team.png`. Pass when the page heading is readable.
- [ ] Lane 9. Open `/settings/integrations`. Save `integrations.png`. Pass when a connected and a disconnected row are distinguishable.
- [ ] Lane 10. Open `/copilot` at 390 pixels. Save `copilot-narrow.png`. Pass when the composer stays on screen.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/library` shows its first card, at trunk and at head.
- [ ] Probe. One cold load of `/library` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1 and lane 2 into `docs/engineering/media/ds-palette-review-library.png` and `docs/engineering/media/ds-palette-review-menu.png`.
- [ ] Record a 30 to 60 second video of a menu open and close on `/library`. Save it as `docs/engineering/media/ds-palette-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-palette to the `dev` stack and the operator lands it.

## Delete latent dark variants (ds-dark-prefix)

**Depends on.** ds-budget.

**Files.**

- [ ] Edit product files under `autogpt_platform/frontend/src` that contain a `dark:` class.
- [ ] Do not add a theme toggle in this PR.

**Build.**

- [ ] Confirm the theme lock before deleting classes. `src/app/providers.tsx` sets `forcedTheme="light"`. `src/app/layout.tsx` passes `attribute="class"` and `defaultTheme="light"`, and leaves `enableSystem` commented out. `tailwind.config.ts` sets `darkMode` to the selector `.dark-mode`, so a `dark:` utility does not follow the `.dark` class next-themes would add. If a user-facing control that switches theme has landed since this plan, stop and tell the operator. The styleguide fixtures that mention dark mode are sample tool output, not a toggle.
- [ ] Remove `dark:` utilities. The role tokens in `.dark` remain for a later theme switch.
- [ ] Lower the `darkPrefix` baseline to the leftover count, which should be 0 in chrome files.

**You see.**

- [ ] `rg -n "dark:" autogpt_platform/frontend/src --glob '!**/__generated__/**'` returns no class in product files, other than a comment if one must stay.
- [ ] Light pages look the same as the parent branch.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `pnpm audit:tokens` reports `darkPrefix` 0 for chrome. Run `pnpm audit:tokens`.
- [ ] `pnpm test:unit src/components/atoms/Button src/components/ui` passes.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Open `/library` at trunk and at head. Save `library.png`. Pass when the light theme matches the parent, with no missing hover color.
- [ ] Lane 2. Open `/home`. Save `home.png`. Pass when the heading matches the parent.
- [ ] Lane 3. Open `/build`. Save `build.png`. Pass when nodes match the parent.
- [ ] Lane 4. Open `/copilot`. Save `copilot.png`. Pass when the composer matches the parent.
- [ ] Lane 5. Open `/login`. Save `login.png`. Pass when the form matches the parent.
- [ ] Lane 6. Open `/signup`. Save `signup.png`. Pass when the form matches the parent.
- [ ] Lane 7. Open `/marketplace`. Save `marketplace.png`. Pass when cards match the parent.
- [ ] Lane 8. Open `/profile`. Save `profile.png`. Pass when the page matches the parent.
- [ ] Lane 9. Open `/team`. Save `team.png`. Pass when the page matches the parent.
- [ ] Lane 10. Open `/artifacts` at 390 pixels. Save `artifacts-narrow.png`. Pass when the page matches the parent.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/library` shows its first card, at trunk and at head.
- [ ] Probe. One cold load of `/library` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time. Deleting unused classes should not add work.

**Review gate.** None. ds-dark-prefix is not review-gated.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-dark-prefix to the `dev` stack and the operator lands it.

## Retire legacy primitives on the signed-in path (ds-legacy)

**Depends on.** ds-button-text.

**Files.**

- [ ] Edit files that import `@/components/__legacy__/ui/button`, `@/components/__legacy__/ui/input`, or `@/components/__legacy__/ui/form`.
- [ ] Delete a legacy file only when its import count is 0.

**Build.**

- [ ] Move login, signup, reset password, and profile API keys onto the atom Button and Input.
- [ ] Do not restyle the workflow builder in this PR. Leave its legacy imports for a follow-up once this slice has lowered `legacyImport`.
- [ ] Lower the baseline only for the imports this PR removes.

**You see.**

- [ ] `/login` no longer imports a legacy button or input.
- [ ] The audit `legacyImport` count is lower than 196.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Login and profile tests pass. Run `pnpm test:unit src/app/(no-navbar)/login src/app/(platform)/profile`.
- [ ] `rg -n "@/components/__legacy__/ui/(button|input|form)" src/app/(no-navbar)/login src/app/(no-navbar)/signup src/app/(platform)/profile` returns no matches.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on `grok-4.7-xhigh-fast` at the PR head, per the boot recipe.

- [ ] Lane 1. Regression lane against trunk. Sign in from `/login` at trunk and at head. Save `login-submit.png`. Pass when the test user reaches a signed-in page on both.
- [ ] Lane 2. Open `/signup`. Save `signup.png`. Pass when the form submits a validation error on an empty email without a crash.
- [ ] Lane 3. Open `/reset-password`. Save `reset.png`. Pass when the email field and the submit button are visible.
- [ ] Lane 4. Open `/profile` API keys. Save `api-keys.png`. Pass when the create action opens a dialog.
- [ ] Lane 5. Open `/library`. Save `library.png`. Pass when the page still renders. This PR does not retarget it.
- [ ] Lane 6. Open `/build`. Save `build.png`. Pass when the canvas still renders. Legacy builder imports stay.
- [ ] Lane 7. Open `/copilot`. Save `copilot.png`. Pass when the composer still accepts text.
- [ ] Lane 8. Open `/settings/api-keys` if that route is the live one, otherwise `/profile`. Save `keys.png`. Pass when a key row or an empty state is visible.
- [ ] Lane 9. Submit login with a wrong password. Save `login-wrong.png`. Pass when the error stays on the form.
- [ ] Lane 10. Open `/login` at 390 pixels. Save `login-narrow.png`. Pass when the submit button is on screen.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Time until `/login` shows the email field, at trunk and at head.
- [ ] Probe. One cold load of `/login` at trunk, then one at head, then one more at trunk.
- [ ] Baseline. Record the first trunk time first.
- [ ] Rule. Fail if the head time is more than 20 percent above the trunk time.

**Review gate.** The operator reviews before merge.

- [ ] Copy lane 1, lane 4, and lane 9 into `docs/engineering/media/ds-legacy-review-login.png`, `docs/engineering/media/ds-legacy-review-keys.png`, and `docs/engineering/media/ds-legacy-review-error.png`.
- [ ] Record a 30 to 60 second video of a failed login and an API key dialog. Save it as `docs/engineering/media/ds-legacy-review.mp4`.
- [ ] Post the screenshots and the video in chat. Stop at merge-ready. Wait for the operator's click.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] Bugbot triage done.
- [ ] Rebased onto current trunk after the verdict, patch-id unchanged.
- [ ] The root appends ds-legacy to the `dev` stack and the operator lands it.

## Close the program

- [ ] Every box above is checked with its evidence.
- [ ] Reply to the operator with the stack root, the stack tip, and a one-line verdict per PR.

## Appendix A. Prototype evidence

No prototype branch was cut. The questions this plan could have prototyped were settled by a count instead.

The question of whether raw colors are the reason the UI looks generic is a measurement. `scripts/audit-design-tokens.mjs` answers it. See Appendix F. A visual prototype would not change that count.

The question of which accent hue to ship is a preference. No run can pick a brand. The default, applied so the program does not wait, is to keep the current light accent `262 83% 58%` and to use that same hue in `.dark`. The operator can name a different accent before ds-values starts. Say the hue in a sentence. There is no code to type back.

## Appendix B. Alternatives rejected

Restyling screens one by one, with no failing count, loses. New screens would add `bg-neutral-900` again. ds-budget exists so the count can only fall.

Adopting StyleX and a three-input theme generator now loses. Linear did that after the roles already existed, and the migration was more than a thousand pull requests. AutoGPT's first failure is three color languages plus stock values. CSS variables already in `globals.css` can hold the roles. A generator is a later program, after ds-legacy.

Banning only hex codes loses. Hex is 447 of the chrome color hits. The custom ramp is 4248, and zinc alone is 3084. A hex ban would leave the generic gray kit in place.

Renaming `zinc` to a private scale in the first PR loses. That is a large diff with no change a person can see. Roles first, then stop reading the ramp from feature code.

Turning on `shadcn/no-raw-colors` before the role migration loses. `@shadcn/lint` 0.1.1 is already a dependency, and `.oxlintrc.json` enables only `shadcn/require-static-classes`. The same package ships `no-raw-colors`, `no-arbitrary-values`, `no-inline-styles`, `no-unknown-classes`, and `no-restyle`. `no-raw-colors` flags palette utilities, and the classes the June 2025 rule approved are palette names such as `bg-zinc-800`. The rule would fail the build on the design ramp. The audit script can count that ramp apart from the default palette. Enable `no-raw-colors` only after feature code uses role utilities.

Combining the shared button restyle with a global HSL edit loses. Pull request 9570 switched the shared button to role utilities and retuned the variables on 5 March 2025. Pull request 9575 reverted that whole change the same day because the restyle broke existing screens. ds-values edits only the token block. ds-button-text edits the atoms after that.

Switching Tailwind's default palette off in one config change loses. Closed testing pull request 15255 tried a remap off `dev` and still had to keep some hues distinct. ds-palette retargets each default-palette class to a role.

## Appendix C. Risks

ds-values changes `--border` and the base rule `* { @apply border-border }`, so every bordered control moves together. The owner watches library, build, and copilot screenshots before asking for review.

`src/app/(platform)/raise/components/ColorStep/helpers.ts` holds 143 default-palette classes and 60 hex codes on purpose, as expert color options. ds-palette does not edit it. A later PR needs a role for an expert's chosen color or the count cannot reach zero.

Artwork is 18 files and 513 hex codes. The script leaves those out of the chrome total. Logos and empty-state illustrations stay literal.

`control-ui` is not available in this environment. Live lanes use Playwright. If a lane cannot sign in, it fails. It does not skip.

The `git show origin/main:pstack/...` lines do not resolve in this repository. The coordinator reads those files from the local plugin. That is a process risk, not a product risk.

ds-dark-prefix deletes `dark:` classes only while `forcedTheme` stays `"light"`. If a user-facing theme control exists, the owner stops. The `.dark` token block stays. Those utilities listen for `.dark-mode`, which is a different selector from `.dark`.

`components.json` currently has `cssVariables` false. Until ds-contract lands, `shadcn add` will generate `neutral` utilities and the count will rise.

The builder still imports legacy primitives after ds-legacy. The follow-up is out of this program on purpose, so this program stays reviewable.

## Appendix D. Links and reading list

Read these before editing tokens.

- `autogpt_platform/frontend/src/app/globals.css`
- `autogpt_platform/frontend/tailwind.config.ts`
- `autogpt_platform/frontend/src/components/styles/colors.ts`
- `autogpt_platform/frontend/src/components/tokens/colors.stories.tsx`
- `autogpt_platform/frontend/src/components/atoms/Button/helpers.ts`
- `autogpt_platform/frontend/src/components/atoms/Text/helpers.ts`
- `autogpt_platform/frontend/CONTRIBUTING.md` styling section
- `autogpt_platform/frontend/scripts/audit-design-tokens.mjs`

Public sources for the architecture are in Appendix G. ds-values and ds-button-text get a design pass with `pstack/skills/how/SKILL.md` before editing, and `pstack/skills/interrogate/SKILL.md` before the review gate. The decision trail for the audit is `docs/engineering/frontend-design-system-audit-decisions.tsv`, per `pstack/skills/show-me-your-work/SKILL.md`.

## Appendix E. What is wrong today

The premise under test was that the UI looks generic because components use raw or hardcoded colors instead of semantic tokens. The premise is half right.

Chrome files are product files under `src` after comment stripping, with tests, stories, and 18 artwork files removed. In those 2549 files the script found 8555 color decisions.

522 of those decisions are role utilities such as `bg-background` and `text-muted-foreground`. That is 6.1 percent.

5034 decisions use the approved ramp in `colors.ts` or its flat colors `white`, `black`, `textGrey`, `textBlack`, and `bgLightGrey`. Zinc alone is 3084. The colors story tells people to style with `bg-slate-500`, not with a role. The Text atom then calls `text-zinc-500` a muted tone. A step number is not a role, so two screens can both be "correct" and still use different grays.

2999 decisions sit outside both systems. 2252 are Tailwind palette families that `colors.ts` does not override. The largest are neutral 791, gray 480, amber 313, violet 206, and blue 185. 225 are arbitrary colors such as `bg-[#fff]`. 447 are hex literals outside the palette file and outside artwork. 75 are `rgb`, `hsl`, or `color-mix` calls. Those 2999 hits sit in 527 files.

1046 of 2549 chrome files contain at least one non-role color. The role layer is not what most screens use.

The role layer that does exist is still the stock kit. Compared with the shadcn zinc template in commit `d5920cc` of `shadcn-ui/ui`, the light block matches 20 of 25 tokens. The five differences are `--background` `0 0% 98%` against `0 0% 100%`, `--accent` `262 83% 58%` against `240 4.8% 95.9%`, the accent foreground, `--input` `85%` against `90%`, and `--ring`. The dark block matches 24 of 24 tokens, so the one custom accent disappears in `.dark` and becomes the stock gray. Sidebar tokens match 8 of 8 in both modes, which is the shadcn sidebar registry.

The page background does not even use `--background`. `body` sets `bg-[#F6F7F8]`. `colors.ts` defines `white` as `#fefefe` and `black` as `#141414`, so `bg-white` is a third white. The outline button also hardcodes `#a6a6a6`.

Two more systems sit beside color. Named radii such as `rounded-small` occur 78 times. Tailwind radii such as `rounded-lg` occur 1751 times. Arbitrary lengths occur 1213 times, and arbitrary font sizes occur 320 times. `Text` variants are rem values written out in `helpers.ts`, including `text-[0.6785rem]` for labels.

Three buttons are imported. The atom is imported 417 times, `src/components/ui/button.tsx` 17 times, and the legacy button 32 times. The shadcn button is the neutral palette plus `dark:` variants. The atom button is the zinc ramp plus a hardcoded border. `components.json` says `cssVariables` is false and `baseColor` is `neutral`, which is how that second button got generated. `CONTRIBUTING.md` already says to prefer tokens and not to import shadcn primitives from feature code. `.oxlintrc.json` only warns on `shadcn/require-static-classes`. Nothing fails the build.

`dark:` appears 501 times in chrome files. `tailwind.config.ts` sets `darkMode` to `["class", ".dark-mode"]` and says those variants are ignored until dark mode is supported. `Providers` forces `forcedTheme="light"`. Layout sets `defaultTheme="light"` and leaves `enableSystem` commented out. A search of `src` found no user-facing theme control. A `dark:` utility follows `.dark-mode`. The stock values in the `.dark` block follow `.dark`. Turning a theme on later would load that stock zinc block unless ds-values has already replaced it.

Legacy imports of `@/components/__legacy__` occur 196 times, against the rule that feature code must not use that folder.

For a person using the product, borders and text grays change between screens, the page background is not the token, and a future dark mode would drop the violet accent. For the team, a new screen has three legal-looking ways to pick a gray, and review cannot catch the wrong one.

The count is a lower bound. Classes built with string concatenation are invisible to the regex. Each file is read once, so the number is not a double count of compiled CSS. Hex inside `colors.ts` is reported separately as 87 palette definitions and is not part of the 447. Family totals sum back to 2252 and 4248, which is the check that the classifier did not drop or double a ramp step.

Regenerate with `pnpm audit:tokens` from `autogpt_platform/frontend`. The committed snapshot is `scripts/audit-design-tokens.report.json`.

## Appendix F. Count reference

Command `pnpm audit:tokens`. Unit is one source match after comments are stripped. Files scanned 3642. Chrome files 2549. Artwork files excluded from chrome 18. Tests and stories are not in the chrome totals.

| Bucket | Count |
| --- | --- |
| Role utilities | 522 |
| Custom ramp | 4248 |
| Custom flat colors | 786 |
| Default palette | 2252 |
| Arbitrary color | 225 |
| Hex outside the palette file and artwork | 447 |
| Functional color | 75 |
| Outside both systems | 2999 |
| Chrome files with a non-role color | 1046 |
| Chrome files outside both systems | 527 |
| `dark:` prefix | 501 |
| Legacy imports | 196 |
| Named radius | 78 |
| Tailwind radius | 1751 |
| Arbitrary length | 1213 |
| Arbitrary type size | 320 |
| Atom Button imports | 417 |
| `ui/button` imports | 17 |
| Legacy button imports | 32 |
| Atom Text imports | 497 |

Light theme tokens identical to the shadcn zinc template, 20. Different, 5. Dark theme identical, 24 of 24. Sidebar identical, 8 of 8 in each mode.

Default palette families in chrome are neutral 791, gray 480, amber 313, violet 206, blue 185, emerald 91, rose 45, sky 38, indigo 36, teal 18, cyan 16, lime 15, fuchsia 15, and stone 3.

Custom ramp families in chrome are zinc 3084, red 399, green 209, purple 208, slate 178, yellow 90, orange 66, and pink 14.

## Appendix G. How other teams avoid a generic kit

Vercel publishes Geist as the product system, not as a suggestion to use Tailwind's default palette. The colors page gives each step a job. Step 100 is the default background, 400 is the default border, 900 is secondary text, and 1000 is primary text. Components use those jobs. The same docs tell you to draw a border once and to keep shadows translucent. Typography is a named set of classes, not a free `text-[13px]`. Source `https://vercel.com/geist/colors` and `https://vercel.com/geist/introduction`, fetched for this audit. The components.build token note describes the same split AutoGPT already half-has, semantic variables over raw values. Source `https://github.com/vercel/components.build/blob/main/skills/building-components/references/design-tokens.mdx`.

shadcn's own theming page says the same thing, and it is the system this repo started from. Override `--background` and `--primary`, and components that use `bg-background` change without a class edit. The default scaffold on that page is a neutral OKLCH theme. AutoGPT kept an older zinc HSL scaffold and then bypassed it with palette classes. Source `https://ui.shadcn.com/docs/theming`, fetched for this audit. The zinc values used for the overlap count are the `cssVarsTemplate` in `https://github.com/shadcn-ui/ui/commit/d5920cc3c1c12ed7d54c384e86fd207b648b6e42`.

Linear's 2024 redesign post says they stopped hand-setting 98 variables per theme. Three inputs, base color, accent color, and contrast, generate the theme in LCH so a red and a yellow at the same lightness look equally light. Light and dark themes share that generator, and the accent does not have to be re-picked per mode. A later StyleX post says custom lint blocks hardcoded values and unsafe overrides, and that a selected row regenerates the theme against its own background. Sources `https://linear.app/now/how-we-redesigned-the-linear-ui` and `https://linear.app/now/styling-linear-for-the-future-stylex`, fetched for this audit, plus `https://linear.app/now/a-design-reset` from the same series.

Cursor has not published a token spec for its own product. The public posts describe the agent interface and a design-engineer role that owns pixel-level UI. They do not name color steps or a generator. Treat that as a thin public record, not as a pattern to copy. Sources `https://cursor.com/blog/cursor-3` and `https://cursor.com/careers/design-engineer`.

xAI's public brand guidelines cover logo use, not product tokens. Third-party pages scrape grok.com and describe a near-white and near-black UI with one orange accent. Those pages are not an xAI specification, and this audit does not treat their hex values as a source. The official page is `https://x.ai/legal/brand-guidelines`. The gap is the finding. A custom system can be real and still unpublished.

What to copy is the structure, not the colors. One primitive palette. Role tokens whose names are jobs. Light and dark defined together so the accent hue survives. A check that fails when a raw value returns. What not to copy yet is Linear's generator and StyleX migration. Appendix B says why.

## Appendix H. Token architecture

Keep three layers, and make the middle one the only layer feature code can see.

This reverses the written rule from 19 June 2025. Pull request 10186 called the ramps in `colors.ts` the only colors, and it called the HSL variables legacy. The config comments that said so were deleted in pull request 11763 while reformatting `tailwind.config.ts`. The colors story still carries the ramp warning. The HSL layer grew after that label was gone, including the sidebar tokens in pull request 11901. Three docs still disagree. The overview example is `text-primary bg-surface`, and there is no `surface` token. The colors story says ramps only. `CONTRIBUTING.md` says prefer a token when one exists. The role layer is the one that can change the product from one file. The ramp names stay, so an existing `bg-zinc-800` class keeps its hex until a later PR stops feature code from using the ramp.

The primitive layer is `src/components/styles/colors.ts`. It holds ramps. Feature code does not import it and does not use `bg-zinc-800`.

The role layer is the variables in `globals.css`, exposed through Tailwind as `bg-background`, `text-foreground`, `border-border`, `bg-primary`, `text-muted-foreground`, `bg-destructive`, and new `success` and `warning` roles when a component needs them. `:root` and `.dark` set the same names. The accent hue is the same in both. Hover and active are extra roles, not a jump to `zinc-700`.

The component layer is the atoms. Button and Text encode the roles once. A page passes a variant, not a color class.

Enforcement is the audit script in `--check` mode, wired by ds-budget. The committed JSON is the ceiling. A pull request that adds a default-palette class, a new hex in chrome, a `dark:` variant, or a legacy import fails. Artwork and `colors.ts` stay excluded. Lower the ceiling only in the PR that deleted the hits.

Do not add a fourth layer of per-component CSS variables until a component has a color that is not a role. The outline button's `#a6a6a6` is not such a case. It is `border-border`.
