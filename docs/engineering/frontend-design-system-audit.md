# Frontend design system audit

Audit of `autogpt_platform/frontend` on 2026-10-07 (branch `newport-beach-v1`, Tailwind 3.4.17, Next 15.5.24, Storybook 9.1.5). Every claim below has a file reference; counts come from grep, `pnpm knip`, `tailwind-merge` and the Tailwind compiler run against the real config. Paths are relative to `autogpt_platform/frontend` unless they start with `.github/` or `docs/`.

The report has five parts: the token layer, the component layer, drift in feature code, enforcement, and a recommended plan. Part 5 is the actionable summary; parts 1 to 4 are the evidence.

---

## Headline

The design system is not one system. It is three colour systems, three component layers, four button implementations, five avatar components, six icon libraries and two sources of truth for fonts, with nothing mechanical holding any of it together.

| Area | Finding |
|---|---|
| Colour | 2,297 uses of Tailwind's default palette (`gray`, `neutral`, `amber`, `violet`...) vs 486 uses of the shadcn semantic variables vs the custom palette in `styles/colors.ts`. 171 raw hex classes. 47 files mix two systems. |
| Radius | Custom radius tokens are about 5% of radius usage (98 vs 1,455 Tailwind defaults and 184 arbitrary values). Five design-system components use them. |
| Typography | 1,060 raw `text-*` size classes and 598 raw `font-*` weight classes in `src/app`, 409 raw `<h*>`/`<p>` with classes. About 90 `!text-zinc-*` overrides fight the `Text` atom's colour. |
| Dark mode | 627 `dark:` classes in 102 files. All dead: the theme provider forces light, and the Tailwind dark selector (`.dark-mode`) does not even match the CSS variable block (`.dark`). |
| Legacy | 115 non-legacy files still import `__legacy__`. 12 design-system atoms and molecules themselves depend on `__legacy__` or `ui/`. 21 legacy files (1,547 lines) are dead. |
| Button | `disabled:opacity-1` appears seven times and emits no CSS. Disabled buttons are double-dimmed. `rounded-[96px]`, `border-[#a6a6a6]`, `min-w-[7.7rem]` on every size. |
| Tooling | No Tailwind lint, no import restrictions, `cn()` cannot dedupe custom radii or shadows, Chromatic is `if: false` in CI, the Storybook test runner is not installed, knip is non-blocking. |
| Docs | The contributing guide references a folder that does not exist (`_legacy__`), promises Chromatic checks that never run, and asks for dark-mode consistency the config says to ignore. No document lists the tokens. |

---

## Part 1: Token layer

### 1.1 Colour: three systems and no owner

**System A, the custom palette** (`src/components/styles/colors.ts`, spread into `tailwind.config.ts:78`). It overrides Tailwind's `slate`, `zinc`, `red`, `orange`, `yellow`, `green`, `purple`, `pink`, and adds `white`, `black`, `textGrey`, `textBlack`, `bgLightGrey`.

Problems inside the file itself:

- `slate.700` and `slate.800` are both `#475263` (`colors.ts:10-11`). 53 class usages are visually identical and the slate ramp has no step between 600 and 900.
- `red.900` is `"#641D1D "` with a trailing space (`colors.ts:36`). Tailwind emits it verbatim.
- `yellow.25` and `yellow.150` exist only because one file wanted them (`organisms/PendingReviewsList/PendingReviewsList.tsx:219`). `pink.100` has zero uses. `orange.800`/`900` are each used once.
- `textGrey` (`#505057`) and `textBlack` (`#1F1F20`) are named colours that duplicate `zinc.700` and a near-black, bypassing both the zinc scale and the `Text` tones. `textBlack` is used 83 times.
- `white` is `#fefefe` and `black` is `#141414`. Every `bg-white` in the app is therefore off-white, which is fine, but it is undocumented and the Link atom hardcodes `#141414` through a CSS variable that nothing defines (`atoms/Link/Link.tsx:14`, `text-[var(--AutoGPT-Text-text-black,#141414)]`).
- Because the palette is applied through `extend`, shades not listed (e.g. `zinc-950`) fall through silently to Tailwind's defaults. `text-zinc-950` is used in `ToggleChip` and `SearchCommandModal`.

**System B, the shadcn HSL variables** (`src/app/globals.css:7-76`, mapped in `tailwind.config.ts:79-105`). `background`, `foreground`, `card`, `popover`, `primary`, `secondary`, `muted`, `accent`, `destructive`, `border`, `input`, `ring`, `sidebar-*`, `chart-*`.

- `--accent` is `262 83% 58%` (a violet), while the brand colour in the custom palette is `purple-500 #7733f5`. Two different purples are both "the accent".
- `--background` is `0 0% 98%` (`#FAFAFA`) but `body` is painted `bg-[#F6F7F8]` (`globals.css:82`). The page background is a hex literal that disagrees with its own token.
- `--chart-1..5` (10 declarations across light and dark) have zero references.
- `components.json` says `cssVariables: false`, which is false.
- `text-muted-foreground` is the most used semantic class (189 uses) while the custom palette has no "muted" concept at all; the `Text` atom's `tone="muted"` maps to `zinc-500`, a different grey.

**System C, Tailwind's default palette.** Nothing stops it, so it won. 2,297 occurrences:

| Family | Uses | Files | Inside atoms/molecules/organisms |
|---|---|---|---|
| `neutral-` | 916 | 151 | 62 |
| `gray-` | 639 | 111 | 71 |
| `amber-` | 204 | 73 | 6 |
| `blue-` | 197 | 72 | 25 |
| `violet-` | 180 | 75 | 2 |
| `emerald-` | 76 | 37 | 5 |
| `rose-` | 41 | 17 | 0 |
| `indigo-` | 26 | 21 | 5 |
| `sky-`, `teal-`, `lime-`, `stone-`, `cyan-`, `fuchsia-` | 18 | | 1 |

`customGray-100..700` in `tailwind.config.ts:107-115` has zero uses anywhere.

**Hex literals.** 171 arbitrary hex classes in 73 files, plus 261 hex literals in `.ts` constants. The same unregistered greys recur across the new admin and settings shells: `#F9F9FA` (6+ files), `#DADADC`, `#EFEFF0`, `#505057`, `#1F1F20`, `#8A8A90`. These are a de-facto grey scale that nobody added to the palette. Two colour maps for node types exist in parallel: `src/lib/utils.ts:81-95` and `build/components/FlowEditor/nodes/helpers.ts:100-197`.

**Same meaning, different colour** (design-system components only):

- Default border: `zinc-200`, `zinc-300`, `#a6a6a6`, `zinc-700`, `zinc-100`, `zinc-200/80`, `border-input`, `border-border`, `neutral-200`, `gray-200`, `gray-300`, bare `border` (Tailwind gray-200), `ring-1 ring-zinc-200`.
- Muted text: `zinc-400`, `zinc-500`, `zinc-600`, `muted-foreground`, `gray-500/600/700`, `neutral-500`, `neutral-600`, `textGrey`, `stone-800`.
- Primary text: `black`, `zinc-900`, `zinc-800`, `zinc-950`, `textBlack`, `gray-900`, `neutral-950`, `foreground`.
- Error: `red-500`, `red-600`, `red-700`, `red-400`, `red-300`, `destructive`, Toast `#f26969`, Alert `#FDECEC80`.
- Success: `emerald-*` (Badge, BillingToggle), Toast `#45b56e`, StatusDot `#22A05B`. The `green` token is used by zero design-system components.
- Warning: `amber-*` (Badge), `yellow-300` border with `orange-600` icon and `#FFF3E680` background (Alert), `yellow-25/150/600` (PendingReviewsList), Toast `#e77b00`.
- Brand: `purple-400`, `purple-500`, `purple-600`, `purple-700/100/50` pills, `purple-800 to purple-600` gradient, `purple-500 to indigo-500` gradient, `violet-50`, `#a78bfa`, `#ddd6fe`, `rgba(168,85,247)`, `#7C3AED`, shadcn `accent`. The primary Button is `bg-zinc-800`, not purple.

### 1.2 Spacing

`tailwind.config.ts:127-170` re-declares Tailwind's entire default spacing scale 0 to 96 with identical values. The only real additions are `18`, `68`, `70`, `71`, `76`, `4.5`, `7.5`, `8.5`. Of those, `70`, `7.5` and `8.5` have zero uses and `71` is used once. The spacing story (`tokens/spacing.stories.tsx:16-72`) hardcodes 35 rows and lists none of the eight additions.

Arbitrary sizes in `src/`: 886 occurrences in 370 files. The same control height is written as `h-[2.625rem]` (17), `h-[46px]` (10) and `h-[2.875rem]` (9). `h-[2.25rem]` (17) is `h-9`. `w-[12rem]` (11) is `w-48`. `h-[3px]` and `p-[1px]` have built-ins. `max-w-[1360px]` appears 18 times with no container token.

### 1.3 Radius

Two radius vocabularies are defined side by side (`tailwind.config.ts:171-182`):

| Token | Value | Tailwind default with the same value |
|---|---|---|
| `rounded-xsmall` | 4px | `rounded-sm` (via `--radius` minus 4px) |
| `rounded-small` | 8px | `rounded-lg` |
| `rounded-medium` | 12px | `rounded-xl` |
| `rounded-large` | 16px | `rounded-2xl` |
| `rounded-xlarge` | 20px | none |
| `rounded-2xlarge` | 24px | `rounded-3xl` |

Usage in `src/`: custom tokens 98 (22 of those are in the token stories), Tailwind defaults 1,455, arbitrary `rounded-[...]` 184. The most common arbitrary values are `rounded-[18px]` (38), `rounded-[10px]` (22), `rounded-[0.5rem]` (20, equals `rounded-lg`), `rounded-[8px]` (16), `rounded-[0.75rem]` (16, equals `rounded-xl`). The custom scale is effectively abandoned. Nine files mix both vocabularies.

Within the design system, five places use the tokens (`Card`, `FileInput` twice, `LLMItem`, `Dialog` styles). Everything else is Tailwind defaults: Button `rounded-full` / `rounded-[96px]` / `rounded-md` / `rounded-lg`; Input and Select `rounded-xl`; DateInput, DateTimeInput and TimeInput `rounded-3xl` (so a form has 12px fields next to 24px fields); TrialCard `rounded-[18px]` / `rounded-[15px]` / `rounded-2xl` in one card.

### 1.4 Typography

The `Text` atom (`atoms/Text/helpers.ts`) is the only typography token source. Every variant is an arbitrary value (`text-[0.875rem] font-[400] leading-[1.375rem]`), so nothing in Tailwind's `fontSize` theme carries the scale and the values cannot be used outside the atom.

- `label` variant is `text-[0.6785rem]` with `tracking-[0.06785rem]` (`helpers.ts:46`). That is 10.86px, almost certainly a typo of `0.6875rem` (11px).
- `h4` is 22px on a 24px line (ratio 1.09) while `h5` is 16px on 24px. `h2` and `h3` share a 40px line height.
- `eyebrow` is `text-zinc-500` while every other variant is `text-black` (`helpers.ts:46-48`).
- `Text` accepts both `variant` and an undocumented `size?: Variant` alias (`Text.tsx:15,40`).
- `As` allows `label`, `kbd`, `li`, `dt`, `dd` but props are typed as `<p>` attributes (`Text.tsx:27`), so `htmlFor` does not type-check.
- No `h6` variant. The `label`/`eyebrow` variants are used by one design-system component while four others hand-roll uppercase tracked labels.
- `unmask` defaults to `true` (`Text.tsx:37`, also `Button`), so user content rendered through `Text` without `unmask={false}` is unmasked in Sentry replays. Examples: `organisms/NeedsAttentionList/components/NeedsAttentionRow.tsx:47,51`, `molecules/InstallWorkflowPicker/InstallWorkflowPicker.tsx:93,97,167,195`, `molecules/Table/Table.tsx:71-76`.

Outside the atom, `src/app` has 1,060 `text-xs..5xl` and 598 `font-*` weight classes, 335 raw `<p className>` and 74 raw `<h1..h6 className>`. `text-[13px]` (71), `text-[11px]` (59), `text-[10px]` (30) and `text-[15px]` (19) are sizes the scale does not have, so people invent them. The `!text-zinc-500` override alone appears 49 times, `!text-zinc-800` 10, `!text-zinc-700` 7, `!text-zinc-400` 7. This is the clearest signal that the `Text` colour API (four tones) is insufficient and that `text-black` as the default is wrong for most body copy.

### 1.5 Shadows and elevation

- `shadow-subtle` is the only shadow token (`tailwind.config.ts:184`). It is used 4 times in `src/` and by zero design-system components.
- Meanwhile three hand-written near-copies exist: `shadow-[0_1px_2px_rgba(15,15,20,0.04)]` (29 uses), `shadow-[0_1px_2px_rgba(0,0,0,0.03)]` (IntegrationsMarquee), and the kbd inset `shadow-[inset_0_-1px_0_rgba(15,15,20,0.04),0_1px_1px_rgba(15,15,20,0.04)]` (6 uses). 88 arbitrary `shadow-[` in total.
- `smooth-shadow-ring-sm` is a custom plugin (`tailwind.config.ts:12-31`) used by 23 call sites but only one design-system component (BriefingCard).
- The `grain-overlay` plugin (`tailwind.config.ts:40-52`), including its inlined SVG data URI, has zero uses.
- Design-system elevation is `shadow-sm` (Button secondary, Switch), `shadow-md` (Card, Popover, DropdownMenu, SecondaryMenu), `shadow-lg` (Switch thumb, DropdownMenuSubContent), `shadow-2xl` (FloatingReviewsPanel, SearchCommandModal), and Tooltip uses `outline-gray-100` instead of a shadow. No elevation scale exists.
- `.agpt-shadow-input` from the legacy input base is cancelled immediately by the atom Input's `shadow-none` (`atoms/Input/Input.tsx:104`).

### 1.6 Motion

- Dead keyframes/animations in `tailwind.config.ts`: `loader` (0 uses), `marquee-x` (0, IntegrationsMarquee animates with framer-motion instead), `caret-blink` (0). `duration-400` and `duration-2000` are each used once.
- Invalid class names in design-system components that silently do nothing: `animate-fadein` (`molecules/Dialog/components/styles.ts:12`, the real name is `fade-in`, so dialog content never animates in), `transition-left transition-right` (`molecules/TabsLine/TabsLine.tsx:94`, `molecules/ScrollableTabs/components/ScrollableTabsList.tsx:40`, so the tab underline snaps instead of sliding).
- Framer durations across the system: 0.12, 0.15, 0.2, 0.22, 0.32, 0.4, 0.45 seconds with four different cubic-bezier curves. Tailwind durations: 150, 200, 300. No motion tokens.
- `useReducedMotion` is honoured by Reveal, SwapFade, ToggleChip, IntegrationsMarquee, BriefingCard; ignored by FadeIn, TypingText, GlassOrb and Confetti.
- `framer-motion` 13.3.0 and `motion` 13.2.0 are both installed. They are the same library under two names at mismatched versions; `motion` has two importers (`ui/vortex.tsx`, `ui/text-generate-effect.tsx`).

### 1.7 Fonts

Two parallel definitions that must be kept in sync by hand:

- `src/components/styles/fonts.ts` uses `next/font` (Poppins from Google, Geist from the `geist` package). This is what the app uses (`src/app/layout.tsx:73`).
- `src/components/styles/fonts.css` does a Google Fonts `@import` for Poppins, declares `@font-face` for Geist from `node_modules`, and redefines `--font-poppins`, `--font-geist-sans`, `--font-geist-mono` on `:root` with different fallback stacks. Only Storybook imports it (`.storybook/preview.tsx:14`).

So Storybook renders with a network-loaded Poppins and different fallbacks from production. The comment at `fonts.css:23` ("matching config from fonts.ts") is the only thing binding them.

`font-poppins` is used 20 times outside the `Text` atom, 11 of them in `__legacy__` with sizes like `text-[48px] leading-[54px]` and `text-[35px]`.

### 1.8 Dark mode: wired three ways, works zero ways

- `src/app/providers.tsx:43` renders `<ThemeProvider forcedTheme="light">`. Dark is unreachable.
- `tailwind.config.ts:55` sets `darkMode: ["class", ".dark-mode"]`, so `dark:` utilities compile to `:is(.dark-mode *)` (verified by compiling).
- `globals.css:43` defines the dark CSS variables under `.dark`, which is what `next-themes` with `attribute="class"` would add.
- The `smoothShadowRing` plugin covers `.dark, .dark-mode, [data-theme="dark"]` (`tailwind.config.ts:21`), a third convention.
- `__legacy__/ThemeToggle.tsx` is the only `useTheme` consumer and has zero importers.

Result: 627 `dark:` occurrences across 102 files (217 in `__legacy__`, 229 in `src/app`, 45 in `ui/`, 42 in molecules, 23 in atoms) and a 33-variable `.dark` block are dead weight, and `frontend/AGENTS.md:49` says "No `dark:` Tailwind classes" while `CONTRIBUTING.md:622` says to keep dark-mode behaviour consistent.

### 1.9 `globals.css`

- `.agpt-rounded-box`, `.agpt-box`, `.agpt-div`, `.agpt-card-selected` have zero uses.
- `.agpt-card` is used by one file (`__legacy__/ui/card.tsx`). `.agpt-rounded-card` by `__legacy__/Button.tsx`, whose only live consumer chain is `__legacy__/Sidebar.tsx`.
- `.agpt-border-input` (7 uses) is where the legacy inputs, and through them the atom `Input`, get `m-0.5` and a `gray-400` focus ring. The atom then overrides the ring with `purple-400`, but the `m-0.5` outer margin survives.
- `body` uses `bg-[#F6F7F8]` against a `--background` of `#FAFAFA` (see 1.1).
- The file also carries Google Picker z-index hacks with `!important`, streamdown table padding, and KaTeX sizing. None of it is tokenised.

---

## Part 2: Component layer

### 2.1 Three component layers, one of them pretending to be dead

| Layer | Files | Lines | Importers outside the layer |
|---|---|---|---|
| `src/components/{atoms,molecules,organisms}` | 124 non-story | | the intended design system |
| `src/components/ui/` (shadcn + Aceternity) | 18 | 2,372 | 58 files (37 in `src/app`, 24 of those in copilot) |
| `src/components/__legacy__/` | 54 | 6,566 | 115 files, 191 import lines |

The `ui/` folder is not mentioned in any guideline. It was created by shadcn's CLI (`components.json` registers an `@aceternity` registry) and it has been growing: `aurora-background.tsx`, `collapsible.tsx`, `dot-distortion-shader.tsx`, `text-generate-effect.tsx`, `vortex.tsx` were all added in the last 180 days. It is a third, undocumented component source that reproduces the legacy folder:

- `ui/separator.tsx` is byte-identical to `__legacy__/ui/separator.tsx`.
- `ui/button.tsx` differs from `__legacy__/ui/button.tsx` by 5 lines; `ui/skeleton.tsx` by 5; `ui/sheet.tsx` by 4; `ui/collapsible.tsx` by 2.
- `ui/sidebar.tsx` is 790 lines, the single largest component in the tree, and the anchor keeping `ui/button`, `ui/input`, `ui/separator`, `ui/sheet`, `ui/skeleton` and `ui/tooltip` alive. It is also the main consumer of the `sidebar-*` colour tokens.

Live duplicates by concept:

| Concept | Implementations |
|---|---|
| Button | `atoms/Button` (305 app files), `__legacy__/ui/button` (29), `ui/button` (8), `__legacy__/Button.tsx` (legacy-only) |
| Skeleton | `atoms/Skeleton` (71), `__legacy__/ui/skeleton` (23), `ui/skeleton` (5) |
| Tooltip | `atoms/Tooltip/BaseTooltip` (60), `ui/tooltip` (12, including the `ToggleChip` atom) |
| Dialog | `molecules/Dialog` (113), `__legacy__/ui/dialog` (10), `molecules/FullscreenDialog`, raw Radix in `SearchCommandModal` |
| Input | `atoms/Input`, `__legacy__/ui/input` (14, including the atom itself), `ui/input` (3) |
| Textarea | `__legacy__/ui/textarea` (4), `ui/textarea` (2), `atoms/Input type="textarea"`, a raw `<textarea>` in VoicePicker |
| Select | `atoms/Select` wraps `__legacy__/ui/select` (10) |
| Popover | `molecules/Popover`, `__legacy__/ui/popover` (used by the Date atoms) |
| Collapsible / Accordion | `molecules/Collapsible` wraps `__legacy__/ui/collapsible`; `ui/collapsible`; `molecules/Accordion` is a pure re-export of `ui/accordion`; raw Radix accordion in `file-tree` |
| Dropdown | `molecules/DropdownMenu`, `molecules/SecondaryMenu` (different radius, colours, z-index 10 vs 50), `__legacy__/ui/dropdown-menu` |
| Tabs | `molecules/TabsLine` (15), `molecules/ScrollableTabs` (1, with a verbatim copy of TabsLine's trigger classes), `__legacy__/ui/tabs` (1) |
| Table | `molecules/Table` wraps `__legacy__/ui/table`; raw `<table>` markup in `WorkOutputSheet` and `PlanComparison`; `__legacy__/ui/data-table` (dead) |
| Avatar | `atoms/Avatar`, `molecules/ExpertAvatar`, `molecules/AutopilotAvatar`, `molecules/NotionAvatar` (dead), `molecules/WorkflowAvatar`, plus `WorkflowTile` and `IntegrationLogo` |
| Time / date | `atoms/TimeInput` (1 consumer) and `molecules/TimePicker` (2); `atoms/DateInput` (1) and `atoms/DateTimeInput` (2) which re-implements the time input inline |
| Spinner | `atoms/LoadingSpinner`, `ui/spinner`, `__legacy__/ui/loading`, and hand-rolled spinners in Button, FileInput (twice), ErrorCard `LoadingState`, SearchInput |
| ShowMore | `molecules/ShowMore` (0 consumers) and `molecules/ShowMoreText` (2) with byte-identical `helpers.ts` |
| Checkbox | `__legacy__/ui/checkbox` only (6 consumers). No atom exists. |

### 2.2 The design system depends on the thing it replaces

Twelve atoms and molecules import from `__legacy__` or `ui/`, and several import from feature code:

- `atoms/Input/Input.tsx:4` and `useInput.ts:1` wrap `__legacy__/ui/input`
- `atoms/Select/Select.tsx:10` wraps `__legacy__/ui/select` (whose content panel uses `neutral-*`)
- `atoms/DateInput/DateInput.tsx:5,11,12` uses `__legacy__/ui/button`, `popover`, `calendar`
- `atoms/DateTimeInput/DateTimeInput.tsx:13,14` uses `__legacy__/ui/popover`, `calendar`
- `atoms/ToggleChip/ToggleChip.tsx:8` uses `ui/tooltip` (dark style) instead of the atom tooltip (white style)
- `atoms/FileInput/FileInput.tsx:7,8` imports a molecule (`Dialog`) and feature code (`contextual/OutputRenderers`)
- `atoms/Input` and `atoms/Select` import `molecules/InformationTooltip` (atom depends on molecule)
- `atoms/Button/helpers.ts:1-5` imports class strings from `atoms/Link`
- `molecules/Accordion/Accordion.tsx:3-8` is a barrel re-export of `ui/accordion`
- `molecules/Collapsible/Collapsible.tsx:9` wraps `__legacy__/ui/collapsible`
- `molecules/Dialog/components/DrawerWrap.tsx:1` uses `__legacy__/ui/button` while the sibling `DialogWrap` uses the atom Button
- `molecules/Table/Table.tsx:9` wraps `__legacy__/ui/table`
- `molecules/file-tree.tsx:13,14` uses `ui/button`, `ui/scroll-area`
- `molecules/ErrorCard/helpers.ts:105,116` calls `sonner` directly, bypassing `molecules/Toast`
- `organisms/SearchCommandModal/*.tsx` uses `ui/button`, `ui/input`, `ui/separator`
- `organisms/WorkOutputSheet/WorkOutputSheet.tsx:14` uses `ui/sheet`
- `organisms/PendingReviewCard/components/ReviewInputFields/ReviewInputFields.tsx:1` imports from `app/(platform)/library/agents/[id]/...`
- `organisms/FloatingReviewsPanel/FloatingReviewsPanel.tsx:10` imports the builder page's Zustand store

Deleting `__legacy__` is therefore impossible today without first rewriting Input, Select, DateInput, DateTimeInput, Collapsible, Table and DrawerWrap.

### 2.3 Button

`atoms/Button/helpers.ts` is the most-used component and has the most defects:

- `disabled:opacity-1` on seven variants (lines 18, 20, 22, 24, 26, 27, 31). There is no `opacity-1` step in Tailwind and the config adds none. Verified by compiling: no CSS is emitted. The base `disabled:opacity-50` (line 13) still applies, so every disabled button gets the explicit disabled colours **and** 50% opacity. This is a live visual bug.
- `border-[#a6a6a6]` for the outline variant (lines 24, 52). Not in the palette.
- `rounded-[96px]` for the icon variant (line 27) where every other pill uses `rounded-full`.
- `min-w-[7.7rem]` (123px) on the base class applies to every size including `xs`, `icon` and `link`. Consequence: organisms override it everywhere (`organisms/ApprovalFields/ApprovalFields.tsx:75`, `components/FieldValue.tsx:121`, `organisms/SubscriptionPlans/components/OfferActions.tsx:76,100,129`, `PlanFooter.tsx:23`) with `h-auto min-w-0 p-0`.
- `focus-visible:ring-neutral-950` uses a non-token grey.
- Size scale is `small | large | icon | xs | icon-xs | icon-sm`. `icon` is `p-3` with no fixed box while `icon-xs`/`icon-sm` are `size-7`/`size-8`. `variant="icon"` and `size="icon"` are separate axes.
- Loading recolours every non-ghost variant to `bg-zinc-500` (`Button.tsx:113-116`), so a loading secondary button turns dark grey. Loading drops `rightIcon` (line 78).
- Three icon props with different types: `leftIcon: ReactNode`, `rightIcon: ReactNode`, `leadingIcon: IconSvgElement`.
- `asChild` is accepted and discarded (`helpers.ts:87`, `Button.tsx:31`).
- The `link` variant inherits `Link`'s `focus:ring-2 focus:ring-blue-500` (`atoms/Link/Link.tsx:25`), which is `focus:` not `focus-visible:` and a non-token blue.

### 2.4 Form inputs: five copies of one style block

`Input`, `Select`, `TimeInput`, `DateInput` and `DateTimeInput` each re-declare the base field style and the size array (`h-[2.25rem]`/`h-[2.875rem]`, `leading-[22px]`, `py-2`/`py-2.5`), plus the "always render an error line with a space" block. Differences that crept in:

| | Input | Select | TimeInput | DateInput | DateTimeInput | SearchInput |
|---|---|---|---|---|---|---|
| Radius | `rounded-xl` | `rounded-xl` | `rounded-3xl` | `rounded-3xl` | `rounded-3xl` | `rounded-md` / `rounded-xl` |
| Focus ring | `purple-400` | `zinc-400` | `zinc-400` | `zinc-400` | `zinc-400` | `ring` (shadcn) |
| Size prop | `small\|medium` | `small\|medium` | `small\|medium` | `default\|small` | `default\|small` | `xsmall\|small\|medium` |
| Label variant | `large-medium` | | `body-medium` | raw `<label>` `text-gray-700` | raw `<label>` | |
| Error colour | `red-500` | `red-500` | `red-500` | `red-500` raw span | `red-500` | |
| Disabled | `opacity-50` | | | | | `opacity-60` |
| Reserved error margin | always `mb-6` | | always `mb-6` | none | only when error (layout shift) | |
| `wrapperClassName` | applied to two nested divs (`Input.tsx:232,256`) | error wrapper only | two divs | | | |

Other defects in this group:

- `!placeholder:text-zinc-500` (`DateInput.tsx:89`, `DateTimeInput.tsx:139`, `Select.tsx:76`) puts the important marker before the variant, which is invalid in Tailwind 3; the class is dead.
- `border-1.5` (`DateInput.tsx:94`, `DateTimeInput.tsx:144`, `Select.tsx:89`) is not in the config; dead.
- `Select` fires `option.onSelect` from `onMouseDown` with `preventDefault` (`Select.tsx:122-127`), so keyboard selection never triggers it. The whole field is a `<label>` wrapping the `InformationTooltip` button (nested interactive, `Select.tsx:157-170`).
- `placeholder={placeholder || label}` in Input, Select and TimeInput uses the label as placeholder, which is an accessibility anti-pattern.
- No `aria-invalid` or `aria-describedby` link from any field to its error text.
- `Input`'s props type is named `TextFieldProps` (`Input.tsx:18`). `as any` twice (`Input.tsx:189,190`).
- `DateInput` places `className` before the size classes (`DateInput.tsx:100-106`) so callers cannot override; `DateTimeInput` places it last.
- `DateTimeInput` re-implements the time input inline (`DateTimeInput.tsx:159-177`) instead of using `TimeInput`.
- `TimePicker` uses static ids `time-hour`, `time-minute`, `time-meridiem` (`TimePicker.tsx:26,47,62`) so two pickers collide, declares `className` and never applies it (`TimePicker.tsx:6,9`).
- `FileInput` (517 lines, no hook split) uses `blue-*` and `gray-*` throughout, three `Button size="small"` overridden to `h-7 w-7 min-w-0 p-0` instead of `size="icon-xs"`, two different clear icons, hand-rolled spinners, `console.error` (`FileInput.tsx:282`), and `getFileLabelFromValue` duplicates `helpers.getFileLabel`.

### 2.5 Size vocabularies

| Component | Prop vocabulary | Rendered |
|---|---|---|
| Button | `small / large / xs / icon / icon-xs / icon-sm` | 36 / 46 / 28 / no box / 28 / 32 px |
| Input, Select, TimeInput | `small / medium` | 36 / 46 |
| DateInput, DateTimeInput | `default / small` | 46 / 36 |
| SearchInput | `xsmall / small / medium` | 28 / 36 / 46 |
| Badge | `small / medium` | 20 / 24 |
| LoadingSpinner | `small / medium / large` | 16 / 24 / 40 |
| Avatar | none (className only) | 40 |
| ExpertAvatar, AutopilotAvatar, NotionAvatar, IntegrationLogo, Emoji, InformationTooltip | `size: number` | 40 / 24 / 96 / 16 / 24 / 24 |
| WorkflowAvatar | `18 \| 36` literal union | |
| Text | `variant` and `size` alias | |
| Dialog, TabsLine, FileInput | `variant: default \| compact` | |
| ExpertIdentityDetails | `size: compact \| card \| page` | layout, not size |
| VoicePicker, ExpertTagline | `compact?: boolean` | |
| `ui/button` (in SearchCommandModal, file-tree) | `default / sm / lg / icon / icon-sm` | 36 / 32 / 40 / 36 / 32 |

"small" means 36px or 28px depending on the component. "compact" is a variant, a size, and a boolean.

### 2.6 Focus rings

Thirteen distinct focus treatments inside the design system:

| Ring | Where |
|---|---|
| `ring-1 ring-neutral-950` | Button, Switch (`ring-2 offset-2`) |
| `ring-1 ring-purple-400` | Input |
| `ring-1 ring-zinc-400` | Select, TimeInput, DateInput, DateTimeInput |
| `ring-2 ring-blue-500 offset-2` (`focus:` not `focus-visible:`) | Link, Button link |
| `ring-2 ring-purple-600 offset-2` | MultiToggle |
| `ring-2 ring-zinc-300` | InformationTooltip, ApprovalFields |
| `ring-neutral-400` | TabsLine trigger, ScrollableTabs trigger |
| `ring-stone-400` | TabsLine content (same file, different grey) |
| `ring-ring` (shadcn) | SearchInput, VoicePicker |
| `ring-gray-400` via `.agpt-border-input` | FileInput, legacy inputs |
| `ring-0` | SearchCommandModal input; `!focus-visible:ring-0` in DrawerWrap is mis-ordered and dead |
| none | ToggleChip, ShowMore, ShowMoreText, BriefingCard toggle, PendingReviewsList toggle, SecondaryMenu items |

Width 1 vs 2, offset 0 vs 2, and `focus:` vs `focus-visible:` all vary.

### 2.7 Disabled states

Button: base `disabled:opacity-50` plus broken `disabled:opacity-1` (see 2.3). Switch, legacy inputs, Input password toggle: `cursor-not-allowed opacity-50`. SearchInput: `opacity-60`. MultiToggle: `disabled:opacity-50` **and** explicit `border-zinc-200 text-zinc-200 opacity-50`. ToggleChip `locked`: `opacity-70` with `aria-disabled` but `onClick` still fires (`ToggleChip.tsx:46-47`). Button link loading `opacity-60` vs disabled `opacity-50`.

### 2.8 Props API

- `className`: `TimePicker` declares and ignores it; `AutoGPTLogo`'s `className ?? "h-10 w-[5.5rem]"` drops the default size when any class is passed; `Emoji`, `GlassPixelBackdrop`, `RunStatusBadge`, `NeedsAttentionList` have none. Extras in circulation: `wrapperClassName`, `labelClassName`, `triggerClassName`, `contentClassName`, `itemWrapperClassName`, `toggleClassName`, `indicatorClassName`, `areaClassName`. `Dialog` takes `styling: CSSProperties` instead of `style`.
- Variant words for "error": `destructive` (Button, Toast, SecondaryMenu), `error` (Badge, Alert), `danger` (Text tone).
- Loading booleans: `loading` (Button, SearchInput) vs `isLoading` (SearchCommandModal, WorkOutputSheet, ReviewInputFields) vs `isSubmitting`, `isStarting`, `isCanceling`, `isProcessing`, `isFetchingMore`.
- `readonly` (DateInput, DateTimeInput) vs `readOnly` (Table). `manualstart` (Confetti). `ariaLabel` camelCase (ToggleChip) vs `"aria-label"` elsewhere.
- Open/close: Collapsible `open/defaultOpen/onOpenChange`; Dialog `controlled:{isOpen,set}` plus `forceOpen` plus `onClose`; InstallWorkflowPicker and SearchCommandModal `open|isOpen` plus `onClose`; WorkOutputSheet `open/onOpenChange`; FullscreenDialog always open with `onClose`.
- Icon props: `ReactNode` (ToggleChip, SelectOption, Button left/right), `IconSvgElement` (Button leading, TabsLine), `ComponentType<{className}>` (Alert, SearchCommandModal, documented as "Phosphor / lucide / heroicons").
- Props naming: `Props` is the rule, yet 14 components export suffixed types (`ButtonProps`, `TextFieldProps`, `SelectFieldProps`, `AvatarProps`, `DateInputProps`, `ProgressProps`, `TextProps`, `ErrorCardProps`, `TableProps`...). `type` vs `interface` is mixed.
- Convention violations: arrow-function or `React.FC` components in `DateInput`, `DateTimeInput`, `TimeInput`, `Form`, `DropdownMenu`, `Alert`, `Popover`, `Progress`, `Switch`, `BaseTooltip`, `SecondaryMenu`, `TabsLine`, `file-tree`; 17 `useCallback`/`useMemo`; `any` in `Input`, `use-toast`, `PendingReviewsList`; `@ts-ignore` x4 in `ScrollableTabs` and `TabsLine`; `console.error` in `FileInput`, `ErrorCard/helpers`, `useTallyPopup`.

### 2.9 Molecules and organisms, notable items

- **Dialog**: `text-md` is not a class (`styles.ts:3`); `animate-fadein` is not defined (`styles.ts:12`); `bg-stone-500/20`, `text-stone-800`; sr-only title and description are the literal word "Dialog" (`DialogWrap.tsx:134,153`, `DrawerWrap.tsx:96`); `BaseTrigger` overwrites the child's own `onClick` (`BaseTrigger.tsx:11`); `withGradient` and `testId` props are never used; `DrawerWrap` uses `hover:bg-gray-200 dark:hover:bg-gray-700` in a template literal.
- **Alert**: hex with alpha `bg-[#FFF3E680]`, `bg-[#FDECEC80]`; lucide icons; warning mixes a yellow border with an orange icon; `AlertTitle` is an `<h5>` typed as a paragraph; `role="alert"` on the default variant; no success/info variants.
- **Badge**: `emerald`/`amber` instead of `green`/`yellow` tokens; `text-[11px]`; `overflow-hidden text-ellipsis` on an `inline-flex` never truncates.
- **Toast**: kebab-case files; a CSS module with six hex colours and `!important` on every rule; a `toastWarning` style that no variant can reach; `useToast` returns a fake `toasts: []`; `(error: any)`.
- **Tooltip**: folder `Tooltip`, file `BaseTooltip.tsx`, export `Tooltip`; not portalled by default (`BaseTooltip.tsx:31-33`) so Button icon tooltips clip inside `overflow-hidden` ancestors like `Table`; delay 10ms here, 400ms in InformationTooltip, 300ms in ReferenceLink.
- **Collapsible**, **ShowMore**, **ShowMoreText**: `flex-end`/`flex-start` are not Tailwind classes (`Collapsible.tsx:57`, `ShowMore.tsx:49-50`, `ShowMoreText.tsx:46-47`). ShowMoreText's toggle is a 16px-tall button labelled "..." with no `aria-label` or `aria-expanded`.
- **TabsLine / ScrollableTabs**: dead transition classes (see 1.6); `ring-neutral-400` and `ring-stone-400` in one file; one `MutationObserver` per trigger; `document.querySelectorAll` global lookup in ScrollableTabs breaks with two instances; no `role="tab"`/`aria-selected` in ScrollableTabs.
- **SearchCommandModal**: the kbd block (200-char className with an arbitrary inset shadow and `text-[11px]`) is pasted five times; a `Kbd` atom is missing. `z-[80]` vs Dialog `z-50`. Overlay `bg-black/20 backdrop-blur-sm` vs Dialog `bg-stone-500/20 backdrop-blur-md`.
- **TrialCard**: `Text variant="h5"` forced to `font-poppins !text-[17px] !font-semibold !text-zinc-800` (`components/TrialTitle/TrialTitle.tsx:12`); five `!text-zinc-*` overrides that should be `tone`; a radial gradient in `rgba(168,85,247)` and `rgba(99,102,241)` (Tailwind purple-500/indigo-500, not the palette); three radii in one card.
- **SubscriptionPlans**: Card's every default overridden (`SubscriptionOffer.tsx:26`); three hand-rolled pills instead of `Badge`; raw `<a>` instead of `Link` (`PlanFooter.tsx:13-18`); Buttons nested inside a `<p>` via `Text` (`OfferActions.tsx:95-107`); `to-indigo-500` gradient; currency formatting duplicated with `TrialCard/helpers.ts`.
- **PendingReviewCard / PendingReviewsList**: `gray-*` throughout, `textBlack`, `textGrey`, `yellow-25`, `yellow-150`; `getShortenedNodeId` and `isPlainObject` duplicated between them (and `isRecord` in WorkOutputSheet); 363-line component with no hook split and `let parsedData: any`; Switches without a label association; raw collapse buttons without `aria-expanded`; a destructive Button forced to `bg-red-600` over the atom's `red-500`.
- **VoicePicker**: the only organism using shadcn `accent`; a raw `<h2 className="text-foreground text-2xl font-semibold">` heading bypassing `Text` and Poppins; its own `<textarea>`; `has-[:focus-visible]:ring-2` plus `focus-within:ring-ring` double ring.
- **WorkOutputSheet**: raw `<img>`, raw `<table>` with inline cell styles, six components in one file with mixed inline and interface prop types.
- **FloatingReviewsPanel**: close button with no accessible name (`FloatingReviewsPanel.tsx:133-140`); `rounded-lg shadow-2xl` panel with no background class; imports the builder's Zustand store.
- **ErrorCard**: `components/LoadingState.tsx` is never imported; `shouldShowError` unused; `handleReportError` calls `sonner` directly; `CardWrapper` builds a className with a template literal and an inline gradient from `colors.zinc` under a comment that says "Purple gradient border".
- **NotionAvatar**: 10 files plus a 279 KB `parts.generated.ts`; `ExpertAvatar/helpers.ts:27` redirects every Notion avatar path to the default, so the module is a retired code path still shipped; knip reports the three component files unused.
- **ExpertAvatar**: `color` prop declared and never used (`ExpertAvatar.tsx:17`), yet `ReferenceCard.tsx:23` passes it. Managed identities render `rounded-xl` while custom ones render `rounded-full`.
- **IntegrationsMarquee**: `h-[200px] w-[340px]`, `h-[58px] w-[180px]`, raw `<img>` with an eslint-disable, three components in one file.
- **TallyPoup**: folder typo; file `TallyPopup.tsx`; both a named `TallyPopupSimple` and a default export; renders `null`; hard-coded Sentry org slug.
- **PlanCard**: a component folder with no component (only `plans.ts`, `countries.ts`, `computePricing.ts`).
- **file-tree.tsx**: a lowercase loose file at the molecules root; five `useCallback`; unused `_ref`, `_handleSelect`, `_className`; exports `File`, shadowing the DOM global.
- **Avatar**: `getAvatarSizeFromClassName` regex-parses Tailwind classes to derive pixel size (`Avatar.tsx:72-81`); exports three Props interfaces plus a default export.
- **Progress**: `bg-gray-200`; no `role="progressbar"` or `aria-value*`.
- **Breadcrumbs**: no `<nav>`, `<ol>` or `aria-current`; `font-[400]`; `key={index}`.
- **Link**: `text-sm font-medium` of its own instead of `Text`; `...props` spread on a type that declares no extra attributes.

### 2.10 Accessibility summary

- No focus style: ToggleChip, ShowMore, ShowMoreText, BriefingCard toggle, PendingReviewsList toggle, SecondaryMenu items.
- No accessible name: FloatingReviewsPanel close, ShowMoreText "...", Switches in PendingReviewCard and PendingReviewsList.
- No state attributes: `aria-expanded` missing on four disclosure toggles; ScrollableTabs lacks tab roles; Progress lacks progressbar semantics; Breadcrumbs lack landmark and current page.
- Meaningless screen-reader text: Dialog title and description "Dialog".
- Form fields: label used as placeholder; no `aria-invalid`/`aria-describedby`; Select option callback on mousedown only; `<label>` wrapping interactive children; duplicate static ids in TimePicker; lucide icons without `aria-hidden` in Date atoms.
- Contrast: with the custom palette, `zinc-400` (`#ADADB3`) on white is about 2.2:1 and `zinc-500` (`#83838C`) about 3.8:1. `Text tone="muted"` at body sizes, `text-zinc-400` at 11 to 14px (`ReferenceCard.tsx:49`, `RunRow.tsx:55`, `Input.tsx:292`, `TimeInput.tsx:114`), a `text-[10px]` pill (`TrialOffer.tsx:24`), and `text-[11px]` kbd labels all likely fail AA.
- Small targets: ShowMoreText toggle 16px, `icon-xs` and ToggleChip 28px.

### 2.11 Storybook coverage

Coverage is 50 of 76 component folders (atoms 21/27, molecules 26/38, organisms 3/11). Missing: `AutoGPTLogo`, `Card`, `DateInput`, `DateTimeInput`, `Icon`, `TimeInput`, `ErrorBoundary`, `Form`, `FullscreenDialog`, `InstallWorkflowPicker`, `IntegrationLogo`, `IntegrationsMarquee`, `NotionAvatar`, `PlanCard`, `Popover`, `RunStatusBadge`, `TallyPoup`, `WorkflowAvatar`, `ApprovalFields`, `BriefingCard`, `FloatingReviewsPanel`, `NeedsAttentionList`, `PendingReviewCard`, `PendingReviewsList`, `VoicePicker`, `WorkOutputSheet`. `TrialCard` has a story only for `TrialStatus`.

Three stories sit outside every Storybook glob and are never loaded: `layout/Navbar/components/AccountMenu/AccountMenu.stories.tsx`, `contextual/IntegrationsPanel/components/AIConnectionsSection/ProviderBox.stories.tsx`, `app/(platform)/artifacts/components/OriginFilter/OriginFilter.stories.tsx`.

`overview.stories.tsx:24-27` itself uses `text-gray-900` and `text-gray-600`. The token stories for border radius, icons and spacing import `lucide-react`. The spacing story hardcodes values that no longer match the config; the radius story omits the shadcn `sm/md/lg` aliases; the typography story imports nothing from the config.

A second, 2,093-line in-app style guide exists at `src/app/(platform)/copilot/styleguide/page.tsx` with its own `bg-[#f8f8f9]`, `text-[13px]`, `text-[1rem]`. It is a third source of truth no document points to.

---

## Part 3: Drift in feature code

Numbers are for `src/` excluding generated code. Per-directory breakdowns show where the pain concentrates.

| Signal | Total | Heaviest directories |
|---|---|---|
| Arbitrary hex colour classes | 171 in 73 files | `contextual/IntegrationsPanel` 75, `settings` 23, `layout` 14, `admin` 13, `copilot` 10 |
| Default-palette classes | 2,297 | `__legacy__` 387, `admin` 264, `copilot` 228, `library` 214, `contextual` 197, `build` 149 |
| Arbitrary sizes | 886 in 370 files | `build` 123, `marketplace` 111, `__legacy__` 82, `copilot` 69 |
| Arbitrary `text-[`, `leading-[`, `tracking-[`, `rounded-[`, `shadow-[`, `z-[` | 385 / 118 / 37 / 184 / 88 / 22 | |
| Raw `<p>`/`<h*>` with classes in `src/app` | 409 in 129 files | `copilot` 139, `admin` 127 |
| Raw `<button className>` in `src/app` | 99 in 81 files | `copilot` 30, `library` 16 |
| Raw `<input className>` in `src/app` | 26 in 12 files | `admin/platform-costs` 8 |
| `!important` utilities | 261 in 127 files | `contextual` 39, `copilot` 38, `library` 34 |
| Inline `style={{` | 160 (104 app, 56 components) | `library` 31, `copilot` 31 |
| Opacity colour hacks (`bg-zinc-900/10`...) | 215 in 107 files | `artifacts` 56 (a hand-rolled grey ramp in `FileIllustration.tsx`) |
| `dark:` classes | 627 in 102 files | all dead |
| `__legacy__` importers | 115 files | `build` 26, `profile` 17, `admin` 16, `layout` 14 |
| `ui/` importers from `src/app` | 37 files | `copilot` 24 |
| Non-Hugeicons icon imports | lucide 42, legacy icons 21, radix 15, react-icons 1, phosphor 1 | |

What is working: zero direct `HugeiconsIcon` usage outside the `Icon` atom (1,174 uses of the atom across 527 files), and `Text` is imported in 336 of 1,385 app files.

Exact duplicate files that will keep drift alive if only one copy is fixed: `contextual/OutputRenderers/renderers/MarkdownRenderer.tsx` and `library/agents/[id]/.../OutputRenderers/renderers/MarkdownRenderer.tsx`; `__legacy__/CreatorInfoCard.tsx` and `marketplace/components/CreatorInfoCard/CreatorInfoCard.tsx`.

Worst files by composite drift score: `admin/diagnostics/components/DiagnosticsContent.tsx` (95), `artifacts/components/ArtifactsList/FileIllustration.tsx` (72), `contextual/ProfileInfoForm/ProfileInfoForm.tsx` (54), `profile/(user)/credits/components/SubscriptionTierSection/SubscriptionTierSection.tsx` (45), both `MarkdownRenderer.tsx` (44 each), `build/components/MCPToolDialog.tsx` (44), `profile/(user)/dashboard/components/AgentTableRow/AgentTableRow.tsx` (38), both `CreatorInfoCard.tsx` (36 each), `admin/memory/components/MemoryVisualizer.tsx` (34), `atoms/FileInput/FileInput.tsx` (31, the worst atom).

---

## Part 4: Enforcement

### 4.1 Nothing mechanical enforces any design-system rule

`.eslintrc.json` has exactly one custom rule family (the IME keyboard rule, tested by `scripts/eslint-keyboard-rules.test.ts`). It proves the team can write and test AST rules. There is:

- no `eslint-plugin-tailwindcss` (`no-arbitrary-value`, `no-custom-classname`, `no-contradicting-classname`);
- no `no-restricted-imports` for `@/components/__legacy__/*`, `@/components/ui/*`, `@radix-ui/react-*` in feature code, `lucide-react`, `@phosphor-icons/react`, `@radix-ui/react-icons`, `react-icons`;
- no stylelint for `globals.css`;
- no rule steering `next/link` to the `Link` atom (95 files use `next/link` directly; `frontend/AGENTS.md:50` actually recommends `next/link`, contradicting the atom's existence).

`frontend/AGENTS.md:52` forbids `eslint-disable` and `@ts-ignore`; `src/` has 166 `eslint-disable` comments and 8 `@ts-ignore`/`@ts-expect-error`.

### 4.2 `cn()` cannot merge the custom tokens

`src/lib/utils.ts:14-16` is `twMerge(clsx(...))` with no `extendTailwindMerge`. Verified with the installed `tailwind-merge`:

```
twMerge("rounded-large rounded-md")   -> "rounded-large rounded-md"   (both kept)
twMerge("rounded-xsmall rounded-full")-> "rounded-xsmall rounded-full" (both kept)
twMerge("shadow-subtle shadow-md")    -> "shadow-subtle shadow-md"    (both kept)
twMerge("text-[0.875rem] text-sm")    -> "text-sm"                     (ok)
twMerge("h-[2.875rem] h-9")           -> "h-9"                         (ok)
```

So for every custom radius and the custom shadow, a consumer's `className` override does not win by position; stylesheet order decides. This is the mechanism behind "I passed `rounded-md` to `Card` and nothing changed", and a reason people reach for `!important`.

### 4.3 Prettier

`.prettierrc` enables `prettier-plugin-tailwindcss` but sets neither `tailwindFunctions` nor `tailwindConfig`. Class sorting therefore only touches JSX `className` literals; the 836 `cn(` and 13 `cva(` call sites are unsorted, and custom classes like `rounded-xsmall` are sorted as unknown.

### 4.4 CI (`.github/workflows/platform-frontend-ci.yml`, `platform-fullstack-ci.yml`)

| Check | Status |
|---|---|
| ESLint and `prettier --check` | runs, blocking |
| `tsc --noEmit` | runs, blocking (full-stack CI only) |
| Vitest | runs, blocking |
| Playwright | runs |
| knip | runs with `continue-on-error: true` (informational) |
| Chromatic | `if: ${{ false }}` (frontend-ci:128); the project token is hardcoded in plaintext at frontend-ci:154 |
| `build-storybook` | never |
| `test-storybook` / a11y | never; `@storybook/test-runner` is not in `package.json` and not in `node_modules`, so `pnpm test-storybook` cannot run; `test-runner-jest.config.js` requires the missing package |

`@storybook/addon-a11y` is registered but with no runner it is a dev-time panel. Zero of 75 stories set `parameters.a11y`; three have `play` functions.

### 4.5 Pre-commit

`.pre-commit-config.yaml` runs Prettier, `tsc` and the API client regen for frontend files. No ESLint. No husky or lint-staged.

### 4.6 Review gates

`.github/PULL_REQUEST_TEMPLATE.md` has no design-system checklist item. `.github/CODEOWNERS` has no entry for `autogpt_platform/frontend/src/components`. No review-bot configuration mentions the design system.

### 4.7 Documentation contradictions

| Claim | Where | Reality |
|---|---|---|
| "Use shadcn/ui components as building blocks when available" | `CONTRIBUTING.md:620` | Same file, line 626: "Do not import shadcn primitives directly in feature code" |
| Never use `src/components/_legacy__` | `CONTRIBUTING.md:185, 853, 867` | The folder is `__legacy__` (double underscore); the path in the doc does not exist |
| "Keep responsive and dark-mode behavior consistent" | `CONTRIBUTING.md:622` | `tailwind.config.ts:55` says ignore `dark:`; `frontend/AGENTS.md:49` says no `dark:`; `providers.tsx:43` forces light |
| "Verify in Chromatic after PR" | `CONTRIBUTING.md:96, 186, 876`; `TESTING.md:9, 174`; `.github/copilot-instructions.md:277` | Chromatic job is disabled |
| `pnpm test-storybook` is the CI runner | `TESTING.md:173`; `README.md:136-140` | not in CI, dependency not installed |
| "Hugeicons only" | root `AGENTS.md:44`; `frontend/AGENTS.md:64, 88-90`; `CONTRIBUTING.md:729` | five other icon packages installed and imported; token stories import lucide |
| `iconLibrary: "radix"`, `cssVariables: false` | `components.json` | Hugeicons rule; globals.css is entirely CSS variables |
| Props should be `interface Props` | root `AGENTS.md` | `frontend/AGENTS.md:81` shows `type Props = { ... }` |
| Use Next.js `<Link>` | `frontend/AGENTS.md:50` | a `Link` atom exists |

No document anywhere lists the token values. `docs/engineering/` has no design-system page. The only token references are `styles/colors.ts`, `tailwind.config.ts` and the Storybook token stories, two of which have drifted from the config.

---

## Part 5: Dead code and dependencies

### 5.1 Dead files (knip, verified)

- 21 files in `__legacy__` totalling 1,547 lines: `AgentImageItem`, `AgentImages`, `BecomeACreator`, `CreatorCard`, `CreatorInfoCard`, `CreatorLinks`, `FeaturedAgentCard`, `FilterChips`, `RatingCard`, `SearchBar`, `SmartImage`, `ThemeToggle`, `action-button-group`, `delete-confirm-dialog`, `types.ts`, `composite/FeaturedCreators`, `composite/FeaturedSection`, `composite/HeroSection`, `ui/data-table`, `ui/radio-group`, `ui/render`.
- `__legacy__/Button.tsx` has five importers, all inside `__legacy__`, and only `Sidebar.tsx` among them is alive.
- `molecules/NotionAvatar/{NotionAvatar,NotionAvatarImage,NotionAvatarSvg}.tsx`, `molecules/ErrorCard/components/LoadingState.tsx`, `molecules/ShowMore` (0 consumers).
- `__legacy__/ui/icons.tsx` is 1,880 lines with 59 exports, 31 of them unused.
- Repo-wide knip: 97 unused files, 196 unused exports, 16 unused types.

### 5.2 Dead config

- Tailwind: `customGray-*`, spacing `70`, `7.5`, `8.5`, `grain-overlay` plugin, `animate-loader`, `animate-marquee-x`, `animate-caret-blink`, the entire default-spacing re-declaration, `darkMode` setting, `--chart-*` variables.
- CSS: `.agpt-rounded-box`, `.agpt-box`, `.agpt-div`, `.agpt-card-selected`, the `.dark` block.
- Exports: `textVariants`, `textTones` (story-only), `ICON_STROKE_WIDTH`, `NUMBER_REGEX`, `PHONE_REGEX` (in-file only), `Confetti` `ConfettiContext`, `Dialog` `useDialogCtx`, `DropdownMenuGroup/Portal/Sub`, `PopoverAnchor`, `ShowMore` default export, `getIconSize`, `shouldShowError`, `AUTOPILOT_AVATAR_BG_CLASS`, `getRunStatusGuidance`, `useTable` `clearAll`/`createEmptyRow`, `RevealGroup` (a no-op `<div>`), `Button` `asChild`, `Dialog` `withGradient`, `DrawerWrap` `testId`, `ExpertAvatar` `color`, `TimePicker` `className`.
- Invalid or dead class names in components: `disabled:opacity-1` (x7), `text-md`, `animate-fadein`, `transition-left`, `transition-right` (x2), `flex-end`/`flex-start` (x5), `border-1.5` (x3), `!placeholder:text-zinc-500` (x3), `!focus-visible:ring-0`, every `dark:` class (51 in the design system).

### 5.3 Dependencies

| Package | Importers | Note |
|---|---|---|
| `@radix-ui/react-radio-group` | 0 live (dead `__legacy__/ui/radio-group`) | remove |
| `@tanstack/react-table` | 0 live (dead `__legacy__/ui/data-table`) | remove |
| `@phosphor-icons/react` | 1 | remove after migrating `ProfileInfoForm` |
| `react-icons` | 2 | remove |
| `motion` | 2 | fold into `framer-motion` |
| `cmdk` | 2, both `__legacy__` | remove with multiselect/command |
| `react-day-picker` | 1 (`__legacy__/ui/calendar`) | only the Date atoms need it, via legacy |
| `embla-carousel-react` | 1 (`__legacy__/ui/carousel`) | |
| `lucide-react` | 42 | migrate to Hugeicons |
| `@radix-ui/react-icons` | 16 (12 in `__legacy__`) | migrate |
| `next-themes` | 2 | installed to force a single theme |
| `webpack` | imported by `.storybook/main.ts` but not declared | add or remove |

17 `@radix-ui/*` packages are installed; several are kept alive only by legacy or `ui/` twins.

---

## Part 6: Recommendations

Ordered so each step makes the next one safer. Steps 1 to 3 are small, fix live bugs, and can ship this week. Steps 4 to 6 are the real work. Steps 7 to 8 keep it fixed.

### Step 1: Fix the bugs that are live today

1. Button: replace `disabled:opacity-1` with `disabled:opacity-100` on the seven variants, or drop `disabled:opacity-50` from the base and rely on the explicit colours. Replace `rounded-[96px]` with `rounded-full`. Replace `border-[#a6a6a6]` with a palette grey (`zinc-300` is the nearest). Move `min-w-[7.7rem]` into the `large` and `small` sizes only.
2. Dialog: `animate-fadein` to `animate-fade-in`; `text-md` to `text-base` or a Text variant; sr-only "Dialog" to real titles.
3. TabsLine and ScrollableTabs: drop `transition-left transition-right`, use `transition-[left,right]` or `transition-all`.
4. Remove `flex-end`/`flex-start`, `border-1.5`, mis-ordered `!placeholder:`/`!focus-visible:` classes.
5. `colors.ts`: give `slate.700` its own value, trim the trailing space on `red.900`.
6. `Text` label variant: `0.6785rem` to `0.6875rem`.
7. `Select`: move `option.onSelect` to the Radix `onValueChange` path so keyboard works; stop wrapping the tooltip button in the `<label>`.
8. `TimePicker`: use `useId` for the three ids; apply the `className` it accepts.
9. `ExpertAvatar`: either use `color` or remove it and its call site.
10. `FloatingReviewsPanel` close button: add `aria-label`.
11. `ToggleChip` locked: stop calling `onClick`.

### Step 2: Make `cn()` and Prettier understand the tokens

1. `src/lib/utils.ts`: use `extendTailwindMerge` with class groups for `rounded-{xsmall,small,medium,large,xlarge,2xlarge}`, `shadow-subtle`, and the Text font sizes once they become theme values. Add a unit test that asserts `cn("rounded-large", "rounded-md") === "rounded-md"`.
2. `.prettierrc`: add `"tailwindFunctions": ["cn", "cva", "clsx"]` and `"tailwindConfig": "./tailwind.config.ts"`.

### Step 3: Delete what is dead

1. The 21 knip-dead `__legacy__` files, `__legacy__/ui/icons.tsx`'s 31 unused exports, `NotionAvatar`, `ShowMore`, `ErrorCard/components/LoadingState.tsx`, the three orphan stories (or add their folders to the Storybook globs).
2. Tailwind: `customGray`, spacing `70`/`7.5`/`8.5` and the default-scale re-declaration, `grain-overlay`, `loader`, `marquee-x`, `caret-blink`.
3. CSS: `.agpt-rounded-box`, `.agpt-box`, `.agpt-div`, `.agpt-card-selected`, `--chart-*`.
4. Every `dark:` class and the `.dark` variable block. Dark mode is a future project; when it comes back it should start from the semantic layer in Step 4, not from 627 hand-placed overrides. Also pick one dark selector and delete the other two.
5. Dependencies: `@radix-ui/react-radio-group`, `@tanstack/react-table`, `@phosphor-icons/react`, `react-icons`, `motion`; declare `webpack` for Storybook.
6. Make knip blocking in CI (or at least block on new findings) so this list does not regrow.

### Step 4: Define the tokens once, in Tailwind, and document them

The goal is one vocabulary per axis, expressed as Tailwind theme keys so that classes, `cn()`, Prettier, Storybook and the Paper export (`export-design-tokens.ts` already reads `colors.ts`, `globals.css`, `tailwind.config.ts` and `Text/helpers.ts`) all see the same thing.

1. **Colour.** Keep the custom palette as the primitive layer and build a semantic layer on top of it in `tailwind.config.ts`, replacing the shadcn HSL variables rather than sitting beside them. A minimal set: `bg-{page,surface,surface-raised,surface-sunken}`, `text-{primary,secondary,muted,disabled,inverse,brand,danger,success,warning}`, `border-{default,strong,subtle,focus,danger}`, `ring-focus`, `bg-{brand,brand-subtle,danger,danger-subtle,success-subtle,warning-subtle,info-subtle}`. Every semantic token maps to a palette step (`text-muted` to `zinc-500`, `border-default` to `zinc-200`, `bg-page` to whichever of `#F6F7F8` / `#FAFAFA` design wants). Delete `textGrey`, `textBlack`, `bgLightGrey`, `yellow-25`, `yellow-150`, `sidebar-*`, `chart-*` once mapped. Decide whether the brand is `purple-500` or the violet `--accent` and keep one. Add the recurring unregistered greys (`#F9F9FA`, `#DADADC`, `#EFEFF0`, `#8A8A90`) to the zinc scale or map them to existing steps.
2. **Disable the default palette.** Set `theme.colors` (not `extend.colors`) to the custom palette plus the semantic layer, so `gray-500`, `neutral-800`, `amber-50`, `violet-600` stop compiling. This turns 2,297 silent drifts into build errors that can be fixed directory by directory behind a temporary allowlist.
3. **Radius.** Keep one scale. The simplest choice that matches existing usage is Tailwind's names with the token values (`sm` 4, `md` 8, `lg` 12, `xl` 16, `2xl` 20, `3xl` 24, `full`), set in `theme.borderRadius` so the shadcn `--radius` aliases and the `xsmall..2xlarge` names disappear. Then `rounded-[18px]` (38 uses) becomes a design decision, not a default.
4. **Spacing.** Delete the re-declaration; keep `extend` with only the additions that survive review (`18`, `68`, `76`, `4.5`). Add `container` or `max-w` tokens for the page widths that recur (`1360px` x18, `42rem` x10, `600px` x9).
5. **Typography.** Move the `Text` variants into `theme.fontSize` as named tuples (`body: ["0.875rem", { lineHeight: "1.375rem" }]`, etc.) so the atom becomes `text-body font-medium` and the same names are available to the rare raw element. Add the sizes people keep inventing if design agrees (`13px`, `11px`), otherwise map them to the scale. Change the variant default colour from `text-black` to `text-primary` and extend tones to cover `secondary`, `muted`, `disabled`, `inverse`, `brand`, `danger`, `success`, `warning`, so `!text-zinc-*` overrides have no reason to exist. Add `h6`.
6. **Elevation.** Define `shadow-{xs,sm,md,lg,xl}` in the theme, fold `shadow-subtle` and the three hand-written shadows into it, and decide whether `smooth-shadow-ring-sm` is the `sm` step or dies.
7. **Control heights.** Define `h-control-{sm,md,lg}` (28, 36, 46) and use them in Button, Input, Select, SearchInput, TimeInput, Date inputs, MultiToggle, ToggleChip.
8. **Motion.** `transitionDuration` `{fast: 150, normal: 200, slow: 300}` and two easings; use the same values in framer props.
9. **Focus.** One utility class (`focus-ring`) in `@layer components`: `focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2`. Every interactive atom uses it.
10. **Fonts.** Delete `fonts.css`. Make Storybook load fonts the same way the app does (Storybook's Next.js framework supports `next/font`).
11. **Document it** in `docs/engineering/design-system.md`: the token tables, the component layers, the rules, and the one place each rule is enforced. Make the token stories read from the theme object instead of hardcoding rows.

### Step 5: Rebuild the atoms on the tokens and cut the inverted dependencies

1. **One field base.** Extract a `fieldStyles` helper (base, size, state, error) and a `FieldWrapper` (label, tooltip, hint, reserved error line with `aria-describedby` and `aria-invalid`) used by Input, Select, TimeInput, DateInput, DateTimeInput, SearchInput, and a new Textarea. Unify the size prop to `sm | md | lg`, the radius, the focus ring and the error colour. Drop `placeholder={label}`.
2. **Input and Select** stop wrapping `__legacy__`. Select calls Radix directly like Popover already does.
3. **Date atoms** use `molecules/Popover` and the atom Button; move the calendar into an atom (or keep `react-day-picker` wrapped there) so `__legacy__/ui/calendar` can go.
4. **Collapsible, Table, DrawerWrap, ToggleChip, Accordion** import Radix or the atoms directly. Accordion stops being a barrel.
5. **Button**: one `icon` prop typed `IconSvgElement` for leading, one for trailing; delete `leftIcon`/`rightIcon`/`asChild`; size scale `sm | md | lg | icon-sm | icon-md`; loading keeps the variant's colours and shows the spinner in place of the leading icon.
6. **Missing atoms**: `Checkbox` (6 legacy consumers), `Textarea`, `Kbd` (pasted 5 times), `Separator`, `ScrollArea`, `Sheet`. Create them in `atoms/`, migrate consumers, then delete the `ui/` and `__legacy__` copies.
7. **Collapse the duplicates** listed in 2.1: one Tooltip (portalled by default), one Skeleton, one Spinner, one Dialog family (Dialog with a `fullscreen` variant), one dropdown (fold SecondaryMenu into DropdownMenu), one tabs (fold ScrollableTabs into TabsLine), one avatar (`Avatar` with `shape` and `size` props; ExpertAvatar and WorkflowAvatar become thin wrappers or disappear), one time input, one date input.
8. **Badge** moves to `green`/`yellow` tokens and gains `brand` and `neutral` variants so the hand-rolled pills in SubscriptionOffer, TrialOffer and RunRow can use it. Fix `text-ellipsis` on the flex container.
9. **Toast** moves off the CSS module onto token classes and the four variants it actually exposes.
10. **Alert** uses Hugeicons through `Icon`, gains `success` and `info`, drops the hex-alpha backgrounds.
11. **Rename and tidy**: `TallyPoup` to `TallyPopup`, `file-tree.tsx` to `FileTree/FileTree.tsx`, `BaseTooltip.tsx` to `Tooltip.tsx`, `TextFieldProps` to `Props`, Toast files to PascalCase, `PlanCard` folder to a `lib/` or `helpers` location since it has no component. Remove default exports. Convert arrow components to function declarations. Split `FileInput`, `PendingReviewsList`, `FloatingReviewsPanel`, `DateInput`, `DateTimeInput` into `useX.ts` + `helpers.ts`.
12. **Stories** for the 26 folders that lack them, with `parameters.a11y` on each.

### Step 6: Migrate feature code, directory by directory

Order by drift density and ownership: `admin` (419 raw type classes, 264 palette hits, mostly tables and forms that map cleanly onto `Table`, `Input`, `Badge`), then `copilot` (highest `ui/` dependence; 24 files), `library`, `contextual/IntegrationsPanel` (75 hex classes in one feature), `profile`, `build`, `marketplace`, `settings`. For each directory: replace raw `<h*>`/`<p>`/`<button>`/`<input>` with atoms, replace palette and hex classes with semantic tokens, delete `dark:`, delete `!important` by fixing the atom instead, delete inline styles that encode colour. Fix both copies of `MarkdownRenderer.tsx` and `CreatorInfoCard.tsx` by deleting one.

The `__legacy__` folder becomes deletable when its importers hit zero. Today that is 115 files; the top five modules (`button` 29, `skeleton` 23, `icons` 16, `form` 14, `input` 12) account for most of them and each has an atom equivalent.

### Step 7: Enforce, so it stays fixed

1. **ESLint `no-restricted-imports`**: `@/components/__legacy__/*` (allowlist current importers in an override block, shrink it as they migrate), `@/components/ui/*` outside `src/components`, `@radix-ui/react-*` outside `src/components/{atoms,molecules,organisms}`, `lucide-react`, `@phosphor-icons/react`, `@radix-ui/react-icons`, `react-icons`, `@hugeicons/react` outside `atoms/Icon`, `sonner` outside `molecules/Toast`, `next/link` outside `atoms/Link` (or delete the atom and say so).
2. **`eslint-plugin-tailwindcss`** with `no-custom-classname` (catches `flex-end`, `text-md`, `animate-fadein`, `transition-left`, `opacity-1`), `no-contradicting-classname`, `no-arbitrary-value` with a short allowlist for layout-only values, and `enforces-negative-arbitrary-values`. Run it on `cn()`/`cva()` call sites via `callees`.
3. **A `no-restricted-syntax` rule** (same pattern as the keyboard rule) for raw `<h1..h6>`, `<p>`, `<button>`, `<input>`, `<textarea>` with a `className` in `src/app`, pointing to the atom. Test it in `scripts/` like the keyboard rule.
4. **Disable the default palette** in the theme (Step 4.2); that is the strongest colour guard and needs no plugin.
5. **CI**: re-enable Chromatic with the token in a secret; install `@storybook/test-runner` and run `test-storybook` with a11y checks; run `build-storybook` on PRs touching `src/components`; make knip blocking; add ESLint to pre-commit.
6. **Review**: a CODEOWNERS line for `autogpt_platform/frontend/src/components/**`; a PR template checkbox "uses design tokens and atoms; no `__legacy__`/`ui/` imports; no `dark:`".
7. **Docs**: fix the nine contradictions in 4.7; point `components.json` at the truth or delete it; retire the in-app `copilot/styleguide` page in favour of Storybook or move it under the Storybook globs.

### Step 8: Track it

A small script (or the knip/eslint output) in CI that prints the counts from Part 3 (`__legacy__` importers, default-palette hits, arbitrary values, raw elements, `!important`, `dark:`) and fails if any number goes up. The numbers in this document are the baseline.

---

## Appendix: verification notes

- Tailwind compilation checks were run against the real `tailwind.config.ts` with `postcss` and `tailwindcss` 3.4.17: `.opacity-1` and `.disabled\:opacity-1` are not generated; `rounded-[96px]`, `gap-4.5`, `min-w-[7.7rem]` are; `dark:` utilities compile to `:is(.dark-mode *)`.
- `tailwind-merge` 2.6.0 results are from the installed package.
- Importer counts come from a resolver that follows `@/` aliases and relative paths; test, story and mock files are excluded unless stated.
- `pnpm knip` ran with the existing `node_modules` and exit code 1 (findings present).
- Git churn in the last 90 days: 79 commits touched atoms/molecules/organisms, 7 touched `__legacy__`, 6 touched `ui/`. Five new files were added to `ui/` in the last 180 days.

---

## Part 7: Beyond the audit, what an industry-standard system also needs

The sections above fix what exists. These are the things mature design systems (Radix Themes, Atlassian, Shopify Polaris, GitHub Primer, Vercel Geist) have that this codebase has no version of yet. Ordered by leverage.

### 7.1 Tokens as data, not as Tailwind config

Industry practice is a three-tier token model (primitive, semantic, component) stored in a platform-neutral JSON format (the W3C Design Tokens Community Group format) and built with Style Dictionary or Tokens Studio into every target: Tailwind theme, CSS variables, Storybook docs, Paper/Figma libraries. Today the only source is `colors.ts` plus `tailwind.config.ts`, and the Paper export script that reads them lives outside the repository in a personal skill folder (`~/.claude/skills/paper-designer/scripts/export-design-tokens.ts`) and references a `pnpm design:tokens` script that `package.json` does not define. Design and code are therefore synchronised by hand. Move the token source into the repo as JSON, generate `tailwind.config.ts` and the Paper payload from it, and check the generated files in so diffs are reviewable.

### 7.2 Decide on Tailwind 4 before rebuilding tokens

Tailwind 4 replaces `tailwind.config.ts` with a CSS-first `@theme` block, drops `extend` semantics, and changes how plugins, `darkMode` and `tailwind-merge` configuration work. Rebuilding the token layer on 3.4 and then migrating means doing the work twice. `eslint-plugin-tailwindcss` support for v4 also lags; `@tailwindcss/vite` and `prettier-plugin-tailwindcss` support it. Make the version decision first, then do Step 4.

### 7.3 Package boundary and dependency direction

Mature systems ship as a package with an explicit public surface. Making `src/components/{atoms,molecules,organisms}` a workspace package (`@autogpt/ui`) with an `exports` map would make deep imports impossible, give knip a real entry point, let Storybook and Chromatic run only on that package, and make the inverted dependencies found in 2.2 (atoms importing from `app/` and `contextual/`) a build error. Short of that, `eslint-plugin-import`'s `no-restricted-paths` or `dependency-cruiser` can enforce the layering: atoms import nothing from the app, molecules import atoms, organisms import both, features import all, and nothing in `src/components` imports from `src/app`.

### 7.4 A component contract

Every atom should satisfy the same checklist before it is considered done, and the checklist should be a PR template section for `src/components`: anatomy documented; variants and sizes from the shared vocabulary; all styling via `cva` with `data-state`, `data-size`, `data-variant` attributes for styling hooks and tests; `className` merged last; `ref` forwarded; controlled and uncontrolled modes where relevant; keyboard behaviour; `prefers-reduced-motion`; a story per state; an a11y pass; a changelog entry. Polaris, Primer and Radix publish exactly this as a per-component spec page. Today there is no template, which is why the props APIs in 2.8 diverge.

### 7.5 Accessibility as a gate, not a panel

Set a target (WCAG 2.2 AA) and test it in three places: `axe` through the Storybook test runner on every story, `vitest-axe` in integration tests, and `@axe-core/playwright` in the E2E flows. Then fix the palette itself: with the custom values, `zinc-400` on white is about 2.2:1 and `zinc-500` about 3.8:1, so `tone="muted"` fails AA at body sizes before any component is involved. Generate ramps with contrast guarantees (OKLCH with APCA or WCAG checks) rather than by eye.

### 7.6 Visual regression and interaction tests

Chromatic is wired but disabled. Either re-enable it or use Playwright component screenshots. Pair it with Storybook `play` functions for every interactive atom (3 of 75 stories have one). Without this, the token rebuild in Step 4 cannot be verified except by eye.

### 7.7 Deprecation policy and ratchets

Nothing is ever formally deprecated here; `__legacy__` has existed long enough that the design system now depends on it. Standard practice: mark with `@deprecated` JSDoc and enforce with `@typescript-eslint/no-deprecated` (new usages fail, existing ones are allowlisted); ship a codemod for mechanical migrations (`__legacy__/ui/button` to `atoms/Button` is one); and run a ratchet in CI that fails when the count of deprecated usages, raw palette classes, or arbitrary values goes up. The numbers in Part 3 are the baseline for that ratchet.

### 7.8 Ownership, governance and a decision log

No CODEOWNERS entry, no named owner, no RFC path for adding a component, no record of why decisions were made (why the primary button is zinc and not purple, why 46px fields). Mature systems keep a short decision log in the repo and require a system owner's review on `src/components`. This matters more here than usual because much of the code is agent-written: agents follow rules that are in lint and in a machine-readable catalog, and ignore rules that are only in prose (166 `eslint-disable` comments despite `AGENTS.md` forbidding them).

### 7.9 A machine-readable component catalog

Agents and new engineers pick `ui/button` or a raw `<button>` because nothing tells them which of four buttons is canonical. Storybook's `index.json` can be the catalog, or a generated `COMPONENTS.md` listing every atom, its props, and the legacy thing it replaces. Point `AGENTS.md` at it, and add a Claude Code or pre-commit hook that rejects new `__legacy__` and `ui/` imports before they reach CI.

### 7.10 Layout and responsive tokens

The docs name breakpoints (375, 768, 1024, 1280) but the theme defines none beyond `container` at 1400px, and page widths are written as `max-w-[1360px]` 18 times. Define `screens`, page-width and gutter tokens, and consider layout primitives (`Stack`, `Inline`, `Grid`) or at least a documented spacing rhythm so padding stops being a per-file decision. Validate every story at the four breakpoints through Storybook's viewport addon.

### 7.11 Icons as a system

One library (already decided: Hugeicons), plus a size scale (16, 20, 24) as tokens, a stroke-width token, and a lint rule that blocks the other five packages. The `Icon` atom is the one part of the system that is used correctly everywhere; protect it.

### 7.12 Internationalisation readiness

The codebase handles IME composition carefully but uses physical properties everywhere (`ml-`, `pr-`, `left-`). If RTL is ever a requirement, logical properties (`ms-`, `pe-`, `start-`) need to be the convention from the token rebuild onward; retrofitting is expensive. If it is not a requirement, write that down.

### 7.13 Performance hygiene

Dead `dark:` classes, two copies of framer-motion, a 1,880-line icon file, a 279 KB generated avatar file for a retired feature, and Storybook loading fonts from Google while the app self-hosts them. A CSS and JS bundle-size check in CI (for example `size-limit` on the main route) would have caught each of these as it landed.

### 7.14 Definition of done for the system itself

A design system is "industry standard" when a new engineer can build a screen without asking anyone which component or colour to use, and when doing it wrong fails a check. Concretely: one token source, one component per concept, every rule in a linter, every component in Storybook with a11y and visual tests, a ratchet on drift, an owner, and a changelog. None of those seven exist today; Steps 1 to 8 plus this section are the path to all seven.

---

## Part 8: Adopting the shadcn theme and upgrading the stack

The team intends to rebuild the design system on the shadcn theme. That is a good fit: shadcn is copy-paste code that you own, its semantic variable set is exactly the layer this codebase is missing, and its Tailwind 4 preset resolves several of the dual-definition problems above by construction. This section turns that intention into a concrete order of operations, with versions checked against npm on 2026-10-07.

### 8.1 What "use the shadcn theme" should mean here

1. **The atoms are the shadcn components.** shadcn is designed to be owned, not wrapped. Today `ui/` holds raw shadcn output and `atoms/` wraps or re-implements it. After the move, each shadcn component is generated once, restyled with the design tokens, and lives in `atoms/` or `molecules/` under the existing naming. `ui/` and `__legacy__/ui/` are deleted. Set `components.json` `aliases.ui` to a scratch folder (for example `src/components/__shadcn_scratch__`) that is git-ignored and lint-blocked, so running the CLI never adds a shipping file by accident.
2. **shadcn's semantic variables become the semantic layer** from Step 4.1: `background`, `foreground`, `card`, `popover`, `primary`, `secondary`, `muted`, `accent`, `destructive`, `border`, `input`, `ring`, `chart-1..5`, `sidebar-*`. Add the handful shadcn lacks and this product needs: `success`, `warning`, `info` (each with `-foreground`), and `brand` if the primary action is to stay zinc while purple remains the brand.
3. **The custom palette stays as the primitive layer** and is defined as Tailwind 4 theme colours, replacing the defaults entirely (`--color-*: initial` followed by the palette). That is how the default palette gets disabled under Tailwind 4; `gray-500` and `amber-50` stop compiling.
4. **Radius follows shadcn's model**: one `--radius` with `rounded-sm/md/lg/xl/2xl/3xl` derived from it. Delete `xsmall..2xlarge`. Decide `--radius` once (the current fields are 12px, the current cards 16 to 24px; `0.75rem` with `xl` and `2xl` for cards is the likely answer).
5. **Dark mode comes from the variable swap**, not from `dark:` classes. shadcn's `.dark` block plus `@custom-variant dark (&:is(.dark *))` lines up with `next-themes` `attribute="class"`, which removes the three-selector mismatch in 1.8. Keep `forcedTheme="light"` until the semantic layer is complete, then delete every hand-placed `dark:` class; they become unnecessary rather than merely dead.
6. **Fonts** are declared in `@theme` (`--font-sans: var(--font-geist-sans)` and so on) and `fonts.css` is deleted.
7. **Icons**: shadcn's `iconLibrary` has no Hugeicons option. Set it to `lucide` so generated code compiles, and replace the icon in each component as it is adopted. The `Icon` atom remains the only sanctioned path, enforced by lint.

### 8.2 Target versions

| Package | Now | Target | Why |
|---|---|---|---|
| `tailwindcss` | 3.4.17 | 4.3.x | CSS-first `@theme`, `@custom-variant`, native `--color-*: initial`; the shadcn v4 preset requires it |
| `@tailwindcss/postcss` | none | 4.3.x | replaces `tailwindcss` + `autoprefixer` in `postcss.config.mjs` |
| `tailwind-merge` | 2.6.0 | 3.7.x | v3 is the Tailwind 4 line; `extendTailwindMerge` still needed for custom groups |
| `tailwindcss-animate` | 1.0.7 | `tw-animate-css` 1.4.x | the shadcn v4 preset uses it; `tailwindcss-animate` is a v3 plugin |
| `tailwind-scrollbar` | 3.1.0 | 4.0.x | v4-compatible release |
| `prettier-plugin-tailwindcss` | 0.7.1 | 0.8.x | v4 config discovery via `tailwindStylesheet`; set `tailwindFunctions` |
| `shadcn` (CLI) | none | 4.21.x | v4 preset, `@theme inline` output, registry support for an internal registry |
| `class-variance-authority` | 0.7.1 | keep | still current; use `cva` in every atom |
| `eslint` | 8.57.1 | 9.x flat config (10.x once `eslint-config-next` for Next 15 supports it) | required by current plugins; `eslint-config-next` 15.5.x supports flat config |
| `eslint-plugin-better-tailwindcss` | none | 4.9.x | Tailwind 4 aware: `no-unregistered-classes` (catches `flex-end`, `text-md`, `animate-fadein`, `opacity-1`), `no-conflicting-classes`, `enforce-consistent-class-order`, `no-restricted-classes` (ban `dark:` and the default palette families during migration) |
| `@typescript-eslint/*` | bundled via next | 8.x | `no-deprecated` for the deprecation ratchet |
| `storybook` | 9.1.5 | 10.x | current major |
| `@storybook/addon-vitest` | none | 10.x | replaces the uninstalled `@storybook/test-runner`; runs stories as Vitest browser tests with axe via `@storybook/addon-a11y` |
| `style-dictionary` | none | 5.x | builds the JSON token source into `@theme` CSS, the Paper payload, and docs tables |
| `@axe-core/playwright`, `vitest-axe` | none | 4.13.x, 0.1.x | a11y gates in E2E and integration tests |
| `size-limit` | none | 14.x | bundle ratchet |
| `framer-motion` / `motion` | 13.3.0 / 13.2.0 | keep one | `motion` is the renamed package; pick it or `framer-motion`, not both |

Not recommended now: `next` 16 or `eslint-config-next` 16; they are a separate upgrade with their own breaking changes and are not needed for any step here.

### 8.3 Order of operations

Each step is its own PR, lands green, and is independently revertable.

1. **Tooling first (no visual change).** ESLint flat config; `eslint-plugin-better-tailwindcss` with `no-unregistered-classes` and `no-conflicting-classes` as errors; `no-restricted-imports` for `__legacy__`, `ui/`, the five icon packages, and `sonner`; `extendTailwindMerge`; Prettier `tailwindFunctions`. Allowlist existing violations by path so the rules block new ones immediately. This is Step 7 of Part 6, moved to the front because every later PR gets cheaper once it exists.
2. **Live bug fixes and deletions** (Part 6 Steps 1 and 3). Small, safe, and they shrink the surface the upgrade codemod has to touch.
3. **Tailwind 4 upgrade.** Run `npx @tailwindcss/upgrade` on a branch; it rewrites `globals.css` to `@import "tailwindcss"` and `@theme`, converts `@apply`, moves plugins to `@plugin`, flips `!` from prefix to suffix, and converts `bg-[--x]` to `bg-(--x)`. Then by hand: replace `darkMode` with `@custom-variant dark`, port the two custom plugins (`smoothShadowRing`, `grainTexture` if it survives) to `@utility`, swap `tailwindcss-animate` for `tw-animate-css`, bump `tailwind-merge` and `tailwind-scrollbar`, add `@reference` to any CSS module that uses `@apply`. Known risks: `theme()` calls are gone, `container` config is gone (use `@utility container`), default border colour is now `currentColor` (the `* { @apply border-border }` rule in `globals.css` already covers this), and `space-x/y` and `ring` defaults changed (ring is 1px, which matches most of the atoms). Verify with Chromatic re-enabled on this PR.
4. **shadcn init with the v4 preset.** `cssVariables: true`, style `new-york`, base colour `zinc`, `aliases.ui` pointing at the scratch folder. Replace the generated OKLCH values with the custom palette mapping decided in 8.1, add `success`/`warning`/`info`, set `--radius`. This PR replaces `styles/colors.ts` plus the `tailwind.config.ts` colour block with the token build output from `style-dictionary` (Part 7.1). After this PR the semantic classes (`bg-background`, `text-muted-foreground`, `border-border`) are the only colour classes new code should use, and the palette families are reachable but linted.
5. **Rebuild atoms on shadcn** (Part 6 Step 5), one component per PR, in this order because of the dependency graph: `Button`, `Input`/`Textarea`/`Select` on a shared field base, `Checkbox` (new), `Tooltip`, `Popover`, `Dialog`, `DropdownMenu`, `Tabs`, `Table`, `Skeleton`, `Badge`, `Alert`, `Toast` (shadcn uses `sonner` too, so this is mostly styling), `Calendar`/`DateInput`, `Avatar`, `Separator`, `Sheet`, `ScrollArea`, `Kbd`. Each PR migrates every importer of the legacy or `ui/` twin and deletes it.
6. **Storybook 10 and `addon-vitest`**, with a11y assertions on by default, run in CI. Chromatic stays on for visual diffs.
7. **Feature-code migration** (Part 6 Step 6) directory by directory, with the lint allowlist shrinking in every PR and the ratchet (Part 6 Step 8) asserting the counts only go down.
8. **Dark mode**, if wanted, is then a `.dark` block review plus removing `forcedTheme`, not a code change across 102 files.

### 8.4 Design decisions (made 2026-10-07)

These were open questions blocking the token mapping; they are now decided and are the inputs to step 4.

| Decision | Answer | Token consequence |
|---|---|---|
| Primary action colour | **Zinc.** The primary Button stays dark neutral. | `--primary` maps to `zinc-800`, `--primary-foreground` to white. Purple remains the brand accent only. |
| Which purple for the accent | Palette `purple-500` (`#7733f5`). The violet `--accent` (`hsl(262 83% 58%)`) is retired. | `--accent` and the focus `--ring` map to the `purple` ramp; delete the violet value. (Assumption: zinc was chosen for primary and no purple was named; the palette purple is already used 217 times, so it wins. Flag if the violet was intended.) |
| Page background | **`#FAFAFA`** (the current `--background`). | `body` drops `bg-[#F6F7F8]` and uses `bg-background`. Surfaces (cards, popovers) stay white. |
| Base radius | **Cards use `xl`.** Under shadcn's model `xl = calc(var(--radius) + 4px)`, so `--radius: 0.75rem` gives 12px fields (`lg`) and 16px cards (`xl`), matching the current atoms. | `--radius: 0.75rem`. Delete `xsmall..2xlarge`. |
| Control heights | **shadcn's 32 / 36 / 40.** | `h-8 / h-9 / h-10` for `sm / md / lg` on Button, Input, Select, SearchInput, Date and Time inputs, MultiToggle. The 46px `large` button and field go away. |
| Muted text | **`zinc-600`** for contrast. | `--muted-foreground` maps to `zinc-600` (`#68686F`, about 5.2:1 on white). `Text tone="muted"` follows. `zinc-500` is reserved for placeholders and disabled text. |
| Dark mode | **Near future, by swapping tokens only.** No `dark:` classes anywhere; the `.dark` block of semantic variables is the entire implementation. | Keep `forcedTheme="light"` until the semantic layer is complete. Delete every hand-placed `dark:` class now. Write the `.dark` block with the palette so enabling the theme is a one-line change later. |
