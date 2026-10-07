---
version: alpha
name: AutoGPT Platform
description: >-
  Design system of the AutoGPT Platform frontend (Next.js, Tailwind 3.4,
  Radix). Tokens below are the decided values; "Tokens today" in the prose
  says where the code still differs.
colors:
  primary: "#3E3E43"
  on-primary: "#FFFFFF"
  accent: "#7733F5"
  on-accent: "#FFFFFF"
  background: "#FAFAFA"
  surface: "#FFFFFF"
  text-primary: "#2C2C30"
  text-secondary: "#505057"
  text-muted: "#68686F"
  text-placeholder: "#83838C"
  border: "#DADADC"
  border-strong: "#C5C5C9"
  danger: "#D93636"
  success: "#149443"
  warning: "#876A00"
  focus-ring: "#7733F5"
typography:
  h1:
    {
      fontFamily: Poppins,
      fontSize: 2.75rem,
      fontWeight: 600,
      lineHeight: 3.5rem,
      letterSpacing: -0.033rem,
    }
  h2:
    {
      fontFamily: Poppins,
      fontSize: 2rem,
      fontWeight: 500,
      lineHeight: 2.5rem,
      letterSpacing: -0.02rem,
    }
  h3:
    {
      fontFamily: Poppins,
      fontSize: 1.75rem,
      fontWeight: 500,
      lineHeight: 2.5rem,
      letterSpacing: -0.01313rem,
    }
  h4:
    {
      fontFamily: Poppins,
      fontSize: 1.375rem,
      fontWeight: 500,
      lineHeight: 1.5rem,
    }
  h5:
    { fontFamily: Poppins, fontSize: 1rem, fontWeight: 500, lineHeight: 1.5rem }
  lead:
    {
      fontFamily: Geist,
      fontSize: 1.25rem,
      fontWeight: 400,
      lineHeight: 1.75rem,
    }
  large:
    { fontFamily: Geist, fontSize: 1rem, fontWeight: 400, lineHeight: 1.625rem }
  body:
    {
      fontFamily: Geist,
      fontSize: 0.875rem,
      fontWeight: 400,
      lineHeight: 1.375rem,
    }
  small:
    {
      fontFamily: Geist,
      fontSize: 0.75rem,
      fontWeight: 400,
      lineHeight: 1.125rem,
    }
  label:
    {
      fontFamily: Geist,
      fontSize: 0.6875rem,
      fontWeight: 500,
      lineHeight: 1.25rem,
      letterSpacing: 0.06875rem,
    }
  eyebrow:
    {
      fontFamily: Geist,
      fontSize: 0.75rem,
      fontWeight: 500,
      lineHeight: 1rem,
      letterSpacing: 0.06em,
    }
rounded:
  base: 0.75rem
  sm: 4px
  md: 8px
  field: 12px
  card: 16px
  full: 9999px
spacing:
  unit: 4px
  control-sm: 32px
  control-md: 36px
  control-lg: 40px
components:
  button-primary:
    {
      backgroundColor: "{colors.primary}",
      textColor: "{colors.on-primary}",
      rounded: "{rounded.full}",
      height: "{spacing.control-lg}",
    }
  button-secondary:
    {
      backgroundColor: "{colors.surface}",
      textColor: "#3E3E43",
      borderColor: "{colors.border}",
      rounded: "{rounded.full}",
    }
  input:
    {
      backgroundColor: "{colors.surface}",
      borderColor: "{colors.border}",
      rounded: "{rounded.field}",
      height: "{spacing.control-md}",
    }
  card: { backgroundColor: "{colors.surface}", rounded: "{rounded.card}" }
---

# AutoGPT Platform design system

## Purpose and scope

This is the one place that says what the frontend's UI is built from: the tokens, the components, and the rules, with the lint rule that enforces each. It covers everything under `autogpt_platform/frontend/src`. It replaces nothing in Storybook (`pnpm storybook`), which shows the components; this file says which ones to use.

History, evidence and the rebuild plan are in [the design system audit](../../docs/engineering/frontend-design-system-audit.md). The front matter follows the [design.md](https://github.com/google-labs-code/design.md) format: it holds the decided semantic values (each a palette step, listed under "Palette"), and the prose below says where the code still differs.

## Decisions

Made 2026-10-07 (audit Part 8.4). New code follows them now; the token rebuild (Tailwind 4 and shadcn, audit Part 8.3) moves the existing code onto them.

| Axis            | Decision                                                                                                            |
| --------------- | ------------------------------------------------------------------------------------------------------------------- |
| Primary action  | Zinc. The primary Button is `zinc-800` with white text. Purple is never the primary action.                         |
| Accent          | Palette `purple-500` (`#7733F5`). The violet in `--accent` (`#7C3BED`) is retired. Focus rings use the purple ramp. |
| Page background | `#FAFAFA`, through `bg-background`. Surfaces (cards, popovers, dialogs) are white.                                  |
| Radius          | Base `--radius: 0.75rem`. Fields are `rounded-lg` (12px), cards are `rounded-xl` (16px), pills are `rounded-full`.  |
| Control heights | 32 / 36 / 40px for `sm` / `md` / `lg` (`h-8` / `h-9` / `h-10`). The 46px size goes away.                            |
| Muted text      | `zinc-600` (about 5.2:1 on white). `zinc-500` is only for placeholders and disabled text.                           |
| Dark mode       | Later, and only by swapping the semantic variables in `.dark`. No `dark:` class anywhere.                           |
| Icons           | Hugeicons only (`@hugeicons/core-free-icons`, stroke-rounded), always through the `Icon` atom.                      |

## Tokens today

The sources are `src/components/styles/colors.ts` (palette), `src/app/globals.css` (shadcn variables), `tailwind.config.ts` (theme) and `src/components/atoms/Text/helpers.ts` (type scale). If this table and those files disagree, the files win and this table is out of date.

### Palette

`colors.ts` overrides Tailwind's ramps of the same name; `gray`, `neutral`, `stone`, `emerald`, `amber`, `violet`, `indigo`, `rose`, `lime` and `fuchsia` are Tailwind defaults and are banned by lint. `blue`, `sky`, `teal` and `cyan` are also defaults, have no decision yet, and are allowed for now. `zinc-950` is not defined and falls through to Tailwind's value; do not use it.

| Step | slate     | zinc      | red       | orange    | yellow    | green     | purple    | pink      |
| ---- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 50   | `#F7F8F9` | `#F9F9FA` | `#FEF5F5` | `#FFF3E6` | `#FEF9E6` | `#E8F6ED` | `#F1EBFE` | `#FDEDF5` |
| 100  | `#EFF1F4` | `#EFEFF0` | `#FDECEC` | `#FFDAB0` | `#FCEBB0` | `#B7E2C7` | `#EFE8FE` | `#F9C6DF` |
| 200  | `#CFD4DB` | `#DADADC` | `#FAC5C5` | `#FFC88A` | `#FAE28A` | `#94D5AC` | `#C0A1FA` | `#F6ABD0` |
| 300  | `#B8BFCA` | `#C5C5C9` | `#F69999` | `#FEAF54` | `#F8D554` | `#63C186` | `#A476F8` | `#F284BB` |
| 400  | `#8A97A8` | `#ADADB3` | `#F26969` | `#FE9F33` | `#F7CD33` | `#45B56E` | `#925CF7` | `#F06DAD` |
| 500  | `#64748B` | `#83838C` | `#EF4444` | `#FE8700` | `#F5C000` | `#16A34A` | `#7733F5` | `#EC4899` |
| 600  | `#5B6A7E` | `#68686F` | `#D93636` | `#E77B00` | `#DFAF00` | `#149443` | `#6C2EDF` | `#D7428B` |
| 700  | `#515E70` | `#505057` | `#AA3030` | `#B46000` | `#AE8800` | `#107435` | `#5424AE` | `#A8336D` |
| 800  | `#475263` | `#3E3E43` | `#832525` | `#8C4A00` | `#876A00` | `#0C5A29` | `#411C87` | `#822854` |
| 900  | `#2A313A` | `#2C2C30` | `#641D1D` | `#6B3900` | `#675100` | `#09441F` | `#321567` | `#631E40` |

Also: `white` is `#FEFEFE`, `black` is `#141414`, `yellow-25` and `yellow-150` exist for one component. `textBlack`, `textGrey` and `bgLightGrey` are deprecated; use `zinc-900`, `zinc-700` and `zinc-100`.

**Role of each step.** Zinc is the neutral ramp and carries most of the UI:

| zinc | Role                                                           |
| ---- | -------------------------------------------------------------- |
| 50   | Hover fill, sunken surface, subtle row                         |
| 100  | Soft fill, skeleton, pressed toggle                            |
| 200  | Default border, disabled fill                                  |
| 300  | Strong border, outline button                                  |
| 400  | Decorative icons only (2.2:1 on white, fails as text)          |
| 500  | Placeholder and disabled text (3.8:1)                          |
| 600  | Muted text (`Text tone="secondary"` today; the muted decision) |
| 700  | Secondary emphasis text                                        |
| 800  | Primary action fill, strong text                               |
| 900  | Primary text (`Text tone="primary"`)                           |

The colour ramps follow one pattern: 50 to 100 tinted backgrounds (alerts, badges), 200 to 300 borders on those backgrounds, 400 to 500 the solid hue (dots, icons, fills), 600 hover of the solid hue and text on white, 700 to 800 text on a 50 to 100 tint. Red is error and destructive, green success, yellow warning (orange for its icon), purple brand and accent, slate a cool neutral used in a few places, pink decorative.

### Semantic variables (shadcn)

Defined as HSL in `globals.css`, exposed as Tailwind colours. Prefer these over palette steps where one fits; the token rebuild remaps them to the palette (the "Target" column).

| Class stem                | Today                 | Target               | Role                                                                   |
| ------------------------- | --------------------- | -------------------- | ---------------------------------------------------------------------- |
| `background`              | `#FAFAFA`             | same                 | Page background. `body` still paints `bg-[#F6F7F8]` until the rebuild. |
| `foreground`              | `#09090B`             | `zinc-900`           | Default text                                                           |
| `card`, `popover`         | `#FFFFFF`             | white                | Raised surfaces                                                        |
| `primary` / `-foreground` | `#18181B` / `#FAFAFA` | `zinc-800` / white   | Primary action                                                         |
| `secondary`, `muted`      | `#F4F4F5`             | `zinc-100`           | Soft fills                                                             |
| `muted-foreground`        | `#71717A`             | `zinc-600`           | Muted text                                                             |
| `accent` / `-foreground`  | `#7C3BED` / white     | `purple-500` / white | Brand accent                                                           |
| `destructive`             | `#EF4444`             | `red-500`            | Destructive action                                                     |
| `border`                  | `#E4E4E7`             | `zinc-200`           | Default border (`*` gets `border-border`)                              |
| `input`                   | `#D6D6DB`             | `zinc-200`           | Field border                                                           |
| `ring`                    | `#18181B`             | `purple-500`         | Focus ring                                                             |
| `sidebar-*`               | various               | to be folded in      | Only `ui/sidebar` uses them                                            |

`--radius` is `0.5rem` today (target `0.75rem`). The `.dark` block exists but nothing activates it: `providers.tsx` forces the light theme.

### Spacing

Tailwind's default 4px scale, plus `4.5` (18px), `18` (72px), `68` (272px), `71` (284px) and `76` (304px). Use the scale before an arbitrary value; `h-[2.25rem]` is `h-9`, `w-[12rem]` is `w-48`. Page width `max-w-[1360px]` recurs and has no token yet.

### Radius

| Class                                | Today                                 | After the rebuild               |
| ------------------------------------ | ------------------------------------- | ------------------------------- |
| `rounded-sm` / `md` / `lg`           | 4 / 6 / 8px (from `--radius: 0.5rem`) | 8 / 10 / 12px (fields use `lg`) |
| `rounded-xl` / `2xl` / `3xl`         | 12 / 16 / 24px (Tailwind)             | 16px cards (`xl`) and up        |
| `rounded-xsmall` … `rounded-2xlarge` | 4 / 8 / 12 / 16 / 20 / 24px           | deleted                         |
| `rounded-full`                       | pill                                  | pill (Button, Badge)            |

Until the rebuild, match the atoms: fields `rounded-xl` (12px), cards `rounded-large` or `rounded-2xl` (16px), buttons `rounded-full`. Do not add new uses of `xsmall` … `2xlarge`.

### Typography

Two families: Poppins for headings (`font-poppins`), Geist Sans for everything else (`font-sans`), Geist Mono for code (`font-mono`), loaded with `next/font` in `src/components/styles/fonts.ts`. All type goes through the `Text` atom: `<Text variant="body" tone="secondary">`.

| Variant                         | Size / line height | Weight          | Notes                 |
| ------------------------------- | ------------------ | --------------- | --------------------- |
| `h1`                            | 44 / 56px          | 600             | Poppins               |
| `h2`                            | 32 / 40px          | 500             | Poppins               |
| `h3`                            | 28 / 40px          | 500             | Poppins               |
| `h4`                            | 22 / 24px          | 500             | Poppins               |
| `h5`                            | 16 / 24px          | 500             | Poppins               |
| `lead`, `-medium`, `-semibold`  | 20 / 28px          | 400 / 500 / 600 |                       |
| `large`, `-medium`, `-semibold` | 16 / 26px          | 400 / 500 / 600 |                       |
| `body`, `-medium`               | 14 / 22px          | 400 / 500       | Default body copy     |
| `small`, `-medium`              | 12 / 18px          | 400 / 500       |                       |
| `label`                         | 11 / 20px          | 500             | Uppercase, tracked    |
| `eyebrow`                       | 12 / 16px          | 500             | Uppercase, `zinc-500` |

Tones: `primary` (`zinc-900`), `secondary` (`zinc-600`), `muted` (`zinc-500`), `danger` (`red-600`). Without a tone the variant is black. Use a tone, not `!text-zinc-*`. `Text` unmasks its content in Sentry replays by default; pass `unmask={false}` for user data. Off-scale sizes (`text-[13px]`, `text-[11px]`) are an open design question; do not add more.

### Shadow and elevation

No elevation scale exists yet. Use, in order of height: `shadow-subtle` (1px hairline), `shadow-sm` (secondary and icon buttons), `shadow-md` (Card, Popover, DropdownMenu), `shadow-lg` (sub-menus), `shadow-2xl` (floating panels, command palette). `smooth-shadow-ring-sm` is a soft shadow plus a 1px ring. Do not write new `shadow-[...]` values.

### Motion

Durations: Tailwind `duration-150` (hover), `duration-200` (default), `duration-300` (panels). Easing: `ease-out`, or `ease-out-quint` for expand and collapse. Animations in the theme: `fade-in`, `fade-up`, `accordion-down/up`, `collapsible-down/up`, `shimmer`, `shimmer-text`, `shake`, `progress-bar`, `grow-line`, `aurora`. Respect reduced motion: `motion-reduce:` classes, or framer-motion's `useReducedMotion`.

## Component catalog

All under `src/components/`. Every folder has a story except the two helper-only ones. "Replaces" names the legacy or `ui/` file it supersedes.

**Atoms**

| Atom               | Purpose                                                                      | Replaces                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| AutoGPTLogo        | Brand logo, plus a white variant                                             |                                                                                                           |
| Avatar             | Image avatar with fallback                                                   |                                                                                                           |
| Badge              | Status and label pill                                                        | `__legacy__/ui/badge`                                                                                     |
| Button             | Every button: variants, sizes, loading, icons, `as="NextLink"`               | `__legacy__/ui/button`, `ui/button`                                                                       |
| Card               | Bordered content container                                                   | `__legacy__/ui/card`                                                                                      |
| DateInput          | Date field with calendar popover                                             | (still uses `__legacy__` popover, calendar, button)                                                       |
| DateTimeInput      | Date and time field                                                          | (still uses `__legacy__` popover, calendar)                                                               |
| Emoji              | Emoji at a given size                                                        |                                                                                                           |
| FadeIn             | Fade-in wrapper                                                              |                                                                                                           |
| FileInput          | File upload with preview                                                     |                                                                                                           |
| GlassPixelBackdrop | Decorative backdrop                                                          |                                                                                                           |
| Icon               | Renders a Hugeicon at the system stroke width                                | `__legacy__/ui/icons`, every other icon library                                                           |
| Input              | Text, password, number, amount and textarea field with label, hint and error | `__legacy__/ui/input`, `ui/input`, `__legacy__/ui/textarea`, `ui/textarea` (still wraps the legacy input) |
| LLMItem            | LLM provider logo and name                                                   |                                                                                                           |
| Link               | In-app and external links                                                    | raw `<a>` and bare `next/link`                                                                            |
| LoadingSpinner     | Spinner                                                                      | `__legacy__/ui/loading`, `ui/spinner`                                                                     |
| OverflowText       | Truncated text with a tooltip                                                |                                                                                                           |
| Progress           | Progress bar                                                                 |                                                                                                           |
| Reveal             | Staggered entrance animation                                                 |                                                                                                           |
| Select             | Labelled select from an options array                                        | `__legacy__/ui/select` (still wraps it)                                                                   |
| Skeleton           | Loading placeholder                                                          | `__legacy__/ui/skeleton`, `ui/skeleton`                                                                   |
| SwapFade           | Cross-fade on key change                                                     |                                                                                                           |
| Switch             | Toggle switch                                                                |                                                                                                           |
| Text               | All typography                                                               | raw `<p>`, `<h1>`…`<h6>` with classes                                                                     |
| TimeInput          | Time field                                                                   |                                                                                                           |
| ToggleChip         | Icon and label toggle chip                                                   |                                                                                                           |
| Tooltip            | Tooltip (`BaseTooltip.tsx`)                                                  | `ui/tooltip`                                                                                              |

**Molecules**

| Molecule                                      | Purpose                                                         | Replaces                                                       |
| --------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| Accordion                                     | Accordion primitives                                            | re-exports `ui/accordion`                                      |
| Alert                                         | Inline alert with icon                                          |                                                                |
| AutopilotAvatar, ExpertAvatar, WorkflowAvatar | Identity avatars                                                |                                                                |
| ExpertAvatarPicker, ExpertIdentityDetails     | Expert identity editing and display                             |                                                                |
| Breadcrumbs                                   | Breadcrumb trail                                                |                                                                |
| Collapsible                                   | Disclosure section                                              | `__legacy__/ui/collapsible` (still wraps it), `ui/collapsible` |
| Confetti                                      | Confetti effect                                                 |                                                                |
| Dialog                                        | Modal, drawer on mobile                                         | `__legacy__/ui/dialog`                                         |
| DropdownMenu                                  | Dropdown menu                                                   | `__legacy__/ui/dropdown-menu`                                  |
| ErrorBoundary, ErrorCard                      | Render-error boundary and error display                         |                                                                |
| Form                                          | react-hook-form bindings                                        | `__legacy__/ui/form`                                           |
| FullscreenDialog                              | Full-screen modal                                               |                                                                |
| GlassOrb, TypingText                          | Decorative effects                                              |                                                                |
| InfiniteList                                  | Infinite-scroll list                                            |                                                                |
| InformationTooltip                            | Info icon with tooltip                                          |                                                                |
| InstallWorkflowPicker                         | Pick an expert or workflow to install                           |                                                                |
| IntegrationLogo, IntegrationsMarquee          | Provider logos                                                  |                                                                |
| MultiToggle                                   | Segmented toggle group                                          |                                                                |
| NotionAvatar                                  | Avatar composition helpers (no component, no story)             |                                                                |
| PlanCard                                      | Pricing data helpers (no component, no story)                   |                                                                |
| Popover                                       | Popover                                                         | `__legacy__/ui/popover`                                        |
| RunStatusBadge                                | Agent run status badge                                          | `__legacy__/Status` (partly)                                   |
| ScrollableTabs                                | Tabs synced to scroll position                                  |                                                                |
| SearchInput                                   | Search field                                                    |                                                                |
| SecondaryMenu                                 | Context menu                                                    |                                                                |
| ShowMoreText                                  | Clamped text with a toggle                                      |                                                                |
| Table                                         | Editable table from column config                               | `__legacy__/ui/table` (still wraps it)                         |
| TabsLine                                      | Underline tabs                                                  | `__legacy__/ui/tabs`                                           |
| TallyPoup                                     | Tally feedback popup (folder name has a typo)                   |                                                                |
| TimePicker                                    | Hour and minute picker                                          |                                                                |
| Toast                                         | Toasts (`useToast`, `Toaster`)                                  | direct `sonner` calls                                          |
| `file-tree.tsx`                               | File tree view (loose file, uses `ui/button`, `ui/scroll-area`) |                                                                |

**Organisms:** ApprovalFields, BriefingCard, FloatingReviewsPanel, NeedsAttentionList, PendingReviewCard, PendingReviewsList, SearchCommandModal (uses `ui/button`, `ui/input`, `ui/separator`), SubscriptionPlans, TrialCard, VoicePicker, WorkOutputSheet (uses `ui/sheet`). Each is a product feature built from atoms and molecules; reuse them rather than copying.

## Rules

One rule per line, with what enforces it. "Allowlisted" means existing violators are listed in `eslint-allowlist.json` and new ones fail.

1. Import nothing from `src/components/__legacy__`. Enforced: `@typescript-eslint/no-restricted-imports` (`legacy`), allowlisted.
2. Import `src/components/ui/*` only from inside `src/components`. Enforced: `no-restricted-imports` (`ui`), allowlisted.
3. Icons are Hugeicons through the `Icon` atom; no `lucide-react`, `@phosphor-icons/react`, `@radix-ui/react-icons`, `react-icons`, or direct `@hugeicons/react` (type imports are fine). Enforced: `no-restricted-imports`, allowlisted.
4. Toasts go through `molecules/Toast`, never `sonner`. Enforced: `no-restricted-imports` (`sonner`), allowlisted.
5. Use only classes Tailwind generates. Enforced: `better-tailwindcss/no-unknown-classes`, allowlisted.
6. Do not combine classes that set the same property. Enforced: `better-tailwindcss/no-conflicting-classes` (reports nothing until Tailwind 4).
7. No `gray`, `neutral`, `stone`, `emerald`, `amber`, `violet`, `indigo`, `rose`, `lime` or `fuchsia` colour classes; map them to `zinc`, `zinc`, `zinc`, `green`, `yellow`, `purple`, `purple`, `red`. Enforced: `better-tailwindcss/no-restricted-classes`, allowlisted.
8. `blue`, `sky`, `teal`, `cyan`: no decision yet; leave existing uses, avoid new ones. Not yet enforced.
9. No `dark:` classes. Not yet enforced.
10. No hex or `rgb()` colour classes (`bg-[#F9F9FA]`); use a palette step. Not yet enforced.
11. Prefer semantic classes (`bg-background`, `text-muted-foreground`, `border-border`) where one fits. Not yet enforced.
12. Typography through `Text`; no raw `<p>` or `<h1>`…`<h6>` with classes. Not yet enforced.
13. Buttons through the `Button` atom; no raw `<button className>`. Not yet enforced.
14. In-app links through the `Link` atom (or `Button as="NextLink"`); `next/link` only inside those atoms. Not yet enforced.
15. No `!important` overrides of an atom's classes; fix or extend the atom. Not yet enforced.
16. No arbitrary values where the scale has one (`h-[2.25rem]` is `h-9`). Not yet enforced.
17. Keyboard handlers use `isKey()` from `@/lib/keyboard`. Enforced: `no-restricted-syntax` (keyboard rules in `eslint.config.mjs`).
18. Components use `interface Props`, function declarations, and no barrel files. Not yet enforced (review).
19. A new or changed design-system component comes with a story. Not yet enforced (PR template checklist, CODEOWNERS review).
20. Never add to `eslint-allowlist.json`. Review only; `scripts/eslint-allowlist.test.ts` fails on stale entries, not new ones.

### The allowlist

`eslint-allowlist.json` lists files that broke a rule before the rule existed, per import restriction and per Tailwind rule. It only shrinks. When a file is fixed, delete its entry, or rebuild the whole list:

```bash
npx tsx scripts/eslint-allowlist-regenerate.ts
```

The script lints everything with the allowlist removed, rewrites the file, and prints what it removed and added. Anything it adds is a new violation: fix it rather than committing the addition.

## Migration status

Wave 1 (merged into `ds/integration`): lint enforcement, live atom fixes, dead code, Storybook coverage (every component folder has a story, `a11y.test: "error"` declared), and the `admin`, `copilot`, `library`, `profile` and `settings` migrations. Wave 2 (in progress): `build`, `marketplace` and the remaining app routes, `components/contextual` and `layout`, new atoms (Checkbox, Textarea, Separator and others), the dead-code backlog, and this document. Counts before and after are in the audit's "Wave 1 and 2 outcome".

Not done yet: the Tailwind 4 upgrade, the shadcn token rebuild (which applies the decisions above), the 32/36/40 control heights, a `test-storybook` runner, and visual regression in CI (the Chromatic job runs only when `CHROMATIC_PROJECT_TOKEN` is set).

## Deprecated

Do not add imports of these. Each has a replacement or is waiting for one.

| Deprecated                                                                                                                                                                                                                        | Use instead                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `__legacy__/ui/button`, `ui/button`                                                                                                                                                                                               | `atoms/Button`                                                                       |
| `__legacy__/ui/input`, `ui/input`                                                                                                                                                                                                 | `atoms/Input`                                                                        |
| `__legacy__/ui/textarea`, `ui/textarea`                                                                                                                                                                                           | `atoms/Input type="textarea"`                                                        |
| `__legacy__/ui/select`                                                                                                                                                                                                            | `atoms/Select`                                                                       |
| `__legacy__/ui/skeleton`, `ui/skeleton`                                                                                                                                                                                           | `atoms/Skeleton`                                                                     |
| `__legacy__/ui/loading`, `ui/spinner`                                                                                                                                                                                             | `atoms/LoadingSpinner`                                                               |
| `__legacy__/ui/badge`                                                                                                                                                                                                             | `atoms/Badge`                                                                        |
| `__legacy__/ui/card`                                                                                                                                                                                                              | `atoms/Card`                                                                         |
| `__legacy__/ui/icons`                                                                                                                                                                                                             | `atoms/Icon` with `@hugeicons/core-free-icons`                                       |
| `ui/tooltip`                                                                                                                                                                                                                      | `atoms/Tooltip`                                                                      |
| `__legacy__/ui/dialog`                                                                                                                                                                                                            | `molecules/Dialog`                                                                   |
| `__legacy__/ui/popover`                                                                                                                                                                                                           | `molecules/Popover`                                                                  |
| `__legacy__/ui/dropdown-menu`                                                                                                                                                                                                     | `molecules/DropdownMenu`                                                             |
| `__legacy__/ui/form`                                                                                                                                                                                                              | `molecules/Form`                                                                     |
| `__legacy__/ui/tabs`                                                                                                                                                                                                              | `molecules/TabsLine`                                                                 |
| `__legacy__/ui/collapsible`, `ui/collapsible`                                                                                                                                                                                     | `molecules/Collapsible`                                                              |
| `ui/accordion`                                                                                                                                                                                                                    | `molecules/Accordion`                                                                |
| `__legacy__/ui/table`                                                                                                                                                                                                             | `molecules/Table` for editable tables; read-only data tables have no replacement yet |
| `__legacy__/Status`                                                                                                                                                                                                               | `molecules/RunStatusBadge` where it fits                                             |
| `__legacy__/ui/checkbox`, `label`, `separator`, `scroll-area`, `sheet`, `calendar`, `carousel`, `command`, `multiselect`, `pagination-controls`; `ui/separator`, `scroll-area`, `sheet`, `sidebar`, `input-group`, `button-group` | No replacement yet (atoms are being added in wave 2). Keep existing uses, add none.  |
| `__legacy__/Sidebar`, `SortDropdown`, `SearchFilterChips`                                                                                                                                                                         | No replacement yet                                                                   |
| `ui/aurora-background`, `vortex`, `dot-distortion-shader`, `text-generate-effect`                                                                                                                                                 | Decorative one-offs; do not reuse                                                    |
| `textBlack`, `textGrey`, `bgLightGrey` colours                                                                                                                                                                                    | `zinc-900`, `zinc-700`, `zinc-100`                                                   |
| `rounded-xsmall` … `rounded-2xlarge`                                                                                                                                                                                              | The Tailwind radius scale (see Radius)                                               |
