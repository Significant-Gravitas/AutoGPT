# Design system rules

The rules for frontend styling in `autogpt_platform/frontend`, and the lint rule that enforces each one. Every `@shadcn/lint` message ends with "See docs/engineering/design-system.md." (`settings.shadcn.note` in `.oxlintrc.json`), which is this page.

`pnpm lint` runs ESLint (`.eslintrc.json`) and oxlint with `@shadcn/lint` (`.oxlintrc.json`). Directories that are not migrated yet get the `shadcn/*` rules at `warn` through the `overrides` in `.oxlintrc.json`, and some get the ESLint import rules at `warn` in `.eslintrc.json`; those lists are the migration checklist. On a pull request, CI also runs `.oxlintrc.strict.json`, which has no `warn` list, on every `.tsx` file the PR adds or changes. Until the migration lands that step is report-only: it lists the violations in the files you touched (job summary and warning annotations) and does not fail CI (SECRT-3153 makes it blocking). Fix what you touch.

## Tokens

Colour comes from the role tokens in `src/app/globals.css` (`--color-<role>`, registered in `tailwind.config.ts`). Use them with any colour utility, opacity modifiers included: `bg-surface`, `text-foreground-secondary`, `border-border-base`, `ring-focus`, `bg-overlay/20`.

- Surfaces: `canvas`, `surface`, `surface-muted`, `surface-hover`, `surface-active`, `overlay`
- Text: `foreground-base`, `foreground-strong`, `foreground-emphasis`, `foreground-body`, `foreground-secondary`, `foreground-tertiary`, `foreground-subtle`, `foreground-disabled`, `placeholder`
- Borders: `separator`, `border-base`, `border-strong`, `focus`
- Action and accent: `action`, `action-hover`, `action-foreground`, `brand`, `brand-hover`, `accent-surface`, `accent-strong`
- Status: `error`, `error-hover`, `error-foreground`, `error-surface`, `warning`, `warning-foreground`, `warning-surface`, `warning-border`, `success`, `success-foreground`, `success-surface`, `info`, `info-foreground`, `info-surface`

The raw palette (`zinc-*`, `neutral-*`, `purple-*`, …) and hex values belong in `globals.css` only. Rules: `shadcn/no-raw-colors` for palette classes, `shadcn/no-arbitrary-values` for `bg-[#fff]`. `white`, `black` and the old `customGray-*` are not flagged yet; write `bg-surface` and `text-foreground-base` anyway.

The shadcn names (`background`, `foreground`, `muted-foreground`, `border`, `input`, `ring`, `primary`, `secondary`, `accent`, `destructive`, `card`, `popover`, `sidebar-*`) still resolve, but several render a different value from the role of the same intent (`TODO(tokens)` in `globals.css`), so prefer the role in new code. When migrating a class, keep the value: same value, new name. If no role has the exact value, leave the class and add `// TODO(tokens)`; never pick the nearest role.

## Primitives

`src/components/{atoms,molecules,organisms}` own their look. At the call site, pick a `variant` or `size` (and a `tone` on `Text`); `className` is for layout only: width, flex and grid placement, margin, position. A treatment you repeat becomes a variant on the primitive, never a class string copied between call sites. `Text` also accepts `truncate`, `line-clamp-*`, `whitespace-*`, `break-*`, `text-balance`, `text-pretty` and text alignment, and `Skeleton` accepts shape. Rule: `shadcn/no-restyle` (off inside the three design-system directories, which define the looks).

Classes on a primitive are static strings the linter can read, so no `` `bg-${tone}` ``; map a prop to whole class names instead. Rule: `shadcn/require-static-classes`. Tooltips use the `Tooltip` atom, not the `title` attribute. Rule: `react/forbid-dom-props`.

## Styles

Tailwind classes go in the component that renders the element. Global CSS is only for DOM we do not render as JSX (Streamdown output, xyflow, KaTeX), keyframes and token definitions. Not enforced by lint; reviewers check it.

No `style={{}}` or `<style>` for static values; pass a value known only at runtime through a CSS custom property. Rule: `shadcn/no-inline-styles`. No arbitrary values except for layout and transitions (`w-[320px]`, `top-[7px]` and `transition-[width]` pass); the error names the scale class with the same value, for example `p-[3px]` → `p-0.75`. Rule: `shadcn/no-arbitrary-values`.

## Typography

Text renders through the `Text` atom by default: `<Text variant="body">`, with `as` for the tag (`<Text variant="small" as="span">`) and `tone` for the colour. The variants are `h1` to `h5`, `lead`, `large`, `body` and `small` (with `-medium` weights, plus `-semibold` on `lead` and `large`), `label` and `eyebrow`. Rule: `shadcn/no-restyle` (the `Text` contract: `<Text>` owns its typography and colour, and `className` takes layout, truncation, wrapping and alignment only).

Use a role class only where `Text` can't render the element identically (same tag, display and margins), and inside primitives. Each role bundles its line-height: `text-tiny` (10/14px), `text-caption-sm` (11/16), `text-caption` (12/18), `text-body-sm` (13/20), `text-body` (14/22), `text-body-md` (15/24), `text-body-lg` (16/26), `text-lead` (20/28), `text-heading-sm` (22/24), `text-heading` (28/40), `text-title` (32/40) and `text-display` (44/56). Override the line-height only with `leading-inherit`, `leading-4.5`, `leading-5.5`, `leading-6.5` or a Tailwind step. Letter-spacing is `tracking-display`, `tracking-heading`, `tracking-title`, `tracking-caps` or `tracking-eyebrow` (all in em), and weights use Tailwind's scale (`font-medium`, `font-semibold`). When you swap a raw element for `Text`, it must render identically; if it can't, keep the role classes. Choosing `Text` over role classes is a review check, not a lint rule.

Never use raw sizes: `text-[13px]`, `leading-[..]`, `tracking-[..]`, `font-[..]` or inline font styles. The rare sizes that are still raw (5, 6, 9, 17, 23, 26, 35, 36, 48 and 64px, fractional rems, the `Text` `label` size and the rem-based `Text` tracking) carry `// TODO(tokens): SECRT-3152` until they are snapped; never round one to the nearest role. Rules: `shadcn/no-arbitrary-values`, `shadcn/no-inline-styles`.

## Dark mode

No `dark:` classes: a theme is a set of token values, not a second set of classes. `tailwind.config.ts` ties `dark:` to a `.dark-mode` class the app never sets, so these classes never render. Not enforced yet: DS4 left the `dark:` rule out and DS4b (SECRT-3130) adds it. Until then only a `dark:` class with a raw colour (`dark:bg-zinc-800`) fails, through `shadcn/no-raw-colors`.

## Icons

Hugeicons only: icon data from `@hugeicons/core-free-icons`, rendered with the `Icon` atom (`<Icon icon={Delete02Icon} size={16} />`), which applies the 2px stroke. Type icon props with `IconSvgElement` from `@hugeicons/react`. Importing `lucide-react`, `@phosphor-icons/react`, `react-icons` or `@radix-ui/react-icons`, or rendering `HugeiconsIcon` directly, is an error. Rule: `no-restricted-imports`. `Icon` takes its colour from the surrounding text; `className` on it is layout only. Rule: `shadcn/no-restyle` (the `Icon` contract).

## Imports

Feature code never imports `@/components/__legacy__/*`, or `@/components/ui/*` (the shadcn primitives `src/components` is built from); import the atom, molecule or organism instead. `extendedButtonVariants` is internal to `Button`. Rule: `no-restricted-imports`.

## Composition reference

Placeholder: DS8 (SECRT-3114) names one composition reference here, a slot-object card (`Root`, `Row`, `Icon`, `Content`, `Actions`) with a `variant` and a density. Not a lint rule.

## Escape hatch

Fix the code first. When a suppression is unavoidable, scope it to one line and one rule, and give the reason on the same line after `--`. `shadcn/*` rules are oxlint-only, so suppress them with `oxlint-disable-next-line`; an `eslint-disable` comment naming a `shadcn/*` rule fails ESLint with "Definition for rule … was not found". ESLint rules take `eslint-disable-next-line`.

```tsx
{/* oxlint-disable-next-line shadcn/no-arbitrary-values -- xyflow positions handles in px */}
```

`pnpm lint` runs oxlint with `--report-unused-disable-directives` and ESLint with `--report-unused-disable-directives-severity warn`, so a suppression that no longer suppresses anything shows up as a warning. The same-line reason is not enforced by a lint rule yet; reviewers check it.

## Evidence gate

Chromatic is off, so every design-system PR puts four sections in its description:

1. **Counts.** `pnpm exec oxlint -c .oxlintrc.strict.json <dir> | wc -l` on the PR's base and on its head, for each directory in scope. Remove a directory from the `warn` overrides in `.oxlintrc.json` only when its count is 0.
2. **Value equivalence.** The `old class | new class | resolved value` table that `scripts/check-token-equivalence.ts` prints for every class replaced; it exits 1 on any difference. A docs-only PR writes "0 classes replaced; no rows".
3. **Screenshots.** `scripts/screenshot-pages.ts` against the running app on the base (a `git worktree`) and on the head, at 1440x900 and 390x844, with 0 changed pixels from its `compare`. Store the images outside the branch (an evidence-only branch on a fork, or PR attachments) and link them.
4. **What improved.** Two lines the team can read.
