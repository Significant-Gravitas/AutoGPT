# Design system rules

The rules for styling in `autogpt_platform/frontend`, and the lint rule that enforces each one. Every `design-system/*` lint message ends with "See docs/engineering/design-system.md.", which is this page.

## How the rules are enforced

`pnpm lint` runs oxlint with `.oxlintrc.json`, which loads two local plugins: `scripts/eslint-rules/plugin.mts` (`autogpt/*`, the React Compiler rules) and `scripts/eslint-rules/design-system.mts` (`design-system/*`, the rules on this page). The `design-system/*` rules apply to `src/**`. They don't apply to tests, `src/tests`, `src/playwright`, `src/mocks` or the generated API client. Stories skip the four styling rules (`no-raw-colors`, `no-arbitrary-values`, `no-restyle`, `no-inline-styles`). CI runs `pnpm lint` in the frontend `lint` job, so a violation fails the pull request.

Six of the rules are [`@shadcn/lint`](https://github.com/shadcn-ui/lint) rules, wrapped by `design-system.mts`. The rest are our own.

### The baseline

The rules were turned on with about 9,000 existing violations, so each rule is a ratchet:

- `scripts/eslint-rules/design-system-baseline.json` holds, per file and rule, how many violations the file had when the rule was turned on.
- A file at or under its count passes. A file over its count fails, and the error lists every violation of that rule in the file, since lint can't tell which one is new.
- A file the baseline doesn't list must have none. That covers every new file.
- A file with fewer violations than its count fails too (`design-system/baseline-outdated`), so a removed violation is locked in: run `pnpm lint:ds-baseline` and commit the lower count.
- A violation suppressed with an `oxlint-disable` or `eslint-disable` comment doesn't count, the same as when the baseline is written.

To update the baseline, run this from `autogpt_platform/frontend`:

```bash
pnpm lint:ds-baseline
```

It rewrites `design-system-baseline.json` from the current tree. Run it in two cases:

- **You removed violations.** Commit the lower counts so they can't come back.
- **You moved, renamed or split a file.** The counts are keyed by path, so a moved file reads as a new file with violations. Say in the pull request that the increase is a move. The baseline is in CODEOWNERS, so a raised count is reviewed.

Never raise a count to make new code pass. Fix the code, or suppress the one line with a reason (see [Escape hatch](#escape-hatch)).

## Tokens

Colour comes from the role tokens in `src/app/globals.css` (`--color-<role>` on `:root`, registered in its `@theme`). Use them with any colour utility, opacity modifiers included: `bg-surface`, `text-foreground-secondary`, `border-border-base`, `ring-focus`, `bg-overlay/20`.

- Surfaces: `canvas`, `surface`, `surface-muted`, `surface-hover`, `surface-active`, `overlay`
- Text: `foreground-base`, `foreground-strong`, `foreground-emphasis`, `foreground-body`, `foreground-secondary`, `foreground-tertiary`, `foreground-subtle`, `foreground-disabled`, `placeholder`
- Borders: `separator`, `border-base`, `border-strong`, `focus`
- Action and accent: `action`, `action-hover`, `action-foreground`, `brand`, `brand-hover`, `accent-surface`, `accent-strong`
- Status: `error`, `error-hover`, `error-foreground`, `error-surface`, `warning`, `warning-foreground`, `warning-surface`, `warning-border`, `success`, `success-foreground`, `success-surface`, `info`, `info-foreground`, `info-surface`
- Builder: `builder-canvas`, `builder-grid`, `node-surface`, `node-border`, `node-selected`, `edge`, `handle`

The raw palette (`zinc-*`, `neutral-*`, `purple-*`, …) lives in `tailwind.v3-preset.ts`, and hex values belong in `globals.css` only. The palette stays out of `globals.css`'s `@theme` because `@shadcn/lint` reads every colour there as a token.

- **Rule `design-system/no-raw-colors`:** palette classes such as `bg-zinc-200`, in `className`, `cn()`/`cva()` and any other string.
- **Rule `design-system/no-arbitrary-values`:** arbitrary colours such as `bg-[#fff]` and `text-[rgb(…)]`.

The error names the token with the same colour when one exists: `"text-zinc-500" … Use "text-foreground-tertiary" instead`. It reads the exact matches from `scripts/codemod-tokens.ts` and the role values from `globals.css`. When no token has the colour, the error lists the roles for that utility. Pick the one whose meaning fits and check the screen, because the colour changes, or add a role.

`white`, `black` and the old `customGray-*` aren't flagged yet; write `bg-surface` and `text-foreground-base` anyway. The shadcn names (`background`, `foreground`, `muted-foreground`, `border`, `primary`, `accent`, `destructive`, …) still resolve, but several render a different value from the role with the same intent (`TODO(tokens)` in `globals.css`), so prefer the role in new code.

When you migrate a class, keep the value: same value, new name. If no role has the exact value, leave the class and add `// TODO(tokens)`. Never pick the nearest role.

`src/components/__tests__/design-system-contract.test.ts` checks that:
- every role on `globals.css`'s `:root` is registered in its `@theme`, and the other way round, and `color-roles.ts` lists them all;
- `.dark` only redefines names that `:root` declares;
- the app's palette values (`APP_COLORS` in `tailwind.v3-preset.ts`) get no new colours, and `globals.css` declares none.

## Dark mode

No `dark:` or `light:` classes. A theme is a set of token values, not a second set of classes. `globals.css` ties `dark:` to a `.dark` ancestor (`@custom-variant dark`), which the app never sets, so these classes never render. Delete them and keep the light value; the role tokens carry the theme. Rule: `design-system/no-dark-variant`.

## Primitives

`src/components/{atoms,molecules,organisms}` own their look. At the call site, pick a `variant` or `size` (and a `tone` on `Text`). Use `className` only for layout: width, flex and grid placement, margin, position. A treatment you repeat becomes a variant on the primitive, never a class string copied between call sites.

`Text` also accepts `truncate`, `line-clamp-*`, `whitespace-*`, `break-*`, `text-balance`, `text-pretty` and text alignment; `Icon` accepts layout only; `Skeleton` also accepts shape. Rule: `design-system/no-restyle`. It is off inside the three design-system directories, which define the looks.

Classes on a primitive must be static strings the linter can read, so no `` `bg-${tone}` ``; map a prop to whole class names instead. Rule: `design-system/require-static-classes`.

Each primitive marks its root element with `data-slot`. The contract test lists the five that predate this, and the list only shrinks.

## Classes Tailwind knows

Every class must be one this project's Tailwind generates: a typo such as `text-md` or `flex-end` silently renders nothing. Rule: `design-system/no-unknown-classes`.

`components.json` keeps `tailwind.css` at `src/app/globals.css`, the stylesheet the app loads. That is where the shadcn CLI writes, and where `@shadcn/lint` reads the colour tokens from. `globals.css` has no `@import "tailwindcss"` (Tailwind 3 compat), so for the grammar `@shadcn/lint` falls back to the one stylesheet that does: `src/app/tailwind-grammar.css`, which imports Tailwind and `globals.css` (and through it `tailwind.config.ts`). Nothing imports that file, so it never ships. `design-system.mts` drops the notice `@shadcn/lint` prints about this fallback. A few class names that aren't Tailwind are allowed in `.oxlintrc.json`: the PostHog and xyflow markers (`ph-no-capture`, `nodrag`, `nowheel`, `nopan`) and the classes `globals.css` defines (`agpt-*`, `markdown-output`).

## Styles

Tailwind classes go in the component that renders the element. Global CSS is only for DOM we don't render as JSX (Streamdown output, xyflow, KaTeX), keyframes and token definitions. Lint doesn't enforce this; reviewers check it.

No `style={{}}` or `<style>` for static values; pass a value known only at runtime through a CSS custom property. Rule: `design-system/no-inline-styles`.

No arbitrary values except for layout and transitions (`w-[320px]`, `top-[7px]` and `transition-[width]` pass). The error names the scale class or token with the same value, for example `rounded-[0.75rem]` → `rounded-xl`, `font-[600]` → `font-semibold`, `text-[13px]` → `text-body-sm leading-inherit`. Rule: `design-system/no-arbitrary-values`.

## Typography

Text renders through the `Text` atom by default: `<Text variant="body">`, with `as` for the tag (`<Text variant="small" as="span">`) and `tone` for the colour. `<Text>` owns its typography and colour; `className` takes layout, truncation, wrapping and alignment only (the `Text` contract of `design-system/no-restyle`).

Use a role class only where `Text` can't render the element identically, and inside primitives. Each role bundles its line-height:

| Role | Size / line-height |
| --- | --- |
| `text-tiny` | 10/14px |
| `text-caption-sm` | 11/16px |
| `text-caption` | 12/18px |
| `text-body-sm` | 13/20px |
| `text-body` | 14/22px |
| `text-body-md` | 15/24px |
| `text-body-lg` | 16/26px |
| `text-lead` | 20/28px |
| `text-heading-sm` | 22/24px |
| `text-heading` | 28/40px |
| `text-title` | 32/40px |
| `text-display` | 44/56px |

- **Line-height:** override it only with `leading-inherit`, `leading-4.5`, `leading-5.5`, `leading-6.5` or a Tailwind step.
- **Letter-spacing:** `tracking-display`, `tracking-heading`, `tracking-title`, `tracking-caps` or `tracking-eyebrow`.
- **Weight:** Tailwind's scale (`font-medium`, `font-semibold`).

Never use raw sizes: no `text-[13px]`, `leading-[..]`, `tracking-[..]`, `font-[..]` or inline font styles. A few rare sizes are still raw (5, 6, 9, 17, 23, 26, 35, 36, 48 and 64px, plus fractional rems). They carry `// TODO(tokens): SECRT-3152` until they are snapped; never round one to the nearest role.

## Icons

Hugeicons only: icon data from `@hugeicons/core-free-icons`, rendered with the `Icon` atom (`<Icon icon={Delete02Icon} size={16} />`), which applies the 2px stroke. Type icon props with `IconSvgElement` from `@hugeicons/react`. Rule `design-system/hugeicons-only` reports:
- importing `lucide-react`, `@phosphor-icons/react`, `react-icons` or `@radix-ui/react-icons`;
- rendering `HugeiconsIcon` outside `atoms/Icon`.

## Legacy components

Feature code never imports from `src/components/__legacy__`, from any other `legacy/` path or from `@/components/ui`. Import the atom, molecule or organism in `src/components` instead; Storybook shows them. `src/components/__legacy__`, `LegacyButton` and the `legacy-*` Button variants and sizes are deleted; the rules below keep them from coming back. The rounded-md look they drew is the Button's `compact-*` variants and sizes. The Button's internal variant helpers are only for the Button atom itself.

- **Rule `design-system/no-legacy-import`:** the imports.
- **Rule `design-system/no-legacy-button-variant`:** the `legacy-*` variants, should one come back. The error names the design-system variant to use instead.

## Escape hatch

Fix the code first. When a suppression is unavoidable, scope it to one line and one rule, and give the reason on the same line after `--`:

```tsx
{/* oxlint-disable-next-line design-system/no-arbitrary-values -- xyflow positions handles in px */}
```

The `design-system/*` rules are oxlint rules, so suppress them with `oxlint-disable-next-line`. Rule `design-system/require-suppression-reason` fails any `eslint-disable` or `oxlint-disable` comment without a `-- reason`. Like every rule here, it grandfathers the files that already had some.

## Pull requests

The pull request template has a design-system checklist. For a change to `src/components/**`, `globals.css` or `tailwind*.ts`, attach before and after screenshots at 1440 and 390 px wide, or write "no pixel change" and say how you checked. There is no automated visual diff; the screenshots are the evidence. `scripts/screenshot-pages.ts` takes and compares them, and `scripts/check-token-equivalence.ts` proves that a class swap renders the same value.

## Owners

`.github/CODEOWNERS` routes these to `@Significant-Gravitas/frontend-maintainers` and `@Significant-Gravitas/design`:
- the design-system directories (`atoms`, `molecules`, `organisms`, `tokens`, `styles`);
- `globals.css` and `tailwind-grammar.css`;
- the Tailwind config files and `components.json`;
- the lint configs and `scripts/eslint-rules/` (which includes the baseline);
- the token scripts;
- this page.
