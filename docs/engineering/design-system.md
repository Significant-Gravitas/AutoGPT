# Design system rules

The rules for styling in `autogpt_platform/frontend`, and the lint rule that enforces each one. Every `design-system/*` lint message ends with "See docs/engineering/design-system.md.", which is this page.

## How the rules are enforced

`pnpm lint` runs oxlint with `.oxlintrc.json`, which loads two local plugins: `scripts/eslint-rules/plugin.mts` (`autogpt/*`, the React Compiler rules) and `scripts/eslint-rules/design-system.mts` (`design-system/*`, the rules on this page). The `design-system/*` rules apply to `src/**`. They don't apply to tests, `src/tests`, `src/playwright`, `src/mocks` or the generated API client. Stories skip four styling rules (`no-raw-colors`, `no-arbitrary-values`, `no-restyle`, `no-inline-styles`) but not the colour-system rules (`no-alias-colours`, `no-palette-colours`, `no-hex-colours`). CI runs `pnpm lint` in the frontend `lint` job, so a violation fails the pull request.

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

The app has one colour system: the role tokens. Their values live in one place, `src/app/tokens/theme.css`, as RGB triplets on `:root` (`--color-<role>`); `src/app/globals.css` registers each in its `@theme inline` block, so `bg-surface` reads `rgb(var(--color-surface) / 1)` at runtime. A theme is a second block that redefines the same names, with no component edit (the T3 Code layout). Use the roles with any colour utility, opacity modifiers included: `bg-surface`, `text-foreground-secondary`, `border-border-base`, `ring-focus`, `bg-overlay/20`. In a string that needs a colour (an inline style, a drag image, an xyflow marker) write `rgb(var(--color-<role>))`.

- Surfaces: `canvas`, `surface`, `surface-muted`, `surface-hover`, `surface-active`, `overlay`, `scrim` (the dark backdrop behind dialogs, and the colour of a coloured shadow)
- Text: `foreground-base`, `foreground-strong`, `foreground-emphasis`, `foreground-body`, `foreground-secondary`, `foreground-tertiary`, `foreground-subtle`, `foreground-disabled`, `placeholder`
- Borders: `separator`, `border-base`, `border-strong`, `focus`
- Action and brand: `action`, `action-hover`, `action-foreground`, `brand`, `brand-hover`, `accent-surface`, `accent-strong`
- Status: `error`, `error-hover`, `error-foreground`, `error-surface`, `warning`, `warning-foreground`, `warning-surface`, `warning-border`, `success`, `success-foreground`, `success-surface`, `info`, `info-foreground`, `info-surface`
- Accents, hues that are not a status (categories, usage meters, data types, badges): `orange`, `orange-foreground`, `yellow`, `yellow-foreground`, `sky`, `sky-foreground`, `indigo`, `pink`, `pink-foreground`. A pale tint or border of a hue is the hue at an opacity: `bg-orange/10`, `bg-pink/10`, `ring-sky/30`.
- Builder: `builder-canvas`, `builder-grid`, `node-surface`, `node-border`, `node-selected`, `edge`, `handle`

`src/components/styles/color-roles.ts` lists every role (the Tokens/Colors story draws it). To add one, write its triplet in `theme.css`, register it in `globals.css`'s `@theme` and list it there; add one only when real, repeated uses need a colour no role has.

Three rules keep the other colour systems out, in every string (`className`, `cn()`/`cva()`, style objects, template literals), stories included:

- **Rule `design-system/no-alias-colours`:** the shadcn names (`bg-background`, `text-muted-foreground`, `border-border`, `ring-ring`, `bg-accent`, `bg-sidebar`, …). The error names the role each one aliased, which renders the same colour. `theme.css` keeps seven of them (`background`, `foreground`, `muted`, `muted-foreground`, `border`, `primary`, `primary-foreground`) for streamdown's prebuilt markup, which Tailwind scans in `node_modules`; they are not for app code.
- **Rule `design-system/no-palette-colours`:** the Tailwind palette (`bg-zinc-200`, `text-orange-600`, `bg-white`, `shadow-black/10`). The error names the role `scripts/ds/palette-map.json` maps the class to.
- **Rule `design-system/no-hex-colours`:** hex colours (`"#fff"`, `bg-[#123456]`).
- `design-system/no-raw-colors` still fails named colours in SVG attributes (`fill="white"`), and `design-system/no-arbitrary-values` arbitrary colours such as `text-[rgb(…)]`.

Artwork, logos, chart series (recharts writes them as SVG attributes, where CSS variables do not resolve), user-picked colour palettes and markup that never reaches the app's stylesheet keep their colours through `ds-colour-exempt.json`: a file glob, optionally the classes or hex values allowed in it, and a reason. It is the only way: a suppression comment for these three rules fails `scripts/__tests__/colour-exempt.test.ts`, which also checks every entry still matches a file and every allowed value is still used, and that no stylesheet other than `theme.css` (and exempt ones) writes a hex colour.

The codemods that did the migration stay for branches that still use the old names: `npx tsx scripts/ds/alias-to-role.ts --write <dir>` (the shadcn names; no pixel moves, `scripts/check-token-equivalence.ts` proves it against a base checkout) and `npx tsx scripts/ds/palette-to-role.ts --write <dir>` (the palette, by `palette-map.json`, whose rows say whether the colour stays the same, shifts a shade, or changes hue). The palette itself (`tailwind.v3-preset.ts`) is still compiled for streamdown and the exempt files.

`src/components/__tests__/design-system-contract.test.ts` checks that:
- every role on `theme.css`'s `:root` is registered in `globals.css`'s `@theme`, and the other way round, and `color-roles.ts` lists them all;
- no shadcn name is left but streamdown's, each aliasing a role;
- `.dark` only redefines names that `:root` declares;
- the app's palette values (`APP_COLORS` in `tailwind.v3-preset.ts`) get no new colours.

## Dark mode

No `dark:` or `light:` classes. A theme is a set of token values, not a second set of classes. `globals.css` ties `dark:` to a `.dark` ancestor (`@custom-variant dark`), which the app never sets, so these classes never render. Delete them and keep the light value; the role tokens carry the theme. Rule: `design-system/no-dark-variant`.

## Primitives

`src/components/{atoms,molecules,organisms}` own their look. At the call site, pick a `variant`, `size`, `tone` or `shape`. Use `className` only for layout: width, flex and grid placement, margin, position. A treatment you repeat becomes a variant on the primitive, never a class string copied between call sites.

The props that replace the usual restyles:

- **`Text`:** `tone` (colour), `weight`, `mono`, `tabular`, `italic`, `decoration`, `transform` and `align`, on top of the `variant`. A line height that differs from the variant's is a variant of its own (`body-tight`, `small-loose`, ...), not a `leading-*` class.
- **`Icon`:** `tone` (the same values as `Text`, from `src/components/styles/tones.ts`; without one the icon takes the colour around it), `spin` for a loading indicator and `motion` (`transform`, `colors`, `opacity`) for an icon that animates a class change, such as a chevron's rotation.
- **`Button`:** the quiet `ghost-muted`, `ghost-subtle` and `ghost-danger` variants, the `row` (32px), `inline`, `inline-xs` (no box, for a link in copy) and `icon-2xs` (24px) sizes, and `shape` (`pill`, `rounded`, `soft`) for the corner radius.

`Text` also accepts `truncate`, `line-clamp-*`, `whitespace-*`, `break-*`, `text-balance`, `text-pretty`, `text-ellipsis`, `gap-*` and text alignment; `Icon` and `Button` accept layout only; `Skeleton` also accepts shape. Rule: `design-system/no-restyle`. Inside the three design-system directories, which define the looks, it checks only `Text`, `Icon` and `Button`, so a molecule restyles them through their props like any call site.

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

Text renders through the `Text` atom by default: `<Text variant="body">`, with `as` for the tag (`<Text variant="small" as="span">`), `tone` for the colour and `weight`, `mono` or `tabular` to adjust the variant. `<Text>` owns its typography and colour; `className` takes layout, truncation, wrapping and alignment only (the `Text` contract of `design-system/no-restyle`).

Use a scale class only where `Text` can't render the element identically, and inside primitives. The type scale is the app's only set of text sizes: its values are in `src/app/tokens/typography.css` (one `:root` block, mapped to `text-<step>` by `@theme` in `globals.css`), so a redesign changes that file and every size follows. Each step bundles its line-height:

| Step | Size / line-height | Replaced (SECRT-3152) |
| --- | --- | --- |
| `text-tiny` | 10/14px | |
| `text-caption-sm` | 11/16px | |
| `text-caption` | 12/18px | Tailwind `xs` (12/16) |
| `text-body-sm` | 13/20px | |
| `text-body` | 14/22px | Tailwind `sm` (14/20) |
| `text-body-md` | 15/24px | |
| `text-body-lg` | 16/26px | Tailwind `base` (16/24) |
| `text-lead` | 20/28px | Tailwind `lg` (18/28) and `xl` (20/28) |
| `text-heading-sm` | 22/24px | |
| `text-heading-md` | 24/32px | Tailwind `2xl` (24/32) |
| `text-heading` | 28/40px | Tailwind `3xl` (30/36) |
| `text-title` | 32/40px | Tailwind `4xl` (36/40) |
| `text-display` | 44/56px | Tailwind `5xl` (48/48) |
| `text-hero` | 64/64px | Tailwind `6xl` to `9xl` |

- **Line-height:** override it only with `leading-inherit`, `leading-4.5`, `leading-5.5`, `leading-6.5` or a Tailwind step.
- **Letter-spacing:** `tracking-display`, `tracking-heading`, `tracking-title`, `tracking-caps` or `tracking-eyebrow`.
- **Weight:** Tailwind's scale (`font-medium`, `font-semibold`).

Tailwind's own sizes are reset in the theme (`--text-*: initial`), so `text-sm` or `md:text-2xl` renders nothing. Rule `design-system/no-raw-text-size` fails them, and arbitrary sizes (`text-[13px]`, `text-[length:..]`), naming the step to use; `scripts/ds/text-size-scale.ts` holds the mapping above and rewrites a file with `--write`. Never use raw sizes: no `text-[13px]`, `leading-[..]`, `tracking-[..]`, `font-[..]` or inline font styles. Streamdown, the chat's markdown renderer, still writes Tailwind's names in its own markup; `globals.css` sizes them from the same steps inside `[data-streamdown]`.

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
