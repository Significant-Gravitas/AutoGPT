# Design guide

How to build and change UI in `autogpt_platform/frontend`. Read this before you write or edit any component, page or style.

This page says what to reach for. The rules behind it, and the lint rule that enforces each one, are in [`docs/engineering/design-system.md`](../../docs/engineering/design-system.md). When the two disagree, that page and `pnpm lint` win; fix this one in the same pull request. `src/components/__tests__/design-md.test.ts` fails when a name list below no longer matches the source.

## Reuse before you build

- Before you write any UI, look for a component that already does the job: `src/components/atoms`, `molecules` and `organisms`, and Storybook (`pnpm storybook`). If one does, use it.
- If one almost fits, add a `variant`, `size`, `tone` or prop to it, with static class strings and a name that says its purpose. Do not restyle it at the call site, and do not copy it.
- Build a new component only when nothing fits. Never make a second version of one that exists: another button, dialog, input, tooltip, avatar or badge.
- A piece only one feature uses is composed from the primitives and lives in that feature's `components/` folder. When a second feature needs it, move it into `src/components`. File structure is in [`CONTRIBUTING.md`](./CONTRIBUTING.md).
- Never import from `src/components/__legacy__`, a `legacy/` path or `@/components/ui`.

## The one idea

Every design value lives in a token file. Components name a token; they never write a value. Change a token file and the whole app follows.

| File                            | Holds                                                                                             |
| ------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/app/tokens/theme.css`      | Every colour, as a role (`--color-surface`, `--color-foreground-body`)                            |
| `src/app/tokens/typography.css` | Text sizes and their line heights                                                                 |
| `src/app/tokens/shape.css`      | Corner radii, shadows, stacking layers, the app shell's sizes                                     |
| `src/app/tokens/motion.css`     | Durations and easings (`src/components/styles/motion.ts` holds the same values for Framer Motion) |

So in a component you never write a hex or `rgb()` colour, a pixel radius, a `shadow-[…]`, a `z-50`, a cubic-bezier, or a font size. If the token you need does not exist, add it (see [Adding a token](#adding-a-token)); do not write the value at the call site.

## Colour: pick the role by what the thing is

Use a role with any colour utility (`bg-`, `text-`, `border-`, `ring-`, `outline-`, `fill-`, `stroke-`, `from-`, `via-`, `to-`, a shadow colour), opacity included: `bg-surface`, `text-foreground-secondary`, `border-border-base`, `bg-orange/10`. In a string that needs a colour, write `rgb(var(--color-<role>))`, with an alpha if needed: `rgb(var(--color-brand) / 0.1)`.

| You are colouring                                                                   | Use                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The page behind everything                                                          | `canvas`                                                                                                                                                                                                                         |
| A card, panel, input or menu, then its hover and pressed fills                      | `surface`, `surface-muted`, `surface-hover`, `surface-active`                                                                                                                                                                    |
| A backdrop behind a dialog; a veil over an image                                    | `scrim`, `overlay`                                                                                                                                                                                                               |
| Text and icons, strongest to weakest                                                | `foreground-base`, `foreground-strong`, `foreground-emphasis`, `foreground-body`, `foreground-secondary`, `foreground-tertiary`, `foreground-subtle`, `foreground-disabled`, `placeholder`                                       |
| A divider, a border, a focus ring                                                   | `separator`, `border-base`, `border-strong`, `focus`                                                                                                                                                                             |
| The primary button, a checked or selected marker, a progress fill, and text on them | `action`, `action-hover`, `action-foreground`                                                                                                                                                                                    |
| An always-dark surface that shows content (toast, tooltip, terminal, code block)    | `surface-inverse`, `surface-inverse-hover`, `foreground-inverse`                                                                                                                                                                 |
| Text or an icon on brand, brand-hover, accent-strong, error-hover, info-foreground  | `foreground-on-color`                                                                                                                                                                                                            |
| A grey dot, bar, track, tick, tile or skeleton, lightest first                      | `fill-subtle`, `fill-soft`, `fill-medium`, `fill-bold`, `fill-strong`                                                                                                                                                            |
| The brand purple                                                                    | `brand`, `brand-hover`, `accent-surface`, `accent-strong`                                                                                                                                                                        |
| A status: the colour, its hover, its text, its pale surface                         | `error`, `error-hover`, `error-foreground`, `error-surface`, `warning`, `warning-foreground`, `warning-surface`, `warning-border`, `success`, `success-foreground`, `success-surface`, `info`, `info-foreground`, `info-surface` |
| Status text on an always-dark fill (toast, terminal, tooltip)                       | `error-inverse`, `success-inverse`                                                                                                                                                                                               |
| A category, meter or data hue that is not a status (a tint is `bg-pink/10`)         | `orange`, `orange-foreground`, `yellow`, `yellow-foreground`, `sky`, `sky-foreground`, `indigo`, `pink`, `pink-foreground`                                                                                                       |
| The app frame and sidebar (not read yet)                                            | `shell`, `shell-foreground`, `shell-hover`, `shell-active`, `shell-border`                                                                                                                                                       |
| The flow editor                                                                     | `builder-canvas`, `builder-grid`, `node-surface`, `node-border`, `node-selected`, `edge`, `handle`                                                                                                                               |

`foreground-on-color` is white, so it is too light for the pale fills: on `warning`, `yellow`, `orange` and `sky` it stays under 3:1, and text there is `foreground-base` (6.6:1 or more). On `error`, `success`, `info`, `indigo` and `pink` it reaches only 3.5 to 4.4:1 today, enough for large text and icons; `theme-contrast.test.ts` lists those pairs until design picks new values.

Three mistakes to avoid, because they break the day a look changes one role:

- **A text role as a fill.** `bg-foreground-subtle` on a dot is wrong; use `bg-fill-subtle`. A surface role is never a text colour.
- **`action` as "the dark surface".** A toast is not a button; use `surface-inverse`.
- **`action-foreground` as "white".** On `bg-brand` or `bg-accent-strong`, use `foreground-on-color`.

Artwork, logos, chart series and colours a person picks are the only colours outside the roles. They are listed, with a reason, in `ds-colour-exempt.json`; never use a suppression comment for a colour.

## Tones

`tone` on `Text`, `Icon`, `Link` and `Button` takes a name from `src/components/styles/tones.ts`. A tone is not a role name: `primary` is `text-foreground-strong` and `muted` is `text-foreground-tertiary`. Text without a tone keeps its variant's colour; an Icon without one takes the colour around it.

| For                                    | Tones                                                                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Text, strongest to weakest             | `base`, `primary`, `emphasis`, `emphasis-soft`, `body`, `secondary`, `muted`, `subtle`, `disabled`                                   |
| On a dark surface; on a saturated fill | `inverse`, `on-color`                                                                                                                |
| The brand purple                       | `brand`, `brand-strong`, `accent`                                                                                                    |
| A status                               | `danger`, `error`, `error-strong`, `warning`, `warning-strong`, `warning-bright`, `success`, `success-strong`, `info`, `info-strong` |
| A status on a dark fill                | `success-light`, `error-light`                                                                                                       |
| An accent hue; the builder's handles   | `yellow`, `orange`, `handle`                                                                                                         |
| The parent's colour                    | `inherit`                                                                                                                            |

## Text

All copy goes through the `Text` atom. The `variant` is the role; everything else is a prop.

- **Variants:** `hero`, `h1`, `h2`, `h3`, `h4`, `h5`, `heading-md`, `lead`, `large`, `body-md`, `body`, `body-sm`, `small`, `caption-sm`, `tiny`, `label`, `eyebrow`.
- The variant also picks the tag; the `as` prop changes it.
- `weight`: `normal`, `medium`, `semibold`, `bold`.
- `leading` (a line height other than the role's): `none`, `tight`, `snug`, `normal`, `relaxed`, `inherit`, `4`, `4.5`, `5`, `6`, `7`, `8`, `9`, `10`.
- `tracking`: `tight`, `normal`, `wide`, `widest`, `title`, `heading`, `display`, `caps`, `eyebrow`.
- `decoration`: `underline`, `line-through`, `none`. `transform`: `uppercase`, `lowercase`, `capitalize`, `none`. `align`: `left`, `center`, `right`, `start`, `end`.
- `narrowSize` (a larger size below md): `body`, `large`. `motion="colors"` animates a colour change. `mono`, `tabular` and `italic` are booleans.

```tsx
<Text variant="small" weight="medium" tone="secondary">
  Last run 2h ago
</Text>
```

`className` on `Text` is for layout, truncation, line clamping, wrapping, gaps and alignment only. Never a size (`text-sm`, `text-[13px]`), a weight class or a colour class.

Outside `Text` (inside a primitive, or where `Text` cannot render the element), a size is a type-scale step: `text-tiny`, `text-caption-sm`, `text-caption`, `text-body-sm`, `text-body`, `text-body-md`, `text-body-lg`, `text-lead`, `text-heading-sm`, `text-heading-md`, `text-heading`, `text-title`, `text-display`, `text-hero`. Each bundles its line height. Fonts are `font-sans` (Geist), `font-mono` (Geist Mono) and `font-poppins` (headings).

## Buttons

Use the `Button` atom and choose four things:

- **`variant`**, the role: `primary` (default), `secondary`, `outline`, `ghost`, `destructive`, `link`, `icon`, and the rarer `toggle`, `floating`, `tray`, `brand`, `warning`, `success`.
- **`size`:** `large` (default), `small`, `row` (32px), `xs` (28px), `inline`, `inline-xs` (no box, inside copy), and `icon`, `icon-sm`, `icon-xs`, `icon-2xs` for an icon alone.
- **`shape`**, the corner: `pill` (default), `rounded`, `soft`, `sharp` (also the default of the toggle and floating variants). Shape is the only place a button gets its radius.
- **`tone`:** a shared tone, or for a ghost that changes colour on hover `quiet`, `quiet-subtle`, `quiet-danger`, `quiet-warning`, `quiet-info`, `quiet-remove`.

`leadingIcon` takes a Hugeicon and sizes it to the button; `loading` shows a spinner; an icon-only button shows its `aria-label` as a tooltip (`withTooltip={false}` turns it off); `as="NextLink"` with `href` renders a link. `className` on a `Button` is layout only.

The `compact-*` family is older. Do not use it in new code:

- Compact variants: `compact-primary`, `compact-destructive`, `compact-outline`, `compact-ghost`.
- Compact sizes: `compact`, `compact-sm`, `compact-icon`, `compact-icon-sm`.

## Icons

Hugeicons only: icon data from `@hugeicons/core-free-icons`, rendered with the `Icon` atom: `<Icon icon={Delete02Icon} size={16} tone="muted" />`. Without `size` it is `1em`. `spin` for a loader; `motion` (`transform`, `colors`, `opacity`) when a class change animates it, such as a chevron's rotation. The atom sets the 2px stroke; do not change it. Type an icon prop as `IconSvgElement` from `@hugeicons/react`. Never import `lucide-react`, `@phosphor-icons/react`, `react-icons` or `@radix-ui/react-icons`, and never render `HugeiconsIcon` yourself.

## Shape: radius

Use a step: `rounded-none`, `rounded-sm` (4px), `rounded` (4), `rounded-md` (6), `rounded-lg` (8), `rounded-xl` (12), `rounded-2xl` (16), `rounded-3xl` (24), `rounded-4xl` (32), `rounded-5xl` (40), and `rounded-2xl-inner` (15) for a `rounded-2xl` card's surface inside its 1px border. Corners take the same steps (`rounded-t-lg`, `rounded-l-pill`). The older names `rounded-xsmall` to `rounded-2xlarge` alias steps of the same size. Never `rounded-[…]`.

**Pill or circle.** Ask: in a look where controls have 8px corners, should this become 8px?

- Yes: it is a **pill**. Use `rounded-pill`. Buttons, icon buttons, chips, badges, tags, tabs, search fields, tracks of a progress bar or switch.
- No, it is round by nature: it is a **circle**. Use `rounded-full`. Avatars, status dots, spinners, the thumb of a switch, radio marks, count bubbles sized square.

## Depth: shadows

Use a named step from `shape.css`. Never write a shadow by hand.

| Need                                                           | Class                                                                                                    |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| A resting card or field; an inset well                         | `shadow-sm`, `shadow`, `shadow-subtle`, `shadow-inner`                                                   |
| A menu, popover, dropdown; a card floating over the composer   | `shadow-md`, `shadow-lg`, `shadow-popover-raised`                                                        |
| A dialog or sheet                                              | `shadow-xl`, `shadow-2xl`                                                                                |
| A card that lifts on hover                                     | `shadow-lift`                                                                                            |
| Under a badge, avatar or filled button                         | `shadow-drop`                                                                                            |
| The chat composer; the chat message table                      | `shadow-composer`, `shadow-message-table`                                                                |
| Above a footer that content scrolls under                      | `shadow-scroll-edge`                                                                                     |
| A keyboard key                                                 | `shadow-keycap`                                                                                          |
| A glossy chip or bar; the sheen of a tab joining the composer  | `shadow-gloss`, `shadow-sheen`                                                                           |
| A 1px outline drawn as a shadow; a selected card's halo        | `shadow-outline`, `shadow-halo`                                                                          |
| A call to action, a status chip or the tour's next click glows | `shadow-glow`, `shadow-status-glow`, `shadow-beacon`                                                     |
| The tour's spotlight edge                                      | `shadow-spotlight`                                                                                       |
| The marketplace's accent chips                                 | `shadow-chip-glow-brand`, `shadow-chip-glow-warning`, `shadow-chip-glow-sky`, `shadow-chip-glow-neutral` |

To tint a one-colour step, add a colour class beside it: `shadow-drop shadow-brand/45`. `smooth-shadow-ring-sm` (in `globals.css`) is a soft layered shadow with a hairline ring.

## Stacking: layers

Use a named layer, lowest first, as a class on the element: `z-overlay`. Never `z-10`, `z-[60]` or an inline `zIndex`. The variable (`var(--z-<name>)`) is only for an API that takes a style object instead of a class; nothing in the app needs it today.

`z-behind` · `z-base` · `z-raised` (lifted over a sibling) · `z-sticky` (bars pinned in a pane) · `z-search-bar` (the library search bar) · `z-floating` (a button over a pane) · `z-header` (the inset chat header) · `z-prompt` (a feedback prompt in a corner) · `z-overlay` (dialogs, sheets, menus, popovers, tooltips) · `z-drawer-scrim` · `z-drawer` · `z-command` (the command palette) · `z-canvas-loading` and `z-canvas-toolbar` (the builder) · `z-blocking` (must cover everything) · `z-tour-scrim` · `z-tour`

If nothing fits, add a layer to `shape.css` with a name that says what it is.

## Motion

- **In CSS:** `duration-instant` (100ms), `duration-fast` (150), `duration-base` (200), `duration-slow` (300), `duration-slower` (400), `duration-slowest` (500); `ease-standard` (also `ease-in-out`), `ease-out`, `ease-in`, `ease-emphasized`, `ease-out-quint` (accordions), `ease-glide` (onboarding scenes), `ease-drawer`.
- **In Framer Motion:** import `duration`, `ease` and `spring` from `@/components/styles/motion`. `spring` has `snappy`, `gentle` and `bouncy`.

```tsx
import { duration, ease, spring } from "@/components/styles/motion";

<motion.div transition={{ duration: duration.base, ease: ease.out }} />
<motion.div transition={{ type: "spring", ...spring.snappy }} />
```

No cubic-bezier arrays, no `"easeOut"` strings, no local `EASE` constants; a test fails them. Endless ambient loops and durations of 0.6s or longer may keep their own numbers.

Reduced motion is handled once at the app root (`MotionConfig reducedMotion="user"`): Framer drops movement and keeps fades. Add your own branch only when an animation needs more than that.

## Primitives own their look

`src/components/{atoms,molecules,organisms}` decide how they look. At a call site you pick a `variant`, `size`, `tone` or `shape`; `className` is for layout: width, flex and grid placement, margin, position. `Skeleton` also takes a shape.

- If you are about to restyle a primitive with classes, look for a prop first.
- If the same treatment appears a second time, make it a prop on the primitive. Do not copy a class string between call sites.
- Never build a class from a variable (`` `bg-${tone}` ``); map the prop to whole class names.
- Arbitrary values are for layout and transitions only (`w-[320px]`, `top-[7px]`, `transition-[width]`).
- No inline `style` for a fixed value. A value known only at runtime goes through a CSS custom property and a class: ``style={{ "--w": `${width}px` } as React.CSSProperties}`` with `w-(--w)`.

## Inputs and copy

- Keyboard handlers use `isKey(e, "Enter")` from `@/lib/keyboard`, never a `e.key` comparison. Root `AGENTS.md` has the details.
- A controlled text field shows exactly what was typed, in the same render. Root `AGENTS.md` says how, and how to test it.
- Text a person reads says workflow and Otto, never agent or graph. Root `AGENTS.md`, "User-facing copy", has the rule.

## Themes and looks

The app is light only today. Two attributes on `<html>` are reserved so that this can change without touching components:

- `data-theme` is the mode, `light` or `dark`. It means nothing else.
- `data-look` is the look, a second set of token values. The only one that exists is `canary`, a development tool.

Never write a `dark:` or `light:` class; they never render. A theme is a file of token values, not a second set of classes. How to add one is in `docs/engineering/design-system.md`.

## Before you say it is done

1. `pnpm format`, `pnpm lint`, `pnpm types`. The `design-system/*` rules run inside `pnpm lint`, and each error names the token or prop to use.
2. Never raise a count in `scripts/eslint-rules/design-system-baseline.json`, and never add a suppression to get past a rule. If you removed violations, run `pnpm lint:ds-baseline` and commit the lower counts.
3. If you touched a token file or a primitive, prove the pipeline still holds: build Storybook, then run `npx tsx scripts/ds/canary-check.ts --storybook-url <url>`. It renders every story in the default and the canary look and lists any colour, radius or font that did not follow the tokens. A run counts only when every target rendered: read the `failed:` lines at the top of `report.txt` (or each target's `error` in `report.json`), not just the leak totals. Your change should add no leaks.
4. For a change to `src/components/**`, `globals.css` or a token file, attach before and after screenshots at 1440 and 390 px to the pull request, or write "no pixel change" and how you checked.

## Adding a token

Add one only when real, repeated uses need a value no token has.

1. Write the value in the token file (`theme.css` for a colour, as an RGB triplet; `typography.css` for a text size and its line height; `shape.css` for a radius, shadow, layer or shell size; `motion.css` for a duration or easing).
2. Register the name in `src/app/globals.css` so Tailwind makes a class for it. Shadows are registered in `shape.css` itself; a new shadow step or layer also goes in `src/lib/tw-merge.ts`.
3. For a colour, list it in `src/components/styles/color-roles.ts`. For a motion value, add it to `src/components/styles/motion.ts`.
4. For a colour, a radius or a text size, give it a value in `src/app/tokens/themes/canary.css`.
5. Name it on this page, in the list where it belongs.
6. Run the token tests: `pnpm vitest run src/components/__tests__ src/components/styles`. They fail a token that is not registered, not listed, not in the canary or not on this page.
