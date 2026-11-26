# Design system rules

The rules for styling in `autogpt_platform/frontend`, and the lint rule that enforces each one. Every `design-system/*` lint message ends with "See docs/engineering/design-system.md.", which is this page.

## How the rules are enforced

`pnpm lint` runs oxlint with `.oxlintrc.json`, which loads two local plugins: `scripts/eslint-rules/plugin.mts` (`autogpt/*`, the React Compiler rules) and `scripts/eslint-rules/design-system.mts` (`design-system/*`, the rules on this page). The `design-system/*` rules apply to `src/**`. They don't apply to tests, `src/tests`, `src/playwright`, `src/mocks` or the generated API client. Stories skip four styling rules (`no-raw-colors`, `no-arbitrary-values`, `no-restyle`, `no-inline-styles`) but not the colour-system rules (`no-shadcn-alias-colours`, `no-raw-palette-classes`, `no-hex-colours`, `no-colour-functions`). CI runs `pnpm lint` in the frontend `lint` job, so a violation fails the pull request.

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

The app has one colour system: the role tokens. Their values live in one place, `src/app/tokens/theme.css`, as RGB triplets on `:root` (`--color-<role>`); `src/app/globals.css` registers each in its `@theme inline` block, so `bg-surface` reads `rgb(var(--color-surface) / 1)` at runtime. A theme is a second block that redefines the same names, with no component edit (the T3 Code layout). Use the roles with any colour utility, opacity modifiers included: `bg-surface`, `text-foreground-secondary`, `border-border-base`, `ring-focus`, `bg-overlay/20`. In a string that needs a colour (an inline style, a drag image, an xyflow marker) write `rgb(var(--color-<role>))`, with an alpha if it needs one (`rgb(var(--color-brand) / 0.1)`); a shadow's colour is `rgb(var(--shadow-color) / <alpha>)`.

- Surfaces: `canvas`, `surface`, `surface-muted`, `surface-hover`, `surface-active`, `overlay`, `scrim` (the dark backdrop behind dialogs, and the colour of a coloured shadow)
- Text: `foreground-base`, `foreground-strong`, `foreground-emphasis`, `foreground-body`, `foreground-secondary`, `foreground-tertiary`, `foreground-subtle`, `foreground-disabled`, `placeholder`
- Borders: `separator`, `border-base`, `border-strong`, `focus`
- Action and brand: `action`, `action-hover`, `action-foreground`, `brand`, `brand-hover`, `accent-surface`, `accent-strong`
- Status: `error`, `error-hover`, `error-foreground`, `error-surface`, `warning`, `warning-foreground`, `warning-surface`, `warning-border`, `success`, `success-foreground`, `success-surface`, `info`, `info-foreground`, `info-surface`, and `error-inverse` and `success-inverse` for status text on the always-dark toast, terminal and tooltip
- Accents, hues that are not a status (categories, usage meters, data types, badges): `orange`, `orange-foreground`, `yellow`, `yellow-foreground`, `sky`, `sky-foreground`, `indigo`, `pink`, `pink-foreground`. A pale tint or border of a hue is the hue at an opacity: `bg-orange/10`, `bg-pink/10`, `ring-sky/30`.
- Builder: `builder-canvas`, `builder-grid`, `node-surface`, `node-border`, `node-selected`, `edge`, `handle`
- Inverse: `surface-inverse`, `surface-inverse-hover` (a dark surface that shows content on the light theme: a toast, a tooltip, a terminal or code block, the user's chat bubble, a dark chip or badge) and `foreground-inverse`, the text and icons on it (`tone="inverse"`)
- On colour: `foreground-on-color` (`tone="on-color"`), text and icons on a saturated fill. It reaches 4.5:1 on `brand`, `brand-hover`, `accent-strong`, `error-hover` and `info-foreground`. On `error`, `success`, `info`, `indigo` and `pink` it reaches only 3.5 to 4.4:1 today, enough for large text and icons but not for body text, and the contrast test lists those pairs until design picks new values. It never goes on `warning`, `yellow`, `orange` or `sky`, where it stays under 3:1 (1.7 to 2.8:1): text there is `foreground-base` (`tone="base"`), 6.6:1 or more.
- Fills, solid marks that are not text (dots, bars, tracks, icon tiles, an illustration's marks), lightest first: `fill-subtle`, `fill-soft`, `fill-medium`, `fill-bold`, `fill-strong`
- Not used yet: `shell`, `shell-foreground`, `shell-hover`, `shell-active`, `shell-border` (the app frame around the page)

#### What a role means

A role is named for what it colours, and a look changes it for that job alone: a look that makes the primary Button purple changes `action`, and nothing else may turn purple with it. So pick the role by meaning, not by the colour it happens to have today:

- `action` and `action-hover` are the primary Button and controls that act like one (the send button, a pressed or selected choice), and the markers of what is selected, current or in progress: a checked box, the selected tab's indicator, the current step, a progress or usage bar, a loader. `action-foreground` is the text and icons on them. An Icon inside a primary Button takes no tone; it inherits the Button's colour.
- A dark surface that shows content is not an action, even when it is the same colour: it is `surface-inverse` (or `-hover`) with `foreground-inverse` on it.
- White on a coloured fill is `foreground-on-color`, not `action-foreground`.
- A text role (`foreground-*`) colours text and icons. A dot, bar, track, divider, tile or skeleton block is a fill (`fill-*`), a backdrop is `scrim`, and a line is `separator` or a `border-*` role.
- A surface role (`canvas`, `surface-*`) is never a text colour.

- **Rule `design-system/no-misused-role`:** a fill (`bg-`, `from-`, `via-`, `to-`) on a text role (`bg-foreground-subtle`), and text (`text-`) on a surface role (`text-surface-muted`). The error names the fill role with the same colour when there is one. A background clipped to its text (`bg-clip-text`), gradient or solid, is that text's colour, so it may read text roles. Lint can't tell an action from a dark surface, since both are fills of roles that share a colour today; reviewers check that `bg-action` is an action and `bg-surface-inverse` is not. The baseline holds only the Button atom (a disabled fill and disabled text), which is being reworked separately.

`src/components/styles/color-roles.ts` lists every role (the Tokens/Colors story draws it). To add one, write its triplet in `theme.css`, register it in `globals.css`'s `@theme` and list it there; add one only when real, repeated uses need a colour no role has.

The roles are the only colour system. Tailwind's palette is reset (`--color-*: initial` at the top of `theme.css`) and `tailwind.config.ts` has no colours, so `bg-red-500`, `text-zinc-600` or `bg-white` generate nothing. Five rules keep the other colour systems out of the source, in every string (`className`, `cn()`/`cva()`, style objects, template literals), stories included:

- **Rule `design-system/no-shadcn-alias-colours`:** the shadcn names (`bg-background`, `text-muted-foreground`, `border-border`, `ring-ring`, `bg-accent`, `bg-sidebar`, …). The error names the role each one aliased, which renders the same colour. `theme.css` keeps seven of them (`background`, `foreground`, `muted`, `muted-foreground`, `border`, `primary`, `primary-foreground`) for streamdown's prebuilt markup, which Tailwind scans in `node_modules`; they are not for app code.
- **Rule `design-system/no-raw-palette-classes`:** the Tailwind palette (`bg-zinc-200`, `text-orange-600`, `bg-white`, `shadow-black/10`). The error names the role `scripts/ds/palette-map.json` maps the class to.
- **Rule `design-system/no-hex-colours`:** hex colours in any string (`"#fff"`, `border:1px solid #E4E4E7`).
- **Rule `design-system/no-colour-functions`:** `rgb()`, `rgba()`, `hsl()`, `hsla()` and `oklch()` with numbers for arguments, signed ones included (`hsl(-30 100% 50%)`), in any string: an inline style (`"rgb(248 250 252)"`), a gradient or shadow inside a class (`bg-[linear-gradient(180deg,rgba(14,165,233,0.1),transparent)]`), a template literal (`` `rgba(255,255,255,${alpha})` ``). A role read (`rgb(var(--color-sky) / 0.1)`) is not one. A class whose whole arbitrary value is the colour (`text-[rgb(20,20,20)]`) is `no-arbitrary-colour`'s. Its baseline is empty: every shadow reads a step in `shape.css`.
- **Rule `design-system/no-arbitrary-colour`:** a colour class whose arbitrary value is a colour (`bg-[#123456]`, `text-[rgb(131,131,140)]`, `border-[color:hsl(…)]`, `[color:#fff]`). The error names the role with the same colour when there is one. A role read through `var()` is not one.
- `design-system/no-raw-colors` still fails named colours in SVG attributes (`fill="white"`).

Artwork, logos, chart series (recharts writes them as SVG attributes, where CSS variables do not resolve), user-picked colour palettes and markup that never reaches the app's stylesheet (an SVG data-URI image, an iframe's document) keep their colours through `ds-colour-exempt.json`: a file glob, optionally the classes, hex values or colour functions allowed in it, and a reason. It is the only way: a suppression comment for these rules fails `scripts/__tests__/colour-exempt.test.ts`, which also checks every entry still matches a file and every allowed value is still used, and that no stylesheet other than `theme.css` (and exempt ones) writes a hex colour.

The codemods that did the migration stay for branches that still use the old names: `npx tsx scripts/ds/alias-to-role.ts --write <dir>` (the shadcn names; no pixel moves, `scripts/check-token-equivalence.ts` proves it against a base checkout) and `npx tsx scripts/ds/palette-to-role.ts --write <dir>` (the palette, by `palette-map.json`, whose rows say whether the colour stays the same, shifts a shade, or changes hue). The palette itself is gone: the exempt user-picked palettes write their hex as arbitrary values (`bg-[#fde68a]`), and the few palette classes streamdown's prebuilt markup writes (its mermaid error box and image hover veil) are drawn by a block at the end of `theme.css`, inside streamdown's markup only.

`src/components/__tests__/design-system-contract.test.ts` checks that:

- every role on `theme.css`'s `:root` is registered in `globals.css`'s `@theme`, and the other way round, and `color-roles.ts` lists them all;
- no shadcn name is left but streamdown's, each aliasing a role;
- `dark:` keys on `[data-theme="dark"]`, and no stylesheet has a `.dark` selector;
- there is no raw palette (no `tailwind.v3-preset.ts`, no `--color-<hue>-<step>`).

`src/components/styles/__tests__/roles.test.ts` checks that every role has its value in the one `:root` block of `theme.css` and a `@theme inline` entry in `globals.css`, that compiling the stylesheet gives every role a class, and that the palette is reset before any colour is registered, so a palette class compiles to nothing.

### The token files

Every design value lives in one of four files in `src/app/tokens/`. Change a value there and every class that uses it follows.

- `theme.css`: colour. The roles above.
- `typography.css`: type. The text sizes and the line height each one bundles (see [Typography](#typography)).
- `shape.css`: radius, elevation, layers and the app shell.
  - Radius: `rounded-sm` to `rounded-5xl`, `rounded-pill` and `rounded-full`. `rounded` is `--radius` (4px). `rounded-4xl` (32px) is the chat composer's and the raise preview's, `rounded-5xl` (40px) the raise flow's stage, and `rounded-2xl-inner` (15px) a `rounded-2xl` card's surface inside its 1px border. The older names (`rounded-xsmall` to `rounded-2xlarge`) alias the step of the same size. Write no radius by hand (`rounded-[…]`, an inline `borderRadius`, a drag image's `border-radius`): use a step, or `var(--radius-<step>)` in CSS. What is left: `rounded-[inherit]` (the ScrollArea viewport, the ExpertAvatar frame), percentage shapes on artwork (the GlassOrb blobs, the file illustration's folded corner), the TrialCard billing skeleton's 18px, and documents the app's CSS does not reach (the React artifact preview's iframe, the MCP OAuth callback page).
  - Pill or circle: both are 9999px today, but a look may square pills and never circles. Ask of each fully rounded element: in a look where controls have 8px corners, should this become 8px? Then it is a pill, `rounded-pill` (and `rounded-t-pill`, `rounded-l-pill` and the other corners). Is it round by nature, in any look? Then it is a circle, `rounded-full`.
    - Pills: every Button, icon-only ones included (`size="icon"`, a round close or send button); chips, tags and badges with horizontal padding (`ToggleChip`, `CategoryChip`, a status pill); segmented controls and pill tabs (`MultiToggle`, the billing toggle, the profile tabs and their sliding indicator); progress, usage and switch tracks and their fills, and a step indicator's widened current step; search fields; skeletons standing in for any of these.
    - Circles: avatars and the rings, overlays and badges drawn on them; presence and status dots, step dots and timeline nodes; count bubbles sized square; spinners and loaders; switch and radio thumbs and a radio's dot; colour swatches; icon medallions (a `size-12` round tile behind an icon); decorative orbs and blurred blobs.
  - Shadows: `shadow-sm` to `shadow-2xl`, `shadow-inner`, `shadow-subtle` and `shadow-message-table`. Their colour is `--shadow-color`, an RGB triplet read at runtime. The steps themselves are in the file's own `@theme` block, because Tailwind reads each shadow's colour from the value to make a coloured shadow (`shadow-md shadow-scrim/10`).
  - Named effects, one per purpose: `shadow-lift` (a card raised on hover), `shadow-popover-raised` (a card floating over the composer), `shadow-composer`, `shadow-drop` (under a badge, avatar or filled button), `shadow-scroll-edge` (above a footer content scrolls under), `shadow-keycap`, `shadow-gloss` (a glossy pill's white top edge and hairline drop), `shadow-sheen` (the inner sheen of a tab joining the composer), `shadow-outline` (a 1px edge that leaves `ring` to focus), `shadow-halo` (a selected card), `shadow-spotlight` (the tour's lit hole), `shadow-glow` (a call to action), `shadow-status-glow` (a status chip), `shadow-beacon` (the tour's next thing to click) and `shadow-chip-glow-brand`, `-warning`, `-sky` and `-neutral` (the marketplace's accent chips). A step with one colour takes another from a shadow colour class: `shadow-drop shadow-brand/45`, `shadow-status-glow shadow-info/70`. That class sets every layer to the one colour and alpha, so a step that mixes a white edge with a tinted glow names its colours (`rgb(var(--color-surface) / 0.9)`) and comes once per accent. A white highlight is `surface`.
  - Write no shadow by hand (`shadow-[…]`, an inline `boxShadow`, a drag image's `box-shadow`): use a step, or add one here named for what it is for. What is left: keyframes that animate a shadow (the orbit loader, the builder tutorial's pulse), the builder tutorial's Shepherd card, the category chip's runtime-coloured selected state, the expert title's runtime-coloured glow, and the exempt artwork (`GlassSurface`, the sticky note); all of them read roles.
  - Layers: see [Layers](#layers).
  - Shell: `--shell-sidebar-width` (18.25rem, the app's sidebar), `--shell-sidebar-width-tour` (19rem, the tour's and the logged-out marketplace's), `--shell-sidebar-width-default` (20rem, a `SidebarProvider` that sets no width), `--shell-sidebar-width-mobile` (20rem, the phone sheet), `--shell-sidebar-width-collapsed` (3rem, the icon rail) and `--shell-content-max`. The Sidebar's constants and the providers in `PlatformChrome` and `TourCopilot` read them as `--sidebar-width: var(--shell-sidebar-width)`; elsewhere read them with `w-(--shell-sidebar-width)`.
- `motion.css`: durations (`duration-instant` 100ms, `-fast` 150ms, `-base` 200ms, `-slow` 300ms, `-slower` 400ms, `-slowest` 500ms) and easings (`ease-standard`, `ease-in`, `ease-out`, `ease-emphasized`, `ease-out-quint`, `ease-glide`, `ease-drawer`; `ease-in-out` is the standard curve). A bare number (`duration-200`) still works. `src/components/styles/motion.ts` has the same values for Framer Motion: `duration` in seconds, `ease` as cubic-bezier numbers, and three `spring` presets.

`tailwind.config.ts` holds no colour, text size, radius, shadow, layer, duration or easing.

To add a token, write its value in the file above that owns it, and register it in `globals.css`'s `@theme inline reference` block, which turns it into a class: `--radius-<step>: var(--radius-<step>)`, `--z-index-<name>: var(--z-<name>)`, `--transition-duration-<name>: var(--duration-<name>)` or `--ease-<name>: var(--ease-<name>)`. A shadow goes in `shape.css`'s own `@theme` block, coloured with `rgb(var(--shadow-color) / <alpha>)` or a role (`rgb(var(--color-<role>) / <alpha>)`), and its name in `src/lib/tw-merge.ts`, so `cn()` treats it as a shadow rather than a colour. A colour role is also listed in `color-roles.ts`. A motion value is also written in `motion.ts`.

`src/components/__tests__/shape-motion-tokens.test.ts` checks that every variable in `shape.css` and `motion.css` is registered and compiles to a class that reads it, that every shadow step reads only `--shadow-color` or a colour role and compiles to a class a shadow colour class can recolour, and that no such value is left in `tailwind.config.ts` or in `globals.css`'s `@theme`. `src/lib/__tests__/tw-merge.test.ts` checks that `cn()` knows every named shadow step. `src/components/styles/__tests__/motion.test.ts` checks that `motion.ts` matches `motion.css`.

### Layers

Every z-index is a named layer from `shape.css`, lowest first:

| Class              | z-index | What sits there                                               |
| ------------------ | ------- | ------------------------------------------------------------- |
| `z-behind`         | -10     | decoration behind its own content                             |
| `z-base`           | 0       |                                                               |
| `z-raised`         | 10      | lifted over a sibling in the same component                   |
| `z-sticky`         | 20      | search bars, banners and bars pinned in a pane                |
| `z-search-bar`     | 21      | the library search bar, over the sticky bars it overlaps      |
| `z-floating`       | 30      | buttons floating over a pane (jump to bottom)                 |
| `z-header`         | 40      | the inset header floating over the chat                       |
| `z-prompt`         | 40      | a feedback prompt fixed in a corner (NPS), under dialogs      |
| `z-overlay`        | 50      | dialogs, sheets, menus, popovers, tooltips                    |
| `z-drawer-scrim`   | 60      | the backdrop of a drawer above dialogs                        |
| `z-drawer`         | 70      | the artifact panel on a phone                                 |
| `z-command`        | 80      | the command palette; popovers over a drawer                   |
| `z-canvas-loading` | 99      | the builder's loading box, over the nodes                     |
| `z-canvas-toolbar` | 100     | the builder's run toolbar, over the nodes and the loading box |
| `z-blocking`       | 100     | modals that must cover everything else                        |
| `z-tour-scrim`     | 110     | the tour's dim                                                |
| `z-tour`           | 120     | the tour's spotlight card and contextual tips                 |

- **Rule `design-system/no-raw-z-index`:** a layer set by number in any string (`z-10`, `-z-10`, `hover:z-[21]`) or a style object (`zIndex: 50`). The error names the layer with that number. Its baseline is empty. A stylesheet reads the variable (`z-index: var(--z-raised)`); a style object does too (`zIndex: "var(--z-overlay)"`), and a class that reads it (`z-(--z-overlay)`, `z-[var(--z-overlay)]`) is not a number either.
- A new layer goes in `shape.css` with the number it needs, a `--z-index-<name>: var(--z-<name>)` entry in `globals.css`, and its name in `src/lib/tw-merge.ts`, so `cn("z-raised", "z-overlay")` keeps the last. Name it for what sits there. Two names share a number only when they are different layers that happen to sit at the same height today (`z-header` and `z-prompt`, `z-canvas-toolbar` and `z-blocking`), so either can move without the other.
- `design-system/no-restyle` allows `z-*` wherever it allows layout classes, as it allowed the numbers.
- Third-party numbers stay as their libraries need them: the Google Drive picker (`[class*="picker-dialog"]`) is forced to `z-index: 10000 !important` in `globals.css`, over its own inline value, and while it is open the app's dialog overlay and popup drop to `z-index: 1 !important` beneath it.

#### How to animate

In Framer Motion, take every value from `@/components/styles/motion`: `transition={{ duration: duration.base, ease: ease.out }}`, and springs as `{ type: "spring", ...spring.snappy }` (`snappy` 520/30, `gentle` 300/25, `bouncy` 200/15; a site may still add `mass`, `bounce` or `restDelta`). A timer that waits for a CSS transition reads the same token: `duration.fast * 1000`. Pick the step nearest what the transition needs; do not add a step for one site. `<MotionConfig reducedMotion="user">` in `src/app/providers.tsx` drops transform and layout animations for people who ask for reduced motion, so a component only needs its own `useReducedMotion` branch when it should do something other than that (fade instead of slide, skip a delay).

Some numbers stay literal, because they are not UI transitions: `duration: 0`, durations of 0.6s and longer, `repeat: Infinity` loops, drawn, typed and ambient effects (a marquee, a typing reveal, a breathing loader), durations computed from data, and `delay` and stagger values. A file with several of them names them in a local constant. A spring far from all three presets (the folder icon's slow settle) stays literal too.

`src/components/styles/__tests__/motion-literals.test.ts` fails on a curve written out in the source: a cubic-bezier array, a `cubicBezier()` call, one of Framer's easing names other than `"linear"` (`"easeOut"`, `"easeInOut"`, ...), or an `*EASE*` array constant. The loops that keep their curve are listed in its `ALLOWED` map, each with a reason.

### The canary look

`src/app/tokens/themes/canary.css` is a second look, a dev tool the app never imports. Under `html[data-look="canary"]` it gives every colour role a loud value no default has (surfaces green, text magenta, borders cyan, each status and accent its own hue), every radius but `--radius-full` 0px (pills go square, circles stay round), `--shadow-color` pure red, the sans font Papyrus and the mono font Comic Sans, and every text size a quarter more. Anything that keeps its normal colour, corner or font under it does not read a token. (`data-theme` is for light and dark; a look is `data-look`.)

- In Storybook, switch the toolbar's **Look** to Canary.
- `npx tsx scripts/ds/canary-check.ts --storybook-url <url>` (or `--base-url <url> --routes /login,/signup` for the app) loads each story or page, screenshots it in both looks, and reports every visible element whose computed colour (text, background, border, outline, SVG fill and stroke, and the colours in a shadow or gradient), corner radius or font is not a canary value, grouped by kind, de-duplicated and counted, as `report.json` and `report.txt`. A full radius on a box that is not square is reported apart, as `border-radius (full)`: a `rounded-full` that should be `rounded-pill`, or a circle stretched out of square (a blurred blob, a voice dot while it speaks). Images, canvas, iframes and the artwork `ds-colour-exempt.json` lists are not checked; third-party markup (xyflow, streamdown, shiki, KaTeX, sonner) is labelled with its library. Public routes and stories are checked signed out; a route behind sign-in needs `SCREENSHOT_EMAIL` and `SCREENSHOT_PASSWORD`, or the run stops before it starts. It exits 1 when a target failed to render. The header of the script lists its flags.
- `src/components/__tests__/canary-look.test.ts` keeps the canary complete: it must redefine every `--color-*` role, every `--radius-*` but `full`, `--shadow-color` and every `--text-*` the token files declare on `:root`, and nothing they don't. A new token needs a canary value.

## Dark mode

The app is light only. There is no second theme and no switch.

No `dark:` or `light:` classes. A theme is a set of token values, not a second set of classes. Rule: `design-system/no-dark-variant`. Delete them and keep the light value; the role tokens carry the theme.

Two attributes on `<html>` set how the app looks. They are independent.

- **`data-theme`** is the mode: `light` or `dark`, and nothing else. next-themes writes it from `src/app/providers.tsx`, which forces `light`. The attribute is kept for the mode because the components we vendor switch their own dark styles on `[data-theme="dark"]`. `globals.css` ties `dark:` to the same selector (`@custom-variant dark`), and `shape.css` keeps its one dark value, `--smooth-ring-color`, under it. Neither renders today.
- **`data-look`** is the look: a set of brand colour and shape values. The canary look, a dev tool, is the one example (see "The canary look").

`src/components/__tests__/theme-contrast.test.ts` computes WCAG contrast from the light triplets: the body text roles on `canvas`, `surface` and `surface-muted`, `action-foreground` on `action`, the inverse text on `surface-inverse`, `foreground-on-color` on `brand`, `brand-hover`, `accent-strong`, `error`, `error-hover`, `success`, `info`, `info-foreground`, `indigo` and `pink`, `foreground-base` on `warning`, `yellow`, `orange` and `sky`, `accent-strong` on `accent-surface`, each status foreground on its surface and `placeholder` on `surface`, all against 4.5:1 (WCAG 1.4.3 holds placeholder text to the same minimum as other text). The pairs that fail today are listed in the test with their ratio.

### Adding a theme later

1. Write a file under `src/app/tokens/themes/` that redefines every colour role under `html[data-theme="dark"]`, and import it from `globals.css` right after `theme.css`.
2. Add it to the `THEMES` list in `theme-contrast.test.ts`, and add a test that it redefines every colour role, like `canary-look.test.ts` does for the canary.
3. Check that every component that draws an always-dark surface reads the inverse roles: `surface-inverse` (or `-hover`) for the surface and `foreground-inverse` for its text. The toast, the tooltip, terminals, code blocks, the solid Badge and the user's chat bubble already do, and anything new must too. One that paints the surface with `action` or its text with a surface role takes the theme's values for those roles, where its text can be unreadable while the contrast test, which checks the inverse roles' own pairs, passes.
4. Stop forcing light in `providers.tsx`.
5. In that file, give the inverse, fill and on-colour roles values of their own. In `theme.css` they repeat the values of `action`, `action-foreground` and of text roles.

## Primitives

`src/components/{atoms,molecules,organisms}` own their look. At the call site, pick a `variant`, `size`, `tone` or `shape`. Use `className` only for layout: width, flex and grid placement, margin, position. A treatment you repeat becomes a variant on the primitive, never a class string copied between call sites.

The props that replace the usual restyles:

- **`Text`:** the `variant` names a type role only (`hero`, `h1` to `h5`, `heading-md`, `lead`, `large`, `body-md`, `body`, `body-sm`, `small`, `caption-sm`, `tiny`, `label`, `eyebrow`). On top of it: `tone` (colour), `weight`, `leading` (a line height that differs from the role's: `none`, `tight`, `snug`, `normal`, `relaxed`, `inherit` or the fixed steps `4`, `4.5`, `5`, `6`, `7`, `8`, `9`, `10`), `tracking`, `mono`, `tabular`, `italic`, `decoration`, `transform`, `narrowSize` (a step up below md, to `body` or `large`) and `align`. So `<Text variant="small" weight="medium" leading="4">`, never a `leading-*` or `font-*` class. `scripts/ds/text-variant-map.ts` lists what each old compound name (`small-medium-tight`, ...) became, and `npx tsx scripts/ds/text-variants.ts --write <dir>` rewrites a branch that still uses them (no pixel moves; `scripts/ds/text-variant-map.test.tsx` proves it).
- **`Icon`:** `tone` (the same values as `Text`, from `src/components/styles/tones.ts`; without one the icon takes the colour around it), `spin` for a loading indicator and `motion` (`transform`, `colors`, `opacity`) for an icon that animates a class change, such as a chevron's rotation.
- **`Button`:** the `variant` names a role only (`primary`, `secondary`, `destructive`, `outline`, `ghost`, `link`, `icon`, `toggle`, `floating`, `tray`, `brand`, `warning`, `success`, and the `compact-*` family below). `size` is one scale: `large`, `small`, `row` (32px), `xs` (28px), `icon`, `icon-sm`, `icon-xs`, `icon-2xs` (24px), and `inline` and `inline-xs` (no box, for a link in copy). `shape` is the corner and the only radius a Button has: `pill` (the default), `rounded` (12px), `soft` (8px) or `sharp` (6px), each a radius token, so a look reshapes every Button; sizes set none, `toggle` and `floating` default to `sharp`, and a link has no box. `tone` is the label colour: the shared tones, plus `quiet`, `quiet-subtle`, `quiet-danger`, `quiet-warning`, `quiet-info` and `quiet-remove` for a ghost Button that changes colour on hover (`<Button variant="ghost" tone="quiet">`). `scripts/ds/button-variant-map.ts` lists what the old names became (`ghost-muted` and the other quiet ghosts, and the radius `xs`, `icon-xs`, `icon-2xs` and `icon-sm` used to set), and `npx tsx scripts/ds/button-variants.ts --write <dir>` rewrites a branch that still uses them (no pixel moves; `scripts/ds/button-variant-map.test.tsx` proves it).

`Text` also accepts `truncate`, `line-clamp-*`, `whitespace-*`, `break-*`, `text-balance`, `text-pretty`, `text-ellipsis`, `gap-*` and text alignment; `Icon` and `Button` accept layout only; `Skeleton` also accepts shape. Rule: `design-system/no-restyle`. Inside the three design-system directories, which define the looks, it checks only `Text`, `Icon` and `Button`, so a molecule restyles them through their props like any call site.

Classes on a primitive must be static strings the linter can read, so no `` `bg-${tone}` ``; map a prop to whole class names instead. Rule: `design-system/require-static-classes`.

Each primitive marks its root element with `data-slot`. The contract test lists the five that predate this, and the list only shrinks.

## Classes Tailwind knows

Every class must be one this project's Tailwind generates: a typo such as `text-md` or `flex-end` silently renders nothing. Rule: `design-system/no-unknown-classes`.

`components.json` keeps `tailwind.css` at `src/app/globals.css`, the stylesheet the app loads. That is where the shadcn CLI writes, and where `@shadcn/lint` reads the colour tokens from. `globals.css` has no `@import "tailwindcss"` (Tailwind 3 compat), so for the grammar `@shadcn/lint` falls back to the one stylesheet that does: `src/app/tailwind-grammar.css`, which imports Tailwind and `globals.css` (and through it `tailwind.config.ts`). Nothing imports that file, so it never ships. `design-system.mts` drops the notice `@shadcn/lint` prints about this fallback, and the one about the `Skeleton` contract's `rounded-pill` entry (its grammar is cn's bundled one, which has no `pill` step, so it warns that the entry matches only that exact class, which is the intent). A few class names that aren't Tailwind are allowed in `.oxlintrc.json`: the PostHog and xyflow markers (`ph-no-capture`, `nodrag`, `nowheel`, `nopan`) and the classes `globals.css` defines (`agpt-*`, `markdown-output`).

## Styles

Tailwind classes go in the component that renders the element. Global CSS is only for DOM we don't render as JSX (Streamdown output, xyflow, KaTeX), keyframes and token definitions. Lint doesn't enforce this; reviewers check it.

No `style={{}}` or `<style>` for static values; pass a value known only at runtime through a CSS custom property. Rule: `design-system/no-inline-styles`.

No arbitrary values except for layout and transitions (`w-[320px]`, `top-[7px]` and `transition-[width]` pass). The error names the scale class or token with the same value, for example `rounded-[0.75rem]` → `rounded-xl`, `font-[600]` → `font-semibold`, `text-[13px]` → `text-body-sm leading-inherit`. Rule: `design-system/no-arbitrary-values`.

## Typography

Text renders through the `Text` atom by default: `<Text variant="body">`, with `as` for the tag (`<Text variant="small" as="span">`), `tone` for the colour and `weight`, `leading`, `mono` or `tabular` to adjust the role. `<Text>` owns its typography and colour; `className` takes layout, truncation, wrapping and alignment only (the `Text` contract of `design-system/no-restyle`).

Use a scale class only where `Text` can't render the element identically, and inside primitives. The type scale is the app's only set of text sizes: its values are in `src/app/tokens/typography.css` (one `:root` block, mapped to `text-<step>` by `@theme` in `globals.css`), so a redesign changes that file and every size follows. Each step bundles its line-height:

| Step              | Size / line-height | Replaced (SECRT-3152)                  |
| ----------------- | ------------------ | -------------------------------------- |
| `text-tiny`       | 10/14px            |                                        |
| `text-caption-sm` | 11/16px            |                                        |
| `text-caption`    | 12/18px            | Tailwind `xs` (12/16)                  |
| `text-body-sm`    | 13/20px            |                                        |
| `text-body`       | 14/22px            | Tailwind `sm` (14/20)                  |
| `text-body-md`    | 15/24px            |                                        |
| `text-body-lg`    | 16/26px            | Tailwind `base` (16/24)                |
| `text-lead`       | 20/28px            | Tailwind `lg` (18/28) and `xl` (20/28) |
| `text-heading-sm` | 22/24px            |                                        |
| `text-heading-md` | 24/32px            | Tailwind `2xl` (24/32)                 |
| `text-heading`    | 28/40px            | Tailwind `3xl` (30/36)                 |
| `text-title`      | 32/40px            | Tailwind `4xl` (36/40)                 |
| `text-display`    | 44/56px            | Tailwind `5xl` (48/48)                 |
| `text-hero`       | 64/64px            | Tailwind `6xl` to `9xl`                |

- **Line-height:** override it only with `leading-inherit`, `leading-4.5`, `leading-5.5`, `leading-6.5` or a Tailwind step.
- **Letter-spacing:** `tracking-display`, `tracking-heading`, `tracking-title`, `tracking-caps` or `tracking-eyebrow`.
- **Weight:** Tailwind's scale (`font-medium`, `font-semibold`).

Tailwind's own sizes are reset in the theme (`--text-*: initial`), so `text-sm` or `md:text-2xl` renders nothing. Rule `design-system/no-raw-text-size` fails them, and arbitrary sizes (`text-[13px]`, `text-[length:..]`), naming the step to use; `scripts/ds/text-size-scale.ts` holds the mapping above and rewrites a file with `--write`. Never use raw sizes: no `text-[13px]`, `leading-[..]`, `tracking-[..]`, `font-[..]` or inline font styles. Streamdown, the chat's markdown renderer, still writes Tailwind's names in its own markup; `globals.css` sizes them from the same steps inside `[data-streamdown]`.

## Icons

Hugeicons only: icon data from `@hugeicons/core-free-icons`, rendered with the `Icon` atom (`<Icon icon={Delete02Icon} size={16} />`), which applies the 2px stroke. Type icon props with `IconSvgElement` from `@hugeicons/react`. Rule `design-system/hugeicons-only` reports:

- importing `lucide-react`, `@phosphor-icons/react`, `react-icons` or `@radix-ui/react-icons`;
- rendering `HugeiconsIcon` outside `atoms/Icon`.

## Legacy components

Feature code never imports from `src/components/__legacy__`, from any other `legacy/` path or from `@/components/ui`. Import the atom, molecule or organism in `src/components` instead; Storybook shows them. `src/components/__legacy__`, `LegacyButton` and the `legacy-*` Button variants and sizes are deleted; the rules below keep them from coming back. The rounded-md look they drew is the Button's `compact-*` variants and sizes. That family stays apart from the roles: besides the corner it has no border, its own shadows and hovers, no disabled colours and renders its children bare, so it can't become a `shape` and size of the pill variants without moving pixels. The Button's internal variant helpers are only for the Button atom itself.

- **Rule `design-system/no-legacy-import`:** the imports.
- **Rule `design-system/no-legacy-button-variant`:** the `legacy-*` variants, should one come back. The error names the design-system variant to use instead.

## Escape hatch

Fix the code first. When a suppression is unavoidable, scope it to one line and one rule, and give the reason on the same line after `--`:

```tsx
{
  /* oxlint-disable-next-line design-system/no-arbitrary-values -- xyflow positions handles in px */
}
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
