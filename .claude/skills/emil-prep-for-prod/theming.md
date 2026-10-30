# Lane G — Theming

The lane for the theme nobody opened. Most apps are built in one color scheme and *have* a second one. The second gets a token flip, a glance at the homepage, and ships with invisible cards, unreadable badges, and a logo that disappears. This lane is a defect sweep of the other theme, not a redesign of it. Owning skills: `emil-surfaces` (shadows, borders, dark surfaces) and `emil-color` (values, contrast).

First, from recon, establish which case you're in:

```
Does the app have a theme toggle, a `dark:` variant in use, or a
`prefers-color-scheme` rule?
├── No → single-theme app. Run G1 and G8 only.
└── Yes
    Is the second theme reachable by users (toggle, or OS setting honored)?
    ├── Yes → run the whole lane
    └── No (dead `dark:` classes, no way to activate) → G2, then G8
```

## G1. A single-theme app says so — SHOULD FIX

A dark-only app that never declares it gets white native scrollbars, light form controls, and a white flash before the CSS loads. A light-only app that declares `light dark` (usually pasted in with a boilerplate `<head>`) gets dark native controls and a dark canvas on any phone set to dark mode, under a UI that was only designed light. Declare exactly the schemes that were designed.

**Hunt for:** `<meta name="color-scheme">` and the CSS `color-scheme` property; a dark-only app with neither; a light-only app declaring `light dark`.

```css
:root { color-scheme: light; }        /* light-only */
:root { color-scheme: dark; }         /* dark-only: native controls and scrollbars go dark */
```

## G2. Half a dark mode is worse than none — BLOCKER when reachable

Some components have `dark:` classes and some don't, and the OS setting or a leftover toggle activates it. The user gets dark cards on a white page with black text on the dark cards.

**Hunt for:** `dark:` utilities or `[data-theme="dark"]` rules present in some components and absent in the layout, the inputs, or the text tokens; Tailwind's `darkMode` config set to `media` when the app was only designed in light.

This is a decision, not a diff: either finish the theme or make it unreachable (remove the toggle, set `color-scheme: light`, stop the `dark:` variant from firing). Put it on the human list with both options and what each costs. Turning off a broken dark mode before launch is a legitimate choice.

## G3. Text is readable in both themes — BLOCKER for body text

Contrast was checked in the theme the app was designed in. The muted grey that passes on white fails on near-black, and the brand color that works as a button fill fails as link text on a dark surface.

**Hunt for:** hardcoded color values inside components (`#6b7280`, `text-gray-500`, `text-black`, `bg-white`) that don't flip; muted / secondary / placeholder / disabled text tokens in the dark theme; colored badges and alerts (tinted background + same-hue text); text over images.

The measurement and the repair are A9: find the real pair, then move the foreground's lightness. Here the job is to check every pair *twice*. Fast triage on a dark surface (L ≤ 0.25): foregrounds want L at or over 0.75.

## G4. Hardcoded colors don't flip — SHOULD FIX

One `bg-white` on a dropdown, one `#fff` in an SVG, one `border-gray-200` on a card: each is a bright rectangle in the dark theme. These are the most common dark-mode defects and the most mechanical to fix.

**Hunt for:** raw hex / `rgb()` / `white` / `black` in component styles; Tailwind palette classes (`bg-white`, `text-gray-900`, `border-gray-200`) in a project that has semantic tokens; inline `fill="#000"` / `stroke="#111"` in SVG icons; `<img>` logos with baked-in dark text.

Fix: point each at the project's existing token. Icons use `currentColor` so they follow the text:

```html
<svg fill="none" stroke="currentColor">…</svg>
```

Never scatter new per-component `dark:` overrides to patch these. If the project flips a numbered variable scale at the theme root, the fix is using the variable (Hard Rule 7). A raster logo with dark text needs a second asset or an SVG; that's the human list.

## G5. Elevation survives the dark theme — SHOULD FIX

Stacked translucent black shadows are invisible against a dark surface, so cards, menus, and popovers that relied on shadow for their edge dissolve into the page. In the dark theme each shadow token collapses to a single low-opacity white ring.

**Hunt for:** `box-shadow` tokens with no dark-theme value; cards and popovers with shadow-only edges; `shadow-*` utilities with no dark counterpart.

```css
:root {
  --elevation-raised: 0 0 0 1px rgb(0 0 0 / 0.07), 0 2px 4px rgb(0 0 0 / 0.06);
}
[data-theme="dark"] {
  --elevation-raised: 0 0 0 1px rgb(255 255 255 / 0.08);
}
```

If the project has shadow tokens, add the dark value at the token. If every component hand-writes its shadow, that's a bigger job; list the surfaces that disappear and name `emil-surfaces`. Image outlines flip too: black at 10% in light, white at 10% in dark, never tinted.

## G6. Images and media work on both backgrounds — SHOULD FIX

A transparent PNG screenshot with dark text, a diagram drawn for a white page, a product shot with a white matte: each is wrong on the other background.

**Hunt for:** transparent PNG / SVG assets containing text or dark linework; screenshots of the app's light UI used on a dark marketing page; embedded charts with hardcoded axis colors; favicons that vanish on a dark tab bar.

```html
<picture>
  <source srcset="/diagram-dark.svg" media="(prefers-color-scheme: dark)" />
  <img src="/diagram-light.svg" alt="…" width="800" height="400" />
</picture>
```

That works when the theme follows the OS. With a class-based toggle, render both and hide one by theme class. A missing second asset can't be invented (Hard Rule 8); flag it.

## G7. Third-party surfaces follow the theme — SHOULD FIX

The app goes dark and the embedded pieces don't: a white code block, a white Stripe element, a white chat widget, a light syntax theme, a map, a CAPTCHA.

**Hunt for:** syntax highlighter themes (Shiki / Prism) configured with one theme; payment, auth, and support widgets with an `appearance` / `theme` option left at default; iframes.

Shiki takes a theme per scheme (`themes: { light, dark }`); Stripe Elements, Clerk, and most widgets take an appearance option. Pass the app's resolved theme through.

## G8. Browser chrome follows the theme — POLISH

`theme-color` per scheme (C18), `color-scheme` on the root so native scrollbars, form controls, and autofill match (G1), a `::selection` color that's legible in both, and a favicon that survives a dark tab bar. Autofilled inputs get the browser's pale yellow or blue background regardless of theme; if that breaks a dark form, override it:

```css
input:-webkit-autofill {
  -webkit-text-fill-color: var(--gray-12);
  box-shadow: 0 0 0 100px var(--gray-2) inset;
}
```

## G9. The theme is right on first paint and switches cleanly — see E5, B14

Flash of the wrong theme on refresh is E5. Every color tweening at a different speed on toggle is B14. Check both here, count them there.

## G10. Open every route in the other theme — always

Code review finds the hardcoded colors. It doesn't find the screen that simply looks wrong. Put on the device list: "switch to the other theme and visit every route, including auth, settings, empty states, error pages, modals, toasts, and emails if they're themed", "open each dropdown, popover, and tooltip in the dark theme and confirm you can see its edge", "toggle the theme with a modal open". Report these as checks to run, never as verified.
