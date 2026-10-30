# Lane A — Accessibility

The lane that excludes people when it's wrong. Fix these first. For anything deeper than the fix shown here, the owning skills are `emil-touch-and-accessibility` (interaction) and `emil-color` (contrast).

Each check has a **Hunt for** line (what to grep or read) and a fix. Severity is the default; Phase 3 of the skill may adjust it with context.

## A1. Icon buttons have a name — BLOCKER

A `<button>` whose only child is an SVG is announced as "button" and nothing else. The label names the **action**, never the element or the icon.

**Hunt for:** `<button>` / `<a>` with an icon component and no text child; `aria-label="icon"`, `aria-label="button"`, `aria-label="close icon"`.

```html
<button aria-label="Close dialog"><CloseIcon /></button>
<button aria-label="Search"><SearchIcon /></button>
```

Decorative icons next to visible text get `aria-hidden="true"` so the label isn't read twice.

## A2. Buttons are buttons — BLOCKER

A `<div onClick>` isn't focusable, isn't reachable by Tab, doesn't fire on Enter or Space, and isn't announced as interactive. `<button>` gets all of it for free.

**Hunt for:** `onClick` on `div`, `span`, `li`, `svg`, `img`; `role="button"` (usually a sign the element should have been a button).

```html
<!-- Bad -->
<div onClick={handleClick}>Save</div>

<!-- Good -->
<button type="button" onClick={handleClick}>Save</button>
```

`type="button"` matters: a bare `<button>` inside a form is a submit button. Navigation is `<a href>`. Actions are `<button>`. A card whose whole surface is one `<a>` reads the entire card as a single link and blocks text selection; give it a distinct CTA link instead.

## A3. Focus is visible — BLOCKER

`outline: none` with no replacement makes keyboard navigation blind. Replace, don't remove. Use `:focus-visible` so mouse clicks don't show the ring, and keep the color grey, black, or white; brand-colored outlines clash with the interface.

**Hunt for:** `outline: none`, `outline: 0`, `outline-none` (Tailwind) without a sibling `focus-visible` rule. A global reset is the usual culprit.

```css
:focus-visible {
  outline: 2px solid var(--gray-12);
  outline-offset: 2px;
}
```

A focus ring clipped by `overflow: hidden` on the parent is the same failure. Move the ring inside with a negative `outline-offset`, or lift the clip.

## A4. Labels are associated — BLOCKER

Clicking a label must focus its input. Without the association, the label is decoration and screen readers announce the input as unlabeled. A placeholder is not a label; it disappears the moment the user starts typing, right when the reminder is needed.

**Hunt for:** `<label>` without `htmlFor`/`for`; inputs with only a `placeholder`; inputs with neither.

```html
<label for="email">Email</label>
<input id="email" type="email" />

<!-- or wrap -->
<label>
  Email
  <input type="email" />
</label>
```

A search field with no room for a visible label still needs a name: `aria-label="Search projects"`.

## A5. Modals manage focus — BLOCKER

When a dialog opens, focus moves into it (the first interactive element, or the dialog itself). While it's open, Tab cannot escape to the page behind. Escape closes it. When it closes, focus returns to the trigger. Content behind the modal is `inert`.

**Hunt for:** hand-rolled modals (`position: fixed` overlays toggled by `useState`) that don't use a dialog primitive; overlays with no `keydown` handler for Escape.

Use a primitive that does this (Base UI, Radix, the native `<dialog>` with `showModal()`). If the modal is hand-rolled, that is the finding. Swapping in the project's existing primitive is a fix; building a focus trap from scratch is bigger than a ship check, so it goes on the human list with `emil-touch-and-accessibility` named.

## A6. Hidden things are out of the tab order — BLOCKER

An off-screen drawer or a collapsed panel that's only `opacity: 0` or `transform: translateX(-100%)` is still tabbable. Keyboard users land on invisible controls, and screen readers read the closed menu aloud.

**Hunt for:** `opacity: 0` / `translate` off-screen on panels that contain links or buttons; mobile nav drawers; accordions that collapse with `height: 0`.

```css
.panel[data-closed] {
  visibility: hidden; /* removed from tab order and the accessibility tree */
}
```

```jsx
<aside inert={!isOpen}>…</aside>
```

`display: none` also works when the panel doesn't animate. To keep an exit animation, delay the flip: `transition: visibility 0s 200ms` on close.

## A7. Errors aren't color-only — BLOCKER

A red border is invisible to colorblind users and silent to screen readers. Three signals: color, an icon, and a text message colocated with the field.

**Hunt for:** `border-color: red` / `border-red-500` toggles with no adjacent message; error summaries at the top of the form only; success and warning states told apart by hue alone.

```jsx
<div className="field">
  <label htmlFor="email">Email</label>
  <input id="email" aria-invalid={!!error} aria-describedby={error ? "email-error" : undefined} />
  {error && <p id="email-error" className="error"><AlertIcon aria-hidden /> {error}</p>}
</div>
```

The message names the fix ("Your email must include an @"), not the category ("Invalid input").

## A8. Reduced motion is honored — BLOCKER for large movement, SHOULD FIX otherwise

Motion can make people sick. `prefers-reduced-motion: reduce` means **remove the movement, keep the meaning**: a modal that scales in becomes one that fades in; a sidebar that slides becomes one that fades; a multi-step form that slides crossfades instead. Never zero out feedback entirely.

**Hunt for:** `transform`/`translate`/`scale` animations and `animate={{ x, y }}` with no `prefers-reduced-motion` or `useReducedMotion` anywhere in the file or the tokens; autoplaying video and GIFs; `scroll-behavior: smooth`; infinite loops; parallax.

```css
.modal { animation: scale-in 200ms var(--ease-out); }

@media (prefers-reduced-motion: reduce) {
  .modal { animation: fade-in 200ms var(--ease-out); }
}
```

```jsx
// App-wide safety net for Motion; the default is "never", so this does nothing until you set it.
<MotionConfig reducedMotion="user">{children}</MotionConfig>
```

Tailwind: `motion-safe:` and `motion-reduce:` variants.

Specific cases:

- **Smooth scrolling** is opt-in under `no-preference`, not opt-out under `reduce`: `@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }`.
- **Autoplaying video**: don't autoplay under `reduce`; show controls. `autoPlay={!prefersReducedMotion} controls={prefersReducedMotion}`.
- **Animated GIF/AVIF**: `<picture>` with a `media="(prefers-reduced-motion: no-preference)"` source and a static `<img>` fallback.
- **Looping animation**: pause it on a representative frame, not frame 0: `animation-play-state: paused; animation-delay: -0.4s`.

The blanket `* { animation: none !important; transition: none !important; }` is a finding, not a fix (Hard Rule 6). It removes the fades that tell the user something happened.

Verify with DevTools → Rendering → *Emulate CSS media feature prefers-reduced-motion*. The common failure is a "reduced" variant that still moves because one transform lives in a shared class.

## A9. Contrast passes in both themes — BLOCKER for body text, SHOULD FIX otherwise

Measure the real pair: the text color against the nearest opaque ancestor background, not against the page. Body text needs 4.5:1 (WCAG AA); text around 24px and larger, and UI components, need 3:1. Sanity-check with APCA where the two specs disagree: |Lc| ≥ 60 for body text, ≥ 45 for large text, ≥ 30 for borders, icons, and focus rings.

**Hunt for:** the lightest grey text tokens on light backgrounds (`--gray-6`…`--gray-9` used for text, `text-gray-400`); placeholder text; text on tinted cards and brand-colored buttons; every one of those again in dark mode; `opacity` on `:disabled` (passes on one background, fails on another; use a specific muted token).

The repair is one-dimensional. Move the foreground's lightness away from the background's and leave hue and chroma alone, so the color keeps its identity:

```css
/* fails: lavender label nearly as light as its lavender card */
.card-label { color: oklch(0.72 0.14 300); }

/* passes: only L moved */
.card-label { color: oklch(0.42 0.14 300); }
```

Raising chroma to "make it pop" does nothing for contrast. Placeholder text isn't exempt: if it carries information it needs 4.5:1 too, which is one more reason it can't be the label.

## A10. Illustrations and decoration are labeled or hidden — SHOULD FIX

Code-built illustrations are announced as a pile of divs. Decorative images are announced by filename.

**Hunt for:** `<img>` without `alt`; decorative `<svg>` without `aria-hidden`; illustration containers with no role.

```jsx
<div role="img" aria-label="Diagram of the request flow" style={{ userSelect: "none", pointerEvents: "none" }} />
<img src="hero.webp" alt="" />          {/* decorative */}
<svg aria-hidden="true">…</svg>          {/* decorative icon */}
```

An informative image gets `alt` that says what it shows. A missing `alt` on the only product shot is a blocker.

## A11. DOM order matches visual order — SHOULD FIX

Screen readers and Tab follow the DOM. `order: -1`, `flex-direction: row-reverse`, or absolute positioning that reorders visually creates a tab sequence that jumps around the screen.

**Hunt for:** `order:`, `row-reverse`, `column-reverse`, Tailwind `order-*` on containers with interactive children; positive `tabindex` values (always wrong).

Fix: reorder the markup, not the CSS.

## A12. Page structure is real — SHOULD FIX

Screen reader users navigate by landmarks and headings. A page of `<div>`s gives them nothing to jump between, and a tab with the same title on every route gives them no way to tell pages apart.

**Hunt for:** routes with no `<main>`; more than one `<h1>`; heading levels that skip (`h1` → `h4`) because the size looked right; `<html>` with no `lang`; client-side route changes that never update `document.title`.

```html
<html lang="en">
  <body>
    <header>…</header>
    <main id="main">
      <h1>Projects</h1>
    </main>
  </body>
</html>
```

Pick the heading level from the outline and the size from the type scale. They're separate decisions.

## A13. Skip link on pages with long navigation — SHOULD FIX

Without one, keyboard users tab through every nav item on every page.

```html
<a href="#main" class="skip-link">Skip to content</a>
<main id="main">…</main>
```

```css
.skip-link {
  position: absolute;
  left: -9999px;
}
.skip-link:focus-visible {
  left: 16px;
  top: 16px;
}
```

## A14. Tooltips hold no interactive content — SHOULD FIX

A tooltip can't be focused, so a link inside it is unreachable. Interactive content goes in a popover triggered by click.

**Hunt for:** `<a>` or `<button>` rendered inside a tooltip component; tooltips as the only place an error or a limit is explained.

## A15. Hover enhances, never enables — SHOULD FIX

A control that only appears on hover is unreachable on touch and by keyboard. Feedback (errors, confirmations, status) is visible without hover.

**Hunt for:** `opacity: 0` → `:hover { opacity: 1 }` and `group-hover:opacity-100` on containers holding buttons; feedback shown only in a hover tooltip.

Fix: always render the control (a lower-emphasis style is fine), or reveal it on `:focus-within` as well as hover.

## A16. Async changes are announced — SHOULD FIX

A toast, a form result, or a "3 results" count that appears silently doesn't exist for a screen reader user. They submitted the form and heard nothing.

**Hunt for:** hand-rolled toast and notification components; inline "Saved" confirmations; search result counts; form-level errors rendered after submit.

```jsx
<div role="status" aria-live="polite">{message}</div>   {/* confirmations, counts */}
<div role="alert">{error}</div>                          {/* errors that need attention now */}
```

The live region must exist in the DOM *before* the message is put into it, or nothing is announced. Sonner and the common toast primitives do this already; the finding is the hand-rolled one.

## A17. Time-limited actions pause when the tab is hidden — SHOULD FIX

A toast countdown, an undo window, or a session timer that keeps running while the user is on another tab expires behind their back.

**Hunt for:** `setTimeout` driving user-visible countdowns with no `visibilitychange` listener; toasts that don't pause on hover or focus.

```js
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pauseTimer();
  else resumeTimer();
});
```

Toast duration scales with content (~200–250 words per minute reading speed), not a flat 2s for every message.

## A18. Keyboard focus is never lost or off-screen — SHOULD FIX

Deleting the focused row, closing a popover, or removing the focused element drops focus to `<body>`, and the keyboard user starts again from the top of the page.

**Hunt for:** delete handlers on list rows; conditional rendering that unmounts the element that was just activated; focusable elements inside horizontally scrolled or clipped containers.

Fix: after a removal, move focus to the next row, the previous one, or the list's heading. Focused elements inside scrollers scroll into view: `e.target.scrollIntoView({ block: "nearest" })`.

## A19. Shortcuts show the right modifier — POLISH

"⌘K" on Windows is a lie. Detect the platform, show `Ctrl` elsewhere, and bind the matching key (`e.metaKey || e.ctrlKey`).

## A20. Menus keep their content in the DOM — POLISH

Header submenus on marketing pages render their content server-side and hide it, rather than mounting on hover. Screen readers and crawlers see the structure; hover only reveals it. See lane I.

## A21. Test with a keyboard and a screen reader — always

From code you can prove attributes exist. You cannot prove the reading order makes sense or that the focus path is sane. Put both on the device list, concretely: "unplug the mouse and complete signup → create project → invite using only Tab, Enter, Escape, and arrows", "turn on VoiceOver (Cmd+F5) and listen to the dashboard top to bottom". Report these as device checks, never as verified.
