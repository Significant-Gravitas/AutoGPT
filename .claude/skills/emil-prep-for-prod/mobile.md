# Lane C — Mobile

The lane where "works in Chrome" means nothing. Every item here is a real-hardware behavior. This is the ship-check subset; the full reasoning behind each fix, and the deeper cases (PWAs, sheets, gestures), live in `emil-mobile-native`.

## C1. Never disable zoom — BLOCKER

`user-scalable=no` and `maximum-scale=1` block low-vision users from zooming. They exist in codebases as a workaround for input zoom (C3). Remove them and fix the input instead.

**Hunt for:** `user-scalable`, `maximum-scale`, `maximumScale`, `userScalable` in the viewport meta tag or the Next.js `viewport` export.

```html
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, interactive-widget=resizes-content" />
```

## C2. Hover is gated — SHOULD FIX

Touch has no hover, so the first tap applies `:hover` and leaves it there. A button that scales on hover stays scaled after being tapped. Gate every hover style behind a capability query and give touch users press feedback through `:active` instead.

**Hunt for:** `:hover` rules outside `@media (hover: hover)`. In Tailwind v4 `hover:` already compiles to `@media (hover: hover)`; in v3 check for `future.hoverOnlyWhenSupported` in the config, and add it if it's missing. That one line is the root fix for every instance.

```css
@media (hover: hover) and (pointer: fine) {
  .button:hover { background: var(--gray-3); }
}
```

## C3. Inputs are 16px on touch — BLOCKER

iOS Safari zooms the page when focus lands on an input under 16px and doesn't zoom back on blur.

**Hunt for:** `font-size` under 16px on `input`, `textarea`, `select`; `text-sm` / `text-xs` on them or on a shared `Input` component; a contenteditable editor with small text.

```css
input, textarea, select { font-size: 16px; }

/* if desktop needs smaller inputs */
@media (pointer: coarse) {
  input, textarea, select { font-size: 16px; }
}
```

Tailwind: `text-base md:text-sm` on the shared input. While there: `inputmode="numeric"` for codes, `inputmode="decimal"` for amounts, `type="email"` / `type="tel"` for their fields, `autocapitalize="none"` + `autocorrect="off"` on usernames and codes, `autocomplete="one-time-code"` on verification inputs, `enterkeyhint="search"` / `"send"` / `"done"` so the return key says what it does.

## C4. Tap targets are 44px — SHOULD FIX

A 28px icon button mis-taps. The visual can stay small; the hit area can't. Two hit areas must never overlap; if an extended area collides with a neighbor, shrink it to the largest non-colliding size.

**Hunt for:** icon buttons with `width` / `height` under 44px (`size-6`, `h-8 w-8`) and no padding or pseudo-element extending the hit area; toast dismiss buttons; table row actions; pagination.

```css
.icon-button {
  position: relative;
  width: 24px;
  height: 24px;
}
.icon-button::before {
  content: "";
  position: absolute;
  inset: -10px;
}
```

Tailwind: `relative before:absolute before:-inset-2.5`.

## C5. Taps feel instant — SHOULD FIX

Two causes stack. `touch-action: manipulation` removes the double-tap-zoom wait so `click` fires immediately. Press feedback on `:active` responds when the finger lands, not when it leaves.

**Hunt for:** tappable elements with no `touch-action`; buttons with no `:active` rule.

```css
button, a, input, [role="button"] { touch-action: manipulation; }

.button { transition: transform 100ms var(--ease-out), background-color 100ms; }
.button:active { transform: scale(0.97); }
```

## C6. Tap highlight is off, and press feedback replaces it — SHOULD FIX

iOS and Android paint a translucent flash on anything with a click handler. It's the loudest "this is a website" signal. Remove it globally, then make sure every control has its own `:active` state (C5), because you've removed the only feedback the browser gave.

```css
html { -webkit-tap-highlight-color: transparent; }
```

## C7. Viewport height is `dvh` or `svh`, not `vh` — SHOULD FIX

`100vh` is the *largest* viewport, with the URL bar collapsed. On load the bar is visible, so a `100vh` shell overflows and a bottom-pinned button sits under it.

**Hunt for:** `100vh`, `h-screen`, `min-h-screen`, `max-h-screen`, `calc(100vh - …)`.

```css
.app  { height: 100dvh; }      /* app shells, drawers: tracks the visible area */
.hero { min-height: 100svh; }  /* marketing heroes: stable, never overflows, no shift on scroll */
```

Tailwind: `h-dvh`, `min-h-svh`. Desktop-only admin tools can keep `vh`; note it and move on.

## C8. Safe areas are respected — SHOULD FIX

Fixed headers, bottom tab bars, toasts, and sheets sit under the notch or the home indicator without this. Both halves are required: without `viewport-fit=cover` every `env()` value is `0`.

**Hunt for:** `position: fixed` / `sticky` elements at `top: 0` or `bottom: 0` with no `env(safe-area-inset-*)`; the toaster's offset; `env()` used without the meta tag.

```css
.app-header { padding-top: env(safe-area-inset-top); }
.bottom-bar { padding-bottom: env(safe-area-inset-bottom); }
.sheet      { padding-bottom: calc(1rem + env(safe-area-inset-bottom, 0px)); }
```

## C9. Scroll doesn't chain or pull-to-refresh in app UI — SHOULD FIX

Inside an app shell, a drawer, or a chat list, scrolling past the end should not rubber-band the page or trigger Android's pull-to-refresh. `none` on the root, `contain` on inner scrollers. Never a `touchmove` + `preventDefault()` listener; it blocks scrolling and makes the listener non-passive.

**Hunt for:** `preventDefault` inside `touchmove`; inner scroll containers (`overflow-y: auto`) without `overscroll-behavior`; modals whose background scrolls when the modal content hits its end.

```css
html, body { overscroll-behavior: none; }        /* skip on a scrolling document where pull-to-refresh is welcome */
.sheet-content { overflow-y: auto; overscroll-behavior: contain; }
```

## C10. Controls aren't selectable; content is — SHOULD FIX

Long-pressing a button selects its label on iOS; long-pressing a link pops the copy callout. Text that is a control isn't selectable. Text that is content must stay selectable, so never put this on `body`.

**Hunt for:** `user-select: none` / `select-none` on `body`, `*`, or a layout wrapper (a defect); buttons and tabs without it (a polish item).

```css
button, [role="button"], .tab, .chip, .drag-handle {
  user-select: none;
  -webkit-user-select: none;
  -webkit-touch-callout: none;
}
```

## C11. Gesture surfaces declare their axis — SHOULD FIX

A horizontal carousel without `touch-action: pan-y` makes the page jitter while the track moves. A custom drag surface without `touch-action: none` fights the browser for every pointer event. And `touch-action: none` on something the user needs to scroll past traps them.

**Hunt for:** carousels, sliders, drag-to-dismiss sheets, sortable lists, and canvases with no `touch-action`.

```css
.carousel     { touch-action: pan-y; }  /* I handle horizontal; browser keeps vertical */
.drag-surface { touch-action: none; }   /* I handle every axis; only on a real gesture surface */
```

For a native-scroll carousel, prefer `scroll-snap-type: x mandatory` on the track and `scroll-snap-align: start` on slides over a JS gesture.

## C12. Video autoplays inline — SHOULD FIX

Without `muted` and `playsinline`, iOS refuses to autoplay or opens the video full screen.

**Hunt for:** `<video` with `autoplay` / `autoPlay` missing either attribute.

```html
<video autoplay muted playsinline loop>
  <source src="video.mp4" type="video/mp4" />
</video>
```

Under `prefers-reduced-motion: reduce`, don't autoplay; show controls (A8).

## C13. No autofocus on touch — SHOULD FIX

`autoFocus` on a touch device opens the keyboard unexpectedly and shoves the layout up. Autofocus the first input in a modal on desktop only.

**Hunt for:** `autoFocus` / `autofocus` / `.focus()` in a mount effect with no pointer gate.

```jsx
const isCoarse = typeof window !== "undefined" && window.matchMedia("(pointer: coarse)").matches;
<input autoFocus={!isCoarse} />
```

## C14. The keyboard doesn't cover the input — SHOULD FIX

Android Chrome overlays the software keyboard on top of the page by default, so a bottom-pinned composer or CTA ends up underneath it. `interactive-widget=resizes-content` in the viewport meta tag (C1) makes the keyboard shrink the layout viewport, so `100dvh` layouts react the way they do on iOS.

**Hunt for:** chat composers, comment boxes, and sticky form footers pinned with `position: fixed; bottom: 0`; the viewport meta tag missing `interactive-widget`.

## C15. Fixed bottom CTAs don't cover the last field — SHOULD FIX

A fixed bottom button sits over the final form field and over the home indicator. Pad the scroll container by the CTA's height plus the safe area, and pad the CTA itself with `env(safe-area-inset-bottom)` (C8). The same goes for cookie banners and chat widgets: check what they sit on top of at 375px wide.

## C16. Sticky headers leave room on a phone — SHOULD FIX

An 80px sticky header on a 667px viewport in landscape leaves almost no content. Cap it, and give anchored elements `scroll-margin-top` so in-page links don't land under it.

```css
[id] { scroll-margin-top: 64px; } /* the sticky header's height; read it from the code */
```

## C17. Nothing scrolls sideways — SHOULD FIX

One element wider than the viewport makes the whole page pan horizontally on a phone, and every screen feels loose. The usual causes: a long URL or email in a card, a `<pre>` block, a wide table, a fixed-width element, `100vw` on a page that has a vertical scrollbar.

**Hunt for:** `width: 100vw` / `w-screen`; fixed pixel widths over ~340px; `<table>` and `<pre>` with no scrolling wrapper; user-generated strings with no `overflow-wrap`.

```css
.card-text { overflow-wrap: break-word; }
.table-wrap, pre { overflow-x: auto; }
```

Fix the element that overflows. `overflow-x: hidden` on `body` hides the symptom and breaks `position: sticky`.

## C18. Status bar matches the page — POLISH

One `theme-color` means light mode gets a dark bar or dark mode gets a white one. One per scheme, matching the color at the very top of the page (the header background, not the brand color).

```html
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />
<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#0a0a0a" />
<meta name="color-scheme" content="light dark" />
```

In Next.js: the `viewport` export's `themeColor: [{ media, color }]`. If theme is toggled by class, update the tag from JavaScript on toggle. Read the colors from the project's tokens; if they aren't obvious, put this on the human list rather than guessing (Hard Rule 8).

## C19. Test on hardware — always

Sticky hover, the tap highlight, `vh` and the URL bar, input zoom, the click delay, overscroll, safe areas, and the software keyboard do not reproduce in device emulation. Connect a phone, run the dev server on `0.0.0.0`, open it by LAN IP, and use Safari's Web Inspector (iOS) or `chrome://inspect` (Android). Test on a phone a few years old, with the keyboard open, and once in landscape. Report this as a device check, never as verified.
