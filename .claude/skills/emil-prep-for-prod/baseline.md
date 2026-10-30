# The Global Baseline

When a project has none of this, ship it first (Phase 4, step 2). It resolves a dozen findings across lanes A, B, C, E, G, and H at once, and it's the floor for any app a phone will open.

Read before pasting:

- **Merge, don't append.** Most projects have half of this already. Add the missing lines to the existing root rules; never create a second global stylesheet or a second viewport tag.
- **Project idiom.** In a Tailwind project this goes in the base layer of the global CSS, using the project's tokens. `var(--gray-12)` below stands for "the project's strongest neutral"; substitute the real token.
- **Every value marked `←` is read from the project, never invented** (Hard Rule 8). If the project doesn't have it, leave the line out and put it on the human list.
- **Conditional lines are marked.** Pasting `color-scheme: light dark` into a light-only app creates the bug in G1.

## Head

```html
<html lang="en">                                       <!-- ← the content's real language -->

<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, interactive-widget=resizes-content" />

<!-- Two-theme apps. Single-theme apps: one theme-color, and color-scheme set to that one scheme. -->
<meta name="theme-color" media="(prefers-color-scheme: light)" content="#ffffff" />   <!-- ← header background, light -->
<meta name="theme-color" media="(prefers-color-scheme: dark)"  content="#0a0a0a" />   <!-- ← header background, dark -->
<meta name="color-scheme" content="light dark" />

<link rel="preload" href="/fonts/inter-var.woff2" as="font" type="font/woff2" crossorigin />  <!-- ← skip with next/font -->
```

Next.js App Router equivalent, in the root layout:

```tsx
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  interactiveWidget: "resizes-content",
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export const metadata: Metadata = {
  metadataBase: new URL("https://example.com"),          // ← the production origin
  title: { default: "Product", template: "%s · Product" }, // ← the product's name
  description: "…",                                       // ← existing copy, or the human list
};
```

`viewport-fit=cover` is what makes `env(safe-area-inset-*)` non-zero. `interactive-widget=resizes-content` makes the software keyboard shrink the layout viewport on Android Chrome, so `100dvh` and bottom-pinned inputs react the way they do on iOS. Neither `maximum-scale` nor `user-scalable` appears, ever.

## Root CSS

```css
html {
  -webkit-text-size-adjust: 100%;            /* no font inflation in landscape */
  -webkit-tap-highlight-color: transparent;  /* every control then needs its own :active */
  font-synthesis: none;                      /* a missing weight fails visibly instead of being faked */
  overscroll-behavior: none;                 /* app shells only; drop on a scrolling document */
}

body {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Keyboard focus: visible, neutral, never removed */
:focus-visible {
  outline: 2px solid var(--gray-12);
  outline-offset: 2px;
}

/* iOS zooms into any input under 16px and never zooms back */
input, textarea, select {
  font-size: 16px;
}

/* No double-tap-zoom wait on controls */
button, a, input, select, textarea, [role="button"] {
  touch-action: manipulation;
}

/* Controls aren't selectable. Content is: never on body */
button, [role="button"] {
  user-select: none;
  -webkit-user-select: none;
}

/* Headings break evenly; paragraphs don't strand a word */
h1, h2, h3 { text-wrap: balance; }
p { text-wrap: pretty; }

/* In-page links clear the sticky header */
[id] { scroll-margin-top: 64px; }            /* ← the sticky header's height */

/* Smooth scrolling is opt-in for people who haven't asked for less motion */
@media (prefers-reduced-motion: no-preference) {
  html { scroll-behavior: smooth; }
}

/* Theme switches don't tween every color at its own speed */
.no-transitions, .no-transitions * { transition: none !important; }
```

## The two gates

These aren't rules to paste. They're the two places the rest of the stylesheet has to move into, and doing it at the token or config level fixes every instance at once.

**Hover.** Every `:hover` rule lives behind a capability query, so a tap never leaves a control stuck in its hover state.

```css
@media (hover: hover) and (pointer: fine) {
  .button:hover { background: var(--gray-3); }
}
```

Tailwind v4's `hover:` variant does this already. Tailwind v3: one line in the config.

```js
// tailwind.config.js
module.exports = { future: { hoverOnlyWhenSupported: true } };
```

**Reduced motion.** Movement swaps to opacity; feedback stays. If the project's animations run through shared keyframes or duration tokens, make the swap there.

```css
@media (prefers-reduced-motion: reduce) {
  .modal   { animation-name: fade-in; }      /* was scale-in */
  .drawer  { transition-property: opacity; } /* was transform */
}
```

```jsx
// Motion: honor the OS setting app-wide. The default is "never".
<MotionConfig reducedMotion="user">{children}</MotionConfig>
```

Never the blanket `* { animation: none !important; transition: none !important; }`. It removes the fades that tell the user something happened.

## Theme provider

```jsx
// next-themes: no flash on load (it injects the blocking script), no color ripple on toggle
<ThemeProvider attribute="class" disableTransitionOnChange>
  {children}
</ThemeProvider>
```

Only when the project already uses `next-themes`. Match its existing `attribute`.

## Dev-only mounts

```jsx
{process.env.NODE_ENV === "development" && <Lapse />}
```

Every inspector, control panel, and devtools drawer found in recon sits behind this gate, imported dynamically so the package stays out of the production bundle (H9).

## What the baseline doesn't cover

It sets the floor; it doesn't sweep the app. After it's in, the per-component findings remain: the icon buttons still need names, the `100vh` shells still need `dvh`, the fixed bars still need `env()` padding, the empty states still need to exist. And with the tap highlight gone, any control without an `:active` state now gives no touch feedback at all, so C5 becomes more urgent, not less.
