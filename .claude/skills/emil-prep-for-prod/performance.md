# Lane B — Performance

The lane where the dev machine lies. Every check here reproduces on a mid-range phone and hides on a fast laptop. Fast UI is mostly work you *don't* do, so most fixes here delete or defer something. For anything deeper, the owning skill is `emil-performance`.

Layout shift from content that arrives late is split between this lane (assets: images, fonts, video) and lane E (data: skeletons, empty states, persisted UI state).

## B1. Media reserves its space — BLOCKER

The single most visible launch-day bug. An image or video without dimensions has zero height until it loads, then shoves the page down under the user's finger.

**Hunt for:** `<img>` / `<video>` / `<iframe>` without `width` and `height` (or an `aspect-ratio`); `next/image` with `fill` inside a parent that has no size; embeds and ad slots.

```html
<img src="hero.webp" width="1200" height="630" alt="…" />
```

```css
.video-frame { aspect-ratio: 16 / 9; }
```

The attributes don't fix the rendered size. They give the browser the ratio so it reserves the box before the bytes arrive.

## B2. Fonts are preloaded, subset, and limited to used weights — SHOULD FIX

An un-preloaded font is discovered late and swaps in after first paint, shifting every line. Eight weights loaded for three used is dead weight on every visit. Declare a fallback stack whose metrics match so the swap is invisible.

**Hunt for:** `@font-face` or `next/font` config; `<link rel="preload">` for fonts in the head; the number of weight/style files loaded vs the weights used in CSS; `.ttf` / `.otf` / `.woff` files served to browsers.

```html
<link rel="preload" href="/fonts/inter-var.woff2" as="font" type="font/woff2" crossorigin />
```

```jsx
import { preload } from "react-dom";
preload("/fonts/inter-var.woff2", { as: "font", type: "font/woff2", crossOrigin: "anonymous" });
```

`crossorigin` is required even for same-origin fonts; without it the preload is fetched and then thrown away. Use `woff2`, subset to the alphabets you render, and prefer one variable font over several static weights. `next/font` handles preload and fallback metrics; if the project uses it, check that only the used weights and subsets are listed. Set `font-synthesis: none` so a weight that never loaded fails visibly instead of being faked (lane H).

## B3. Above-the-fold images are preloaded; below-the-fold are lazy — SHOULD FIX

The hero image should be in flight before the CSS finishes parsing. Everything under the fold should wait.

**Hunt for:** the LCP image (hero, first product shot) with no preload; gallery and list images without `loading="lazy"`; `loading="lazy"` on the hero (delays the one image that matters).

```html
<link rel="preload" as="image" href="/hero.webp" />
<img src="/gallery-12.webp" loading="lazy" decoding="async" width="600" height="400" alt="…" />
```

In Next.js, `priority` on the hero `<Image>`, nothing on the rest. Never `priority` on every image; that's the same as none.

## B4. Images are the size they're shown at — SHOULD FIX

A 4000px PNG decoded to show a 64px avatar costs bandwidth, decode time, and memory on exactly the devices that have the least of each.

**Hunt for:** raw `<img>` in a project that has an image component; `.png` / `.jpg` photos in `public/` over ~300KB; avatars and thumbnails pointing at the original upload; `unoptimized` on `next/image`.

Fix: the framework's image component with `sizes`, or `srcset` by hand; WebP or AVIF for photos; SVG for anything drawn. A raster image scaled *up* is the mirror-image finding: it ships blurry on every retina screen.

## B5. Only `transform` and `opacity` animate — BLOCKER on frequent interactions, SHOULD FIX otherwise

Animating `width`, `height`, `margin`, `padding`, `top`, `left`, or `border-width` triggers layout + paint + composite on every frame. `box-shadow`, `border-radius`, and `color` skip layout but repaint every frame. `transform`, `opacity`, `filter`, and `clip-path` composite only.

**Hunt for:** `transition:` and `@keyframes` declaring any of the properties above; `animate={{ height }}`, `animate={{ width }}`; sidebars that animate `width`; accordions that animate `height`.

| Instead of | Animate |
| --- | --- |
| `padding` / `width` / `height` to grow or shrink | `scale()` |
| `margin` / `top` / `left` to move | `translate()` |
| `visibility` / `display` swaps | `opacity` |
| `box-shadow` | opacity of a pseudo-element that holds the larger shadow |
| `border-radius` | `clip-path: inset(0 round 12px)` |

`height: auto` isn't animatable. Measure and animate the pixel value once, animate `grid-template-rows: 0fr → 1fr`, or clip an inner element. A sidebar that opens forty times a session and animates `width` is a blocker; a rarely opened settings accordion animating `height` is a should-fix.

## B6. No `transition: all` — SHOULD FIX

`all` makes every style change a candidate animation: properties you never meant to tween start tweening (including layout ones), and a theme switch animates every color at a different timing.

**Hunt for:** `transition: all`, `transition-all`, and Tailwind's bare `transition` class, which compiles to a broad property list in disguise.

```css
/* Bad */
.button { transition: all 200ms ease; }

/* Good */
.button { transition: background-color 150ms ease, transform 150ms var(--ease-out); }
```

```tsx
// Tailwind
<a className="transition-[translate,opacity] duration-150 ease-out">
```

`transition-transform` covers the whole transform family. Mixing in non-transform properties takes the arbitrary form: `transition-[opacity,filter,scale]`.

## B7. React doesn't re-render per frame — BLOCKER when found

`setState` inside a scroll, drag, or `requestAnimationFrame` handler re-renders the component and its subtree every 16ms. Write to the element instead.

**Hunt for:** `setX` / `setPosition` / `setProgress` / `setScrollY` inside `onScroll`, `onPointerMove`, `onMouseMove`, `requestAnimationFrame`; scroll-linked headers and progress bars.

```jsx
// Drops frames
setY(nextY);

// Smooth
ref.current.style.transform = `translateY(${nextY}px)`;
// or a Motion value: y.set(nextY)
```

## B8. Motion under load runs off the main thread — SHOULD FIX

Anything that animates while the page is busy (route transitions, tab switches during navigation, animations that fire alongside data loading or hydration) must not depend on `requestAnimationFrame`. Motion's `x` / `y` / `scale` shorthands run on the main thread and are not hardware-accelerated; the full `transform` string, CSS, or WAAPI is.

**Hunt for:** `motion.div` with `layoutId` or `animate={{ x }}` on navigation elements (active tab indicators, page transitions); animations triggered in the same tick as a fetch or a route change.

```jsx
<motion.div animate={{ x: 100 }} />                          // main thread, can jank
<motion.div animate={{ transform: "translateX(100px)" }} />  // hardware accelerated
```

For predetermined motion during navigation, move it to a CSS transition or animation.

## B9. CSS variables don't drive child transforms — SHOULD FIX

An animated CSS variable on a parent recalculates styles for every descendant on every frame. A drag that gets slower as the list grows is this.

**Hunt for:** `style.setProperty("--…")` or `style={{ "--x": … }}` updated per frame on a container whose children read the variable in `transform`.

Fix: set `transform` directly on the element that moves.

## B10. Blur stays under 20px — SHOULD FIX

`filter: blur()` cost explodes past ~20px, especially in Safari. 2–5px masks a crossfade; anything larger on an animating or full-screen element is a frame-rate problem.

**Hunt for:** `blur(` with a value over 20px on an animated element; `backdrop-filter: blur()` on a full-viewport overlay or on a sticky header over a long scrolling page; `blur-3xl` decorative glows.

## B11. `will-change` is targeted, not sprinkled — POLISH

Browsers already promote `transform` / `opacity` animations to their own layer. `will-change` everywhere costs GPU memory for nothing. Add it only to an element that visibly hitches on its first frame (Safari is the usual offender), only for compositor properties (`transform`, `opacity`, `filter`, `clip-path`), and keep that layer small.

**Hunt for:** `will-change` on more than a handful of selectors, on `*`, as `will-change: all`, or naming layout properties (`height`, `color`), where it's a no-op.

## B12. Long lists are virtualized — SHOULD FIX, BLOCKER when the list is unbounded

Rendering 2,000 rows renders 2,000 rows. Only what's visible should exist in the DOM. The dev database had forty rows; the first real customer imports ten thousand.

**Hunt for:** `.map()` over data with no cap, no pagination, and no virtualizer, in tables, feeds, logs, comboboxes, and pickers.

Use the project's existing virtualizer (`@tanstack/react-virtual`, Virtuoso). Adding a virtualizer to a table is bigger than a one-line fix: apply it when the list is simple, otherwise put it on the human list with `emil-performance` named. Never churn an existing dependency without being asked.

## B13. Off-screen work pauses — POLISH

Looping animations, video, canvases, and polling that keep running while scrolled out of view or in a background tab burn battery and frame budget for nothing.

**Hunt for:** `<video autoPlay loop>`, infinite keyframes, `<canvas>` render loops, and `setInterval` polling with no `IntersectionObserver` or `visibilitychange` anywhere near them.

```js
const observer = new IntersectionObserver(([entry]) => {
  entry.isIntersecting ? start() : pause();
});
observer.observe(element);
```

## B14. Theme switches don't animate every color — SHOULD FIX

Each component has its own transition duration, so a theme toggle animates every surface at a different speed and the page ripples. Disable transitions for the switch.

**Hunt for:** a theme toggle with no transition suppression; `next-themes` `ThemeProvider` without `disableTransitionOnChange`.

```jsx
<ThemeProvider disableTransitionOnChange>…</ThemeProvider>
```

Without `next-themes`:

```js
function setTheme(theme) {
  document.documentElement.classList.add("no-transitions");
  document.documentElement.setAttribute("data-theme", theme);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.documentElement.classList.remove("no-transitions");
    });
  });
}
```

```css
.no-transitions, .no-transitions * { transition: none !important; }
```

## B15. Content routes are static — SHOULD FIX

Blog posts, docs, changelogs, and pricing don't change per request. Fetching them at request time adds latency to every visit for content that could be built once and revalidated.

**Hunt for:** `fetch()` in server components / loaders for content routes with no caching; `dynamic = "force-dynamic"`; `cache: "no-store"` on CMS calls; `cookies()` or `headers()` read in a marketing layout (it opts the whole subtree out of static rendering).

```jsx
export const revalidate = 3600;
export async function generateStaticParams() { … }
```

An auth-aware CTA (lane I) is the usual reason a marketing layout went dynamic. Keep the page static and resolve the CTA on the client.

## B16. Search and resize handlers are debounced — SHOULD FIX

A request per keystroke hammers the server and creates race conditions where an older response overwrites a newer one. Debounce at ~300ms and discard stale responses.

**Hunt for:** `onChange` that calls `fetch` directly; `resize` / `scroll` listeners doing layout reads (`getBoundingClientRect`, `offsetHeight`) with no throttle; scroll listeners not marked `{ passive: true }`.

```js
useEffect(() => {
  const controller = new AbortController();
  const id = setTimeout(() => search(query, controller.signal), 300);
  return () => { clearTimeout(id); controller.abort(); };
}, [query]);
```

## B17. Heavy, rarely-seen UI is split out — SHOULD FIX

A chart library, a rich-text editor, a syntax highlighter, or a confetti effect imported at the top of the root layout is downloaded and parsed by every visitor, including the ones who never open the screen that uses it.

**Hunt for:** large dependencies imported in the root layout or a shared provider; modals and editors imported statically by the page that *might* open them; whole icon packs or `lodash` imported as a namespace.

```jsx
const Editor = dynamic(() => import("./editor"), { loading: () => <EditorSkeleton /> });
```

The `loading` fallback reserves the editor's box (lane E). If the project has a bundle analyzer configured, run it and report the three largest client chunks; if it doesn't, don't add one during a ship check.

## B18. Nothing dev-only ships in the bundle — see lane H

Devtools, inspectors, and debug panels cost bytes as well as embarrassment. The check lives in lane H (H9); count it once.

## B19. Verify on a mid-range phone — always

None of the above is proven on the dev machine. Profile with the DevTools Performance panel on a real phone over remote debugging, not with CPU throttling; throttling models a slow CPU, not a weak GPU or a small memory budget. Run Lighthouse against the *production build*, never the dev server. Report this as a device check, never as verified.
