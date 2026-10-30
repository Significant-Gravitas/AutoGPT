---
name: emil-performance
description: "Rules for making UI fast — virtualization, transition specificity, GPU compositing, preloading, lazy loading, layout-shift prevention, and keeping animation out of React renders. Use when building or reviewing UI that renders long lists, animates, loads fonts/images/video, switches themes, or feels janky or slow. Triggers on: performance, slow, janky, lag, stutter, fps, frame drops, virtualization, virtual list, infinite scroll, long list, re-render, re-render storm, transition all, will-change, GPU, compositing, layout shift, CLS, layout thrash, reflow, preload, prefetch, lazy loading, loading=lazy, image optimization, video, font loading, FOUT, skeleton, perceived performance, IntersectionObserver, static generation, SSG, theme switch, blur filter, CSS variables, Framer Motion performance."
---

# Interface Performance

Fast UI is mostly about work you *don't* do: don't render DOM nodes nobody can see, don't make the browser watch properties that never change, don't animate what the GPU can't composite, don't fetch at request time what you could build at build time, and don't move layout after the user has started reading it. Most jank is self-inflicted — the fix is usually deleting or deferring work, not adding cleverness.

All CSS below is written as plain CSS for clarity. When implementing, match the project's existing styling system (Tailwind, CSS Modules, styled-components, etc.) rather than imposing a new one.

## Core Principles

### 1. Transition only what changes — never `transition: all`

With `transition: all` (or Tailwind's bare `transition` class, which compiles to `transition-property: all`), every style change becomes a candidate animation: the browser must track the entire style object, properties you never meant to tween (padding, shadow, colors) start tweening whenever they change, and the engine loses the optimizations a narrow, known property list allows.

```css
/* Do — name each property that actually animates */
.menu-trigger {
  transition-property: translate, opacity;
  transition-duration: 120ms;
  transition-timing-function: ease-out;
}

/* Don't — a shorthand that watches everything */
.menu-trigger {
  transition: all 120ms ease-out;
}
```

```tsx
// Do — the property list is explicit
<a className="transition-[translate,opacity] duration-100 ease-out">

// Don't — bare `transition` is `transition-property: all` in disguise
<a className="transition duration-100 ease-out">
```

Tailwind note: `transition-transform` expands to the whole transform family (`transform, translate, scale, rotate`), so reach for it when only transforms move. Mixing in non-transform properties? Use the arbitrary-value form, e.g. `transition-[opacity,filter,scale]`.

### 2. Animate only compositor-friendly properties

- Animate `transform` and `opacity`; avoid animating `height`, `width`, `padding`, `margin` (they trigger layout).
- `blur()` filters above 20px are expensive, especially in Safari. Keep blurs subtle and off frequently-animating elements.
- Pause looping animations when off-screen (principle 8).

### 3. Use `will-change` sparingly, and only where it helps

By default the browser waits until motion actually begins before lifting an element onto a dedicated compositor layer, and that just-in-time layer creation occasionally eats the opening frame — a tiny hitch right as movement starts. `will-change` requests the layer up front so the element is already composited when the animation fires. The hint only matters for properties the compositor owns; declaring it for layout-bound properties is a no-op.

```css
/* Sensible — the panel slides, so promote its transform ahead of time */
.slide-panel { will-change: transform; }

/* Sensible — a slide paired with a fade */
.slide-panel { will-change: transform, opacity; }

/* Wrong — `all` defeats the point of the hint */
.slide-panel { will-change: all; }

/* Useless — these run through layout and paint regardless */
.slide-panel { will-change: height, color; }
```

| Rendering path | Properties | Does `will-change` pay off? |
| --- | --- | --- |
| Compositor | `opacity`, `transform`, `filter` (blur, brightness), `clip-path` | Yes — pre-promotion smooths the first frame |
| Paint | `color`, `background`, `border` | No |
| Layout | `width`, `height`, `top`/`left`, `margin` | No |

Browsers have gotten good at managing layers themselves, so treat this as a targeted fix: reach for it after you've observed a first-frame hiccup (Safari is the usual offender), never as routine seasoning. Each promoted layer holds GPU memory, so a page peppered with `will-change` pays real cost for stutters it never had.

### 4. Virtualize long lists

Don't render hundreds of DOM nodes when only a handful are visible:

```jsx
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualList({ items }) {
  const parentRef = useRef(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
  });

  return (
    <div ref={parentRef} style={{ height: '400px', overflow: 'auto' }}>
      <div style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: virtualItem.start,
              height: virtualItem.size,
            }}
          >
            {items[virtualItem.index]}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 5. Keep animation out of React's render cycle

State updates per frame are a re-render storm. Drive frame-by-frame values through refs and direct DOM writes:

```jsx
// Bad — causes re-render on every frame
const [position, setPosition] = useState(0);

// Good — refs + direct DOM manipulation
const elementRef = useRef(null);

useEffect(() => {
  let frame;
  function animate() {
    elementRef.current.style.transform = `translateX(${position}px)`;
    frame = requestAnimationFrame(animate);
  }
  frame = requestAnimationFrame(animate);
  return () => cancelAnimationFrame(frame);
}, []);
```

Framer Motion: the transform string form is hardware accelerated, the shorthand is not:

```jsx
// Hardware accelerated (uses transform string)
<motion.div animate={{ transform: "translateX(100px)" }} />

// NOT hardware accelerated (more readable but slower)
<motion.div animate={{ x: 100 }} />
```

Also avoid animating CSS variables in deep component trees — every variable change triggers style recalculation for all descendants.

### 6. Zero layout shift from dynamic content

- Hardcode dimensions on images and videos.
- Reserve space for async content with skeletons.
- Use `font-variant-numeric: tabular-nums` on numbers that change, so digits don't reflow.
- Never change font weight on hover.
- Preload fonts so text doesn't reflow when they land:

```jsx
import { preload } from 'react-dom';

preload('/fonts/inter-var.woff2', {
  as: 'font',
  type: 'font/woff2',
  crossOrigin: 'anonymous',
});
// Plain HTML equivalent:
// <link rel="preload" href="/fonts/inter.woff2" as="font" type="font/woff2" crossorigin />
```

### 7. Preload critical, lazy-load the rest

Preload above-the-fold images:

```html
<link rel="preload" as="image" href="/hero.webp" />
```
Below-the-fold images get `loading="lazy"` so they don't compete with critical resources.

### 8. Stop work that's off-screen

Pause looping animations, video, and any resource-intensive operation when the element isn't visible:

```js
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      startAnimation();
    } else {
      pauseAnimation();
    }
  });
});

observer.observe(element);
```

### 9. Generate static content at build time

Blog posts, changelogs, docs — anything that doesn't change per request should never be fetched at request time:

```jsx
// Next.js example
export async function getStaticProps() {
  const posts = await fetchPosts();
  return {
    props: { posts },
    revalidate: 3600, // Revalidate hourly
  };
}
```

### 10. Theme switches must not animate

Flipping themes with transitions enabled makes every color on the page tween independently. Disable transitions for the swap, re-enable after paint:

```js
function setTheme(theme) {
  document.documentElement.classList.add('no-transitions');
  document.documentElement.setAttribute('data-theme', theme);
  // Re-enable after paint (double rAF)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.documentElement.classList.remove('no-transitions');
    });
  });
}
```

```css
.no-transitions,
.no-transitions * { transition: none !important; }
```

## Review Checklist

- [ ] No `transition: all` or bare Tailwind `transition` anywhere
- [ ] Animations touch only `transform`, `opacity`, `filter`, `clip-path`
- [ ] No blur filters above 20px on animating elements
- [ ] `will-change` reserved for transform/opacity elements that showed first-frame stutter
- [ ] Lists over ~50 items are virtualized
- [ ] No per-frame `useState`; frame updates go through refs
- [ ] Images/videos have explicit dimensions; async content has skeletons
- [ ] Changing numbers use `tabular-nums`; hover never changes font weight
- [ ] Fonts and hero images preloaded; below-fold images lazy
- [ ] Off-screen loops paused via IntersectionObserver
- [ ] Static content (blog, docs, changelog) generated at build time
- [ ] Theme switching disables transitions during the swap

## Common Mistakes

- **Tailwind's `transition` class everywhere** — it's `transition-property: all` in disguise; dark-mode toggles and layout changes animate by accident.
- **Animating `height` to expand panels** — triggers layout every frame; animate `transform: scaleY()` or use a clip, or measure and animate once.
- **`will-change` as a lucky charm** — added to every card "for performance," costing layers and memory while fixing nothing.
- **Rendering a 2,000-row table and wondering why scroll is choppy** — virtualize.
- **`useState` driven drag/scroll handlers** — the whole tree re-renders at pointer speed.
- **Fonts loaded without preload** — visible text reflow (FOUT) on every cold load.
- **Videos playing in background tabs and below the fold** — wasted CPU and battery; pause them.
