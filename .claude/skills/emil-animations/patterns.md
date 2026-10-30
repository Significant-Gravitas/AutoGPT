# UI Animation Patterns

Exact recipes for the common interface components. All values assume the defaults from SKILL.md: `ease-out` for enter/exit, exits faster than enters, transform/opacity only, and a `prefers-reduced-motion` path for everything. Adapt the syntax to the project's stack (CSS, Tailwind, Motion) — keep the values.

## Modal / Dialog

Enter: fade + slight scale-up from `0.95`–`0.97`, 200–250ms ease-out. Exit: same but faster (~150–180ms). Overlay fades with **identical duration and easing** — they move as one unit.

```css
.overlay {
  opacity: 0;
  transition: opacity 200ms cubic-bezier(0.32, 0.72, 0, 1);
}
.modal {
  opacity: 0;
  transform: scale(0.97);
  transition:
    opacity 200ms cubic-bezier(0.32, 0.72, 0, 1),
    transform 200ms cubic-bezier(0.32, 0.72, 0, 1);
}
.overlay[data-open] { opacity: 1; }
.modal[data-open] { opacity: 1; transform: scale(1); }
```

Avoid: sliding modals in from off-screen, scaling from 0, bounce on a dialog (it's an interruption, not a toy).

## Drawer / Sheet

Slides from its edge with a translate transition (interruptible — users close drawers mid-open). 250–300ms ease-out enter, faster exit. Backdrop fades with the same timing.

```css
.drawer {
  transform: translateX(100%);
  transition: transform 300ms cubic-bezier(0.32, 0.72, 0, 1);
}
.drawer[data-open] { transform: translateX(0); }
```

If the drawer is draggable (mobile sheet), use a spring so release velocity carries into the animation: `{ type: "spring", duration: 0.5, bounce: 0 }`. For drag-to-dismiss, trigger dismissal when velocity (`swipeAmount / timeTaken`) exceeds ~0.10, not only on distance thresholds.

## Popover / Dropdown Menu

Fade + scale from `0.95`, 150–200ms ease-out, **origin-aware**: scale from the trigger, not the center.

```css
.dropdown {
  transform-origin: var(--radix-dropdown-menu-content-transform-origin);
  /* Base UI: var(--transform-origin) */
  opacity: 0;
  transform: scale(0.95);
  transition: opacity 150ms ease-out, transform 150ms ease-out;
}
.dropdown[data-open] { opacity: 1; transform: scale(1); }
```

A small directional nudge (`translateY(-4px)` → `0` when opening downward) adds polish. Keyboard navigation *inside* the menu gets no animation — highlight moves instantly.

## Tooltip

Fast and subtle: fade + `scale(0.97)`, ~125ms ease-out, origin-aware. The critical detail is the **instant-group behavior**: the first tooltip gets a hover delay and animation; while one is open, sibling tooltips appear instantly with no delay and no animation.

```css
.tooltip {
  transform-origin: var(--transform-origin);
  transition: transform 125ms ease-out, opacity 125ms ease-out;
}
.tooltip[data-starting-style],
.tooltip[data-ending-style] { opacity: 0; transform: scale(0.97); }
.tooltip[data-instant] { transition-duration: 0ms; }
```

Radix UI and Base UI support this via the `data-instant` attribute.

## Toast

Enters with translate + fade from its screen edge (bottom edge → `translateY(16px)` → `0`), 200–250ms ease-out. Exit: fade + small translate toward the edge, faster. When stacking, move existing toasts with `transform` (never re-layout), same easing, and use a spring if toasts are swipeable-to-dismiss.

## Entrances: Split, Then Stagger

For page/section entrances, never move one monolithic wrapper. Carve the content into meaningful pieces — heading, supporting copy, call to action — and offset each by roughly 100ms. The softest layered enter pairs a fade with a short upward travel (`translateY(8px)` → 0) while a light `blur(2px)` melts away. Hero headlines can go finer-grained: per-word offsets of ~80ms.

```tsx
// Motion
const list = { show: { transition: { staggerChildren: 0.09 } } };
const piece = {
  hide: { opacity: 0, filter: "blur(2px)", y: 8 },
  show: { opacity: 1, filter: "blur(0px)", y: 0 },
};

<motion.section variants={list} initial="hide" animate="show">
  {sections.map((s) => (
    <motion.div key={s.id} variants={piece}>{s.content}</motion.div>
  ))}
</motion.section>
```

```css
/* CSS-only — set --n per child (0, 1, 2…) in markup */
.entrance > * {
  opacity: 0;
  translate: 0 8px;
  filter: blur(2px);
  animation: rise-in 350ms ease-out forwards;
  animation-delay: calc(var(--n, 0) * 90ms);
}
@keyframes rise-in {
  to { opacity: 1; translate: 0 0; filter: none; }
}
```

## Exit Animations

An exit deserves less emphasis than an enter — by the time something leaves, attention has already moved elsewhere.

- Nudge a fixed few pixels (`translateY(-8px)` or so), never the element's own height.
- A hint of direction tells the eye where the thing went; a pure fade loses that.
- Run it shorter: ~150ms out against ~250–300ms in.
- Don't cut straight to `display: none` — instant disappearance severs spatial context.

```tsx
<motion.div exit={{ opacity: 0, y: -8, filter: "blur(2px)",
  transition: { duration: 0.14, ease: "easeIn" } }} />
```

Reserve a full off-screen slide (`x: "-100%"`) for moments where the destination itself is the information — a swiped-away list card, a closing drawer.

## Contextual Icon Swaps

Icons that change with state (play → pause, like → liked, copy → check) deserve a real cross-fade, not a visibility toggle. The recipe: shrink deep, fade, add a whisper of blur, and drive it with a bounce-free spring.

- `scale`: `0.3` → `1` — a deep scale-down reads as a swap; a timid one (0.8, 0.9) reads as a glitch
- `opacity`: fades `0` → `1` alongside the scale
- `filter`: `blur(3px)` sharpening to `blur(0px)`
- spring: `{ type: "spring", duration: 0.35, bounce: 0 }` — keep bounce at zero, icons aren't toys

```tsx
<AnimatePresence mode="popLayout" initial={false}>
  <motion.span key={copied ? "check" : "copy"}
    initial={{ scale: 0.3, opacity: 0, filter: "blur(3px)" }}
    animate={{ scale: 1, opacity: 1, filter: "blur(0px)" }}
    exit={{ scale: 0.3, opacity: 0, filter: "blur(3px)" }}
    transition={{ type: "spring", duration: 0.35, bounce: 0 }}>
    {copied ? <CheckIcon /> : <CopyIcon />}
  </motion.span>
</AnimatePresence>
```

Without Motion: render **both icons at once**, layering one absolutely over the other, and swap classes to cross-fade — nothing ever unmounts, so the reverse direction animates just as well. A firm ease-out such as `cubic-bezier(0.25, 0.1, 0, 1)` over ~300ms stands in for the spring. The icon left in normal flow determines the component's size.

Good candidates: actions revealed on hover, state toggles, icons inside contextual toolbars, loading → success indicators. Poor candidates: permanent navigation icons, purely decorative marks, anything always on screen, text labels.

## Press Feedback (Scale on Press)

`scale(0.96)` on `:active`, animated with a ~150ms ease-out CSS transition so a release mid-press glides back instead of snapping. Going under `0.95` starts to look cartoonish.

```tsx
<button className="active:scale-[0.96] transition-[scale] duration-150 ease-out">
```

Not every button needs it — dense or high-frequency UI can skip it. Bake an opt-out into your Button component, e.g. a `still` prop:

```tsx
function Button({ still = false, className, ...rest }) {
  const classes = cn(
    "transition-[scale] duration-150 ease-out",
    !still && "active:not-disabled:scale-[0.96]",
    className,
  );
  return <button className={classes} {...rest} />;
}
```

## Don't Animate on Page Load

If a component is showing its resting state when the page first paints, it must not play an entrance — reserve enter animations for state changes the user causes afterward. In Motion, `initial={false}` on `AnimatePresence` handles this for icon swaps, toggles, tabs, and segmented controls.

Exception: components whose whole point is a first-run entrance (staggered hero, loading state) rely on their `initial` prop — there, `initial={false}` would wipe out the whole entrance. After applying it, hard-refresh the page and confirm nothing that should animate went static.

## AnimatePresence Grouping

When an exiting element sits in a group (list items, toasts, tags), use `mode="popLayout"` so siblings slide into place while the exit plays, instead of waiting for it.

## Looping Animations

Pause looping animations (spinners, shimmer, marquees) when off-screen — `IntersectionObserver` or `animation-play-state: paused` — to save CPU/battery.
