---
name: emil-touch-and-accessibility
description: "Rules for making interfaces work on touch devices and for keyboard and screen-reader users — tap targets, hover vs touch, focus management, aria labels, reduced motion, iOS Safari quirks. Use when building or reviewing any interactive UI (buttons, modals, tooltips, menus, video, timers) or fixing mobile behavior. Triggers on: touch, mobile, tap target, hit area, hover, hover state, iOS, Safari, iPhone, iPad, keyboard navigation, tab order, tabbing, focus, focus trap, aria, aria-label, accessibility, a11y, screen reader, prefers-reduced-motion, reduced motion, inert, touch-action, double-tap zoom, pinch zoom, playsinline, autoplay video, tooltip delay, submenu, dropdown, dialog, modal, Cmd, Ctrl, keyboard shortcut."
---

# Touch & Accessibility

A large share of users will experience your interface through a thumb, a keyboard, or a screen reader — never a mouse. UI that only works with a fine pointer and perfect vision is broken UI, no matter how polished it looks. The rules below are not enhancements to add later; they are the baseline. Hover should enhance, never enable: if functionality is only reachable via hover, it is unreachable on touch.

All CSS below is written as plain CSS for clarity. When implementing, match the project's existing styling system (Tailwind, CSS Modules, styled-components, vanilla-extract, etc.) rather than imposing a new one.

## Core Principles

### 1. Gate hover effects behind capability queries

Touch devices fire `:hover` on tap, causing stuck or false-positive hover states. Only apply hover styles on devices that actually support hovering:

```css
@media (hover: hover) and (pointer: fine) {
  .element:hover {
    transform: scale(1.05);
  }
}
```

Never rely on hover for the UI to function — hover reveals and embellishes, it must not be the only way to reach an action.

### 2. Tap targets are at least 44px

Visual size can stay small; the hit area cannot. Expand it with a pseudo-element:

```css
.icon-button {
  /* Visual size can be smaller */
  width: 24px;
  height: 24px;
  position: relative;
}

.icon-button::before {
  content: '';
  position: absolute;
  inset: -10px; /* 24px + 2*10px = 44px hit area */
}
```

Or with sizing directly:

```css
.small-button {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
}
```

### 3. Control `touch-action`

Prevent double-tap zoom on interactive controls:

```css
button, a, input {
  touch-action: manipulation;
}
```

For custom components that implement their own pan/zoom gestures (canvases, sliders, drawing surfaces), disable native gestures entirely so they don't fight yours:

```css
.custom-canvas {
  touch-action: none;
}
```

### 4. `muted` + `playsinline` for autoplaying video

Without both attributes, iOS either refuses to autoplay or opens fullscreen:

```html
<video autoplay muted playsinline loop>
  <source src="video.mp4" type="video/mp4" />
</video>
```

### 5. Every icon-only button gets an `aria-label`

```html
<button aria-label="Close dialog">
  <CloseIcon />
</button>

<button aria-label="Search">
  <SearchIcon />
</button>
```

Same for illustrations built out of divs/CSS — give them a role and label:

```jsx
<div
  role="img"
  aria-label="Abstract geometric pattern"
  className="decorative-illustration"
/>
```

### 6. Tab order only includes visible elements

Users must never tab into something they can't see. Hide inactive panels from the tab order with `visibility: hidden` or the `inert` attribute:

```css
.hidden-panel {
  visibility: hidden;
}
```

```jsx
<div inert={!isVisible}>...</div>
```

And keyboard focus must never land off-screen — scroll focused elements into view:

```jsx
function handleFocus(e) {
  e.target.scrollIntoView({
    behavior: 'smooth',
    block: 'nearest',
  });
}
```

### 7. Manage focus across modal boundaries

When a modal or dialog opens, move focus into it (first interactive element, or the dialog itself). When it closes, return focus to the element that triggered it. Never leave focus stranded on a removed node.

### 8. Respect `prefers-reduced-motion` everywhere

Every animation needs a reduced-motion path. For video, that means offering a play button instead of autoplaying:

```jsx
const prefersReducedMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)'
).matches;

<video
  autoPlay={!prefersReducedMotion}
  controls={prefersReducedMotion}
  muted
  playsinline
/>
```

### 9. Freeze time-limited actions when the tab is hidden

A countdown that keeps running in a background tab punishes the user for switching away. Pause on `visibilitychange`:

```js
let timeoutId;
let remainingTime;
let startTime;

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // Pause the timer
    clearTimeout(timeoutId);
    remainingTime -= Date.now() - startTime;
  } else {
    // Resume the timer
    startTime = Date.now();
    timeoutId = setTimeout(callback, remainingTime);
  }
});
```

### 10. Show shortcuts for the user's OS

Display `Cmd` on macOS, `Ctrl` elsewhere — and bind the matching key:

```js
const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
const modKey = isMac ? 'Cmd' : 'Ctrl';

// Display: "Save (Cmd+S)" on Mac, "Save (Ctrl+S)" on Windows
```

## Tooltips & Menus

**Delay before opening.** Tooltips need a delay so they don't fire on incidental mouse travel:

```css
.tooltip {
  transition-delay: 200ms;
}
```

**Warm state for sequential tooltips.** Once one tooltip has opened, moving to a neighboring trigger should open its tooltip instantly — no delay, no entrance animation. Track a shared "warm" flag; clear it ~300ms after the last tooltip closes:

```jsx
const [isWarm, setIsWarm] = useState(false);

// When any tooltip opens, set warm state
// Clear warm state after 300ms of no tooltip being open
```

**Submenu safe area.** Users move the cursor diagonally from a parent item toward its submenu. Without a safe zone, that diagonal path exits the trigger and the submenu closes. Carve out a triangular safe area with `clip-path`:

```css
.submenu-trigger::after {
  content: '';
  position: absolute;
  /* Creates a "safe zone" for cursor movement */
  clip-path: polygon(0 0, 100% 0, 100% 100%);
  /* Adjust based on submenu position */
}
```

## Feedback

Feedback components (errors, confirmations, status) must be visible directly on the page. Never hide feedback behind hover states or inside modals — if the user has to hunt for it, it failed.

## Review Checklist

- [ ] Hover styles wrapped in `@media (hover: hover) and (pointer: fine)`
- [ ] No functionality reachable only via hover
- [ ] Every tap target ≥ 44px (visual size may be smaller; hit area may not)
- [ ] `touch-action: manipulation` on buttons, links, inputs
- [ ] `touch-action: none` on custom gesture surfaces
- [ ] Autoplaying videos have `muted` and `playsinline`
- [ ] Icon-only buttons have `aria-label`; code-built illustrations have `role="img"` + `aria-label`
- [ ] Tab order skips hidden content (`visibility: hidden` or `inert`)
- [ ] Keyboard focus scrolls into view; modals move and restore focus
- [ ] Every animation and autoplaying video respects `prefers-reduced-motion`
- [ ] Timers pause on `visibilitychange`
- [ ] Shortcut hints show Cmd/Ctrl per OS

## Common Mistakes

- **Stuck hover states on mobile** — hover styles applied unconditionally, so a tap leaves the element "hovered" until the next tap elsewhere.
- **24px icon buttons with 24px hit areas** — visually fine, physically un-tappable.
- **Videos that open fullscreen on iPhone** — missing `playsinline`.
- **Tabbing into closed drawers and off-screen carousels** — hidden with `opacity: 0` or `transform` instead of `visibility: hidden`/`inert`, which keeps them focusable.
- **Focus lost when a dialog closes** — focus falls back to `<body>`, forcing keyboard users to re-tab from the top.
- **Tooltips with no delay** — flicker on across the page as the cursor passes; or with a delay on every tooltip in a row, making toolbars feel sluggish (use the warm state).
- **Submenus that close mid-flight** — no diagonal safe area.
