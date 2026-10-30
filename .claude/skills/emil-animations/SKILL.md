---
name: emil-animations
description: "Animation for product interfaces — when to animate (and when not to), easing, duration, springs, enter/exit transitions, and performance. Use when adding or reviewing motion in UI components (modals, popovers, drawers, toasts, dropdowns, tooltips, buttons), choosing an easing curve or duration, fixing animations that feel slow, janky, or wrong, or implementing hover/press feedback and reduced-motion support. Triggers on: animation, transition, easing, ease-out, ease-in-out, cubic-bezier, spring, bounce, duration, timing, keyframes, transform, opacity, fade, slide, scale, stagger, hover, press, active state, microinteraction, modal, dialog, drawer, sheet, popover, dropdown, tooltip, toast, Framer Motion, motion/react, AnimatePresence, WAAPI, will-change, GPU, janky, choppy, feels slow, feels off, make it smooth, prefers-reduced-motion."
---

# Interface Animations

Animation in a product interface is feedback, not decoration. The best UI animations go unnoticed — they make the interface feel fast, connected, and physical, then get out of the way. Default to less: a fast, well-eased 200ms transition beats an elaborate 500ms sequence every time. When in doubt, don't animate. When you do animate, be exact — easing and duration are decisions, not defaults you inherit from a snippet.

**Match the project's stack.** Before writing any animation code, check what the project already uses: plain CSS transitions/keyframes, Tailwind, Motion (`motion`/`framer-motion` in package.json), React Spring, or WAAPI. Write animations in that system. Never add an animation library for something CSS transitions can do, and never hand-roll spring physics when Motion is already installed.

## Quick Reference

| Topic | Where |
| --- | --- |
| Modals, drawers, popovers, dropdowns, toasts, tooltips — exact recipes | [patterns.md](patterns.md) |
| Enter/exit + stagger, icon swaps, press feedback, page-load rules | [patterns.md](patterns.md) |
| Springs, performance, interruptibility, easing curve library, debugging | [techniques.md](techniques.md) |

## Decision Framework 1: Should this animate at all?

Ask how often the user will trigger it:

- **100+ times a day** (command menu, keyboard navigation, tab switching in a power tool) → **No animation.** Speed beats smoothness. Raycast doesn't animate its menu for exactly this reason.
- **Keyboard-initiated action** (arrow-key list navigation, shortcuts, focus movement) → **Never animate.** Keyboard users are moving fast; animation makes the UI feel disconnected from their input.
- **Occasional interaction** (opening a modal, a dropdown, a drawer) → Standard animation, under 300ms.
- **Rare or first-time moment** (onboarding, empty state, success celebration) → Can be more elaborate.
- **Marketing page vs. product**: marketing pages may use longer, showier motion; product UI must stay fast and purposeful.

Animate when it adds information: spatial continuity (where did this come from?), state feedback (did my click register?), or connection between cause and effect. If the animation answers no question, cut it.

## Decision Framework 2: Which easing?

1. **Element entering or exiting the screen** (modal, dropdown, toast, tooltip) → `ease-out`. The fast start reads as instant response to the user's action.
2. **Element already on screen moving or resizing** (reordering, layout shift, slider) → `ease-in-out`. Natural accelerate-then-brake motion.
3. **Hover / color / shadow change** → `ease` (or a soft ease-out), 150ms.
4. **Constant-speed motion only** (marquee, spinner, progress-as-time) → `linear`.
5. **`ease-in` alone → almost never.** The slow start delays feedback and feels sluggish. Its only legitimate use is as the exit half of a pairing (subtle exit with `ease-in` while enter uses `ease-out`).

Built-in CSS keywords are weak. Prefer custom curves — a good default for enters is `cubic-bezier(0.32, 0.72, 0, 1)`; the full weak-to-strong curve library is in [techniques.md](techniques.md).

Use a **spring** instead of a curve when motion is gesture-driven or interruptible (drag-to-dismiss, sheets, anything that follows a finger). Springs preserve velocity when interrupted; CSS keyframes restart from zero. Default spring: `{ type: "spring", duration: 0.5, bounce: 0 }` — add bounce (0.1–0.3, never more) only for playful, physical interactions.

## Decision Framework 3: What duration?

| What | Duration |
| --- | --- |
| Hover, press, color/shadow changes | 100–150ms |
| Tooltips, dropdowns, popovers | 150–250ms |
| Modals, drawers, toasts | 200–300ms |
| Page-level transitions | 300–400ms |

Rules of thumb:

- **UI animations stay under 300ms.** If it feels slow, it is slow — cut duration before anything else.
- **Exits run ~20–30% faster than enters** (e.g. enter 250ms, exit 180ms). The user has already decided to leave; don't make them wait.
- **Bigger elements and longer travel get more time**; a full-screen sheet needs more than a tooltip.
- Perceived speed matters as much as real speed — a snappy 180ms dropdown makes the whole app feel faster.

## Core Principles

1. **Animate only `transform` and `opacity`.** They run on the GPU and skip layout/paint. Never animate `width`, `height`, `margin`, `padding`, `top/left`, or box-shadow spread directly — animate `transform: scale/translate` and cross-fade instead. Keep `filter: blur()` under 20px (expensive, worst on Safari).

2. **Interactive state gets CSS transitions; one-shots get keyframes.** Transitions retarget smoothly when the user reverses mid-animation (open → close while still opening); keyframes restart and look broken. Reserve `@keyframes` for enter sequences and loaders that always run to completion.

   ```css
   .drawer { transform: translateX(-100%); transition: transform 250ms cubic-bezier(0.32, 0.72, 0, 1); }
   .drawer[data-open] { transform: translateX(0); }
   ```

3. **Never enter from `scale(0)`.** Elements should always have a visible shape — start at `scale(0.95)` (or `0.97` for small elements) with `opacity: 0`, and settle to `scale(1)`, `opacity: 1`.

4. **Make scaling origin-aware.** Popovers, dropdowns, and tooltips must scale from their trigger, not from center. Set `transform-origin` to the trigger side — Radix and Base UI expose it as a CSS variable:

   ```css
   .popover { transform-origin: var(--radix-popper-transform-origin, var(--transform-origin, top)); }
   ```

5. **Paired elements share timing.** Modal + overlay, drawer + backdrop, tooltip + arrow — if they move as a unit, give them identical duration and easing. Mismatched timing reads as broken.

6. **Press feedback: `scale(0.96)` on `:active`.** Use a CSS transition (150ms, ease-out) so releasing mid-press smoothly returns. Never go below `0.95` — it looks exaggerated. Not every button needs it; skip it on high-frequency controls.

   ```css
   .pressable { transition: scale 150ms ease-out; }
   .pressable:active { scale: 0.96; }
   ```

7. **Exits are quieter than enters.** The user's attention is already on the next thing. Fade out with a small directional nudge (`translateY(-12px)`, opacity 0, ~150ms ease-in) — not a dramatic full-height slide, and not an instant `display: none` that severs spatial context.

8. **Don't animate initial page load state.** Anything that's simply sitting in its resting position when the page appears has no reason to play an entrance — save the motion for state changes the user causes afterward. In Motion, set `initial={false}` on the `AnimatePresence` wrapper for toggles/tabs/icon swaps (but not on genuinely first-run entrances like a staggered hero, where it would skip the whole thing).

9. **Respect `prefers-reduced-motion` — always.** Every animation you add gets a reduced-motion path. Disable all of it, including opacity fades; no exceptions:

   ```css
   @media (prefers-reduced-motion: reduce) {
     .modal { animation: none; transition: none; }
   }
   ```

   In Motion, gate `initial`/`animate` values with `useReducedMotion()`.

10. **Gate hover animations to real hover devices.** Touch devices fire hover on tap. Wrap hover motion in `@media (hover: hover) and (pointer: fine)` (Tailwind v4's `hover:` does this automatically).

11. **Disable transitions during theme switches.** Toggling dark mode should not fire every color transition on the page — temporarily disable transitions when swapping themes.

## Common Mistakes

| Smell | Fix |
| --- | --- |
| `transition: all 400ms ease-in` | Name the properties, 200ms, `ease-out` |
| Enter from `scale(0)` or `translateY(100vh)` | `scale(0.95)` + `opacity: 0`, small travel |
| Animating `height`/`width`/`margin` | `transform` + `opacity`; grid-rows or clip tricks for expansion |
| Popover scales from center | `transform-origin` from the trigger |
| Modal and overlay use different durations | Identical duration + easing for paired elements |
| Keyframe animation on an open/close toggle | CSS transition (interruptible) |
| Animating arrow-key navigation / keyboard actions | No animation on keyboard-initiated changes |
| Exit animation longer or louder than enter | Exit ~20–30% faster, subtler |
| Element flickers on hover | Animate a child element so the hover area stays still |
| Shaky 1px jitter at animation start/end | `will-change: transform` on the animated element |
| Sequential tooltips each wait for a delay | First tooltip delays; siblings open instantly |
| No reduced-motion handling | Add the media query / `useReducedMotion()` the moment you add motion |
| New dependency for a simple fade | Use the project's existing stack |

## Review checklist

When reviewing existing animation code, check in order: (1) should this animate at all (frequency)? (2) transform/opacity only? (3) easing matches the enter/move/hover rule? (4) duration under 300ms with faster exit? (5) interruptible? (6) origin-aware? (7) reduced-motion path exists? Report findings as a Before/After markdown table.
