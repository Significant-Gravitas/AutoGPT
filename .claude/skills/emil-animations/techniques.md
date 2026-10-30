# Animation Techniques Reference

Deeper material behind the SKILL.md rules: the easing curve library, spring configuration, interruptibility, performance, and debugging fixes.

## Easing Curve Library

Built-in CSS keywords (`ease-out`, `ease-in-out`) are usually too weak to feel intentional. Use custom curves, sorted weak → strong. Stronger curves feel snappier and more designed; start around cubic/quart and adjust by eye.

```css
/* ease-out — enters, exits, user-initiated UI */
--ease-out-quad:  cubic-bezier(0.25, 0.46, 0.45, 0.94);
--ease-out-cubic: cubic-bezier(0.215, 0.61, 0.355, 1);
--ease-out-quart: cubic-bezier(0.165, 0.84, 0.44, 1);
--ease-out-quint: cubic-bezier(0.23, 1, 0.32, 1);
--ease-out-expo:  cubic-bezier(0.19, 1, 0.22, 1);
--ease-out-circ:  cubic-bezier(0.075, 0.82, 0.165, 1);

/* ease-in-out — elements moving/morphing on screen */
--ease-in-out-quad:  cubic-bezier(0.455, 0.03, 0.515, 0.955);
--ease-in-out-cubic: cubic-bezier(0.645, 0.045, 0.355, 1);
--ease-in-out-quart: cubic-bezier(0.77, 0, 0.175, 1);
--ease-in-out-quint: cubic-bezier(0.86, 0, 0.07, 1);
--ease-in-out-expo:  cubic-bezier(1, 0, 0, 1);
--ease-in-out-circ:  cubic-bezier(0.785, 0.135, 0.15, 0.86);
```

Why ease-out works for enters: the fast start delivers immediate visual feedback — the element "jumps" toward its destination, then settles. Ease-in inverts that and back-loads the movement, which is why the same duration feels slower and sluggish.

`linear` is only for constant-rate motion: marquees, tickers, and time visualization (a hold-to-delete progress ring should be linear because it *is* time).

## Springs

Springs have no fixed duration — they simulate physics, which makes them feel alive and lets them preserve velocity when interrupted.

**When to use a spring instead of a curve:**

- Drag interactions with momentum (sheets, swipe-to-dismiss, sliders)
- Gestures the user can interrupt mid-motion
- Elements meant to feel physical/alive (Dynamic Island-style morphs)
- Playful, organic interfaces

**Configuration — prefer the duration + bounce API** (Apple's approach; far easier to reason about than mass/stiffness/damping):

```js
{ type: "spring", duration: 0.5, bounce: 0.2 }
// vs the physics API you rarely need:
{ type: "spring", mass: 1, stiffness: 100, damping: 10 }
```

**Bounce rules:** default to `bounce: 0` in product UI. Add bounce only for playful, physical interactions (drag-to-dismiss release, a toy-like toggle) and keep it subtle: 0.1–0.3, never more.

**Interruptibility is the killer feature:** when a gesture reverses mid-flight, a spring carries the current velocity into the new animation. A CSS keyframe animation restarts from zero and looks broken. If the interaction is gesture-driven, that alone justifies a spring.

## Interruptibility: Transitions vs. Keyframes

A transition interpolates from wherever the element currently is toward its newest target — interrupt it and it retargets from the live position, taking only as long as the remaining distance needs. A keyframe animation plays a fixed timeline from frame one; interrupt it and it snaps back to the start and replays. That's why interactive, reversible state (hover, open/close, toggles) belongs to transitions, while fire-and-forget sequences (entrances, loaders) belong to keyframes.

```css
/* Interruptible — hitting close mid-open reverses from the current position */
.side-panel { transform: translateX(100%); transition: transform 240ms ease-out; }
.side-panel[data-open] { transform: translateX(0); }

/* Not interruptible — closing mid-open replays or jumps */
.side-panel[data-open] { animation: panel-in 240ms ease-out forwards; }
```

Rule: anything the user can toggle gets a transition. Keyframes only for sequences that always run to completion.

## Performance

**The golden rule: animate only `transform` and `opacity`.** They composite on the GPU and skip layout and paint entirely.

Never animate:

- `width`, `height`, `padding`, `margin`, `top/left` — trigger layout for the whole subtree
- `filter: blur()` above 20px — expensive, especially Safari
- CSS custom properties consumed deep in a component tree — invalidates broadly

**Fix jitter with `will-change`:** elements can shift ~1px at the start/end of transform animations due to the CPU↔GPU rendering handoff. `will-change: transform` keeps the element on the GPU throughout. Apply it to elements that animate, not globally.

**React-specific:**

- Never drive per-frame animation through React state — a re-render per frame drops frames. Animate outside the render cycle: refs + direct style writes, WAAPI, or a library's animation loop.
- Motion: `animate={{ transform: "translateX(100px)" }}` (string transform) is hardware-accelerated; `animate={{ x: 100 }}` is more readable but runs on the JS thread. Prefer the accelerated form for elements that animate while the main thread is busy.

**CSS vs. JS engines:**

- CSS transitions/animations run off the main thread — smoother under load. Use for simple, predetermined state changes.
- JS engines (Motion, React Spring) run on `requestAnimationFrame` — required for dynamic values, springs, and gesture-driven motion.
- Pick the simplest tool that supports the interaction; don't pay for a JS animation loop to fade a tooltip.

## Reduced Motion

Every animated element needs its own reduced-motion handling — this is part of shipping the animation, not a follow-up.

```css
.modal { animation: fadeIn 200ms ease-out; }

@media (prefers-reduced-motion: reduce) {
  .modal { animation: none; }   /* transition: none for transitions; no !important */
}
```

- Disable **all** motion — no exceptions for opacity or color fades.
- Replace autoplaying video with a play button.
- Motion: `useReducedMotion()` and pass `initial={false}` / static values when it returns true.

```tsx
const reduce = useReducedMotion();
<motion.div initial={reduce ? false : { opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} />
```

## Debugging & Polish Fixes

| Symptom | Fix |
| --- | --- |
| Animation feels off, can't say why | Screen-record it and scrub frame by frame — details invisible at full speed show up |
| Shaky/jittery start or end | `will-change: transform` |
| Hover state flickers | The hover moved the element out from under the cursor — animate a child, keep the parent's hover area static |
| Transition between two states looks disconnected | Add a subtle `blur(2px)` during the transition to bridge the states (keep well under 20px) |
| Hover animations firing on phones | Wrap in `@media (hover: hover) and (pointer: fine)` |
| Everything transitions when theme toggles | Temporarily disable transitions during the theme swap |
| Small icon buttons hard to hit | Expand the hit area to a 44px minimum with an absolutely-positioned pseudo-element — don't scale the visual |
| Sequential tooltips feel sluggish | Delay + animate only the first; siblings open instantly (`data-instant`) |
| Spinner feels slow even though load time is fixed | Spin it faster — perceived speed is real UX |

### Hover flicker fix

```html
<div class="card"><div class="card-inner">…</div></div>
```

```css
.card-inner { transition: transform 200ms ease; }
.card:hover .card-inner { transform: translateY(-20%); } /* parent hover area never moves */
```

### 44px hit area

```css
.icon-button { position: relative; }
.icon-button::after {
  content: "";
  position: absolute;
  width: max(100%, 44px);
  height: max(100%, 44px);
  top: 50%;
  left: 50%;
  translate: -50% -50%;
}
```

## Final Calibration

Don't ship an animation in the same sitting you wrote it. Come back with fresh eyes — the best motion is refined over days, not hours. And use your own product daily: the animations that become annoying through repetition are the ones to remove. Details that go unnoticed are the goal; users should finish their task, not admire the transition.
