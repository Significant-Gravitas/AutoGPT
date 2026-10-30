# Lane F — Motion

The ship-check subset of motion: the mistakes that read as *broken* or *slow* to a user, not the ones that read as unrefined to a designer. This lane doesn't re-time the app or pick nicer curves. It catches what's objectively wrong and hands taste to `emil-animations` after launch. If you need to see what an animation is doing frame by frame before judging it, that's `emil-ask-lapse`.

Layout-property animation and `transition: all` are performance checks (B5, B6). Reduced motion is an accessibility check (A8). Count each once.

Product UI and marketing pages have different bars. Apply F1–F8 to product routes strictly; marketing routes get more room on duration and are swept by lane I for scroll theatrics.

## F1. High-frequency and keyboard actions don't animate — BLOCKER when it delays input, SHOULD FIX otherwise

An animation the user sees a hundred times a day is a tax they pay a hundred times a day. Keyboard users are moving fast; animation makes the UI feel disconnected from their input. Raycast doesn't animate its menu for this reason.

**Hunt for:** enter animations on the command palette; animated highlight movement on arrow-key navigation; transitions on tab switches in a tool used all day; anything animated in response to a keyboard shortcut.

Walk it per animation:

```
How often does the user trigger it?
├── 100+/day, or from the keyboard → no animation
├── Occasionally (modal, dropdown, drawer) → standard, under 300ms
└── Rarely or once (onboarding, success) → may be more elaborate
```

Removing an animation changes something the user may have chosen, so ask before deleting one that isn't plainly a mistake. Cutting a command palette's 300ms entrance to 0 is a fix; removing a success celebration is a decision.

## F2. Nothing enters with `ease-in` — SHOULD FIX

`ease-in` starts slow, which delays the movement at the exact moment the user is watching for a response to their click. A dropdown with `ease-in` feels slower than the same dropdown with `ease-out` at the same duration.

**Hunt for:** `ease-in` (not `ease-in-out`) in `transition`, `animation`, and Motion `ease` props; Tailwind `ease-in`.

```
Entering or exiting the screen?     → ease-out
Already on screen, moving/resizing? → ease-in-out
Hover, color, shadow?               → ease, ~150ms
Constant motion (spinner, marquee)? → linear
ease-in alone                       → only as the exit half of an enter/exit pair
```

Use the project's easing tokens when they exist (Hard Rule 7). When they don't, a strong default for entrances is `cubic-bezier(0.32, 0.72, 0, 1)`.

## F3. UI animation stays under 300ms — SHOULD FIX

If it feels slow, it is slow. Past 300ms the user is waiting for the interface to finish performing before they can act.

**Hunt for:** durations of 400ms and up (`duration-500`, `duration: 0.5`, `0.6s`) on product UI; `transition` with a long `delay`.

| What | Duration |
| --- | --- |
| Hover, press, color and shadow changes | 100–150ms |
| Tooltips, dropdowns, popovers | 150–250ms |
| Modals, drawers, toasts | 200–300ms |
| Page-level transitions | 300–400ms |

Exits run 20–30% faster than their entrance; the user has already decided to leave. A full-screen sheet or a very steep curve can justify more. A 500ms fade on a dropdown can't.

## F4. Nothing enters from `scale(0)` or from off the planet — SHOULD FIX

An element scaling up from nothing reads as a glitch; real objects always have a visible shape. The same goes for a toast that travels the full viewport height to arrive.

**Hunt for:** `scale(0)`, `scale: 0`, `initial={{ scale: 0 }}`; `translateY(100vh)` / `translateY(100%)` on small elements.

```css
/* Bad */
@keyframes pop { from { transform: scale(0); } }

/* Good */
@keyframes pop { from { transform: scale(0.95); opacity: 0; } }
```

`0.95` for panels, `0.97` for small elements. Travel distances stay small (8–16px) unless the element really is coming from off-screen, like a drawer.

## F5. Popovers scale from their trigger — SHOULD FIX

A dropdown that grows from its own center looks detached from the button that opened it. Anchored surfaces scale from the trigger side. Viewport-centered modals with no anchored trigger are the exception and stay centered.

**Hunt for:** popovers, dropdowns, tooltips, and context menus with a scale animation and no `transform-origin`.

```css
.popover {
  transform-origin: var(--radix-popper-transform-origin, var(--transform-origin, top));
}
```

Radix and Base UI expose the origin as a CSS variable; use it.

## F6. Toggled UI is interruptible — SHOULD FIX

`@keyframes` restart from zero when retriggered. A drawer closed while it's still opening jumps to fully open and then plays the close. A toast stack that re-runs its entrance whenever a sibling is added flickers. Transitions and springs retarget from wherever the element currently is.

**Hunt for:** `@keyframes` / `animation:` on anything with an open/closed toggle (drawers, accordions, toggles, tabs, toasts); gesture-driven motion (drag-to-dismiss, sheets) using a fixed-duration tween instead of a spring.

```css
.drawer { transform: translateX(-100%); transition: transform 250ms cubic-bezier(0.32, 0.72, 0, 1); }
.drawer[data-open] { transform: translateX(0); }
```

Reserve keyframes for one-shots that always run to completion: loaders, a first-run entrance. Put on the device list: "open and close each drawer, menu, and accordion as fast as you can and watch for jumps".

## F7. Paired elements share timing — SHOULD FIX

A modal at 300ms over a backdrop at 150ms reads as two unrelated things happening near each other. Modal + overlay, drawer + backdrop, tooltip + arrow: identical duration and easing.

**Hunt for:** overlay and content components with different `duration` / `transition` values in the same dialog or drawer.

## F8. Exits exist — SHOULD FIX

An element that animates in and then vanishes instantly severs the spatial context the entrance built. In React the cause is usually conditional rendering with no presence wrapper: the node unmounts before any exit can play.

**Hunt for:** `{open && <motion.div initial animate>}` with an `exit` prop but no `AnimatePresence` ancestor (the `exit` is silently dead); CSS enter keyframes with `display: none` on close.

Exits are quieter than entrances: opacity to 0 with a small nudge, ~150ms.

## F9. Resting elements don't animate on page load — SHOULD FIX

A tab indicator that slides into place on first render, an icon that plays its swap animation on mount, a sidebar that animates open on every navigation. Nothing the user didn't cause should move when the page appears.

**Hunt for:** `AnimatePresence` around toggles, tabs, and icon swaps without `initial={false}`; `layoutId` indicators that animate from `0,0` on mount; enter animations on elements that are present at first paint.

```jsx
<AnimatePresence initial={false}>…</AnimatePresence>
```

Not on a real first-run entrance like a staggered hero, where it would skip the whole thing.

## F10. Hover motion doesn't make the target run away — SHOULD FIX

An element that translates or scales on hover moves its own hit area. At the edge, the cursor falls off, hover ends, the element returns, hover starts again: a flicker loop.

**Hunt for:** `:hover { transform: translateY(…) }` or a hover `scale` above ~1.02 applied to the hovered element itself.

Fix: keep the hovered element still and animate a child, so the hit area never moves. Hover motion is also gated behind `(hover: hover)` (C2), and a hover scale on something that isn't clickable is a false affordance; remove it.

## F11. Sequential tooltips don't each wait — POLISH

A tooltip delay (~200ms) stops tooltips firing on incidental mouse travel. But once one is open, moving along a toolbar should open the next instantly, with no delay and no entrance animation. Most primitives have this built in (Radix `Tooltip.Provider` with `skipDelayDuration`, Base UI's tooltip provider); the finding is a provider that's missing or mounted per tooltip instead of once.

## F12. Staggers are short — POLISH

Eight items entering at once read as a flash; staggered at 30–50ms they read as arrival. A 150ms stagger across twelve items makes the user wait two seconds for a list. Cap the total: stagger × count stays under ~400ms.

## F13. Watch it on a slow device, slowed down — always

Motion correctness can't be proven from code. Put on the device list: "open each animated surface on a mid-range phone and watch the first frame", "in DevTools → Animations, slow playback to 10% and watch each enter/exit for a jump at the start or the end", "with Lapse installed, record a take of the drawer opening and check for dropped frames". Report these as checks to run, never as verified.
