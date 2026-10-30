# UI Review Standards

The cross-dimensional rule catalog for reviews. Each section names the building skill that owns the topic — defer there for the full treatment. This file exists so a finding can cite an exact threshold without loading five skills.

A finding is a place where the code violates one of these. Rank by the priority order in SKILL.md: broken interaction and accessibility failures first, polish last.

## Interaction correctness

- Every action a user can take with the mouse must also work with the keyboard. Tab reaches it, Enter/Space activates it, Escape dismisses overlays, arrow keys move within menus and lists.
- Focus is never trapped except intentionally (modals), and never lost — after closing an overlay, focus returns to the trigger.
- Nothing important lives behind hover alone. Hover is an enhancement; touch and keyboard users never see it.
- Disabled controls look disabled and are actually unfocusable-or-explained, not just visually greyed while still firing.

## Accessibility floor · see `emil-touch-and-accessibility`, `emil-color`

- Tap targets are ≥ 44×44px, or have an expanded hit area via padding or a pseudo-element.
- Every icon-only button has an accessible name (`aria-label` or visually-hidden text).
- Focus is always visible on keyboard navigation — never `outline: none` without a replacement.
- `prefers-reduced-motion` is honored: keep opacity/color, drop movement; disable decorative-only motion entirely.
- Hover-driven motion is gated behind `@media (hover: hover) and (pointer: fine)`.
- Text contrast passes its threshold — 4.5:1 for body, 3:1 for large text (WCAG), or the equivalent APCA level. Check both themes.
- Inputs are ≥ 16px font size so iOS Safari doesn't zoom on focus.

## Layout stability · see `emil-ui-polish`

- Dynamic content causes no layout shift. Reserve space for async data, images (width/height or aspect-ratio), and anything that appears after load.
- Changing numbers use `font-variant-numeric: tabular-nums` so they don't jitter the layout.
- No font-weight change on hover/active/selected — swapping weight reflows the text by a pixel. Use color, background, or a non-reflowing indicator instead.
- Skeletons and spinners occupy the same box the loaded content will.

## Motion · see `emil-animations`

- Every animation answers "why does this move?" — feedback, spatial continuity, state change, or preventing a jarring jump. "Looks cool" on a frequently-seen element is a block.
- Frequency-appropriate: keyboard-initiated and 100+/day actions get no animation; occasional gets standard; rare/first-run can have delight.
- Easing: entering/exiting → `ease-out` or a strong custom curve; on-screen moves → `ease-in-out`; hover/color → `ease`; `linear` only for constant motion. `ease-in` on UI is a block. Built-in curves are usually too weak — expect custom cubic-beziers.
- Duration under 300ms for UI unless justified by size, distance, or a very steep curve. Exits are shorter than entries. Duration and easing move together.
- Origin-correct: popovers/dropdowns/tooltips scale from their trigger via `transform-origin`, not center. Never animate from `scale(0)` — start at `scale(0.9–0.97)` + opacity. Press is `scale(0.97)`; hover scale is 1–2%.
- Animate `transform` and `opacity` only. Layout properties off the GPU are a performance finding.
- Rapidly-triggered or gesture motion (toasts, toggles, drawers, drags) is interruptible — transitions or springs that retarget, not `@keyframes` restarting from zero.

## Spacing & hierarchy · see `emil-design-foundations`

- Spacing follows a consistent scale, not arbitrary one-off values.
- Edges align. Optical alignment beats mathematical when they disagree.
- One clear primary action per view; secondary actions are visually quieter.
- Restraint: if everything is emphasized, nothing is. Fewer borders, fewer shadows, fewer competing focal points.

## Typography · see `emil-typography`

- A small, sane type scale; sizes come from it, not from arbitrary pixel values.
- Body line length caps around 60–75 characters; body leading 1.5–1.6, headings tighter (~1.1).
- Headings use `text-wrap: balance`; no single orphaned word on a heading's last line.
- `font-synthesis: none` so a missing weight or italic fails visibly instead of faking it.
- Negative tracking on large headings; slight positive tracking on small uppercase labels; body needs neither.

## Surface & depth · see `emil-surfaces`, `emil-color`

- Shadows and borders read as one elevation system, not a grab-bag of values.
- Nested rounded elements use concentric radii: inner radius = outer radius − padding.
- Dark mode is a designed surface set (lighter surfaces read as closer), not a mechanical inversion. Pure `#000`/`#fff` are rarely right.
- Gradients and fades are smooth — banding and hard stops are findings.

## Component API · see `emil-component-design` (only when a shared component is introduced)

- Composition over configuration: prefer children and compound parts over a wall of boolean props.
- Support both controlled and uncontrolled state when the component holds state.
- Props are named for intent, and the common case is a one-liner.
- The component is hard to misuse — no required props that are easy to forget, no footguns.

## Performance · see `emil-performance`

- Lists that can grow large are virtualized.
- Images are sized and lazy-loaded below the fold; nothing decodes a huge asset to show a thumbnail.
- No render storms — animation runs off refs/motion values, not state updates per frame; no CSS variable on a shared parent driving many children.
- Nothing expensive blocks first paint.

## Final polish · see `emil-ui-polish`

- Custom `::selection` where the default clashes.
- Custom scrollbars only inside small scoped elements, never the whole page.
- Empty states and error states are designed, not blank.
- `z-index` comes from a small fixed scale or `isolation: isolate`, never a random large number.
