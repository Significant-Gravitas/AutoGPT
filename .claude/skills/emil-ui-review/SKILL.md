---
name: emil-ui-review
description: "Review interface code the way a design engineer would — catch the spacing, motion, accessibility, and polish mistakes AI leaves behind. Use when reviewing a UI diff, a generated component, or a pull request that touches the interface, when someone asks \"does this look right / feel right\", or after an agent builds UI and you want it held to a craft bar before it ships. Triggers on: review, code review, UI review, design review, PR review, critique, \"does this look right\", \"is this good\", \"feels off\", \"catch mistakes\", AI-generated UI, generated component, polish pass, quality bar, ship it, sign off, nitpick, design QA."
---

# UI Review

A specialized review skill. It does one thing: judge interface code against a high craft bar and report what's wrong, ranked by how much it hurts. It doesn't build features, fix unrelated bugs, or review backend logic. Point it at a diff, a component, or a page.

This is the reviewing counterpart to the building skills. Where `emil-typography`, `emil-color`, `emil-surfaces`, `emil-ui-polish`, `emil-forms-and-inputs`, `emil-touch-and-accessibility`, `emil-performance`, `emil-animations`, and `emil-component-design` tell an agent how to *build* an interface, this skill tells it how to *catch* the interface that was built wrong. When a finding needs a precise value or a deeper rule, defer to the matching skill — this one owns the review *method* and the cross-dimensional bar, not every detail.

## Operating posture

Review like a senior design engineer with a low tolerance for "it works." Code that renders but feels off — inconsistent spacing, a sluggish transition, a tap target you'll miss on a phone, text that shifts as a number ticks — is a regression, not a pass. Default to flagging. Approval is earned, not assumed.

This matters most on AI-generated UI, which is the common case now. Models produce interfaces that run on the first try and look plausible in a screenshot, then fall apart under a real cursor, a real keyboard, a slow network, or a second look at the spacing. Your job is to be the second look. Assume the happy path was tested and the details were not.

## What to review, in priority order

Read the whole diff first, then judge it against these dimensions. The order is the triage order: a broken interaction outranks a soft shadow.

1. **Correctness of interaction** — does every control actually do its job with mouse, keyboard, and touch? Focus reachable, Enter/Escape wired, no dead hover-only affordances.
2. **Accessibility floor** — 44×44px tap targets, icon buttons labeled, focus visible, `prefers-reduced-motion` honored, contrast that passes. This is a floor, not a nice-to-have; a violation is a finding. See `emil-touch-and-accessibility`, `emil-color`.
3. **Layout stability** — no layout shift from dynamic content, tabular numbers on changing figures, reserved space for async content, no weight change on hover/active. See `emil-ui-polish`.
4. **Motion** — justified, frequency-appropriate, strong easing, sub-300ms, GPU-only properties, interruptible where it needs to be. See `emil-animations`.
5. **Spacing & hierarchy** — consistent scale, aligned edges, clear primary action, restraint. See `emil-design-foundations`.
6. **Type** — sane scale, comfortable line length and leading, no orphaned words in headings, no fake weights. See `emil-typography`.
7. **Surface & depth** — shadows and borders that read as one system, concentric radii, dark-mode surfaces that aren't just inverted. See `emil-surfaces`, `emil-color`.
8. **Component API** — is the thing being added reusable without forking, or is it a pile of boolean props? Only when the diff introduces a shared component. See `emil-component-design`.
9. **Performance** — big lists virtualized, images sized, no render storms, nothing blocking paint. See `emil-performance`.
10. **Final polish** — selection, scrollbars, empty states, loading states, the details in `emil-ui-polish`.

## Escalation triggers — flag on sight

These are almost always wrong. You don't need to deliberate:

- `transition: all` — unbounded property animation.
- Hover-only access to a core action, with no keyboard or touch equivalent.
- An icon-only button with no accessible label.
- Animating `width`/`height`/`margin`/`top`/`left` instead of `transform`/`opacity`.
- `ease-in` on a UI interaction, or a built-in named easing on a deliberate animation (usually too weak).
- Animation on a keyboard shortcut, command-palette toggle, or 100+/day action.
- A dynamic number or async value with no reserved space or tabular numbers — it will shift the layout.
- `<input>` with a font size under 16px — iOS zooms on focus.
- `z-index: 9999` or any random large z-index instead of a scale.
- A raster image scaled up, or an untuned SVG dropped in at the wrong size.
- Font weight that changes on hover or when active — the text reflows by a pixel.
- A list that can grow unbounded rendered without virtualization.
- Missing `prefers-reduced-motion` handling on anything that moves.

## Output format

Lead with the verdict, then the findings. Don't bury the one thing that matters under ten nitpicks.

- **Verdict** — one line: **Blocked**, **Approve with changes**, or **Ship it**. Blocked means at least one finding breaks interaction or the accessibility floor.
- **Blocking** — findings that must be fixed before it ships. Each: what's wrong, where (`file:line`), why it hurts, and the fix.
- **Should fix** — real craft problems that aren't blockers. Same shape.
- **Nits** — small polish, clearly labeled as optional.

Be specific and cite the exact value. "The drawer transition is too slow" is weak; "the drawer uses `500ms ease-in-out`; at that curve and distance it should be ~250ms with a steeper curve like `cubic-bezier(0.32, 0.72, 0, 1)` — see the `emil-animations` skill" is a review. When a fix has a precise rule in one of the building skills, name the skill so the fix is grounded, not invented.

For the full cross-dimensional rule catalog, see [STANDARDS.md](STANDARDS.md). Load it whenever a finding needs an exact threshold.

## What earns approval

A diff ships when every interaction works across input methods, the accessibility floor is met, nothing shifts the layout, motion is justified and performant, and the spacing, type, and surfaces read as deliberate. Approval is the absence of findings, not the absence of effort. If you can't find anything wrong, say so plainly — a fast, clean "Ship it" is a real outcome, and manufacturing nitpicks to look thorough wastes the reader's trust.
