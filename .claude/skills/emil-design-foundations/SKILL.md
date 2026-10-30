---
name: emil-design-foundations
description: "Core visual design judgment — hierarchy, spacing, typography, color, layout, information architecture, interaction states, and UI copy, expressed as paired right/wrong decisions. Use when building, styling, or reviewing any interface; when choosing type, colors, borders, shadows, spacing, icons, empty states, error messages, or button hierarchy; or when a UI \"looks off\" and needs a design audit. Triggers on: design, UI, visual hierarchy, spacing, padding, margin, gap, whitespace, typography, font, kerning, tracking, letter-spacing, line-height, line length, color, palette, dark mode, contrast, WCAG, border, shadow, border-radius, icon, iconography, illustration, layout, grid, breakpoint, alignment, empty state, error message, microcopy, copywriting, button, tooltip, popover, badge, tag, toast, form, input, focus ring, accessibility, a11y, information architecture, navigation, tabs, onboarding, looks off, feels cheap, polish, design review, design critique."
---

# Design Foundations

Taste is trained, not innate. It's the practiced ability to see past "it works" to "it's right" — and you build it by studying great interfaces, asking why they feel good, and sweating decisions users will never consciously notice. That's the core bet of this skill: most details are invisible individually, but they compound. A half-pixel-snapped icon, a tabular-nums price column, a border that recedes instead of sitting on top — a thousand barely audible voices all singing in tune, as Paul Graham put it. In a world where everyone's software is functional, this aggregate of invisible correctness is the differentiator. This skill encodes those calls, drawn from Emil Kowalski's design engineering practice, as paired right/wrong decisions: apply them when producing UI, cite them when reviewing it.

**Styling systems:** when a task involves writing styles, detect and match the project's existing approach — Tailwind, plain CSS, CSS Modules, CSS-in-JS, vanilla-extract, whatever is already there. Never impose a different styling system. The values below (durations, ratios, alpha borders) translate to any of them.

## Quick Reference

| Topic | File |
| --- | --- |
| Iconography and illustration craft (stroke weights, optical alignment, icon semantics, illustration cohesion) | [icons-and-illustrations.md](icons-and-illustrations.md) |

## Core Principles

1. **One primary action per view.** If three buttons carry equal visual weight, none of them is the next step. Demote: primary, secondary ghost, text link.
2. **Hierarchy is subtraction.** Make the important thing prominent by making everything else recede — don't make everything louder.
3. **Restraint reads as intent.** Fewer colors, fewer font weights, one icon library, three illustration colors instead of six. A limited palette looks decided; a full one looks indecisive.
4. **Specific beats generic.** "Save changes" beats "Submit". "Your email must include an @ symbol" beats "Invalid input". Empty states say why and what to do next.
5. **Borders and shadows should recede, not sit on top.** Alpha-based borders (`rgba(0,0,0,0.08)` in light mode) and layered shadows read as physical; solid hex borders and single big shadows read as pasted on.
6. **Use semantic tokens and perceptual color spaces.** `--color-border-subtle` over raw hex; OKLCH over HSL when comparing or deriving lightness; `color-mix` over hand-picked hover hexes.
7. **Space with system, not eyeballing.** `gap` over per-child margins, paragraph spacing tied to line-height, line length capped at 65ch, inner radius = outer radius − padding.
8. **Every state is designed.** Default, hover, focus, active, disabled, loading, empty, error — each visually distinct. A missing pressed state makes a button feel dead.
9. **Accessibility is a floor.** Focus rings, labels wired to inputs, real `<button>` elements, color-plus-icon-plus-text for errors. Non-negotiable, not a polish pass.
10. **Know when to break the rules.** Several pairs below are trick questions on purpose — tabs, baseline grids, contrast specs. The rule is the default; context decides.

## How to Review UI

When auditing existing UI, present findings as a single markdown table with `| Before | After | Why |` columns — one row per issue, the Why column giving the one-line reason. Don't use stacked "Before:/After:" lists. And review with fresh eyes when possible: flaws invisible at the end of a build session are obvious the next day.

## Typography

1. Display headlines: metric kerning leaves awkward gaps between certain letter pairs — enable optical kerning for large type.
2. Don't set paragraph spacing to an arbitrary value like 12px; set it to 1× the line-height so the vertical rhythm stays consistent.
3. Cap body text at 65ch. Full-container-width lines are exhausting to read.
4. Uppercase labels need loosened tracking. Default tracking on all-caps looks cramped and amateur.
5. Always declare a fallback font stack designed to match the webfont's x-height and weight — an unstyled fallback causes layout shift on load.
6. Light typeface weights are size-dependent: illegible at 11px, elegant at 32px. Don't use light weights small.
7. Pair for contrast, not similarity. Two humanist sans-serifs together create sameness; a sans with a serif creates hierarchy.
8. Number columns (prices, data tables) get `font-variant-numeric: tabular-nums`. Default proportional figures don't stack and the column reads as chaos.
9. Kill widows: a single orphaned word on the last line looks like a mistake — `text-wrap: balance` (or `pretty`) fixes it.
10. For hover emphasis without layout shift, increase weight via a variable font axis rather than swapping to a fixed heavier weight.
11. Default `<sup>` is too large and throws off line spacing — size and baseline-shift footnote markers manually.
12. `text-wrap: balance` isn't always right: on a 3-line headline it can leave line 1 awkwardly short. Sometimes a manual break wins.
13. Italic is for citations and linguistic stress; **bold** is for UI emphasis. Italic as UI hierarchy is a publishing convention in the wrong medium.
14. Load only the font weights you actually use. Shipping 8 weight/style combinations when 3 are used adds page weight and delays render for nothing.
15. Never underline non-links. Underline is a hyperlink affordance; using it for emphasis trains users to click text that goes nowhere.
16. Never type `...` in markup. Use the `…` character, or `text-overflow: ellipsis` when truncation should respond to container width.

## Color

1. Light-mode 1px borders: `rgba(0,0,0,0.08)` recedes naturally; solid `#e0e0e0` looks pasted on top of the surface.
2. Dark-mode 1px borders: the opposite — solid (`#2a2a2a`-style) sits quietly, while `rgba(255,255,255,0.1)` glows.
3. Card shadows: stack three shadows at different blur/opacity levels. One large drop shadow reads as fake; layered shadows read as physically real.
4. Derive hover states with `color-mix(in oklch, var(--color) 85%, black)`, not a hand-picked darker hex — hardcoded hexes drift hue unexpectedly.
5. HSL lies about lightness: blue at 50% L and yellow at 50% L are nowhere near equal perceptually. Use OKLCH when lightness must actually match.
6. Desaturate brand colors 20–30% for dark mode. Carried over at full saturation they vibrate off a dark background.
7. Dark mode is not inverted light mode: preserve the layering hierarchy — the topmost "canvas" surface should stay the brightest (lightest dark grey).
8. Build light tints by reducing chroma in OKLCH, not by dropping opacity — opacity tints go grey and lifeless.
9. Semantic tokens (`--color-border-subtle`) over raw hex scattered through the codebase. Hardcoded values break the moment anything changes.
10. Contrast trick question: a 3:1 ratio fails WCAG AA for body text but can pass APCA for large text. The specs disagree; judge in context.
11. Neutrals should agree with the brand hue: blue UIs want cool greys. Warm greys fight blue.
12. Never use pure `#808080` grey — it reads as placeholder. Tint neutrals with a slight hue bias and they read as designed.
13. If green is your primary brand color, don't also use green for success states — users can't distinguish "primary action" from "it worked". Pick a distinct confirmation color.
14. Disabled states: use a dedicated muted color token, not `opacity: 0.4`. Opacity-based disabling passes contrast checks on some backgrounds and fails on others; the token is predictable.

## Layout & Spacing

1. Nested radii: inner radius = outer radius − padding. Reusing the parent's radius on an inner card creates a visibly wrong gap at the corners.
2. Space siblings with `gap` on the parent, not `margin-bottom` on every child — margin leaves a trailing gap after the last item.
3. Baseline grids are an editorial tool: they shine in long-form layouts and are usually overkill in dense product UI. (Trick question — know which context you're in.)
4. Cap content containers (~1200px, centered) on wide monitors. Uncapped layouts produce unreadable line lengths and sparse, drifting compositions.
5. Set breakpoints where the content breaks, not at device sizes (768/1024). Device-named breakpoints routinely miss the actual failure point.
6. A simple marketing page doesn't need a 12-column grid — 8 columns (or fewer) does the job without configuration overhead.
7. Inside components, prefer `gap` over `margin`: margin bleeds past the component boundary, gap stays contained.
8. Constrain sticky headers with `max-height` and `dvh` units. An 80px header on a 667px landscape viewport leaves almost no room for content.
9. Symmetric two-column layouts are static; a wider column offset by a narrower one creates tension and interest.
10. Fixed bottom CTAs on mobile must account for `env(safe-area-inset-bottom)` or they sit on top of the last form field on phones with home indicators.
11. Text over full-bleed images requires a scrim, color-matching, and contrast checks. Text in a column beside or below the image just works — default to that.

## Information Architecture

1. Forty settings on one flat scrollable page buries everything equally. Group with a sidebar nav and progressive sections — flat lists have no hierarchy.
2. Never place "Delete" directly beside "Confirm". Proximity implies equivalence; separate destructive actions with whitespace and a visual break.
3. Name things by the user's mental model, not internal jargon: "Send messages in the background", not "Enable async relay mode".
4. Tabs trick question: "Account / Billing / Notifications" tabs are arguably fine; "This week / This month / This year" is the textbook-correct use — tabs are for views of the same thing.
5. Empty states explain and act: "You haven't created a project yet" + a "Create project" button, not "No projects found".
6. Onboarding that creates the user's first item in one step beats an 8-stop feature tour. The tour teaches the product; the shortcut delivers value.
7. Three levels of nested dropdowns lose people. Prefer a flatter structure with breadcrumbs where depth is unavoidable.
8. Search data is IA feedback: if users keep searching "invoices" while the nav says "Billing", rename the nav item.
9. Error pages: explain what went wrong and offer a specific recovery action. "Something went wrong" + a back button ends the journey; specificity continues it.
10. Don't bury the primary action at the bottom of a long page — anchor it persistently in view. Scroll-depth data will confirm most users never reach the bottom.

## Interaction

1. Hover states should confirm affordance, not just decorate: pair the color change with a cursor change and a subtle underline where appropriate.
2. Never ship `outline: none` bare. Replace the default focus ring with a custom `outline` plus `outline-offset: 2px` — removing it is an accessibility failure.
3. Buttons need a pressed state (a slight depress/scale on click). Without it the button feels unresponsive.
4. Long list loads get skeletons, not spinners — the skeleton holds the page shape; a spinner collapses it.
5. Prefer optimistic updates that revert on error over waiting for the server before updating the UI. Optimistic feels instant.
6. Debounce search input at ~300ms. Firing a request per keystroke hammers the server and creates race conditions.
7. Touch targets: 44×44px minimum. 28×28px causes mis-taps — the minimum is a guideline, not a suggestion.
8. Trigger actions on `mouseup`, not `mousedown`. Mousedown fires before the user commits; they lose the ability to cancel by dragging away.
9. Auto-advancing inputs (verification codes) feel clever until someone needs to correct a digit. Waiting for tab is safer.
10. Copy-to-clipboard buttons must confirm: swap to a checkmark for ~1.5s. Without feedback, users click three more times to check it worked.
11. Sliders update their value live during drag. On-release-only feels disconnected and makes precise values hard to hit.
12. Labels toggle their checkboxes — proper `<label>` association is one attribute and fixes an unnecessarily small hit target.
13. Show validation errors inline next to the offending field, not stacked above the form. Top-of-form errors make users hunt.

## Motion Timing

1. Hover transitions: 150ms feels native; 400ms feels like the UI is thinking.
2. Eight list items entering at once reads as a flash; staggered at 40ms intervals reads as arrival.
3. Any wait over 400ms needs an indicator (spinner or skeleton) — silence past that reads as broken.
4. Entrances use `ease-out`, never `ease-in` — ease-in delays the initial movement, exactly when the user is watching.
5. Exits accelerate away (`ease-in`), they don't coast leisurely off screen (`ease-out`).
6. Exit-and-enter should be asymmetric — an exit is departure, not an entrance played in reverse.
7. Modals and popovers should scale up from the element that triggered them (shared origin), not pop from screen center — origin creates continuity. (Exception: viewport-centered modals with no anchored trigger stay centered.)
8. `transition: all 0.3s` is a bug factory: it silently animates width, height, and color, triggering layout recalculation. List the properties: `transition: opacity 0.3s, transform 0.3s`.
9. Choreograph: when three elements would move simultaneously and clash, let one move at a time. Competing parallel motion is visual noise.
10. Disable transitions during theme switches — every component has its own transition timing, so a theme change becomes a ragged cascade. With `next-themes`, set `disableTransitionOnChange` on the `ThemeProvider`.
11. Respect `prefers-reduced-motion`: without the check, movement-heavy UI triggers vestibular symptoms for some users.

## Accessibility

1. `aria-label` describes the action, not the element: `aria-label="Search"`, never `aria-label="icon"`.
2. Error states need three signals: red border + error icon + message text. Color alone is invisible to colorblind users.
3. `<button>` over `<div role="button" onClick>` — the div isn't keyboard-focusable without extra work; the button gets everything for free.
4. Pages with long navs need a visually hidden "Skip to content" link that appears on focus, or keyboard users tab through the whole nav on every page.
5. Wire `<label for>` to the input `id`. Unconnected labels don't focus their input when clicked.
6. Modals trap focus. If focus can escape to the page behind, background content becomes reachable and disorienting for screen reader users.
7. Keep DOM order matching visual order. CSS reordering (`order: -1`) desynchronizes what screen readers announce from what sighted users see.
8. Contrast trick question: WCAG AA 4.5:1 alone isn't the whole story — APCA is more nuanced for large text and UI components. Neither spec is complete by itself.

## Copywriting

1. Label buttons with the specific outcome: "Save changes", not "Submit". Specificity builds trust.
2. Error messages tell users how to fix it: "Your email must include an @ symbol", not "Invalid input".
3. Placeholder text is not a label — it disappears the moment typing starts, right when the reminder is needed. Use a persistent label above the field.
4. Empty states give context and a next step: "You haven't added anything yet — start by creating a project", not "No items".
5. Sentence case for UI, not Title Case Throughout The Whole Product — title case on long labels reads like a legal document.
6. Front-load the important word: "Export ready, download it", not a 20-word sentence with "export" buried in the middle.
7. Drop "please" from imperative UI copy: "Confirm", not "Please confirm your selection before proceeding". It's your product — own it.

## Components

1. Button hierarchy: one primary, one secondary ghost, one text link. Three equally weighted primaries in a row means no clear next action.
2. Inputs need visually distinct default, hover, focus, and error states. If they all look identical, users can't tell active from broken.
3. Badge vs tag semantics: a badge implies a count, a tag implies a category. A "New" feature marker is neither — it needs its own treatment.
4. Tooltips can't contain interactive elements — a "Learn more" link inside a hover tooltip is unreachable. Use a click-triggered popover instead.
5. In data tables, right-aligned tabular numbers do the work zebra striping was compensating for. Fix alignment before adding stripes.
6. Toast duration scales with content: compute it from word count (~200–250 wpm reading speed) instead of a flat 2s for every message.
7. Don't wrap an entire card in one `<a>`: it kills text selection and screen readers announce the whole card as a single link. Give the card a distinct CTA link.

## Common Mistakes Checklist

Run through this when a UI "looks off":

- [ ] More than one visually primary button in view
- [ ] Solid hex borders in light mode / glowing alpha borders in dark mode
- [ ] Raw hex values instead of semantic tokens
- [ ] Body text lines wider than 65ch
- [ ] Uppercase labels without loosened tracking
- [ ] Proportional figures in number columns
- [ ] Per-child margins instead of parent `gap`
- [ ] Inner radius equal to outer radius on nested cards
- [ ] Focus outline removed without replacement
- [ ] Placeholder used as the only label
- [ ] "Submit" / "Invalid input" / "Something went wrong" generic copy
- [ ] Color-only error signaling
- [ ] Two icon libraries mixed in one surface
- [ ] `transition: all` anywhere
- [ ] No `prefers-reduced-motion` handling
