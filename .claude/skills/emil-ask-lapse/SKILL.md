---
name: emil-ask-lapse
description: Answer questions about Lapse — the drop-in animation inspector that slows time, records takes, scrubs frames, names moments, diffs motion, and exports animation state for agents — and wire it into a project. Use when the user mentions Lapse, @aiforui/lapse, lapse-mcp, lapse-playwright, the Lapse panel or extension; asks how to slow a page's animations down, record or scrub an interaction, scope a take to elements, record a page load, annotate a frame, copy the LLM export, address time by named moments, diff two moments or two builds, read a take's jank report or trigger causality; or when slow-mo, a take, or the panel isn't behaving.
---

# Lapse

Lapse is a drop-in animation inspector for web apps: slow every animation down, record an interaction as a **take**, scrub it frame by frame against the real DOM, pin notes to the frame that feels wrong, and copy structured animation state an agent can act on. Framework-neutral — React is one integration, not a requirement.

It answers the one question a screenshot, a video, or the source file cannot: *what did this element actually look like 184ms into the transition, and how did that differ from a moment earlier?*

## Answering posture

Answer from this skill. When a question reaches past it — an exact type, a method signature, a default you're unsure of — read the installed package rather than guessing: `node_modules/@aiforui/lapse/dist/*.d.ts` is ground truth, and the panel's own behavior is in `dist/panel.mjs`. **Never invent an API surface.** Lapse's data contracts are deliberately honest about what they don't know (see *Grades*), and a plausible-sounding invented method is the one failure mode this skill exists to prevent.

If the user's project has Lapse installed, check the installed version before answering version-sensitive questions.

## Name and install

The package is **`@aiforui/lapse`**, and it is not on npmjs.org — it ships with the course from a private registry at `aiforui.dev`. Anyone reading a public `npm install lapse` line is reading the unscoped public naming; in this course everything is scoped.

```
# project .npmrc — safe to commit, holds no secret
@aiforui:registry=https://aiforui.dev/npm/
```

```bash
# the token lives USER-level, never in the project
npm config set "//aiforui.dev/npm/:_authToken" "$AIFORUI_TOKEN"
npm install -D @aiforui/lapse
```

Three rules that cause nearly every install failure:

1. **Never `${AIFORUI_TOKEN}` in the project `.npmrc`.** pnpm 11 refuses to expand credentials from a committed file (a committed `.npmrc` can point a scope at any host) and warns instead. Keep the credential user-level and npm, pnpm, and Bun all read it identically. Yarn Berry ignores `.npmrc` entirely and needs `npmScopes.aiforui` in `.yarnrc.yml` — Yarn *does* expand env vars, so that one keeps the `${AIFORUI_TOKEN}` form.
2. **Never alias it to a bare `lapse`** with npm's `npm:` syntax. The MCP server and the Playwright adapter both depend on the scoped name, so an alias installs two copies and `getSharedEngine()` returns a different engine in each — the panel would drive one and app code the other.
3. **401 means the token isn't reaching npm** (nearly always an unexported variable, not a bad token). **404 from npmjs.org** means the scope line isn't being picked up — check `.npmrc` is in the project root.

The companions live in the same scope and are covered by the same `.npmrc`: `@aiforui/lapse-mcp`, `@aiforui/lapse-playwright`.

## Mounting

Entry points: `@aiforui/lapse` (React + core re-exports) · `/core` (headless, dependency-free) · `/panel` (framework-neutral mount, React bundled inside) · `/install` (side-effect timing patches only).

```tsx
import { Lapse } from '@aiforui/lapse'
;<><YourApp /><Lapse /></>
```

```ts
// any framework, or none — React ships inside this entry
import { mountLapse } from '@aiforui/lapse/panel'
const lapse = mountLapse({ position: 'top-right' })  // await lapse.unmount() later
```

```html
<!-- no build step; first in <head>. The IIFE build exposes window.lapse.
     Serve dist/panel.global.js from the installed package — a scoped private
     package is not on a public CDN. -->
<script src="/vendor/lapse-panel.global.js"></script>
<script>lapse.mountLapse()</script>
```

`mountLapse()` is idempotent, returns `{ engine, unmount() }`, and takes the same options as `<Lapse>` (full prop table in [API.md](API.md)): `position` (8 edge slots, default `'bottom-left'` — the panel is draggable, snaps, and the dragged slot persists and wins over the prop), `engine` (defaults to the process-wide shared singleton, so `getSharedEngine()` in app code drives the same instance the panel does), `enter`, and `defaults`.

### The early-load race — the reason people think slow-mo "doesn't work"

Lapse slows time by patching `requestAnimationFrame`, `performance.now`, `Date.now`, `setTimeout`, and `setInterval`. **Any library that cached a reference to those before the patch keeps the real clock and keeps running at full speed.** Framer Motion and GSAP both cache.

`mountLapse()` and `import '@aiforui/lapse/install'` patch immediately (the panel itself waits for the DOM), so either can go first in the entry. But `<Lapse>` patches only when the component *mounts* — after everything imported above it. With the React component, the side-effect entry goes first:

```tsx
// main.tsx — must be the first import
import '@aiforui/lapse/install'

import { createRoot } from 'react-dom/client'
import { App } from './App'
```

**GSAP imported as an ES module** is invisible to auto-detection — `window.gsap` doesn't exist — so hand Lapse your instance once:

```ts
import { gsap } from 'gsap'
import { getSharedEngine } from '@aiforui/lapse'

getSharedEngine().registerGSAP(gsap)
```

What slow-mo covers: CSS transitions and `@keyframes`, the Web Animations API, rAF loops, JS timers, `<video>`/`<audio>`, SVG SMIL, GSAP, and anything reading the clock (Framer Motion). CSS/WAAPI scale via `playbackRate`, GSAP via global-timeline `timeScale`, and SMIL — which has no playback rate — is paused and pumped forward by the virtual clock each frame.

## The panel

Every shortcut takes <kbd>Shift</kbd> on purpose: plain keys belong to the page, so recording an interaction that uses <kbd>Space</kbd> or <kbd>Esc</kbd> never ends the take. <kbd>S</kbd> is the one plain-key exception, guarded against text entry. None fire while typing in an input.

| Key | Action |
|---|---|
| <kbd>⇧S</kbd> | Cycle speed — 1×, 0.75×, 0.5×, 0.25×, 0.1× |
| <kbd>⇧R</kbd> | Start / stop recording |
| <kbd>⇧⌥R</kbd> | Record a page load — reloads with recording live from `document_start` |
| <kbd>⇧Space</kbd> | Stop recording; otherwise pause/play (freezes time, or plays a video take) |
| <kbd>S</kbd> | Scope picker while idle — hold for a momentary session, tap to latch, <kbd>Esc</kbd> drops it |
| <kbd>⇧V</kbd> | Toggle video capture mode |
| <kbd>⇧C</kbd> | Copy the LLM export (while scrubbing) |
| <kbd>⇧D</kbd> | Cycle export detail level (while scrubbing) |
| <kbd>⇧A</kbd> | Toggle annotate mode (while scrubbing) |
| <kbd>⇧Esc</kbd> | Stop recording / clear the timeline |

Panel styles live in a shadow root and never leak into the app. Takes cap at 60s (plus a frame-count memory bound) and auto-stop when the tab has been hidden more than 5s — the take is kept and a toast names what ended it. Holding <kbd>⌥</kbd> over the panel reveals a bug button that copies a diagnostic bundle written for an agent to file; nothing is ever sent anywhere automatically.

## Takes

A take is one recorded interaction. Four kinds, and picking the right one is most of the skill:

- **Plain** (<kbd>⇧R</kbd>) — the whole page. Scrubbing freezes the page at the frame you're on and it is the *real DOM*, not a picture: hover elements, open devtools on them mid-transition.
- **Scoped** (<kbd>S</kbd>, then record) — hold <kbd>S</kbd> and click elements to build a selection, then record; only those are captured and everything else stays frozen during playback. On a page with a shimmer, a spinner, and a marquee this is the difference between a readable take and noise. Pause first (<kbd>⇧Space</kbd>) to catch a fast-moving element. Capture is gated by *geometry* — an element belongs to the scope when its center sits inside a scoped element's live-tracked box — with portalled dropdowns triggered from the scope captured through a separate channel.
- **Page load** (<kbd>⇧⌥R</kbd>) — the entrance choreography from the first paintable frame, with DCL / load / LCP / fonts as timeline markers. The one take you cannot get any other way, because by the time you press record normally the intro is over.
- **Video** (<kbd>⇧V</kbd>) — pixel truth for content DOM replay can't reproduce: canvas, WebGL, cross-origin iframes, `<video>`. The DOM recorder keeps running underneath, so markers and the export survive; what changes is that the page stays live and scrubbing seeks the recording instead of freezing the DOM. Annotations are replay-only. The panel suggests this once per page load if a visible cross-origin iframe is present when a DOM take starts.

## Annotations and the export

While scrubbing, <kbd>⇧A</kbd> turns the cursor into a picker over the frozen page: click an element mid-animation, type a note, and it pins to *that element at that moment* as a numbered marker. Annotations ride the export carrying each element's animation state at its own timestamp — so "the dropdown feels heavy on the way out" becomes a note on the dropdown at 340ms, where the exit is still at 92% opacity while the translate has already finished. Copying hands them over and clears them (setting-gated).

<kbd>⇧C</kbd> copies the export. Three filters and four detail levels:

| Filter | Contents |
|---|---|
| `active` | Only what's animating at the playhead |
| `interacted` | **Default this.** Only elements your interaction touched or set off; everything ambient collapses to one census line |
| `all-animations` | Everything |

| Detail | Adds |
|---|---|
| `brief` | One line per element — property names and timing only |
| `moderate` | **Default this.** Full values, ranges, interaction state |
| `detailed` | CSS variables, `@keyframes` source, transition conflicts |
| `granular` | Full structural element paths, environment (viewport, URL, UA, DPR) |

Reach for `detailed` when the bug is "two things are fighting over the same property" — that is exactly what the transition-conflict section reports. The export collapses repetition on its own: 26 dots staggering in become one `26 ×` line with the phase pattern, not 26 near-identical blocks.

## Moments

Nobody thinks "frame 137" — they think "right after the hover". Every take derives its own addressable vocabulary, and **every time-taking API accepts a moment id in place of milliseconds**, including offsets:

```ts
engine.seekTo('anim-1-peak')
engine.diff('click-1', 'settle')
engine.diff({ moment: 'jank-1', offsetMs: -50 }, 'jank-1')
```

Kinds: `start` · `end` · interaction (`click-1`) · `hover-1` / `hover-1-end` · `focus-1` · `anim-1-start` / `anim-1-peak` / `anim-1-end` · `settle` · milestones (page-load takes) · `note-1` · `jank-1` · `shift-1` · `route-1` · `state-1`.

Ids are deterministic within a capture (ordinals per kind, in time order) and deliberately **not** stable across captures — cross-capture alignment is the take-diff's job. The vocabulary describes the take you actually recorded, not a template: ambient animations get no start moment, and a take that never rests gets no `settle`. An unknown id fails with the take's full vocabulary in the message, so an agent that guesses wrong learns from the error.

## Diffs

**Two moments of one take** — `engine.diff(a, b)` / `engine.diffForLLM(a, b)`. Styles with signed deltas (`opacity 0.92 → 1 (+0.08)`), transforms decomposed into named channels (`transform.translateX: 34px → 48px (+14px)` — never raw matrix strings), mount/unmount, text changes, animation lifecycle (started / progressed / completed / ran entirely inside the window), and interaction state. Per-unit noise floors keep float churn out, so a diff is real changes rather than a wall of `0.9999 → 1`.

**Two builds** — record the same interaction before and after a code change and diff the takes. A `TakeRecord` is capture + inventory, plain JSON, diffable anywhere with no browser:

```ts
import { diffTakes, formatTakeDiffForLLM } from '@aiforui/lapse/core'
formatTakeDiffForLLM(diffTakes(before, after))
// ### CTA Button (match confidence 0.94)
// - "slide" (CSSTransition): duration 200ms → 320ms; easing ease-out → linear
// - at `settle`: opacity 1 → 0.8 (-0.2)
```

Two decisions make it trustworthy: elements are matched by identity features (role, accessible name, text, classes, rect, animated props) with per-pair confidence always reported and sub-threshold candidates listed for review rather than silently guessed; and time aligns by **moment name, never milliseconds** — `settle` vs `settle` is the question actually being asked, and "A settles but B never does" comes back as a finding rather than a mismatch.

This is the step people skip, and it is the one that catches the animation that got fixed while three others quietly changed.

## Jank and triggers

Both are recorded in every take — neither is opt-in.

**Why it stuttered.** Long frames with per-script attribution including forced synchronous layout (*"41ms in `recalcGrid`, of which 34ms forced synchronous layout"*), dropped-frame clusters, layout shifts resolved onto the take's own elements, and every animated property classified by the rendering work it forces (`height` = layout, `transform` = compositor). Stutters mint moments, so `diff('jank-1 -50ms', 'jank-1')` asks "what changed going into the stutter". The report always states its **tier** (`loaf` on Chromium, `frames` elsewhere — "no long frames" never quietly means "couldn't see them") and the **speed it recorded at**, because jank at 0.25× is not jank at 1×. `engine.setCaptureProfile('perf')` thins sampling so the recorder's own overhead barely touches the numbers, at the cost of coarser scrubbing.

**What set it off.** Every DOM change carries an edge back to the input that preceded it: *"click on 'Toggle panel' → `.open` added 4ms later (handler) → height transition started."* **Grades** are measurements, not heuristics — Lapse dispatches every page timer and rAF callback itself and counts real display frames between tasks:

- `handler` — the change happened in the input handler's own task. An 80ms handler keeps this grade.
- `scheduled` — it went through a timer or rAF, with the count carried.
- `unattributed` — Lapse won't claim a link it can't prove. A `fetch().then()` mutation never gets a `handler` grade, even at 7ms latency.

The word "caused" appears nowhere in the data contract. Typed text is suppressed (`(text input)`); shortcuts survive. Take-diffs gain a trigger-timing axis, which is what catches a debounce regression: *"class +open: 4ms (handler) → 210ms (scheduled)."* For state a framework owns there is an escape hatch, not introspection: `engine.markState('cart.count', 3)` mints a `state-N` moment.

## The review loop

When the user wants motion reviewed, run this — the order is the point:

1. **Build it with the `emil-animations` skill** so the first attempt already uses sensible easing and duration instead of defaults.
2. **Drop to 0.25× and watch.** Most "feels off" becomes nameable at quarter speed — the scale starts too small, the opacity finishes long before the movement does, the exit is as slow as the entrance when it should be quicker. Worth doing before writing a single note, and worth doing with no agent involved at all.
3. **Record the interaction** (scoped, if the page is busy).
4. **Scrub to the frame where it stops feeling right** and annotate that element in plain language.
5. **Copy** — `interacted` at `moderate`. The agent now has the note, the element, the moment, and every property value at that moment; it does not have to reconstruct which element or which moment was meant.
6. **Record the same interaction again and diff the two takes** to confirm the change did what was asked and nothing else.

## Deeper reference

| Question | File |
|---|---|
| Headless / programmatic control, engine methods, core exports, types | [API.md](API.md) |
| Letting an agent drive it — MCP tools, Playwright adapter, CI assertions, the browser extension | [AGENTS.md](AGENTS.md) |
| Slow-mo not working, empty or noisy takes, scrub or install failures | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
