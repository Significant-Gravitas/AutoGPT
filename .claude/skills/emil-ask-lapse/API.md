# Programmatic API

Everything the panel does is available headlessly on the engine — no panel, no React, no UI dependency. `@aiforui/lapse/core` is dependency-free.

## Getting an engine

```ts
import { getSharedEngine } from '@aiforui/lapse'      // or '@aiforui/lapse/core'
const engine = getSharedEngine()   // the process-wide singleton <Lapse> uses by default
```

```ts
import { LapseEngine } from '@aiforui/lapse/core'
const engine = new LapseEngine()  // isolated — only when you deliberately want a second one
```

`@aiforui/lapse/install` installs the shared engine, so importing it and calling `getSharedEngine()` always refer to the same instance. Pass a private engine to `<Lapse engine={…}>` / `mountLapse({ engine })` only to isolate; otherwise app code and the panel would drive different engines.

## Engine

Anywhere a time is taken, the parameter is a `TimeRef`: virtual ms, a moment id, or `{ moment, offsetMs }`.

### Speed and installation

```ts
engine.setSpeed(0.25)          // arbitrary factor; the panel's own steps are 1 / .75 / .5 / .25 / .1
engine.getSpeed()
engine.install()               // patch timing APIs now without changing speed — wins the early-load race
engine.registerGSAP(gsap)      // ES-module GSAP (window.gsap fallback otherwise)
```

`setSpeed` and `startRecording` install on demand, so `install()` is only needed up front.

### Recording

```ts
engine.startRecording()                                        // whole page
engine.startRecording(null, [document.querySelector('.hero')!]) // element-scoped
engine.startPageLoadRecording({ video: true })                 // from document_start (see pageload.ts)
const capture = engine.stopRecording()                         // → TimelineCapture
engine.getCapture()
engine.state                                                   // 'idle' | 'recording' | 'scrubbing'
engine.autoStopReason                                          // why THIS take auto-stopped, or null
engine.takeToken                                               // compact unique id of the current take
engine.markState('cart.count', 3)                              // → a `state-N` moment; false when not recording
```

### Scrubbing and playback

```ts
engine.seekTo('anim-1-peak')
engine.previewSeek(340)          // show a frame WITHOUT moving the playhead
engine.toggleTakePlayback()      // one control surface for DOM and video takes
engine.takePlaying               // …takePlayable, videoTake
engine.setPlaybackRegion({ startMs: 100, endMs: 480 })   // loop region, session-only; null clears
engine.subscribeTakeTime((virtualMs) => {})              // playhead stream while playing
engine.release()                 // unfreeze page time, drop the take, back to live
```

After `stopRecording()` on a DOM take the page's JS time is **frozen** for scrubbing. Interacting with the page again needs `release()` first.

### Moments, inventory, diffs

```ts
engine.moments()                       // MomentIndex | null — the take's vocabulary
engine.inventory()                     // Inventory | null — structural path, label, role, accessible
                                       // name, live rect, visibility, animated props per element
engine.diff('click-1', 'settle', opts?)          // FrameDiff | null
engine.diffForLLM('hover-1', 'settle', 'moderate')
```

The inventory is what an agent's vision maps "the rocket" onto: pair it with a screenshot of the same seeked moment, then take the resolved `path` into every later call. Semantic selection stays agent-side by design — no OCR or vision ships in Lapse.

### Annotations

```ts
engine.addAnnotation(el, 'ease this out slower', 500, point?)   // replay takes only, while scrubbing
engine.annotations                                              // readonly records, in numbering order
engine.updateAnnotation(id, comment)
engine.removeAnnotation(id)                                     // survivors renumber contiguously
engine.clearAnnotations()
```

### Export

```ts
engine.generateExport(time, filter?)                            // TimelineExport | null (structured)
engine.exportForLLM(time, filter?, detail?, audience?)          // markdown
```

Defaults: `filter = 'active'`, `detail = 'moderate'`, `audience = 'agent'`. The audience matters — `'chat'` applies the user's per-section export trims (the panel's copy button, where the user owns the paste budget); `'agent'` always carries every section, because an integration that never heard of audiences must not silently inherit someone's trims. Suppressed sections leave a breadcrumb, never a silent gap.

### Settings (the extension's options panel sets these)

```ts
engine.setCaptureMode('video')            // 'replay' | 'video' — per-origin, persisted
engine.setDefaultCaptureMode('replay')    // global default; loses to any explicit per-origin choice
engine.setCaptureProfile('perf')          // 'standard' | 'perf' — thinned sampling for clean jank numbers
engine.setScopeAutoRecord(true)           // scope pick starts recording immediately (default false)
engine.setClearAnnotationsOnExport(false) // keep annotations after copying (default true)
engine.setExportScreenshot(true)          // save a screenshot on copy (extension-only capability)
engine.setExportSections({ perf: false }) // chat-channel section toggles
engine.setMaxTakeDurationMs(30_000)       // default 60_000
```

### Lifecycle and diagnostics

```ts
engine.subscribe(() => {})     // state change listener → unsubscribe fn
engine.getDiagnostics()        // state, speed, captureMode, timingInstalled, gsapRegistered, capture counts
engine.destroy()
```

## Core module exports

From `@aiforui/lapse/core` (the main entry re-exports most of these):

| Export | Purpose |
|---|---|
| `LapseEngine`, `getSharedEngine`, `resetSharedEngine` | the engine |
| `diffFrames`, `formatDiffForLLM`, `decomposeTransform`, `FRAME_DIFF_VERSION` | intra-take diff |
| `computeMoments`, `resolveTimeRef`, `unknownMomentMessage`, `MOMENTS_VERSION` | moments |
| `buildInventory`, `animatedPropsByPath`, `INVENTORY_VERSION` | element inventory |
| `diffTakes`, `formatTakeDiffForLLM`, `scorePair`, `TAKE_DIFF_VERSION` | inter-take diff (pure, no browser) |
| `classifyPropertyCost`, `formatPerfForLLM`, `summarizePerf`, `PERF_VERSION` | perf report |
| `generateExport`, `formatExportForLLM`, `getFrameAtTime`, `DEFAULT_EXPORT_SECTIONS` | export |
| `generateBugReport`, `collectDiagnostics`, `logEvent`, `getLogEntries` | diagnostics |
| `fitSpringFromEasing`, `describeFittedSpring` | recover spring params from a baked `linear()` trace |
| `realPerfNow`, `realRaf`, `realDelay`, `withRealTimeout` | unpatched clock — for machinery that must ignore slow-mo |
| `armPageLoadRecording`, `maybeStartPageLoadRecording` | page-load arming |
| `DEFAULT_MAX_TAKE_DURATION_MS` | `60_000` |

Every structured payload carries a schema version (`v: 1`) so an external parser can tell when a contract changed.

## Types

```ts
import type {
  AnimationInfo, TimelineCapture, TimelineExport, FrameSnapshot, ElementSnapshot,
  PropertySnapshot, FrameAnimation, CaptureMilestone, Rect,
  ExportFilter, OutputDetailLevel,
  Moment, MomentIndex, MomentKind, TimeRef,
  FrameDiff, ElementDiff, AnimationDelta, StyleDelta, StateDiff, DiffOptions,
  Inventory, InventoryEntry, TakeRecord, TakeDiff, TakeDiffOptions,
  PerfReport, PerfTier, PropertyCostTier, LongFrame, FrameGapCluster, PerfLayoutShift,
  CaptureEvent, CaptureEventLog, CausalityLog, CausalityMutation, TriggerGrade,
} from '@aiforui/lapse'
```

## React hooks

**Every hook requires a `<LapseProvider>` above it** — `useLapseEngine` throws without one, and `useTimeline` / `useSpeed` both go through it. `<Lapse>` renders its own provider around the panel *inside a portal*, so it does not cover your app's tree: wrap the components that call these hooks yourself.

```tsx
import { LapseProvider, useLapseEngine, useTimeline, useSpeed } from '@aiforui/lapse'

<LapseProvider>          {/* engine defaults to the shared singleton */}
  <YourControls />
</LapseProvider>

const { speed, isPaused, setSpeed, togglePause } = useSpeed()

const {
  state,           // 'idle' | 'recording' | 'scrubbing'
  capture,         // TimelineCapture | null
  scrubTime,       // ms
  startRecording, stopRecording, seek, release, exportLLM,
} = useTimeline()
```

## Panel props

`<Lapse>` and `mountLapse()` take the same object:

| Prop | Default | Meaning |
|---|---|---|
| `position` | `'bottom-left'` | One of 8 slots. The panel is draggable and the dragged slot persists across reloads, **overriding this prop** |
| `engine` | shared singleton | The engine the panel drives |
| `enter` | `false` | Spring the panel in from its slot's edge on mount — for a deliberate summon; silent mounts must not re-announce |
| `defaults` | — | `LapseUiDefaults`: `exportFilter`, `detailLevel`, and per-layer `timeline` visibility. A starting point for the panel's session state, not an override — the user's in-session switches always win |

Timeline layers (`bursts`, `hoverBursts`, `milestones`, `annotations`, `shifts` default on; `heat`, the dropped-frame strip, defaults off) are **painting only** — the take always captures everything, so flipping a layer on later re-reveals data from the same recording.
