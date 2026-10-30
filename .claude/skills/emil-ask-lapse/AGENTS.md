# Letting an agent drive Lapse

Lapse deliberately ships **no click, hover, or type tools**. The calling agent already has browser tooling for that. Lapse is the instrument, not the hand: your tools do the interacting, Lapse does the seeing.

## MCP server

```bash
claude mcp add lapse -- npx @aiforui/lapse-mcp --launch
```

`npx` resolves through the same `.npmrc`, so if the install worked this does too. Modes: `--cdp http://localhost:9222` attaches to a browser you already run (the primary mode); `--launch [--headed]` lets the server launch one.

### The flow

1. `lapse_attach` → `lapse_goto` — the page boots instrumented (patches land before page scripts cache the clock).
2. `lapse_record_start` → *interact with your own browser tools* → `lapse_record_stop`. The stop response hands back the take's **moment vocabulary**.
3. Ask by name, never by frame number:
   - `lapse_diff { a: "click-1", b: "settle" }`
   - `lapse_export { ref: "anim-1-peak" }`
   - `lapse_inventory { ref: "settle" }` — pair with `lapse_screenshot_at` at the same ref to map what you *see* onto a `path` you can inspect
   - `lapse_inspect { path, ref }`
4. `lapse_release` — unfreeze page time before interacting again.

### Tools

| Tool | What it does |
|---|---|
| `lapse_status` | connection + engine state |
| `lapse_attach` | attach to a page (`urlIncludes` / `pageIndex`, or a fresh one) |
| `lapse_goto` | navigate the attached page |
| `lapse_record_start` | begin a take — `scope` selectors, `speed`, `profile: "perf"`, `trace: true` |
| `lapse_record_stop` | end the take → summary + moment vocabulary (+ CDP renderer summary when traced) |
| `lapse_moments` | the named landmarks |
| `lapse_seek` | freeze the page at a moment |
| `lapse_export` | animation-state markdown at a moment |
| `lapse_diff` | what changed between two moments (markdown or structured) |
| `lapse_inspect` | one element's state at a moment |
| `lapse_inventory` | the element cast list |
| `lapse_screenshot_at` | seek + screenshot (inline image or file path) |
| `lapse_mark_state` | record an app-state change into the take (→ `state-N` moment) |
| `lapse_take_save` / `lapse_take_diff` | save takes; motion changelog between two saved takes |
| `lapse_release` | unfreeze page time |

Errors teach: an unknown moment id comes back with the take's full available vocabulary in the message.

### Git diff for motion, from an agent

```
lapse_take_save { path: "before.json" }
      …change the code, reload, record the same interaction…
lapse_take_save { path: "after.json" }
lapse_take_diff { a: "before.json", b: "after.json" }
```

When both takes carry perf data the changelog gains a performance axis (`long frames 12 → 2`). Cross-tier or cross-speed comparisons are **refused with the reason**, never fudged.

### The cleanest jank measurement

```
lapse_record_start { profile: "perf", trace: true }
      …interact…
lapse_record_stop
```

`profile: "perf"` thins frame sampling so Lapse's own overhead barely touches the measurement. `trace: true` runs a CDP trace (Chromium) and the stop response gains layout / style / paint / raster / GPU costs bucketed between moments — *"between `jank-2` and `jank-3`: 43.7ms layout"*.

## Playwright adapter

Same primitives against a page your own harness drives — which is how you assert on motion in CI. Not "does the class get added", but "did the duration change".

```ts
import { attach } from '@aiforui/lapse-playwright'

// Attach BEFORE goto: the inject bundle boots at document_start, so the
// timing patches land before page scripts cache the clock.
const sac = await attach(page)
await page.goto('https://your-app.test')

await sac.record()                    // or { scope: ['.hero'] }, { profile: 'perf' }
await page.click('.cta')              // drive the page yourself
await page.waitForTimeout(800)
const take = await sac.stop()         // summary + the moment vocabulary

take.moments
await sac.diff('click-1', 'settle')   // markdown
await sac.diffData('start', 'end')    // structured FrameDiff
await sac.seek('anim-1-peak')
await sac.screenshotAt('anim-1-peak', { path: 'peak.png' })
await sac.inventory('settle')
await sac.inspect(path, 'anim-1-start')
await sac.release()                   // unfreeze before interacting again
```

| Method | Returns |
|---|---|
| `attach(page)` | `LapseHandle` — registers the init script; before navigation is higher fidelity |
| `record(opts?)` | start a take — `{ scope }` CSS selectors, `{ profile: 'perf' }` |
| `stop()` | `TakeSummary` — duration, counts, full moment vocabulary |
| `perfTrace()` | CDP renderer trace around a take — start **before** `record()`, `trace.stop()` **after** `stop()` |
| `moments()` | the named landmarks |
| `seek(ref)` | freeze at a moment (ms, id, or `{ moment, offsetMs }`) |
| `exportMarkdown(ref?, opts?)` | animation-state report (`detail`: brief → granular) |
| `diff(a, b, opts?)` / `diffData(a, b)` | markdown / structured diff |
| `inspect(path, ref?)` | one element's snapshot + animation rows |
| `inventory(ref?)` | the cast list; a ref seeks first so rects belong to a named moment |
| `take()` | portable `TakeRecord` for inter-capture diffing |
| `markState(name, value)` | app-state change → `state-N` moment |
| `screenshotAt(ref, opts?)` | seek + screenshot of the frozen frame (PNG buffer) |
| `setSpeed(x)` | slow-mo capture |
| `release()` | exit scrub mode, page time live again |

`peerDependency`: `playwright-core` (or `playwright`) ≥ 1.40. Everything crossing the page↔Node boundary is plain JSON — the engine and live elements never leave the page. Unknown moment ids throw with the take's vocabulary in the message.

Take records are plain JSON, so one can be committed and diffed against later with `diffTakes` — no browser at diff time.

## Browser extension

Loads Lapse on **any website**, injected into the page's own JS context at `document_start`, so it wins the early-load race even against libraries that cache time functions.

The toolbar icon toggles the panel on the current tab; timing is patched silently at page load regardless, so animations are covered the moment the panel opens. Clicking again hides it and resets the page to 1×.

Right-click the icon → **Options** (synced via `chrome.storage`, reaching open tabs live): panel position and entrance animation · site access (everywhere / listed sites / everywhere except listed) · recording defaults (capture mode, auto-record on scope pick, max take duration) · timeline layer visibility (bursts, hover bursts, milestones, annotations, layout-shift dots, and the off-by-default dropped-frame heat strip — painting only, takes record everything either way) · export defaults (detail, filter, clear-annotations-on-copy, per-section includes).

It won't run on `chrome://` pages, the Chrome Web Store, or `data:`/`about:` URLs — Chrome blocks content scripts there.

The extension is also the fastest way to study motion you like: slow someone's transition to 0.25×, record it, scrub to the middle, and read the actual easing and durations instead of guessing from a video. "Why does this feel better than mine" is usually a question about timing, and timing is the thing you can't eyeball.
