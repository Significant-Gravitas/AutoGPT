# Troubleshooting

## Slow-mo doesn't affect my animations

Nearly always the **early-load race** — see SKILL.md for the fix (`/install` first, or `registerGSAP`).

Two things confirm the diagnosis before you go changing imports:

- **"CSS slows down but my JS animation doesn't"** is the signature. CSS transitions and `@keyframes` scale via `playbackRate` and don't depend on the patches at all, so a split like that means a library cached the clock.
- `engine.getDiagnostics()` reports `timingInstalled` and `gsapRegistered` — what actually happened, rather than what the entry file looks like it does.

## The panel controls one thing and my app code controls another

Two engines exist. Causes, in order of likelihood:

1. The package was **aliased** to a bare `lapse` with npm's `npm:` syntax, so two copies are installed and each has its own module-level singleton. Remove the alias — the MCP server and the Playwright adapter both depend on the scoped name.
2. A private `new LapseEngine()` was passed to `<Lapse engine={…}>` or `mountLapse({ engine })`. That is what the option is *for*; drop it to share the singleton.
3. Duplicate copies from a monorepo hoisting split. Check `node_modules/@aiforui/lapse` resolves to one path.

## The panel never appears

- `mountLapse()` throws outside a browser. In Next.js or any SSR framework, render `<Lapse />` from a client component, or call `mountLapse()` inside an effect / a client-only entry.
- `mountLapse()` is idempotent — a second call while mounted returns the existing handle rather than mounting twice.
- The extension can't inject on `chrome://` pages, the Chrome Web Store, or `data:`/`about:` URLs.

## A shortcut does nothing

Every hotkey takes <kbd>Shift</kbd>, none fire while the focus is in a text input, and several are state-gated: <kbd>⇧C</kbd>, <kbd>⇧D</kbd>, and <kbd>⇧A</kbd> only work **while scrubbing**; <kbd>S</kbd> only works **while idle**. Annotations are replay-only — <kbd>⇧A</kbd> is inert on a video take.

## Install fails

Match the symptom, then apply the setup in SKILL.md:

- **401** — the token isn't reaching npm. Almost always an unexported variable, not a bad token.
- **404 from npmjs.org** — the scope line isn't being picked up. Check `.npmrc` is in the project root.
- **pnpm warns and ignores the credential** — `${AIFORUI_TOKEN}` is in the project `.npmrc`. Move it user-level.
- **Yarn Berry ignores everything** — it doesn't read `.npmrc` at all; it needs `.yarnrc.yml`.
- **CI** — add `AIFORUI_TOKEN` as a secret and expose it to the install step.

## The take is unreadable noise

A page with a shimmer, a spinner, and a marquee buries the one transition you care about. Two independent fixes, use both:

- **Scope the take.** Hold <kbd>S</kbd>, click the elements that matter, then record. Everything else stays frozen during playback.
- **Filter the export** to `interacted`. Ambient animations collapse into a single census line instead of eating the context budget.

Can't click a fast-moving element with the scope picker? Freeze the page first with <kbd>⇧Space</kbd>, then pick.

## Scrubbing shows nothing / the wrong thing

- **Canvas, WebGL, cross-origin iframes, `<video>` content** cannot be reproduced by DOM replay. Use a video take (<kbd>⇧V</kbd>) — the DOM recorder keeps running underneath, so markers and the export survive. The panel suggests this once per page load when a visible cross-origin iframe is present at the start of a DOM take.
- **Playing `<video>`/`<audio>` is paused while you scrub** (video pixels aren't DOM-replayable) and resumes from where it stopped when the timeline is cleared. SMIL timelines are seeked exactly via `setCurrentTime`.
- **The entrance animation isn't in the take** — by the time you pressed record it was over. That's what the page-load take (<kbd>⇧⌥R</kbd>) exists for.

## The page is frozen / I can't interact after recording

Expected. A DOM take freezes page JS time for the scrub session — rAF callbacks and timer handlers would otherwise keep animating live under the replay. `release()` (or <kbd>⇧Esc</kbd>) returns time to live. From MCP or Playwright, `lapse_release` / `sac.release()` before interacting again.

Video takes are the exception: the page stays deliberately live behind the overlay.

## The recording stopped on its own

Takes cap at 60s (plus a frame-count memory bound) and auto-stop when the tab has been hidden more than 5s. The take is kept and a toast beside the panel names what ended it; `engine.autoStopReason` carries it programmatically. Raise the cap with `engine.setMaxTakeDurationMs(ms)` or the extension's max-take-duration option.

If a **video** take clears itself instead, webm assembly timed out (10s watchdog) — the log names it.

## The jank numbers look wrong

- **Check the recording speed the report states.** Jank at 0.25× is not jank at 1×, and the report always says which it measured.
- **Check the tier.** `loaf` (Chromium's `long-animation-frame` observer) sees per-script attribution; `frames` is the fallback everywhere else. "No long frames" never quietly means "couldn't see them" — the tier tells you which claim you're reading.
- **The recorder is in the measurement.** `engine.setCaptureProfile('perf')` (or `profile: 'perf'` from MCP/Playwright) thins frame sampling so the recorder's own main-thread share shrinks, at the cost of coarser scrubbing. Lapse also reports its own directly-measured overhead.

Recording is heavy on very large or complex pages — it snapshots the DOM every frame. Speed control alone costs nothing and works everywhere.

## An unknown moment id

The error carries the take's full vocabulary — read it rather than guessing again. Two legitimate absences: ambient animations get no start moment, and a take that never comes to rest gets no `settle`. The vocabulary describes the take you actually recorded.

Moment ids are deterministic within one capture but **not stable across captures**. Comparing two takes is `diffTakes`'s job, and it aligns by moment *name*, never by millisecond.

## A trigger says `unattributed`

That is the answer, not a failure. Grades are measurements: `handler` means the mutation happened in the input handler's own task, `scheduled` means it went through a timer or rAF, and `unattributed` means Lapse won't claim a link it can't prove. A `fetch().then()` mutation never earns a `handler` grade even at 7ms latency. For state a framework owns, `engine.markState(name, value)` is the escape hatch — Lapse never introspects renderers.

## Filing a bug

Hold <kbd>⌥</kbd> over the panel to reveal the bug button. It copies a diagnostic bundle — engine state, recent internal events, environment — written for an agent to file. Nothing is sent anywhere automatically. Programmatically: `generateBugReport()` from `@aiforui/lapse/core`.
