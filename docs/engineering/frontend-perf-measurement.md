# Frontend performance measurement

How to show that a frontend change made the app faster or slower: run a fixed scenario before and after the change and compare the medians. The tooling lives in `autogpt_platform/frontend` and only ever runs on a developer machine. It is not part of CI, and nothing it adds reaches a production bundle.

Saved baseline tables for the current scenarios: [frontend-perf-baseline.md](frontend-perf-baseline.md).

## The three numbers

- **Re-renders.** React draws the screen from small components (a chat row, a button, a sidebar entry). When data changes, React redraws the affected components; each redraw is a *render*, and one batch of them is a *commit*. Code often redraws far more than it needs to, for example every row of a 200-message chat when only the streaming row changed. Every wasted render costs CPU time.
- **JS CPU.** The milliseconds the browser's main thread spends running our JavaScript. The main thread also handles clicks, layout and painting, so 80 ms of our code in one go freezes the page for 80 ms. A task of 50 ms or more is a *long task*, which is what people feel as jank: stuttering, typing lag, jerky scrolling.
- **JS heap.** The memory our JavaScript holds after garbage collection. A heap that grows and never shrinks slows the tab down and eventually crashes it. It grows when code keeps references it no longer needs, such as a finished chat session that is never dropped.

They are one problem seen from three sides: wasted renders burn CPU, CPU burns the 16 ms frame budget, and leftover renders hold memory.

## What is there

| Piece | Where | What it does |
| --- | --- | --- |
| Render counter | `public/devtools/render-counter.js` | Hooks into React and, after every commit, records which components rendered and why. Loaded only when `NEXT_PUBLIC_RENDER_COUNTER=1`. |
| Measurement script | `scripts/perf/measure.ts` (`pnpm perf`) | Opens the app in Chromium, runs one scenario, and reads the counter plus Chrome DevTools Protocol metrics. Writes one JSON per run. |
| Compare script | `scripts/perf/compare.ts` (`pnpm perf:compare`) | Prints the medians of two labelled sets side by side as markdown. |
| Scenarios | `scripts/perf/scenarios/*.ts` | One file per scenario: setup, the measured interaction, and the window length. |
| Fixtures | `scripts/perf/fixtures/`, `scripts/perf/lib/*-fixture.ts` | Recorded or generated API responses served with `page.route`, so every run sees the same data and the same streamed reply. |
| Intercepted URLs | `scripts/perf/lib/routes.ts` | The one list of URLs the harness answers itself. Everything else goes to the real dev server and backend. |

## Run one scenario

1. Start the backend as usual (Postgres for Better Auth, plus `rest_server` and `websocket_server`). Copilot, builder and library data come from fixtures, so the executors, ClamAV and the LLM are not needed. On a 16 GB machine, stop `clamav`, `executor`, `copilot_executor`, `scheduler_server` and `notification_server`: the dev server needs about 5 GB and gets killed for memory otherwise.
2. Start the frontend with the counter. `NEXT_PUBLIC_*` values are read when the dev server starts, so restart it after changing the flag:

   ```bash
   cd autogpt_platform/frontend
   NEXT_PUBLIC_RENDER_COUNTER=1 pnpm dev
   ```

3. Install Playwright's Chromium once (`npx playwright install chromium`), or pass `--channel chrome` to use an installed Chrome.
4. In another terminal:

   ```bash
   pnpm perf --list                                            # scenarios
   pnpm perf --scenario copilot-stream-reply --label before    # 1 warm-up + 3 runs
   ```

Each run prints a table and saves `.perf/<scenario>/<label>-<timestamp>-<run>.json` (`.perf/` is gitignored). When there is more than one run, the script also prints the median, range and spread of each metric.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--scenario <name>` | required | Which scenario to run. |
| `--label <text>` | `any` | Name of the set, e.g. `before`, `after`, `baseline`. Letters, digits, `.`, `_`, `-`. |
| `--runs <n>` | `3` | Saved runs. |
| `--warmup <n>` | `1` | Unsaved runs first, so a dev-server compile or a cold cache after a restart or a code edit does not land in a saved run. |
| `--base-url <url>` | `http://localhost:3000` | Frontend to measure. |
| `--profile` | off | Also save a `.cpuprofile` for each run. Open it in Chrome DevTools → Performance → Load profile. |
| `--screenshot` | off | Save a PNG of the page at the end of each window, to check the scenario did what it says. |
| `--headed` | off | Show the browser. |
| `--channel <name>` | Playwright Chromium | Browser channel, e.g. `chrome`. Also `PERF_BROWSER_CHANNEL`. |
| `--storage-state <file>` | none | Reuse a signed-in Playwright state, e.g. `.auth/states/smokeLibrary.json` from the e2e global setup. |

Without `--storage-state`, the script signs in through Better Auth as `perf.harness@example.com` and creates that account the first time. Set `PERF_USER_EMAIL` and `PERF_USER_PASSWORD` to use another account. It also marks onboarding and the first-visit tab intros as done, so no welcome card covers the page being measured.

If `window.__renderCounter` is missing, the script stops and tells you to restart the dev server with the flag.

## Before and after

1. Check out the commit **before** the change and run the scenario the change targets with `--label before`.
2. Check out the commit **with** the change and run it with `--label after`.
3. Compare, and paste the output in the PR description:

   ```bash
   pnpm perf:compare --a before --b after --scenario copilot-stream-reply
   ```

Leave out `--scenario` to compare every scenario that has runs under both labels.

When the change adds the scenario it is measured with (as `copilot-tool-wait` was), the commit before it has no such scenario. Stay on the branch, so `scripts/perf/` keeps the scenario and its fixture, and swap only the app files back to the base for the `before` runs:

```bash
git checkout origin/dev -- <app files the change touches>   # e.g. the files under src/
pnpm perf --scenario copilot-tool-wait --label before
git checkout HEAD -- <the same files>
pnpm perf --scenario copilot-tool-wait --label after
```

The dev server hot-reloads each swap, and the default warm-up run absorbs the recompile. Swap only files under `src/`, never `scripts/perf/`.

The output has three tables (metrics, top components and top render causes), each with `before`, `after` and `Δ` columns, followed by the raw runs in a collapsed block. Every value is the median of the runs, not the average.

Tips for numbers you can trust:

- Compare on the same machine and the same build type. Dev React is slower than production React, but a before/after delta on the same build type is still valid.
- Alternate the two checkouts when you can (before, after, before, after) so a machine that slows down over time does not favour one side.
- Keep the rest of the setup equal on both sides. `.env.default` turns on the React Query devtools and the token devtool, and they render too, so leave them as they are or change them for both runs.
- Read total renders and the per-component counts first: they are deterministic, and on the baseline most scenarios repeat them exactly from run to run. The exception is the end of a copilot turn, which now and then does one full-transcript re-render fewer (see the baseline). Commits, and so renders per commit, depend on how React batches updates into frames, so they can shift once a change makes rendering cheaper. Timing metrics move by a few percent. Treat any change smaller than the spread the script prints as noise.

## Reading the causes column

Each render is attributed to one cause, checked in this order:

| Cause | Meaning | Where to look |
| --- | --- | --- |
| `mount` | The component appeared this commit. | Expected on navigation; suspicious if it repeats while nothing appears (a changing `key`, or a parent that unmounts and remounts it). |
| `state#N` | Its own `useState`/`useReducer` (or a store read through `useSyncExternalStore`, such as zustand or React Query) changed. `N` is the hook's position in the component, counting every hook. | The Nth hook call in the component. |
| `props:a,b` | Its parent passed different values for these props (up to four, then `+N`). | Usually a new object, array or function created on every parent render, or an over-broad prop. |
| `parent-same-props<-Parent` | `Parent` re-rendered and passed props with identical values, but the component is not memoised. | Wrapping in `memo`, or moving state down so `Parent` stops re-rendering. |
| `context` | Nothing above changed, so a context it reads did. | A provider whose `value` is a new object every render, or a context that carries too much. |

A long run of `props:children` usually means the parent rebuilds its JSX every commit. Look for the first component in the list that is not a design-system wrapper (Radix `Presence`, `Popper` and `Slot` mostly pass through what they are given).

Names come from the component function. Third-party code bundled already minified shows short names such as `u0`; it still counts, but the name does not help.

## Scenarios

| Name | Setup | Measured | Window |
| --- | --- | --- | --- |
| `copilot-idle` | Open the 200-message chat | Nothing | 20 s |
| `copilot-stream-reply` | Open the 200-message chat | Type "go", press Enter, stream the 20-line fixture reply at about 33 chunks a second | Until the reply has been sent and the page has refetched the session, + 1 s |
| `copilot-background-stream` | Start the same reply slowly (200 ms a chunk) in the 200-message chat, click Home, and wait 3 s for Home's entrance animations | Nothing | 15 s |
| `copilot-type` | Open the 200-message chat and focus the composer | Type 19 characters, 50 ms apart | 2 s after the last key |
| `copilot-switch-threads` | Open the 200-message chat and list the 200 / 50 / 10-message chats in the sidebar | Click 50 → 10 → 200, three rounds, 2 s on each | Until done |
| `builder-drag-running` | Open a 50-node workflow whose run is RUNNING (the page polls it every second) | Drag one node for 2 s with 60 pointer moves | 2 s |
| `library-running` | Open the library with 20 workflows, all with a RUNNING run | Nothing | 30 s |
| `copilot-stream-code` | Open the 10-message chat | Type "go", press Enter, stream a reply that is one 130-line code block (`fixtures/reply-long-code.sse`) at about 33 chunks a second | Until the last line of the reply is on screen and the page has refetched the session, + 1 s |
| `builder-connect-edge` | Open the same 50-node workflow with no run (nothing polls) | Drag one new edge from StoreValue node 4's `output` to Calculator node 5's `a` input, 20 pointer moves over 0.6 s | 2 s after the drop |
| `library-running-events` | Open the same library at `/library?sort=updatedAt`, with the websocket mocked | Push a `graph_execution_event` for one of the 20 runs every 250 ms, 80 in all; print the `subscribe_graph_executions` frames and `GET /executions` requests seen | 20 s of events + 2 s |
| `builder-run-events` | Open the same 50-node workflow with its RUNNING run; the harness answers the websocket subscribe for that run itself | 40 node execution events over the websocket, one every 50 ms: the 10 unfinished nodes go RUNNING then COMPLETED, each event sent twice. The execution GET returns what has been sent so far, as the backend would | 2 s after the last event |
| `copilot-tool-wait` | Open the 200-message chat, send "go", and wait until the reply's one tool call shows "Running Perf lookup…". The reply then sends only `: heartbeat` comments, so the turn stays on that tool | Nothing | 20 s |
| `copilot-resize-drag` | Open the 200-message chat and click "Open files" so the files panel docks on the right | Drag the panel's resize handle 120 px wider with 60 pointer moves over 1 s | 1 s after the drop |
| `builder-run-beads` | Open the same 50-node workflow with its RUNNING run; the harness answers the websocket subscribe for that run itself | 10 new node executions over the websocket (a second, finished run of nodes 30 to 39 with new `node_exec_id`s), one every 150 ms, so each edge into one of those StoreValue nodes gets a new bead that travels the edge and is removed on arrival | 2 s after the last event |
| `library-agent-runs` | Open a workflow's page with 20 runs; the newest is RUNNING and selected, so its detail polls every 1.5 s. The websocket is mocked | Three new RUNNING runs arrive at the top of the list, 2.5 s apart: each is pushed as a `graph_execution_event`, the list refetches and the scenario waits for its row; prints the run-list and run-detail `GET`s seen | 1 s after the last run's 2.5 s |
| `builder-type-field` | Open the same 50-node workflow with no run, click the first node's text field and clear it | Type 20 characters, 50 ms apart | 2 s after the last key |
| `builder-block-menu-search` | Open the same 50-node workflow with no run and open the block menu | Type "text" into its search, 120 ms a key, wait for the first result, then scroll the results 6 times (800 px, 400 ms apart) so more pages load. Results come from the real backend's block search | 2 s after the last scroll |
| `builder-load-graph` | Open a 1-node workflow | Switch to a saved 100-node workflow with `history.pushState`, as opening another workflow does, until its last node is on the canvas | 2 s after the last node appears |
| `copilot-open-long-thread` | Open the 10-message chat and list the 2,000-message one in the sidebar. The 2,000 rows are the 200 recorded ones repeated, with fresh ids and sequences past 200 | Click the 2,000-message chat until its newest message is on screen | 2 s after that |

Each scenario's `readyWhen` field says exactly what setup waits for before the window opens. Most wait for the page's own content, then for network idle (at most 15 s, because some pages poll), then one more second.

Each run uses a fresh browser context at 1440 × 900, with the `en-US` locale and UTC time. Feature flags are pinned to their code defaults (`/api/feature-flags` answers `{flags: {}, configured: false}`), so a PostHog setup on one machine does not change what renders.

## What a run measures

1. Sign in, open the page, set up the scenario (`scenario.setup`).
2. Reset the counter, start a Chrome trace (`devtools.timeline`), optionally start the CPU profiler, read `Performance.getMetrics`, start the counter.
3. Run the interaction, then wait `windowMs`.
4. Stop the counter, read `Performance.getMetrics` again, stop the trace and the profiler.
5. Force a garbage collection (`HeapProfiler.collectGarbage`), wait 500 ms and read `JSHeapUsedSize`.

| Metric | Source |
| --- | --- |
| commits, total renders, renders per commit, max renders in one commit | render counter |
| top components, top render causes | render counter (the JSON keeps the top 50) |
| script, task, layout and style recalc time (ms) | `Performance.getMetrics` after minus before |
| long tasks (count and total ms) | trace `RunTask` events of 50 ms or more on the page's renderer main thread |
| heap used after GC (MB) | `JSHeapUsedSize` after a forced GC |

## Using the render counter by hand

With the dev server started with the flag, open the browser console on any page:

```js
__renderCounter.reset();
// …click around…
__renderCounter.print(); // console.table of the top components and causes
__renderCounter.report(); // { commits, totalRenders, rendersPerCommit, maxRendersPerCommit, components, causes }
```

`stop()` and `start()` pause and resume recording. It records from page load until you call `stop()`.

How it works: React reports every commit to `window.__REACT_DEVTOOLS_GLOBAL_HOOK__` if that object exists before React loads, which is why the script is loaded with `strategy="beforeInteractive"`. The script wraps an existing hook (the React DevTools extension, or Fast Refresh in dev) or installs a minimal one. On each commit it walks the fiber tree and counts function, class, `forwardRef` and simple `memo` components whose `PerformedWork` flag is set. It skips subtrees React bailed out of, because their fibers keep stale flags from older commits. `memo` wrappers that hold another component (tag 14) are not counted, because React flags both the wrapper and the component inside it and one render would count twice.

Production stays untouched: the file is in `public/` and is never bundled, and the `<Script>` tag in `src/app/layout.tsx` is only rendered when `NEXT_PUBLIC_RENDER_COUNTER` is exactly `"1"`. The root layout is a server component, so a production server reads the variable from its own environment at request time; no deployment sets it. After `pnpm build` without the flag, `grep -r __renderCounter .next/static` finds nothing.

## Fixtures

The copilot scenarios do not need an LLM:

- `fixtures/session-200.json`: 200 message rows (text, code blocks, tables, lists and tool calls). The 50- and 10-message chats are its first 50 and 10 rows.
- `fixtures/reply-20-lines.sse`: one streamed reply in the backend's SSE format, one word per `text-delta`.
- `fixtures/reply-long-code.sse`: a reply that is mostly one 130-line TypeScript block, streamed the same way, so nearly every chunk lands inside an open code fence.
- `lib/stream-server.ts`: a small local HTTP server that replays the `.sse` file one frame every N ms. The stream request is redirected there with `route.continue({ url })`, so the reply arrives at a steady, repeatable pace. The copilot's own word-by-word smoothing (`copilotStreamSmoothing.ts`) then runs on top, as it does in production.
- `lib/copilot-fixture.ts` keeps state: once a streamed turn ends, the session GET returns the new user and assistant rows, as the backend would. Without that, the page's refetch after each turn would wipe the reply it just streamed.

The builder and library fixtures (`lib/builder-fixture.ts`, `lib/library-fixture.ts`) are built in code: a 50-node workflow owned by the signed-in user with a RUNNING run (30 nodes done, 3 running, 7 queued), and 20 workflows each with a RUNNING run. Block schemas still come from the real backend.

**Regenerating the synthetic copilot fixtures** (deterministic, same output every time):

```bash
npx tsx scripts/perf/fixtures/generate.ts
```

**Re-recording from a real stack** when the wire format changes:

```bash
npx tsx scripts/perf/record.ts --session-id <id>                    # the chat's messages → session-200.json
npx tsx scripts/perf/record.ts --session-id <id> --message "go"     # …and one streamed reply → reply-20-lines.sse
```

The session is saved before the message is sent, so it holds the history the reply follows. Use a throwaway chat, because the files are committed. To record without paying for a model, start `rest_server` and `copilot_executor` with `CHAT_TEST_MODE=true`, which swaps the LLM for the backend's dummy service. `--session-out` and `--reply-out` write to other file names under `fixtures/`. A recorded fixture breaks the `fixtures.test.ts` check that the committed files match the generator, so update that test when you commit one.

## Adding a scenario

1. Add `scripts/perf/scenarios/<name>.ts` exporting a `Scenario` (`lib/scenario.ts`): a `name`, a `description`, a `readyWhen` sentence, a `windowMs`, and `setup(ctx)`. `setup` navigates and waits until the page is ready, then returns `{ run() }` with the measured interaction (`noInteraction` for an idle window).
2. Serve its data from a fixture: add the URL pattern to `ROUTE_TABLE` in `lib/routes.ts`, and the handler next to the other fixtures. Call `route.fallback()` for requests the fixture does not own.
3. Reuse the Playwright page objects in `src/playwright/pages/` for selectors.
4. Register it in `scenarios/registry.ts` and add a row to the table above.
5. Run it with `--runs 3 --screenshot`, check the screenshot, and check that renders per commit is stable across runs.
6. Add its baseline to [frontend-perf-baseline.md](frontend-perf-baseline.md).
