# Frontend performance baseline

Medians of `pnpm perf` for every scenario on `dev`, saved so later work has something to compare against. How to produce and read these tables: [frontend-perf-measurement.md](frontend-perf-measurement.md).

| | |
| --- | --- |
| App code | `dev` at `03dfdcabf` (2026-10-19). Measured on the branch that added the tooling, which changes no app behaviour. |
| Build | `NEXT_PUBLIC_RENDER_COUNTER=1 pnpm dev` (Next 15.5 with Turbopack, React 18.3 development build) |
| Browser | Playwright Chromium headless shell 141, 1440 × 900, `en-US`, UTC |
| Machine | Linux cloud sandbox, 8 vCPU Intel Xeon 2.9 GHz, 16 GB RAM, Node 24 |
| Runs | 1 warm-up, then 3 saved runs per scenario (`--label baseline`) |

Absolute timings depend on the machine, so compare a branch against these numbers only on similar hardware. To compare against your own machine, run the same scenarios on `dev` first with `--label baseline`. Render counts and commits do not depend on the machine.

What stands out:

- **Streaming a reply re-renders the whole conversation.** In `copilot-stream-reply`, one reply costs about 210,000 renders and 5.7 s of script time over about 55 commits. The biggest commits render more than 11,000 components: the 200 existing rows, mostly their tooltips, icons and Radix wrappers. Most of the cost comes after the stream ends, when the page refetches the session and re-renders the transcript several times. `copilot-idle` shows the same chat at rest: about 40 renders per commit.
- **The end of a turn is not fully deterministic.** One run in three to five does one fewer full-transcript re-render (about 11,000 renders, 5%), whatever the run's timing. The median absorbs it; use `--runs 5` when a change to this scenario is close to that size.
- **Switching threads is the most expensive scenario:** about 270,000 renders and 11.7 s of script time over nine switches.
- **Dragging a node during a running execution** renders about 500 components per pointer move, with 39 long tasks in 2 s.
- **The library with 20 running workflows** and **Home with a reply streaming in the background** are quiet.

## builder-drag-running · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 251 | 249 – 258 | 3.6% |
| total renders | 128,816 | 128,768 – 128,857 | 0.1% |
| renders per commit | 513.2 | 499.4 – 517.1 | 3.4% |
| max renders in one commit | 11,029 | 11,029 – 11,029 | 0.0% |
| script time (ms) | 5,327 | 5,254 – 5,354 | 1.9% |
| task time (ms) | 6,811 | 6,676 – 6,838 | 2.4% |
| layout time (ms) | 37 | 37 – 38 | 3.3% |
| style recalc time (ms) | 112 | 112 – 115 | 2.8% |
| long tasks (≥ 50 ms) | 39 | 38 – 47 | 23.1% |
| long task total (ms) | 5,319 | 5,220 – 5,567 | 6.5% |
| heap used after GC (MB) | 103.5 | 103.4 – 103.6 | 0.2% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Presence` | 6,501 |
| `Text` | 6,150 |
| `HugeiconsIcon` | 5,560 |
| `Icon` | 5,560 |
| `Tooltip` | 5,023 |
| `TooltipContent` | 5,021 |
| `TooltipProvider` | 4,949 |
| `Primitive.div` | 4,275 |
| `Popper` | 3,608 |
| `PopperAnchor` | 3,608 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Presence props:children` | 6,501 |
| `HugeiconsIcon parent-same-props<-Icon` | 5,560 |
| `Tooltip props:children` | 5,023 |
| `TooltipProvider props:children` | 4,949 |
| `Primitive.div props:children` | 3,489 |
| `Primitive.div.Slot props:children` | 3,096 |
| `Primitive.div.SlotClone props:children` | 3,096 |
| `TooltipContent props:children` | 2,876 |
| `Primitive.button.Slot props:children,onPointerMove,onPointerLeave,onPointerDown,+4` | 2,654 |
| `Primitive.button.SlotClone props:onPointerMove,onPointerLeave,onPointerDown,onFocus,+4` | 2,654 |

## copilot-background-stream · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 6 | 6 – 6 | 0.0% |
| total renders | 443 | 443 – 443 | 0.0% |
| renders per commit | 73.8 | 73.8 – 73.8 | 0.0% |
| max renders in one commit | 275 | 275 – 275 | 0.0% |
| script time (ms) | 62 | 53 – 71 | 30.1% |
| task time (ms) | 230 | 229 – 294 | 28.3% |
| layout time (ms) | 0 | 0 – 0 | 0.0% |
| style recalc time (ms) | 0 | 0 – 0 | 0.0% |
| long tasks (≥ 50 ms) | 1 | 1 – 1 | 0.0% |
| long task total (ms) | 73 | 65 – 81 | 21.5% |
| heap used after GC (MB) | 56.0 | 55.9 – 56.3 | 0.7% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Text` | 35 |
| `motion.div` | 21 |
| `Card` | 18 |
| `motion.g` | 18 |
| `Presence` | 18 |
| `HugeiconsIcon` | 15 |
| `Icon` | 15 |
| `Primitive.div` | 13 |
| `Primitive.button` | 12 |
| `Popper` | 11 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `motion.div props:initial,animate,transition,children` | 21 |
| `Card parent-same-props<-motion.g` | 18 |
| `motion.g props:initial,animate,transition,children` | 18 |
| `Presence props:children` | 18 |
| `HugeiconsIcon parent-same-props<-Icon` | 15 |
| `Primitive.div props:children` | 12 |
| `Text parent-same-props<-motion.div` | 12 |
| `Primitive.div.Slot props:children` | 11 |
| `Primitive.div.SlotClone props:children` | 11 |
| `Tooltip props:children` | 10 |

## copilot-idle · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 5 | 5 – 5 | 0.0% |
| total renders | 201 | 201 – 201 | 0.0% |
| renders per commit | 40.2 | 40.2 – 40.2 | 0.0% |
| max renders in one commit | 100 | 100 – 100 | 0.0% |
| script time (ms) | 22 | 21 – 22 | 4.1% |
| task time (ms) | 52 | 49 – 52 | 5.4% |
| layout time (ms) | 0 | 0 – 0 | 0.0% |
| style recalc time (ms) | 0 | 0 – 0 | 0.0% |
| long tasks (≥ 50 ms) | 0 | 0 – 0 | 0.0% |
| long task total (ms) | 0 | 0 – 0 | 0.0% |
| heap used after GC (MB) | 83.7 | 83.6 – 84.8 | 1.4% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Presence` | 12 |
| `Primitive.div` | 10 |
| `Primitive.button` | 8 |
| `Anonymous` | 6 |
| `HugeiconsIcon` | 6 |
| `Icon` | 6 |
| `Popper` | 6 |
| `PopperAnchor` | 6 |
| `PopperProvider` | 6 |
| `Primitive.button.Slot` | 6 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Presence props:children` | 12 |
| `Primitive.div props:children` | 8 |
| `HugeiconsIcon parent-same-props<-Icon` | 6 |
| `Primitive.div.Slot props:children` | 6 |
| `Primitive.div.SlotClone props:children` | 6 |
| `Tooltip props:children` | 6 |
| `LinkComponent props:children,onPointerMove,onPointerLeave,onPointerDown,+4` | 4 |
| `MenuProvider props:scope,children` | 4 |
| `Popper props:children` | 4 |
| `PopperAnchor props:children` | 4 |

## copilot-stream-reply · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 53 | 52 – 55 | 5.7% |
| total renders | 209,396 | 198,389 – 209,496 | 5.3% |
| renders per commit | 3,815.2 | 3,809.0 – 3,950.9 | 3.7% |
| max renders in one commit | 11,149 | 11,149 – 11,149 | 0.0% |
| script time (ms) | 5,709 | 5,396 – 5,760 | 6.4% |
| task time (ms) | 6,158 | 5,842 – 6,252 | 6.7% |
| layout time (ms) | 20 | 19 – 24 | 22.8% |
| style recalc time (ms) | 186 | 185 – 202 | 9.5% |
| long tasks (≥ 50 ms) | 18 | 17 – 18 | 5.6% |
| long task total (ms) | 5,867 | 5,549 – 5,931 | 6.5% |
| heap used after GC (MB) | 86.4 | 85.9 – 86.6 | 0.9% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 16,230 |
| `TooltipContent` | 16,228 |
| `HugeiconsIcon` | 9,061 |
| `Icon` | 9,061 |
| `Presence` | 8,299 |
| `Primitive.div` | 8,196 |
| `Popper` | 8,194 |
| `PopperAnchor` | 8,194 |
| `PopperProvider` | 8,194 |
| `Primitive.div.Slot` | 8,194 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 16,183 |
| `HugeiconsIcon parent-same-props<-Icon` | 9,054 |
| `Presence props:children` | 8,293 |
| `Primitive.div props:children` | 8,188 |
| `Primitive.div.Slot props:children` | 8,187 |
| `Primitive.div.SlotClone props:children` | 8,187 |
| `PopperAnchor props:children` | 8,169 |
| `Popper props:children` | 8,164 |
| `PopperProvider props:children` | 8,164 |
| `TooltipProvider props:children` | 8,141 |

## copilot-switch-threads · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 288 | 286 – 310 | 8.3% |
| total renders | 269,370 | 261,379 – 295,570 | 12.7% |
| renders per commit | 935.3 | 913.9 – 953.5 | 4.2% |
| max renders in one commit | 13,364 | 13,364 – 13,364 | 0.0% |
| script time (ms) | 11,714 | 11,169 – 11,978 | 6.9% |
| task time (ms) | 13,451 | 12,973 – 13,686 | 5.3% |
| layout time (ms) | 143 | 140 – 148 | 5.8% |
| style recalc time (ms) | 376 | 355 – 381 | 6.8% |
| long tasks (≥ 50 ms) | 26 | 23 – 27 | 15.4% |
| long task total (ms) | 9,939 | 9,240 – 10,108 | 8.7% |
| heap used after GC (MB) | 90.4 | 90.4 – 92.6 | 2.4% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 19,082 |
| `TooltipContent` | 19,024 |
| `HugeiconsIcon` | 10,827 |
| `Icon` | 10,827 |
| `Presence` | 10,554 |
| `Primitive.div` | 10,049 |
| `Popper` | 9,955 |
| `PopperAnchor` | 9,955 |
| `PopperProvider` | 9,955 |
| `Primitive.div.Slot` | 9,955 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 13,934 |
| `HugeiconsIcon parent-same-props<-Icon` | 8,920 |
| `Presence props:children` | 8,763 |
| `Primitive.div props:children` | 8,265 |
| `Primitive.div.Slot props:children` | 8,218 |
| `Primitive.div.SlotClone props:children` | 8,218 |
| `PopperAnchor props:children` | 8,095 |
| `Primitive.button props:children,ref,onPointerMove,onPointerLeave,+4` | 7,962 |
| `Primitive.button.Slot props:children,onPointerMove,onPointerLeave,onPointerDown,+4` | 7,962 |
| `Primitive.button.SlotClone props:onPointerMove,onPointerLeave,onPointerDown,onFocus,+4` | 7,962 |

## copilot-type · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 38 | 38 – 38 | 0.0% |
| total renders | 2,869 | 2,869 – 2,869 | 0.0% |
| renders per commit | 75.5 | 75.5 – 75.5 | 0.0% |
| max renders in one commit | 151 | 151 – 151 | 0.0% |
| script time (ms) | 302 | 298 – 331 | 11.0% |
| task time (ms) | 490 | 488 – 523 | 7.2% |
| layout time (ms) | 34 | 32 – 34 | 5.4% |
| style recalc time (ms) | 27 | 26 – 28 | 7.0% |
| long tasks (≥ 50 ms) | 0 | 0 – 0 | 0.0% |
| long task total (ms) | 0 | 0 – 0 | 0.0% |
| heap used after GC (MB) | 85.3 | 84.3 – 85.4 | 1.3% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Presence` | 171 |
| `Popper` | 133 |
| `PopperAnchor` | 133 |
| `PopperProvider` | 133 |
| `Primitive.div` | 133 |
| `Primitive.div.Slot` | 133 |
| `Primitive.div.SlotClone` | 133 |
| `Tooltip` | 114 |
| `TooltipContent` | 114 |
| `Primitive.button` | 95 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Presence props:children` | 171 |
| `Primitive.div props:children` | 133 |
| `Primitive.div.Slot props:children` | 133 |
| `Primitive.div.SlotClone props:children` | 133 |
| `Popper props:children` | 114 |
| `PopperAnchor props:children` | 114 |
| `PopperProvider props:children` | 114 |
| `Tooltip props:children` | 114 |
| `HugeiconsIcon parent-same-props<-Icon` | 76 |
| `Icon parent-same-props<-Primitive.button.SlotClone` | 57 |

## library-running · baseline — median of 3 runs

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 30 | 30 – 30 | 0.0% |
| total renders | 516 | 516 – 516 | 0.0% |
| renders per commit | 17.2 | 17.2 – 17.2 | 0.0% |
| max renders in one commit | 100 | 100 – 100 | 0.0% |
| script time (ms) | 69 | 64 – 75 | 14.9% |
| task time (ms) | 496 | 496 – 575 | 16.0% |
| layout time (ms) | 0 | 0 – 0 | 0.0% |
| style recalc time (ms) | 91 | 91 – 91 | 0.3% |
| long tasks (≥ 50 ms) | 0 | 0 – 0 | 0.0% |
| long task total (ms) | 0 | 0 – 0 | 0.0% |
| heap used after GC (MB) | 47.1 | 47.0 – 47.2 | 0.6% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `u0` | 60 |
| `Presence` | 20 |
| `on` | 18 |
| `oo` | 18 |
| `Primitive.div` | 15 |
| `da` | 12 |
| `em` | 12 |
| `oC` | 12 |
| `Primitive.button` | 12 |
| `r$` | 12 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `u0 context` | 60 |
| `Presence props:children` | 20 |
| `da state#0` | 12 |
| `em props:children` | 12 |
| `oC parent-same-props<-r$` | 12 |
| `on context` | 12 |
| `oo props:isDevBuilding` | 12 |
| `Primitive.div props:children` | 12 |
| `r$ props:state,children` | 12 |
| `rV props:state,children` | 12 |

## builder-connect-edge · baseline — median of 10 runs

Added later than the other tables: app code is `dev` at `a1578ad9a` (2026-10-19); build, browser and machine are the same as above. Ten saved runs instead of three. A few runs have extra commits in the window; total renders hardly move.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 20 | 20 – 36 | 80.0% |
| total renders | 9,197 | 9,197 – 9,457 | 2.8% |
| renders per commit | 459.9 | 262.7 – 459.9 | 42.9% |
| max renders in one commit | 8,682 | 8,682 – 8,682 | 0.0% |
| script time (ms) | 516 | 489 – 633 | 27.9% |
| task time (ms) | 673 | 641 – 863 | 33.0% |
| layout time (ms) | 3 | 2 – 4 | 48.4% |
| style recalc time (ms) | 7 | 6 – 10 | 59.1% |
| long tasks (≥ 50 ms) | 1 | 1 – 2 | 100.0% |
| long task total (ms) | 412 | 396 – 571 | 42.6% |
| heap used after GC (MB) | 85.4 | 84.0 – 85.8 | 2.0% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Text` | 491 |
| `HugeiconsIcon` | 417 |
| `Icon` | 417 |
| `Primitive.div` | 401 |
| `Tooltip` | 370 |
| `TooltipContent` | 370 |
| `HandleComponent` | 346 |
| `SelectItem` | 296 |
| `TooltipProvider` | 295 |
| `Presence` | 240 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 370 |
| `HugeiconsIcon parent-same-props<-Icon` | 362 |
| `HandleComponent state#8` | 344 |
| `SelectItem props:onMouseDown,children` | 296 |
| `TooltipProvider props:children` | 295 |
| `Presence props:children` | 240 |
| `Primitive.div props:children` | 231 |
| `Primitive.div.Slot props:children` | 231 |
| `Primitive.div.SlotClone props:children` | 231 |
| `Text props:children` | 224 |

## library-running-events · baseline — median of 3 runs

Added after the table above, so it was measured on `dev` at `a1578ad9a` (2026-10-19), on the same machine. The websocket is mocked; each run prints what it saw, and all three printed the same: 80 events in 20 s, counted until 2 s after the last one, caused 1,600 `subscribe_graph_executions` frames (all 20 workflows on every event) and 80 `GET /executions` refetches. Each refetch re-renders the whole card grid.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 242 | 242 – 242 | 0.0% |
| total renders | 143,052 | 143,052 – 143,052 | 0.0% |
| renders per commit | 591.1 | 591.1 – 591.1 | 0.0% |
| max renders in one commit | 1,768 | 1,768 – 1,768 | 0.0% |
| script time (ms) | 8,442 | 8,191 – 8,797 | 7.2% |
| task time (ms) | 10,585 | 10,338 – 10,975 | 6.0% |
| layout time (ms) | 55 | 55 – 58 | 6.5% |
| style recalc time (ms) | 160 | 139 – 175 | 22.7% |
| long tasks (≥ 50 ms) | 80 | 80 – 80 | 0.0% |
| long task total (ms) | 8,288 | 8,053 – 8,635 | 7.0% |
| heap used after GC (MB) | 43.5 | 43.5 – 43.5 | 0.1% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `HugeiconsIcon` | 9,760 |
| `Icon` | 9,760 |
| `Presence` | 8,642 |
| `Dialog` | 6,881 |
| `Text` | 5,440 |
| `motion.div` | 4,880 |
| `DialogPortal` | 3,441 |
| `DialogPortalProvider` | 3,441 |
| `DialogProvider` | 3,441 |
| `BaseContent` | 3,440 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `HugeiconsIcon parent-same-props<-Icon` | 9,760 |
| `Presence props:children` | 8,642 |
| `Icon parent-same-props<-motion.div` | 4,800 |
| `Dialog props:onOpenChange,children` | 3,441 |
| `DialogPortal props:children` | 3,441 |
| `DialogPortalProvider props:children` | 3,441 |
| `DialogProvider props:children` | 3,441 |
| `BaseContent props:children` | 3,440 |
| `DialogWrap props:styling,handleOpen,handleClose,children` | 3,440 |
| `LinkComponent props:children` | 3,200 |

## builder-run-events · baseline — median of 3 runs

Added with SECRT-3003: measured on `dev` app code at `a1578ad9a` on a similar 16 GB cloud sandbox.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 158 | 157 – 159 | 1.3% |
| total renders | 58,440 | 58,438 – 58,440 | 0.0% |
| renders per commit | 369.9 | 367.5 – 372.2 | 1.3% |
| max renders in one commit | 10,628 | 10,628 – 10,628 | 0.0% |
| script time (ms) | 2,528 | 2,367 – 2,774 | 16.1% |
| task time (ms) | 3,038 | 2,766 – 3,207 | 14.5% |
| layout time (ms) | 17 | 15 – 17 | 7.6% |
| style recalc time (ms) | 27 | 22 – 28 | 19.4% |
| long tasks (≥ 50 ms) | 6 | 6 – 10 | 66.7% |
| long task total (ms) | 1,868 | 1,588 – 2,065 | 25.5% |
| heap used after GC (MB) | 107.9 | 107.4 – 110.7 | 3.1% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Presence` | 3,032 |
| `Text` | 2,667 |
| `HugeiconsIcon` | 2,438 |
| `Icon` | 2,438 |
| `Tooltip` | 2,266 |
| `TooltipContent` | 2,266 |
| `TooltipProvider` | 2,214 |
| `Primitive.div` | 1,962 |
| `Primitive.button` | 1,636 |
| `Popper` | 1,605 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Presence props:children` | 2,932 |
| `HugeiconsIcon parent-same-props<-Icon` | 2,398 |
| `Tooltip props:children` | 2,226 |
| `TooltipProvider props:children` | 2,174 |
| `Primitive.div props:children` | 1,554 |
| `Primitive.div.Slot props:children` | 1,375 |
| `Primitive.div.SlotClone props:children` | 1,375 |
| `TooltipContent props:children` | 1,344 |
| `Primitive.button.Slot props:children,onPointerMove,onPointerLeave,onPointerDown,+4` | 1,217 |
| `Primitive.button.SlotClone props:onPointerMove,onPointerLeave,onPointerDown,onFocus,+4` | 1,217 |

## copilot-stream-code · baseline — median of 5 runs

Added after the table above was saved: `dev` at `a1578ad9a`, same machine type, 1 warm-up and 5 saved runs. The reply is one 130-line code block, so most of the time is the transcript re-rendering on every streamed word, plus Shiki highlighting the open block on each flush.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 2,573 | 2,565 – 2,588 | 0.9% |
| total renders | 579,560 | 576,640 – 580,000 | 0.6% |
| renders per commit | 224.5 | 224.1 – 226.1 | 0.9% |
| max renders in one commit | 1,010 | 1,010 – 1,010 | 0.0% |
| script time (ms) | 46,940 | 46,115 – 50,794 | 10.0% |
| task time (ms) | 55,268 | 54,101 – 58,986 | 8.8% |
| layout time (ms) | 461 | 428 – 486 | 12.8% |
| style recalc time (ms) | 1,616 | 1,523 – 1,690 | 10.3% |
| long tasks (≥ 50 ms) | 377 | 350 – 407 | 15.1% |
| long task total (ms) | 33,039 | 31,008 – 37,272 | 19.0% |
| heap used after GC (MB) | 70.8 | 70.4 – 73.5 | 4.3% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 35,872 |
| `TooltipContent` | 35,871 |
| `Presence` | 24,642 |
| `Popper` | 20,406 |
| `PopperAnchor` | 20,406 |
| `PopperProvider` | 20,406 |
| `Primitive.button` | 20,406 |
| `Primitive.button.Slot` | 20,406 |
| `Primitive.button.SlotClone` | 20,406 |
| `Primitive.div` | 20,406 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 34,456 |
| `Presence props:children` | 24,631 |
| `Primitive.div props:children` | 20,397 |
| `Primitive.div.Slot props:children` | 20,397 |
| `Primitive.div.SlotClone props:children` | 20,397 |
| `TooltipProvider props:children` | 20,388 |
| `HugeiconsIcon parent-same-props<-Icon` | 19,744 |
| `PopperAnchor props:children` | 19,676 |
| `Popper props:children` | 19,667 |
| `PopperProvider props:children` | 19,667 |

## copilot-tool-wait · baseline — median of 3 runs

Added with SECRT-2995 and measured on `dev` at `a1578ad9a`, before that change. Each of the 20 long tasks is one tick of the thinking timer, which re-rendered the whole transcript once a second while the tool ran.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 45 | 45 – 49 | 8.9% |
| total renders | 183,109 | 183,109 – 183,145 | 0.0% |
| renders per commit | 4,069.1 | 3,737.7 – 4,069.1 | 8.1% |
| max renders in one commit | 9,134 | 9,134 – 9,134 | 0.0% |
| script time (ms) | 5,475 | 5,448 – 5,719 | 4.9% |
| task time (ms) | 7,248 | 7,058 – 7,546 | 6.7% |
| layout time (ms) | 0 | 0 – 0 | 0.0% |
| style recalc time (ms) | 190 | 168 – 196 | 14.2% |
| long tasks (≥ 50 ms) | 20 | 20 – 20 | 0.0% |
| long task total (ms) | 5,439 | 5,412 – 5,693 | 5.2% |
| heap used after GC (MB) | 91.8 | 89.2 – 96.6 | 8.0% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 13,532 |
| `TooltipContent` | 13,532 |
| `HugeiconsIcon` | 7,754 |
| `Icon` | 7,754 |
| `TooltipProvider` | 6,786 |
| `Presence` | 6,780 |
| `Popper` | 6,776 |
| `PopperAnchor` | 6,776 |
| `PopperProvider` | 6,776 |
| `Primitive.button` | 6,776 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 13,532 |
| `HugeiconsIcon parent-same-props<-Icon` | 7,754 |
| `TooltipProvider props:children` | 6,786 |
| `Presence props:children` | 6,780 |
| `Primitive.div props:children` | 6,776 |
| `Primitive.div.Slot props:children` | 6,776 |
| `Primitive.div.SlotClone props:children` | 6,776 |
| `Popper props:children` | 6,770 |
| `PopperAnchor props:children` | 6,770 |
| `PopperProvider props:children` | 6,770 |

## copilot-resize-drag · baseline — median of 3 runs

Added with the scenario (SECRT-2999): `dev` at `6d793e53d`, same machine type, 1 warm-up and 3 saved runs. Every pointer move of the drag wrote the panel width to the copilot UI store, and the page hook read the whole store, so each move re-rendered the whole 200-message transcript: about 3,000 renders per commit and one long task per move.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 193 | 190 – 196 | 3.1% |
| total renders | 574,244 | 574,001 – 574,251 | 0.0% |
| renders per commit | 2,975.4 | 2,929.9 – 3,021.1 | 3.1% |
| max renders in one commit | 9,498 | 9,498 – 9,498 | 0.0% |
| script time (ms) | 17,819 | 17,358 – 18,387 | 5.8% |
| task time (ms) | 20,508 | 19,966 – 20,952 | 4.8% |
| layout time (ms) | 575 | 575 – 577 | 0.4% |
| style recalc time (ms) | 34 | 34 – 35 | 2.1% |
| long tasks (≥ 50 ms) | 60 | 60 – 60 | 0.0% |
| long task total (ms) | 19,661 | 19,213 – 20,218 | 5.1% |
| heap used after GC (MB) | 86.5 | 84.1 – 95.6 | 13.4% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 41,418 |
| `TooltipContent` | 41,418 |
| `HugeiconsIcon` | 24,264 |
| `Icon` | 24,264 |
| `Presence` | 22,172 |
| `Popper` | 21,204 |
| `PopperAnchor` | 21,204 |
| `PopperProvider` | 21,204 |
| `Primitive.button` | 21,204 |
| `Primitive.button.Slot` | 21,204 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 41,298 |
| `HugeiconsIcon parent-same-props<-Icon` | 24,264 |
| `Presence props:children` | 22,172 |
| `Primitive.div props:children` | 21,204 |
| `Primitive.div.Slot props:children` | 21,204 |
| `Primitive.div.SlotClone props:children` | 21,204 |
| `Popper props:children` | 20,955 |
| `PopperAnchor props:children` | 20,955 |
| `PopperProvider props:children` | 20,955 |
| `TooltipProvider props:children` | 20,949 |

## builder-run-beads · baseline — median of 5 runs

Added with SECRT-3009: `dev` app code at `6d793e53d` on a similar 16 GB cloud sandbox, 2 warm-ups and 5 saved runs. On this code every bead re-renders `JSBeads` once per animation frame (about 150 `JSBeads state#0` renders in the window); the heavy commits are the nodes re-rendering on each new execution.


| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 84 | 79 – 85 | 7.1% |
| total renders | 54,037 | 53,947 – 54,078 | 0.2% |
| renders per commit | 642.7 | 636.2 – 684.0 | 7.4% |
| max renders in one commit | 7,285 | 7,285 – 7,285 | 0.0% |
| script time (ms) | 2,332 | 2,142 – 2,505 | 15.6% |
| task time (ms) | 2,791 | 2,567 – 3,057 | 17.6% |
| layout time (ms) | 39 | 38 – 43 | 13.0% |
| style recalc time (ms) | 31 | 29 – 32 | 10.9% |
| long tasks (≥ 50 ms) | 16 | 11 – 17 | 37.5% |
| long task total (ms) | 2,274 | 2,069 – 2,613 | 23.9% |
| heap used after GC (MB) | 99.4 | 97.1 – 103.1 | 6.0% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Text` | 2,954 |
| `TooltipProvider` | 2,819 |
| `HugeiconsIcon` | 2,675 |
| `Icon` | 2,675 |
| `Primitive.div` | 2,411 |
| `Tooltip` | 2,322 |
| `TooltipContent` | 2,322 |
| `Presence` | 1,561 |
| `SelectItem` | 1,512 |
| `Primitive.div.Slot` | 1,427 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `TooltipProvider props:children` | 2,759 |
| `HugeiconsIcon parent-same-props<-Icon` | 2,323 |
| `Tooltip props:children` | 2,282 |
| `SelectItem props:onMouseDown,children` | 1,512 |
| `Presence props:children` | 1,461 |
| `Primitive.div props:children` | 1,437 |
| `Primitive.div.Slot props:children` | 1,407 |
| `Primitive.div.SlotClone props:children` | 1,407 |
| `Popper props:children` | 1,269 |
| `PopperAnchor props:children` | 1,269 |

## library-agent-runs · baseline — median of 3 runs

Added with SECRT-3010 and measured on `dev` at `ede2df2ae`, before that change. The websocket is mocked; every run printed the same: 3 new runs caused 3 `GET` refetches of the run list and the selected RUNNING run polled its detail 5 times. Each new run re-rendered every row of the list about twice (20 to 23 cards, each with its actions menu and two dialogs). Total renders move by one full-list re-render (about 1,300 renders) from run to run.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 40 | 38 – 41 | 7.5% |
| total renders | 9,722 | 8,462 – 9,722 | 13.0% |
| renders per commit | 237.1 | 222.7 – 243.1 | 8.6% |
| max renders in one commit | 1,252 | 1,252 – 1,260 | 0.6% |
| script time (ms) | 785 | 720 – 789 | 8.7% |
| task time (ms) | 978 | 918 – 999 | 8.2% |
| layout time (ms) | 9 | 8 – 10 | 24.7% |
| style recalc time (ms) | 13 | 12 – 13 | 8.9% |
| long tasks (≥ 50 ms) | 6 | 6 – 6 | 0.0% |
| long task total (ms) | 579 | 539 – 588 | 8.5% |
| heap used after GC (MB) | 43.3 | 43.2 – 44.0 | 1.8% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Presence` | 926 |
| `Dialog` | 687 |
| `Text` | 390 |
| `DialogPortal` | 344 |
| `DialogPortalProvider` | 344 |
| `DialogProvider` | 344 |
| `BaseContent` | 343 |
| `DialogWrap` | 343 |
| `MenuProvider` | 302 |
| `Primitive.div` | 259 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Presence props:children` | 926 |
| `Dialog props:onOpenChange,children` | 344 |
| `DialogPortal props:children` | 344 |
| `DialogPortalProvider props:children` | 344 |
| `DialogProvider props:children` | 344 |
| `BaseContent props:children` | 343 |
| `DialogWrap props:styling,handleOpen,handleClose,children` | 343 |
| `Dialog props:controlled,styling,children` | 328 |
| `MenuProvider props:scope,children` | 302 |
| `Primitive.div props:children` | 224 |

## builder-type-field · baseline — median of 3 runs

Added with SECRT-3012: `dev` app code at `8eb406275` on a similar 16 GB cloud sandbox, Chrome 141 (`--channel chrome`), 1 warm-up and 3 saved runs. On this code every keystroke re-renders `<Flow>` and the panels beside the canvas (`BuilderChatPanel`, `FloatingReviewsPanel`, `DraftRecoveryPopup`, `FloatingSafeModeToggle`): 20 renders each.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 120 | 120 – 120 | 0.0% |
| total renders | 9,680 | 9,680 – 9,680 | 0.0% |
| renders per commit | 80.7 | 80.7 – 80.7 | 0.0% |
| max renders in one commit | 207 | 207 – 207 | 0.0% |
| script time (ms) | 826 | 702 – 929 | 27.5% |
| task time (ms) | 1,013 | 862 – 1,114 | 24.9% |
| layout time (ms) | 16 | 14 – 17 | 16.1% |
| style recalc time (ms) | 2 | 2 – 2 | 10.0% |
| long tasks (≥ 50 ms) | 4 | 0 – 7 | 175.0% |
| long task total (ms) | 257 | 0 – 500 | 194.2% |
| heap used after GC (MB) | 82.1 | 82.0 – 82.1 | 0.2% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Primitive.div` | 440 |
| `Tooltip` | 360 |
| `TooltipContent` | 360 |
| `TooltipProvider` | 340 |
| `Presence` | 320 |
| `HugeiconsIcon` | 300 |
| `Icon` | 300 |
| `SelectItem` | 280 |
| `Popper` | 260 |
| `PopperAnchor` | 260 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 360 |
| `TooltipProvider props:children` | 340 |
| `Presence props:children` | 320 |
| `HugeiconsIcon parent-same-props<-Icon` | 280 |
| `SelectItem props:onMouseDown,children` | 280 |
| `Primitive.div props:children` | 260 |
| `Primitive.div.Slot props:children` | 260 |
| `Primitive.div.SlotClone props:children` | 260 |
| `Popper props:children` | 220 |
| `PopperAnchor props:children` | 220 |

## builder-block-menu-search · baseline — median of 3 runs

Added with SECRT-3012: `dev` app code at `8eb406275`, same machine and settings. Results come from the real backend's block search, so the counts depend on the blocks the backend has. On this code each new page of results re-renders every `Block` row already in the list (`Block parent-same-props<-InfiniteScroll`). One run in three loaded one page fewer within the window (1,461 renders).

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 64 | 59 – 64 | 7.8% |
| total renders | 1,843 | 1,461 – 1,843 | 20.7% |
| renders per commit | 28.8 | 24.8 – 28.8 | 14.0% |
| max renders in one commit | 211 | 179 – 211 | 15.2% |
| script time (ms) | 268 | 195 – 274 | 29.4% |
| task time (ms) | 937 | 810 – 992 | 19.5% |
| layout time (ms) | 11 | 9 – 13 | 28.8% |
| style recalc time (ms) | 41 | 36 – 41 | 13.3% |
| long tasks (≥ 50 ms) | 0 | 0 – 1 | 0.0% |
| long task total (ms) | 0 | 0 – 78 | 0.0% |
| heap used after GC (MB) | 84.1 | 83.6 – 85.3 | 2.0% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Button` | 302 |
| `HugeiconsIcon` | 292 |
| `Icon` | 292 |
| `Block` | 256 |
| `u0` | 120 |
| `AnimatePresence` | 42 |
| `FilterChip` | 35 |
| `PopChild` | 35 |
| `PopChildMeasure` | 35 |
| `PresenceChild` | 35 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `HugeiconsIcon parent-same-props<-Icon` | 239 |
| `Icon parent-same-props<-Button` | 218 |
| `Block parent-same-props<-InfiniteScroll` | 208 |
| `Button props:onDragStart,onClick,children` | 208 |
| `u0 context` | 120 |
| `Button mount` | 54 |
| `HugeiconsIcon mount` | 53 |
| `Icon mount` | 53 |
| `Block mount` | 48 |
| `AnimatePresence state#5` | 30 |

## builder-load-graph · baseline — median of 3 runs

Added with SECRT-3012: `dev` app code at `8eb406275`, same machine and settings. The window opens on the 1-node workflow and closes 2 s after the last of the 100 nodes is on the canvas. On this code the switch also shows the "Unsaved changes found" popup for the new workflow.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 71 | 71 – 74 | 4.2% |
| total renders | 80,164 | 80,164 – 80,193 | 0.0% |
| renders per commit | 1,129.1 | 1,083.7 – 1,129.1 | 4.0% |
| max renders in one commit | 25,160 | 25,160 – 25,160 | 0.0% |
| script time (ms) | 3,100 | 3,022 – 3,244 | 7.2% |
| task time (ms) | 3,706 | 3,616 – 3,881 | 7.1% |
| layout time (ms) | 51 | 48 – 52 | 7.8% |
| style recalc time (ms) | 111 | 107 – 111 | 3.4% |
| long tasks (≥ 50 ms) | 4 | 4 – 4 | 0.0% |
| long task total (ms) | 3,062 | 2,989 – 3,202 | 7.0% |
| heap used after GC (MB) | 105.0 | 104.8 – 108.1 | 3.1% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `TooltipProvider` | 3,700 |
| `Text` | 3,370 |
| `Tooltip` | 3,103 |
| `TooltipContent` | 3,103 |
| `Primitive.div` | 3,075 |
| `HugeiconsIcon` | 3,017 |
| `Icon` | 3,017 |
| `Presence` | 2,825 |
| `Popper` | 2,823 |
| `PopperProvider` | 2,823 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `TooltipProvider props:children` | 2,120 |
| `Presence props:children` | 2,044 |
| `HugeiconsIcon parent-same-props<-Icon` | 1,706 |
| `Primitive.div props:children` | 1,663 |
| `Primitive.div.Slot props:children` | 1,663 |
| `Primitive.div.SlotClone props:children` | 1,663 |
| `Tooltip props:children` | 1,660 |
| `PopperAnchor props:children` | 1,417 |
| `Text mount` | 1,260 |
| `Text props:children` | 1,154 |

## copilot-open-long-thread · baseline — median of 3 runs

Measured on `dev` at `a7fe56d05` (2026-10-20), before the transcript was windowed (SECRT-3011), with `--channel chrome`. Opening the 2,000-message chat mounts 924 rows in one commit; the longest task is about 2.5 s.

| metric | median | range | spread |
| --- | --- | --- | --- |
| commits | 27 | 27 – 28 | 3.7% |
| total renders | 59,027 | 59,025 – 59,042 | 0.0% |
| renders per commit | 2,186.1 | 2,108.6 – 2,186.2 | 3.5% |
| max renders in one commit | 38,844 | 38,844 – 38,844 | 0.0% |
| script time (ms) | 2,546 | 2,537 – 2,708 | 6.7% |
| task time (ms) | 3,337 | 3,240 – 3,387 | 4.4% |
| layout time (ms) | 72 | 71 – 80 | 12.5% |
| style recalc time (ms) | 209 | 194 – 230 | 16.9% |
| long tasks (≥ 50 ms) | 8 | 8 – 8 | 0.0% |
| long task total (ms) | 3,007 | 2,986 – 3,155 | 5.6% |
| heap used after GC (MB) | 119.8 | 119.7 – 119.8 | 0.1% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 3,337 |
| `TooltipContent` | 3,337 |
| `Streamdown` | 2,772 |
| `Presence` | 2,303 |
| `TooltipProvider` | 2,238 |
| `Popper` | 2,235 |
| `PopperAnchor` | 2,235 |
| `PopperProvider` | 2,235 |
| `Primitive.button` | 2,235 |
| `Primitive.button.Slot` | 2,235 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip mount` | 2,194 |
| `TooltipContent mount` | 2,194 |
| `Streamdown state#1` | 1,848 |
| `Presence props:children` | 1,196 |
| `Block mount` | 1,177 |
| `Ie mount` | 1,177 |
| `HugeiconsIcon mount` | 1,146 |
| `Icon mount` | 1,146 |
| `Primitive.div props:children` | 1,136 |
| `Primitive.div.Slot props:children` | 1,136 |
