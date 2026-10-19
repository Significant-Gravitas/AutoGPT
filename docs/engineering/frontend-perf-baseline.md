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
