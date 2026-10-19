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

- **Streaming a reply re-renders the whole conversation.** In `copilot-stream-reply`, one reply costs about 210,000 renders and 5.7 s of script time over 56 commits. The biggest commits render more than 11,000 components: the 200 existing rows, mostly their tooltips, icons and Radix wrappers. Most of the cost comes after the stream ends, when the page refetches the session and re-renders the transcript several times. `copilot-idle` shows the same chat at rest: about 40 renders per commit.
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
| script time (ms) | 60 | 60 – 62 | 4.9% |
| task time (ms) | 250 | 239 – 259 | 7.7% |
| layout time (ms) | 0 | 0 – 0 | 0.0% |
| style recalc time (ms) | 0 | 0 – 0 | 0.0% |
| long tasks (≥ 50 ms) | 1 | 1 – 2 | 100.0% |
| long task total (ms) | 80 | 58 – 116 | 71.7% |
| heap used after GC (MB) | 56.4 | 55.9 – 56.4 | 1.0% |

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
| commits | 56 | 55 – 56 | 1.8% |
| total renders | 209,512 | 198,596 – 209,513 | 5.2% |
| renders per commit | 3,741.3 | 3,546.4 – 3,809.3 | 7.0% |
| max renders in one commit | 11,149 | 11,149 – 11,149 | 0.0% |
| script time (ms) | 5,725 | 5,344 – 5,788 | 7.8% |
| task time (ms) | 6,179 | 5,819 – 6,309 | 7.9% |
| layout time (ms) | 21 | 21 – 24 | 11.5% |
| style recalc time (ms) | 195 | 181 – 226 | 23.1% |
| long tasks (≥ 50 ms) | 18 | 17 – 18 | 5.6% |
| long task total (ms) | 5,876 | 5,506 – 5,980 | 8.1% |
| heap used after GC (MB) | 84.8 | 84.6 – 86.7 | 2.5% |

**Top components (median renders)**

| name | renders |
| --- | --- |
| `Tooltip` | 16,233 |
| `TooltipContent` | 16,229 |
| `HugeiconsIcon` | 9,064 |
| `Icon` | 9,064 |
| `Presence` | 8,305 |
| `Primitive.div` | 8,201 |
| `Popper` | 8,197 |
| `PopperAnchor` | 8,197 |
| `PopperProvider` | 8,197 |
| `Primitive.div.Slot` | 8,197 |

**Top render causes (median renders)**

| name | renders |
| --- | --- |
| `Tooltip props:children` | 16,186 |
| `HugeiconsIcon parent-same-props<-Icon` | 9,057 |
| `Presence props:children` | 8,299 |
| `Primitive.div props:children` | 8,192 |
| `Primitive.div.Slot props:children` | 8,190 |
| `Primitive.div.SlotClone props:children` | 8,190 |
| `PopperAnchor props:children` | 8,171 |
| `Popper props:children` | 8,166 |
| `PopperProvider props:children` | 8,166 |
| `TooltipProvider props:children` | 8,143 |

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
