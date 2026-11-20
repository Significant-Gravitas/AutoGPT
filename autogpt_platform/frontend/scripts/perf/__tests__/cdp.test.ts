import { describe, expect, it } from "vitest";
import { summarizeLongTasks, type TraceEvent } from "../lib/cdp";

function task(pid: number, tid: number, durMs: number): TraceEvent {
  return { name: "RunTask", ph: "X", pid, tid, ts: 0, dur: durMs * 1000 };
}

function threadName(pid: number, tid: number, name: string): TraceEvent {
  return { name: "thread_name", ph: "M", pid, tid, ts: 0, args: { name } };
}

describe("summarizeLongTasks", () => {
  it("counts tasks of 50 ms or more on the page's renderer main thread", () => {
    const events: TraceEvent[] = [
      {
        name: "TracingStartedInBrowser",
        ph: "I",
        pid: 1,
        tid: 1,
        ts: 0,
        args: {
          data: { frames: [{ processId: 7 }, { processId: 9, parent: "f" }] },
        },
      },
      threadName(7, 70, "CrRendererMain"),
      threadName(7, 71, "Compositor"),
      threadName(9, 90, "CrRendererMain"),
      task(7, 70, 80),
      task(7, 70, 50),
      task(7, 70, 49),
      task(7, 71, 200),
      task(9, 90, 300),
    ];
    expect(summarizeLongTasks(events)).toEqual({ count: 2, totalMs: 130 });
  });

  it("falls back to every renderer main thread without frame info", () => {
    const events = [threadName(3, 30, "CrRendererMain"), task(3, 30, 60)];
    expect(summarizeLongTasks(events)).toEqual({ count: 1, totalMs: 60 });
  });
});
