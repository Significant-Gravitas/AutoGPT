import type { BrowserContext, CDPSession, Page } from "@playwright/test";

const TRACE_CATEGORIES =
  "devtools.timeline,disabled-by-default-devtools.timeline";
const LONG_TASK_US = 50_000;
const GC_SETTLE_MS = 500;

export interface TraceEvent {
  name: string;
  ph: string;
  pid: number;
  tid: number;
  ts: number;
  dur?: number;
  args?: {
    name?: string;
    data?: { frames?: { processId?: number; parent?: string }[] };
  };
}

type Metrics = Record<string, number>;

export async function openCDP(context: BrowserContext, page: Page) {
  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable");
  await cdp.send("HeapProfiler.enable");
  return cdp;
}

export async function readMetrics(cdp: CDPSession): Promise<Metrics> {
  const { metrics } = await cdp.send("Performance.getMetrics");
  return Object.fromEntries(metrics.map(({ name, value }) => [name, value]));
}

function threadKey(event: TraceEvent) {
  return `${event.pid}:${event.tid}`;
}

/**
 * Main threads of the page's renderer: `CrRendererMain` in the process that
 * hosts the top-level frame (falling back to every renderer main thread if
 * the trace does not say which process that is).
 */
function mainThreads(events: TraceEvent[]) {
  const frames =
    events.find((event) => event.name === "TracingStartedInBrowser")?.args?.data
      ?.frames ?? [];
  const pagePIDs = new Set(
    frames.filter((frame) => !frame.parent).map((frame) => frame.processId),
  );
  return new Set(
    events
      .filter(
        (event) =>
          event.ph === "M" &&
          event.name === "thread_name" &&
          event.args?.name === "CrRendererMain" &&
          (pagePIDs.size === 0 || pagePIDs.has(event.pid)),
      )
      .map(threadKey),
  );
}

export function summarizeLongTasks(events: TraceEvent[]) {
  const threads = mainThreads(events);
  const long = events.filter(
    (event) =>
      event.name === "RunTask" &&
      event.ph === "X" &&
      (event.dur ?? 0) >= LONG_TASK_US &&
      threads.has(threadKey(event)),
  );
  return {
    count: long.length,
    totalMs: long.reduce((sum, event) => sum + (event.dur ?? 0), 0) / 1000,
  };
}

function durationMs(before: Metrics, after: Metrics, name: string) {
  return ((after[name] ?? 0) - (before[name] ?? 0)) * 1000;
}

export async function startWindow(cdp: CDPSession, profile: boolean) {
  const events: TraceEvent[] = [];
  function onData({ value }: { value: object[] }) {
    events.push(...(value as TraceEvent[]));
  }
  cdp.on("Tracing.dataCollected", onData);
  await cdp.send("Tracing.start", {
    categories: TRACE_CATEGORIES,
    transferMode: "ReportEvents",
  });
  if (profile) {
    await cdp.send("Profiler.enable");
    await cdp.send("Profiler.start");
  }
  const before = await readMetrics(cdp);
  return { events, before, profile, onData };
}

type OpenWindow = Awaited<ReturnType<typeof startWindow>>;

export async function stopWindow(cdp: CDPSession, window: OpenWindow) {
  const after = await readMetrics(cdp);
  const complete = new Promise<void>((resolve) =>
    cdp.once("Tracing.tracingComplete", () => resolve()),
  );
  await cdp.send("Tracing.end");
  await complete;
  cdp.off("Tracing.dataCollected", window.onData);
  const cpuProfile = window.profile
    ? (await cdp.send("Profiler.stop")).profile
    : undefined;
  const longTasks = summarizeLongTasks(window.events);
  return {
    scriptMs: durationMs(window.before, after, "ScriptDuration"),
    taskMs: durationMs(window.before, after, "TaskDuration"),
    layoutMs: durationMs(window.before, after, "LayoutDuration"),
    recalcStyleMs: durationMs(window.before, after, "RecalcStyleDuration"),
    longTaskCount: longTasks.count,
    longTaskMs: longTasks.totalMs,
    cpuProfile,
  };
}

export async function heapAfterGC(cdp: CDPSession, page: Page) {
  await cdp.send("HeapProfiler.collectGarbage");
  await page.waitForTimeout(GC_SETTLE_MS);
  const metrics = await readMetrics(cdp);
  return (metrics.JSHeapUsedSize ?? 0) / (1024 * 1024);
}
