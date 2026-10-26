import type { Page } from "@playwright/test";
import type { RankedEntry } from "./report";

export interface CounterReport {
  commits: number;
  totalRenders: number;
  rendersPerCommit: number;
  maxRendersPerCommit: number;
  components: RankedEntry[];
  causes: RankedEntry[];
}

interface RenderCounter {
  start(): void;
  stop(): void;
  reset(): void;
  report(): CounterReport;
}

type CounterWindow = Window & { __renderCounter?: RenderCounter };

export async function assertCounterLoaded(page: Page) {
  const present = await page.evaluate(
    () => typeof (window as CounterWindow).__renderCounter === "object",
  );
  if (!present) {
    throw new Error(
      "window.__renderCounter is missing. Start the dev server with NEXT_PUBLIC_RENDER_COUNTER=1 pnpm dev (see docs/engineering/frontend-perf-measurement.md).",
    );
  }
}

export function counter(page: Page, action: "start" | "stop" | "reset") {
  return page.evaluate(
    (name) => (window as CounterWindow).__renderCounter?.[name](),
    action,
  );
}

export async function counterReport(page: Page) {
  const report = await page.evaluate(() =>
    (window as CounterWindow).__renderCounter?.report(),
  );
  if (!report) throw new Error("window.__renderCounter disappeared mid-run");
  return report;
}
