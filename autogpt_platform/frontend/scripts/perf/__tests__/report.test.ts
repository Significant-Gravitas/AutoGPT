import fs from "fs";
import os from "os";
import path from "path";
import { describe, expect, it } from "vitest";
import {
  formatComparison,
  formatDelta,
  loadRuns,
  median,
  safeLabel,
  spreadPercent,
  summarize,
  writeRun,
  type RunResult,
} from "../lib/report";

function run(label: string, index: number, renders: number): RunResult {
  return {
    scenario: "copilot-stream-reply",
    label,
    run: index,
    startedAt: `2026-01-05T09:00:0${index}.000Z`,
    gitSha: "abcdef1234567890",
    baseURL: "http://localhost:3000",
    windowMs: 1000,
    metrics: {
      commits: 10,
      totalRenders: renders,
      rendersPerCommit: renders / 10,
      maxRendersPerCommit: renders / 2,
      scriptMs: renders / 4,
      taskMs: 0,
      layoutMs: 0,
      recalcStyleMs: 0,
      longTaskCount: 1,
      longTaskMs: 60,
      heapUsedMB: 80,
    },
    components: [
      { key: "TimelineRow", count: renders / 2 },
      { key: "ChatInput", count: index },
    ],
    causes: [{ key: "TimelineRow props:message", count: renders / 2 }],
    pageErrors: 0,
  };
}

describe("median", () => {
  it("takes the middle value, or the mean of the two middle values", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBe(0);
  });
});

describe("spreadPercent", () => {
  it("is the range as a share of the median", () => {
    expect(spreadPercent([95, 100, 105])).toBeCloseTo(10);
    expect(spreadPercent([7])).toBe(0);
  });
});

describe("formatDelta", () => {
  it("shows the signed difference and percentage", () => {
    expect(formatDelta(200, 50, 0)).toBe("-150 (-75.0%)");
    expect(formatDelta(10, 12.5, 1)).toBe("+2.5 (+25.0%)");
    expect(formatDelta(0, 0, 0)).toBe("0");
    expect(formatDelta(0, 3, 0)).toBe("+3");
  });
});

describe("summarize", () => {
  it("takes per-metric medians and median counts per component", () => {
    const summary = summarize([
      run("before", 1, 100),
      run("before", 2, 300),
      run("before", 3, 200),
    ]);
    expect(summary.metrics.totalRenders).toBe(200);
    expect(summary.components[0]).toEqual({ key: "TimelineRow", count: 100 });
    expect(summary.components[1]).toEqual({ key: "ChatInput", count: 2 });
  });
});

describe("formatComparison", () => {
  it("prints both medians with the delta, rankings and raw runs", () => {
    const output = formatComparison({
      scenario: "copilot-stream-reply",
      a: { label: "before", runs: [run("before", 1, 2000)] },
      b: { label: "after", runs: [run("after", 1, 200)] },
    });
    expect(output).toContain("| metric | before | after | Δ |");
    expect(output).toContain(
      "| total renders | 2,000 | 200 | -1,800 (-90.0%) |",
    );
    expect(output).toContain("| `TimelineRow` | 1,000 | 100 | -900 (-90.0%) |");
    expect(output).toContain("Raw runs");
  });
});

describe("run files", () => {
  it("round-trips runs and filters them by label", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "perf-report-"));
    writeRun(dir, run("before", 2, 10));
    writeRun(dir, run("before", 1, 20));
    writeRun(dir, run("before-x", 1, 30));
    const loaded = loadRuns(dir, "before");
    expect(loaded.map((result) => result.run)).toEqual([1, 2]);
    expect(loadRuns(path.join(dir, "missing"), "before")).toEqual([]);
    fs.rmSync(dir, { recursive: true });
  });

  it("rejects labels that are not safe file name parts", () => {
    expect(safeLabel("before-fix_2")).toBe("before-fix_2");
    expect(() => safeLabel("../before")).toThrow();
  });
});
