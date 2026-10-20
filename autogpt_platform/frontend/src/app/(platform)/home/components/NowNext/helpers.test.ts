import { describe, expect, it } from "vitest";
import { formatRunningFor, formatUntil } from "./helpers";

const NOW = new Date("2026-08-09T09:00:00Z");

describe("formatUntil", () => {
  it("renders imminent and same-day runs distinctly", () => {
    expect(formatUntil(new Date(NOW.getTime() + 30_000), NOW)).toBe("now");
    expect(formatUntil(new Date(NOW.getTime() + 90 * 60_000), NOW)).toBe(
      "in 1h 30m",
    );
  });

  it("reads an overdue run as due now rather than a negative countdown", () => {
    expect(formatUntil(new Date(NOW.getTime() - 5 * 60_000), NOW)).toBe("now");
  });
});

describe("formatRunningFor", () => {
  it("scales from minutes to hours", () => {
    const startedAt = (minutes: number) =>
      new Date(NOW.getTime() - minutes * 60_000);
    expect(formatRunningFor(startedAt(0), NOW)).toBe(
      "Running for less than a minute",
    );
    expect(formatRunningFor(startedAt(4), NOW)).toBe("Running for 4m");
    expect(formatRunningFor(startedAt(60), NOW)).toBe("Running for 1h");
    expect(formatRunningFor(startedAt(75), NOW)).toBe("Running for 1h 15m");
    expect(formatRunningFor(null, NOW)).toBeNull();
  });
});
