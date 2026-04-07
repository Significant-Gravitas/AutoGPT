import { describe, expect, it } from "vitest";
import { formatResetTime } from "../UsagePanelContent";

describe("formatResetTime", () => {
  const now = new Date("2025-06-15T12:00:00Z");

  it("returns 'now' when reset time is in the past", () => {
    expect(formatResetTime("2025-06-15T11:00:00Z", now)).toBe("now");
  });

  it("returns minutes only when under 1 hour", () => {
    const result = formatResetTime("2025-06-15T12:30:00Z", now);
    expect(result).toBe("in 30m");
  });

  it("returns hours and minutes when under 24 hours", () => {
    const result = formatResetTime("2025-06-15T16:45:00Z", now);
    expect(result).toBe("in 4h 45m");
  });

  it("returns formatted date when over 24 hours away", () => {
    const result = formatResetTime("2025-06-17T00:00:00Z", now);
    expect(result).toMatch(/Tue/);
  });

  it("accepts a Date object for resetsAt", () => {
    const resetDate = new Date("2025-06-15T14:00:00Z");
    expect(formatResetTime(resetDate, now)).toBe("in 2h 0m");
  });
});
