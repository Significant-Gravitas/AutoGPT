import { describe, expect, it } from "vitest";

import {
  EASE_OUT,
  formatCents,
  formatRelativeReset,
  formatShortDate,
} from "../helpers";

describe("EASE_OUT", () => {
  it("is a 4-tuple of cubic-bezier control points", () => {
    expect(EASE_OUT).toEqual([0.16, 1, 0.3, 1]);
  });
});

describe("formatCents", () => {
  it("formats positive cents as USD with thousands separators", () => {
    expect(formatCents(1234)).toBe("$12.34");
    expect(formatCents(343434)).toBe("$3,434.34");
    expect(formatCents(0)).toBe("$0.00");
  });

  it("uses the locale-correct negative prefix", () => {
    expect(formatCents(-100)).toBe("-$1.00");
    expect(formatCents(-1234)).toBe("-$12.34");
  });
});

describe("formatRelativeReset", () => {
  it("returns the empty placeholder when target is null/undefined/empty", () => {
    expect(formatRelativeReset(null)).toEqual({ prefix: "Resets", value: "—" });
    expect(formatRelativeReset(undefined)).toEqual({
      prefix: "Resets",
      value: "—",
    });
    expect(formatRelativeReset("")).toEqual({ prefix: "Resets", value: "—" });
  });

  it("returns the empty placeholder when the date is unparseable", () => {
    expect(formatRelativeReset("not-a-date")).toEqual({
      prefix: "Resets",
      value: "—",
    });
  });

  it("returns 'soon' when the target is in the past", () => {
    const past = new Date(Date.now() - 60_000);
    expect(formatRelativeReset(past)).toEqual({
      prefix: "Resets",
      value: "soon",
    });
  });

  it("returns hours+minutes when the target is within 24h", () => {
    const inAFewHours = new Date(Date.now() + 2 * 60 * 60 * 1000 + 5 * 60_000);
    const result = formatRelativeReset(inAFewHours);
    expect(result.prefix).toBe("Resets in");
    expect(result.value).toMatch(/^\d+h \d+m$/);
  });

  it("returns a weekday + time label when the target is more than 24h away", () => {
    const inAWeek = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
    const result = formatRelativeReset(inAWeek);
    expect(result.prefix).toBe("Resets");
    // Locale-dependent format, but always contains a weekday short name.
    expect(result.value.length).toBeGreaterThan(3);
  });

  it("accepts an ISO string as well as a Date instance", () => {
    const iso = new Date(Date.now() + 30 * 60_000).toISOString();
    const fromIso = formatRelativeReset(iso);
    expect(fromIso.prefix).toBe("Resets in");
  });
});

describe("formatShortDate", () => {
  it("returns the empty placeholder for null/undefined/empty input", () => {
    expect(formatShortDate(null)).toBe("—");
    expect(formatShortDate(undefined)).toBe("—");
    expect(formatShortDate("")).toBe("—");
  });

  it("returns the empty placeholder for unparseable input", () => {
    expect(formatShortDate("not-a-date")).toBe("—");
  });

  it("formats Date instances and ISO strings the same way", () => {
    const date = new Date(2026, 3, 28); // Apr 28, 2026 local
    const fromDate = formatShortDate(date);
    const fromIso = formatShortDate(date.toISOString());
    expect(fromDate).toBe(fromIso);
    expect(fromDate).toMatch(/Apr 28, 2026/);
  });
});
