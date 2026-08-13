import { afterEach, describe, expect, it, vi } from "vitest";
import { formatBriefingDate, isInternalLink } from "./helpers";

describe("formatBriefingDate", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("labels today's briefing as this morning", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-07T09:00:00Z"));

    expect(formatBriefingDate("2026-08-07")).toBe("This morning");
  });

  it("does not shift a date-only string a day back", () => {
    // The API sends "2026-08-07" with no time part, so the client's date
    // transformer leaves it a string. `new Date(...)` would read it as UTC
    // midnight — i.e. Aug 6 in the evening for anyone west of UTC.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-09T12:00:00Z"));

    expect(formatBriefingDate("2026-08-07")).toBe("August 7");
  });

  it("accepts an already-parsed Date", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-09T12:00:00Z"));

    expect(formatBriefingDate(new Date(2026, 7, 7, 9, 0))).toBe("August 7");
  });
});

describe("isInternalLink", () => {
  it("accepts an app-relative path", () => {
    expect(isInternalLink("/library/agents/lib-1")).toBe(true);
  });

  it.each(["https://evil.example", "javascript:alert(1)", "//evil.example"])(
    "rejects %s",
    (link) => {
      expect(isInternalLink(link)).toBe(false);
    },
  );
});
