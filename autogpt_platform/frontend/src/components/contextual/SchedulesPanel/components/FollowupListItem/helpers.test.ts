import { describe, expect, test } from "vitest";
import { formatNextRunTitle } from "./helpers";

describe("formatNextRunTitle", () => {
  test("returns undefined when next_run_time is missing", () => {
    expect(formatNextRunTitle(null, "Asia/Jakarta")).toBeUndefined();
    expect(formatNextRunTitle(undefined, "Asia/Jakarta")).toBeUndefined();
    expect(formatNextRunTitle("", "Asia/Jakarta")).toBeUndefined();
  });

  test("returns undefined for an invalid ISO string", () => {
    expect(formatNextRunTitle("not-a-date", "Asia/Jakarta")).toBeUndefined();
  });

  test("formats the time in the schedule's user_timezone and labels the tz", () => {
    const out = formatNextRunTitle("2026-05-24T11:00:00+00:00", "Asia/Jakarta");
    expect(out).toBeDefined();
    expect(out).toContain("Asia/Jakarta");
    // 11:00 UTC is 18:00 in Asia/Jakarta (+07:00) — the formatted output must
    // reflect the schedule's tz, not the browser's tz.
    expect(out).toMatch(/6:00/);
  });

  test("falls back to the browser tz when the schedule timezone is empty", () => {
    const out = formatNextRunTitle("2026-05-24T11:00:00+00:00", "  ");
    expect(out).toBeDefined();
    const browserTz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    expect(out).toContain(browserTz);
  });

  test("falls back to UTC ISO when timeZone is invalid (Intl throws)", () => {
    const out = formatNextRunTitle(
      "2026-05-24T11:00:00+00:00",
      "Not/A_Real_Timezone",
    );
    expect(out).toBe("Next run: 2026-05-24T11:00:00.000Z (UTC)");
  });
});
