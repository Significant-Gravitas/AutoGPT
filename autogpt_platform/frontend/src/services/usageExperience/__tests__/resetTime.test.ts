import { describe, expect, it } from "vitest";
import {
  ceilToMinute,
  formatExactReset,
  formatWeekdayDate,
  needsDate,
} from "../resetTime";

// Local times, so the copy reads the same in any time zone.
const NOW = new Date(2026, 10, 4, 20, 31); // Wednesday 8:31 PM
const DAY_MS = 24 * 60 * 60 * 1000;

describe("a reset shown to the minute", () => {
  it("is rounded up, so the time shown is never before the reset", () => {
    expect(ceilToMinute(new Date(2026, 10, 12, 8, 30, 56))).toEqual(
      new Date(2026, 10, 12, 8, 31),
    );
    expect(ceilToMinute(new Date(2026, 10, 12, 8, 30, 0, 1))).toEqual(
      new Date(2026, 10, 12, 8, 31),
    );
  });

  it("is left alone when it is already on the minute", () => {
    const onTheMinute = new Date(2026, 10, 12, 8, 30);
    expect(ceilToMinute(onTheMinute)).toEqual(onTheMinute);
  });

  it("rolls over the hour and the day", () => {
    expect(ceilToMinute(new Date(2026, 10, 12, 23, 59, 30))).toEqual(
      new Date(2026, 10, 13, 0, 0),
    );
  });
});

describe("whether a weekday alone says which day", () => {
  it.each([
    [1, false],
    [5.9, false],
    [6, true],
    [7.75, true],
  ])("a reset %s days away needs its date: %s", (days, expected) => {
    const reset = new Date(NOW.getTime() + days * DAY_MS);
    expect(needsDate(reset, NOW)).toBe(expected);
  });
});

describe("the exact reset on the billing page", () => {
  it("gives the weekday, date and time in the reader's own time zone", () => {
    expect(formatExactReset(new Date(2026, 10, 12, 8, 30, 56))).toBe(
      "Thu, Nov 12, 8:31 AM",
    );
  });

  it("reads an ISO string like a date", () => {
    const reset = new Date(2026, 10, 19, 20, 30, 56);
    expect(formatExactReset(reset.toISOString())).toBe("Thu, Nov 19, 8:31 PM");
  });

  it.each([null, undefined, "", "not-a-date"])(
    "has nothing to say for %s",
    (value) => {
      expect(formatExactReset(value)).toBeNull();
    },
  );

  it("names a day with its date", () => {
    expect(formatWeekdayDate(new Date(2026, 10, 12, 8, 31))).toBe(
      "Thu, Nov 12",
    );
  });
});
