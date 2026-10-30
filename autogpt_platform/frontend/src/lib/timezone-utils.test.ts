import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  formatInTimezone,
  formatScheduleTime,
  getTimezoneAbbreviation,
  getTimezoneDisplayName,
} from "./timezone-utils";

// A midwinter and a midsummer instant, used to pin DST-observing zones to a
// known standard-time / summer-time offset regardless of when the tests run.
const WINTER = new Date("2024-01-15T12:00:00Z");
const SUMMER = new Date("2024-07-15T12:00:00Z");

describe("getTimezoneAbbreviation", () => {
  // These zones do not observe DST, so their short name is a fixed GMT offset
  // year-round — making the assertions stable regardless of when the tests run.
  it("returns the GMT offset for whole-hour offset zones", () => {
    expect(getTimezoneAbbreviation("Asia/Shanghai")).toMatch(
      /^GMT\+0?8(?::00)?$/,
    );
    expect(getTimezoneAbbreviation("Asia/Tokyo")).toMatch(/^GMT\+0?9(?::00)?$/);
  });

  it("returns the GMT offset for half-hour offset zones", () => {
    expect(getTimezoneAbbreviation("Asia/Kolkata")).toMatch(/^GMT\+0?5:30$/);
  });

  it("does not fall back to the raw IANA id for offset zones", () => {
    for (const tz of ["Asia/Shanghai", "Asia/Kolkata", "Asia/Tokyo"]) {
      expect(getTimezoneAbbreviation(tz)).not.toBe(tz);
    }
  });

  // DST-observing zones resolve to standard time in winter and summer time in
  // summer, so the clock is pinned to make each abbreviation deterministic.
  describe("DST-observing zones", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("returns PST in winter and PDT in summer for America/Los_Angeles", () => {
      vi.setSystemTime(WINTER);
      expect(getTimezoneAbbreviation("America/Los_Angeles")).toBe("PST");

      vi.setSystemTime(SUMMER);
      expect(getTimezoneAbbreviation("America/Los_Angeles")).toBe("PDT");
    });

    // en-US has no short abbreviation for Central European time, so the
    // formatter falls back to the GMT offset (CET -> GMT+1, CEST -> GMT+2)
    // rather than emitting "CET"/"CEST".
    it("returns the GMT offset for Central European Time / Summer Time", () => {
      vi.setSystemTime(WINTER);
      expect(getTimezoneAbbreviation("Europe/Berlin")).toMatch(
        /^GMT\+0?1(?::00)?$/,
      );

      vi.setSystemTime(SUMMER);
      expect(getTimezoneAbbreviation("Europe/Berlin")).toMatch(
        /^GMT\+0?2(?::00)?$/,
      );
    });
  });
});

describe("getTimezoneDisplayName", () => {
  it("labels offset zones with their GMT offset, not the raw id", () => {
    expect(getTimezoneDisplayName("Asia/Tokyo")).toMatch(
      /^Tokyo \(GMT\+0?9(?::00)?\)$/,
    );
    expect(getTimezoneDisplayName("Asia/Kolkata")).toMatch(
      /^Kolkata \(GMT\+0?5:30\)$/,
    );
  });

  describe("DST-observing zones", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("labels America/Los_Angeles with its seasonal abbreviation", () => {
      vi.setSystemTime(WINTER);
      expect(getTimezoneDisplayName("America/Los_Angeles")).toBe(
        "Los Angeles (PST)",
      );

      vi.setSystemTime(SUMMER);
      expect(getTimezoneDisplayName("America/Los_Angeles")).toBe(
        "Los Angeles (PDT)",
      );
    });

    it("labels Europe/Berlin with its GMT offset in the CEST period", () => {
      vi.setSystemTime(SUMMER);
      expect(getTimezoneDisplayName("Europe/Berlin")).toMatch(
        /^Berlin \(GMT\+0?2(?::00)?\)$/,
      );
    });
  });
});

describe("formatInTimezone", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    undefined,
    null,
    "",
    "not-a-date",
    "2024-13-50T12:00:00Z",
    new Date(NaN),
    new Date(Infinity),
    false,
    true,
    0,
    NaN,
    Infinity,
    1n,
    [],
    {},
    Symbol("date"),
    Object.create(Date.prototype),
    { getTime: () => 0 },
  ])(
    "returns a placeholder for invalid runtime date case %#",
    (date: unknown) => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      for (const timezone of ["UTC", "Invalid/Timezone"]) {
        expect(formatInTimezone(date as string, timezone)).toBe(
          "Not scheduled",
        );
      }
      expect(warn).not.toHaveBeenCalled();
    },
  );

  it("formats a valid ISO string in the given timezone", () => {
    expect(formatInTimezone("2024-01-15T12:00:00Z", "UTC")).toBe(
      "Jan 15, 2024, 12:00 PM",
    );
  });

  it.each([
    [WINTER, "America/New_York", "Jan 15, 2024, 07:00 AM"],
    [SUMMER, "America/New_York", "Jul 15, 2024, 08:00 AM"],
    [WINTER, "Asia/Kolkata", "Jan 15, 2024, 05:30 PM"],
    [new Date(0), "UTC", "Jan 1, 1970, 12:00 AM"],
  ] as const)("formats %s in %s", (date, timezone, expected) => {
    expect(formatInTimezone(date, timezone)).toBe(expected);
  });

  it("honors caller formatting options", () => {
    expect(
      formatInTimezone(WINTER, "America/New_York", {
        month: "long",
        hour12: false,
        timeZoneName: "short",
      }),
    ).toBe("January 15, 2024 at 07:00 EST");
  });

  it.each(["not-set", "Invalid/Timezone"])(
    "uses the local timezone for %s",
    (timezone) => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      const options: Intl.DateTimeFormatOptions = {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      };
      expect(formatInTimezone(WINTER, timezone, options)).toBe(
        new Intl.DateTimeFormat("en-US", options).format(WINTER),
      );
    },
  );

  it("rejects objects without invoking their date coercion methods", () => {
    const coerce = vi.fn(() => {
      throw new Error("Unexpected coercion");
    });
    const date = { getTime: coerce, [Symbol.toPrimitive]: coerce };
    expect(formatInTimezone(date as unknown as Date, "UTC")).toBe(
      "Not scheduled",
    );
    expect(coerce).not.toHaveBeenCalled();
  });

  it("keeps the schedule formatter safe and preserves its timezone label", () => {
    expect(formatScheduleTime("not-a-date", "UTC")).toBe("Not scheduled");
    expect(formatScheduleTime(WINTER, "UTC")).toBe(
      "Jan 15, 2024, 12:00 PM UTC",
    );
  });
});
