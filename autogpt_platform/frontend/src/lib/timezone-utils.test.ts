import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
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
