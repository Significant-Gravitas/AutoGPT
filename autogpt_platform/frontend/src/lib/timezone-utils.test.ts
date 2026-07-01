import { describe, expect, it } from "vitest";

import {
  getTimezoneAbbreviation,
  getTimezoneDisplayName,
} from "./timezone-utils";

// These zones do not observe DST, so their short name is a fixed GMT offset
// year-round — making the assertions stable regardless of when the tests run.
describe("getTimezoneAbbreviation", () => {
  it("returns the GMT offset for whole-hour offset zones", () => {
    expect(getTimezoneAbbreviation("Asia/Shanghai")).toBe("GMT+8");
    expect(getTimezoneAbbreviation("Asia/Tokyo")).toBe("GMT+9");
  });

  it("returns the GMT offset for half-hour offset zones", () => {
    expect(getTimezoneAbbreviation("Asia/Kolkata")).toBe("GMT+5:30");
  });

  it("does not fall back to the raw IANA id for offset zones", () => {
    for (const tz of ["Asia/Shanghai", "Asia/Kolkata", "Asia/Tokyo"]) {
      expect(getTimezoneAbbreviation(tz)).not.toBe(tz);
    }
  });
});

describe("getTimezoneDisplayName", () => {
  it("labels offset zones with their GMT offset, not the raw id", () => {
    expect(getTimezoneDisplayName("Asia/Tokyo")).toBe("Tokyo (GMT+9)");
    expect(getTimezoneDisplayName("Asia/Kolkata")).toBe("Kolkata (GMT+5:30)");
  });
});
