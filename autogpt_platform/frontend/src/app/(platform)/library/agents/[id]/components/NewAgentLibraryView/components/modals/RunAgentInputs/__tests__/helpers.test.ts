import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import { afterEach, describe, expect, it } from "vitest";
import { toCalendarDate, withCalendarDates } from "../helpers";

// Deleting TZ leaves Node in the last zone set, so restore the zone by name.
const START_ZONE =
  process.env.TZ ?? Intl.DateTimeFormat().resolvedOptions().timeZone;
afterEach(() => {
  process.env.TZ = START_ZONE;
});

describe("toCalendarDate", () => {
  it.each([
    { value: "2026-03-15", zone: "Pacific/Auckland", day: "2026-03-15" },
    // Each is midnight as written, which the backend reads as 15 March;
    // Los Angeles is still on 14 March at each moment.
    {
      value: "2026-03-15T00:00:00.000Z",
      zone: "America/Los_Angeles",
      day: "2026-03-15",
    },
    {
      value: "2026-03-15T00:00:00+13:00",
      zone: "America/Los_Angeles",
      day: "2026-03-15",
    },
    {
      value: "2026-03-15T00:00",
      zone: "America/Los_Angeles",
      day: "2026-03-15",
    },
    // A two-digit offset is midnight as written too.
    {
      value: "2026-03-15T00:00:00+13",
      zone: "America/Los_Angeles",
      day: "2026-03-15",
    },
    {
      value: "2026-03-15T00:00-08",
      zone: "Pacific/Auckland",
      day: "2026-03-15",
    },
    { value: "2024-02-29", zone: "UTC", day: "2024-02-29" },
    // Local midnight on 15 March, as the run form sent it before #17305.
    {
      value: "2026-03-14T11:00:00.000Z",
      zone: "Pacific/Auckland",
      day: "2026-03-15",
    },
    {
      value: "2026-03-15T07:00:00.000Z",
      zone: "America/Los_Angeles",
      day: "2026-03-15",
    },
    { value: "next Tuesday", zone: "UTC", day: "" },
    { value: "2026", zone: "UTC", day: "" },
    { value: null, zone: "UTC", day: "" },
    { value: 20260315, zone: "UTC", day: "" },
    // Days that don't exist, which would otherwise pass as written or roll
    // over into the next month.
    { value: "2026-02-30", zone: "UTC", day: "" },
    { value: "2026-13-45", zone: "UTC", day: "" },
    { value: "2026-02-29", zone: "UTC", day: "" },
    { value: "2026-02-30T00:00:00Z", zone: "UTC", day: "" },
    { value: "2026-02-30T10:00:00Z", zone: "UTC", day: "" },
    // Offsets that don't exist: hours past 23, minutes past 59.
    { value: "2026-03-15T00:00+99", zone: "UTC", day: "" },
    { value: "2026-03-15T00:00:00+24:00", zone: "UTC", day: "" },
    { value: "2026-03-15T00:00:00+13:60", zone: "UTC", day: "" },
  ])("reads $value in $zone as $day", ({ value, zone, day }) => {
    process.env.TZ = zone;
    expect(toCalendarDate(value)).toBe(day);
  });
});

describe("withCalendarDates", () => {
  const schema: { properties: Record<string, BlockIOSubSchema> } = {
    properties: {
      date: { type: "string", format: "date" },
      optional_date: {
        anyOf: [{ type: "string", format: "date" }, { type: "null" }],
      },
      note: { type: "string" },
    },
  };

  it("reads only a Date input's value as a day", () => {
    process.env.TZ = "Pacific/Auckland";

    expect(
      withCalendarDates(schema, {
        date: "2026-03-14T11:00:00.000Z",
        optional_date: "2026-03-14T11:00:00.000Z",
        note: "2026-03-14T11:00:00.000Z",
        not_in_schema: "2026-03-14T11:00:00.000Z",
        // Named after members every object inherits, not fields.
        constructor: "2026-03-14T11:00:00.000Z",
        toString: "2026-03-14T11:00:00.000Z",
      }),
    ).toEqual({
      date: "2026-03-15",
      optional_date: "2026-03-15",
      note: "2026-03-14T11:00:00.000Z",
      not_in_schema: "2026-03-14T11:00:00.000Z",
      constructor: "2026-03-14T11:00:00.000Z",
      toString: "2026-03-14T11:00:00.000Z",
    });
  });

  // The field shows nothing for these, so they must count as empty, and an
  // explicit null keeps a template's stored value from being merged back in.
  it.each(["soon", "2026-02-30", "2026-13-45", 20260315, { day: 15 }])(
    "seeds a Date input's value it can't show as a day as null: %j",
    (value) => {
      const seeded = withCalendarDates(schema, {
        date: value,
        optional_date: value,
        note: "kept",
      });

      expect(seeded).toStrictEqual({
        date: null,
        optional_date: null,
        note: "kept",
      });
    },
  );

  it.each([null, ""])("keeps an empty Date input's value: %j", (value) => {
    expect(withCalendarDates(schema, { date: value })).toStrictEqual({
      date: value,
    });
  });
});
