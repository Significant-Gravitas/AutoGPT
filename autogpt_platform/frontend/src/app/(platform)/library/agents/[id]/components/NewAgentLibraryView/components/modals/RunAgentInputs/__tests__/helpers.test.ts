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
  ])("reads $value in $zone as $day", ({ value, zone, day }) => {
    process.env.TZ = zone;
    expect(toCalendarDate(value)).toBe(day);
  });
});

describe("withCalendarDates", () => {
  it("reads only a Date input's value as a day, and keeps one it can't read", () => {
    process.env.TZ = "Pacific/Auckland";
    const schema: { properties: Record<string, BlockIOSubSchema> } = {
      properties: {
        date: { type: "string", format: "date" },
        optional_date: {
          anyOf: [{ type: "string", format: "date" }, { type: "null" }],
        },
        unreadable_date: { type: "string", format: "date" },
        note: { type: "string" },
      },
    };

    expect(
      withCalendarDates(schema, {
        date: "2026-03-14T11:00:00.000Z",
        optional_date: "2026-03-14T11:00:00.000Z",
        unreadable_date: "soon",
        note: "2026-03-14T11:00:00.000Z",
        not_in_schema: "2026-03-14T11:00:00.000Z",
        // Named after members every object inherits, not fields.
        constructor: "2026-03-14T11:00:00.000Z",
        toString: "2026-03-14T11:00:00.000Z",
      }),
    ).toEqual({
      date: "2026-03-15",
      optional_date: "2026-03-15",
      unreadable_date: "soon",
      note: "2026-03-14T11:00:00.000Z",
      not_in_schema: "2026-03-14T11:00:00.000Z",
      constructor: "2026-03-14T11:00:00.000Z",
      toString: "2026-03-14T11:00:00.000Z",
    });
  });
});
