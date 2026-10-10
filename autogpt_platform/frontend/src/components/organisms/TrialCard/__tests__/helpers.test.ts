import { afterEach, describe, expect, it } from "vitest";
import {
  describeTrialTimeLeft,
  formatTrialDays,
  formatTrialEndTime,
  getTrialDaysLeft,
} from "../helpers";

const realTimeZone = process.env.TZ;
afterEach(() => {
  if (realTimeZone === undefined) delete process.env.TZ;
  else process.env.TZ = realTimeZone;
});

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;
const end = new Date("2030-09-17T15:00:00Z");

describe("getTrialDaysLeft", () => {
  it.each([
    [5 * DAY, 5],
    [4 * DAY + 1, 5],
    [DAY, 1],
    [3 * HOUR, 1],
    [0, 0],
    [-HOUR, 0],
  ])("rounds %ims left up to %i days", (left, days) => {
    expect(getTrialDaysLeft(end, end.getTime() - left)).toBe(days);
  });

  it("counts no days without an end", () => {
    expect(getTrialDaysLeft(null)).toBe(0);
  });
});

describe("formatTrialDays", () => {
  it.each([
    [1, "1 day"],
    [2, "2 days"],
    [5, "5 days"],
  ])("reads %i as %s", (days, label) => {
    expect(formatTrialDays(days)).toBe(label);
  });
});

describe("describeTrialTimeLeft", () => {
  it.each([
    [5 * DAY, 5],
    [DAY, 1],
    [DAY + HOUR, 2],
  ])("counts whole days when %ims are left", (left, days) => {
    expect(describeTrialTimeLeft(end, new Date(end.getTime() - left))).toEqual({
      kind: "days",
      days,
    });
  });

  it("reads under a day on the same local date as today", () => {
    const localEnd = new Date(2030, 8, 17, 14, 10);
    expect(
      describeTrialTimeLeft(localEnd, new Date(2030, 8, 17, 11, 10)),
    ).toEqual({ kind: "today", time: formatTrialEndTime(localEnd) });
  });

  it("reads under a day past local midnight as tomorrow", () => {
    const localEnd = new Date(2030, 8, 17, 1, 0);
    expect(
      describeTrialTimeLeft(localEnd, new Date(2030, 8, 16, 22, 0)),
    ).toEqual({ kind: "tomorrow", time: formatTrialEndTime(localEnd) });
  });

  it("dates an end under a day away that falls two calendar days out", () => {
    process.env.TZ = "America/New_York";
    const now = new Date(2026, 2, 7, 23, 30);
    const localEnd = new Date(2026, 2, 9, 0, 10);
    expect(localEnd.getTime() - now.getTime()).toBeLessThan(DAY);
    expect(describeTrialTimeLeft(localEnd, now)).toEqual({
      kind: "days",
      days: 1,
    });
  });

  it.each([
    ["earlier the same day", new Date(2030, 8, 17, 14, 0)],
    ["just before midnight", new Date(2030, 8, 16, 23, 58)],
    ["right now", new Date(2030, 8, 17, 15, 0)],
  ])("never reads an end %s as today", (_, localEnd) => {
    expect(
      describeTrialTimeLeft(localEnd, new Date(2030, 8, 17, 15, 0)),
    ).toEqual({ kind: "ended" });
  });
});
