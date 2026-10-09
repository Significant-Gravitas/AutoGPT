import { describe, expect, it } from "vitest";
import {
  describeTrialTimeLeft,
  formatTrialDays,
  formatTrialEndTime,
  getTrialDaysLeft,
} from "../helpers";

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
});
