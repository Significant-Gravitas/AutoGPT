import { describe, expect, test } from "vitest";
import { getPaidMilestones, getRenewalDate } from "../helpers";

function day(date: Date) {
  return [date.getFullYear(), date.getMonth() + 1, date.getDate()];
}

describe("getRenewalDate", () => {
  test("renews on the same day next month", () => {
    expect(day(getRenewalDate(new Date(2026, 9, 23), "month"))).toEqual([
      2026, 11, 23,
    ]);
  });

  test("clamps a month-end signup to the last day of a shorter month", () => {
    expect(day(getRenewalDate(new Date(2027, 0, 31), "month"))).toEqual([
      2027, 2, 28,
    ]);
    expect(day(getRenewalDate(new Date(2028, 0, 31), "month"))).toEqual([
      2028, 2, 29,
    ]);
  });

  test("rolls December into January of the next year", () => {
    expect(day(getRenewalDate(new Date(2026, 11, 15), "month"))).toEqual([
      2027, 1, 15,
    ]);
  });

  test("clamps a leap-day yearly signup to 28 February", () => {
    expect(day(getRenewalDate(new Date(2028, 1, 29), "year"))).toEqual([
      2029, 2, 28,
    ]);
  });
});

describe("getPaidMilestones", () => {
  test("starts today and renews a period later", () => {
    const now = new Date(2026, 9, 27, 9, 0, 0);
    const [today, renew] = getPaidMilestones({
      price: "$50",
      cadence: "month",
      now,
    });

    expect(today.title).toBe("Your plan starts");
    expect(today.detail).toBe(
      "You pay $50, plus any applicable tax. Full access starts straight away.",
    );
    expect(renew.date).toEqual(getRenewalDate(now, "month"));
    expect(renew.detail).toBe("$50 a month until you cancel in Settings.");
  });
});
