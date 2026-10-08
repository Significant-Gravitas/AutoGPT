import { describe, expect, it } from "vitest";

import {
  clampCustomIntervalValue,
  humanizeCronExpression,
  makeCronExpression,
  safeHumanizeCronExpression,
} from "./cron-expression-utils";

describe("humanizeCronExpression", () => {
  it("renders comma-separated weekdays", () => {
    expect(humanizeCronExpression("0 9 * * 1,3,5")).toBe(
      "Every Monday, Wednesday, Friday at 09:00",
    );
  });

  it("renders weekday ranges (e.g. Mon-Fri) instead of Unknown(NaN)", () => {
    expect(humanizeCronExpression("0 9 * * 1-5")).toBe(
      "Every Monday, Tuesday, Wednesday, Thursday, Friday at 09:00",
    );
  });

  it("renders mixed ranges and lists", () => {
    expect(humanizeCronExpression("30 8 * * 1-3,5")).toBe(
      "Every Monday, Tuesday, Wednesday, Friday at 08:30",
    );
  });

  it("renders comma-separated months", () => {
    expect(humanizeCronExpression("0 12 1 1,6,12 *")).toBe(
      "Every year on the 1st day of January, June, December at 12:00",
    );
  });

  it("renders month ranges (e.g. Mar-May) without Unknown(NaN)", () => {
    expect(humanizeCronExpression("0 12 1 3-5 *")).toBe(
      "Every year on the 1st day of March, April, May at 12:00",
    );
  });
});

describe("safeHumanizeCronExpression", () => {
  it("uses a generic label for malformed cron expressions", () => {
    expect(safeHumanizeCronExpression("not-a-cron")).toBe("Scheduled");
  });
});

describe("makeCronExpression custom interval (#15275)", () => {
  function custom(unit: string, value: number) {
    return makeCronExpression({
      frequency: "custom",
      customInterval: { unit, value },
      minute: 15,
      hour: 9,
    });
  }

  it.each([
    ["minutes", 5, "*/5 * * * *"],
    ["minutes", 59, "*/59 * * * *"],
    ["minutes", 60, "0 * * * *"],
    ["minutes", 90, "0 * * * *"],
    ["hours", 6, "0 */6 * * *"],
    ["hours", 24, "0 0 * * *"],
    ["hours", 48, "0 0 * * *"],
    ["days", 2, "15 9 */2 * *"],
    ["days", 31, "15 9 */30 * *"],
    ["minutes", 0, "*/1 * * * *"],
    ["minutes", Number.NaN, "*/1 * * * *"],
    ["hours", -3, "0 */1 * * *"],
  ])("%s every %s → %s", (unit, value, expected) => {
    expect(custom(unit, value)).toBe(expected);
  });

  it("never emits a step at or above the field range", () => {
    for (const unit of ["minutes", "hours", "days"]) {
      for (let value = -1; value <= 100; value++) {
        const step = /\*\/(\d+)/.exec(custom(unit, value))?.[1];
        if (step === undefined) continue;
        const limit = { minutes: 60, hours: 24, days: 31 }[unit]!;
        expect(Number(step)).toBeGreaterThanOrEqual(1);
        expect(Number(step)).toBeLessThan(limit);
      }
    }
  });

  it("clamps input values per unit", () => {
    expect(clampCustomIntervalValue("minutes", 75)).toBe(60);
    expect(clampCustomIntervalValue("days", 45)).toBe(30);
    expect(clampCustomIntervalValue("hours", Number.NaN)).toBe(1);
    expect(clampCustomIntervalValue("hours", 3.7)).toBe(3);
  });
});
