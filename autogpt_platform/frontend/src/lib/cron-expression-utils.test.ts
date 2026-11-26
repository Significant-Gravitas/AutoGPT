import { describe, expect, it } from "vitest";

import {
  type CustomIntervalUnit,
  getCustomIntervalError,
  humanizeCronExpression,
  makeCronExpression,
  parseCustomInterval,
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
  function custom(unit: CustomIntervalUnit, value: number) {
    return makeCronExpression({
      frequency: "custom",
      customInterval: { unit, value },
      minute: 15,
      hour: 9,
    });
  }

  it.each([
    ["minutes", 1, "*/1 * * * *"],
    ["minutes", 59, "*/59 * * * *"],
    ["minutes", 60, "0 * * * *"],
    ["hours", 23, "0 */23 * * *"],
    ["hours", 24, "0 0 * * *"],
    ["days", 2, "15 9 */2 * *"],
    ["days", 30, "15 9 */30 * *"],
  ] as const)("%s every %s → %s", (unit, value, expected) => {
    expect(custom(unit, value)).toBe(expected);
  });

  it.each([
    ["minutes", 61],
    ["hours", 25],
    ["days", 31],
    ["minutes", 0],
    ["hours", -3],
    ["days", 1.5],
    ["minutes", Number.NaN],
  ] as const)("%s every %s has no cron", (unit, value) => {
    expect(custom(unit, value)).toBe("");
  });

  it("never emits a step at or above the field range", () => {
    const limits = { minutes: 60, hours: 24, days: 31 } as const;
    for (const unit of ["minutes", "hours", "days"] as const) {
      for (let value = -1; value <= 100; value++) {
        const step = /\*\/(\d+)/.exec(custom(unit, value))?.[1];
        if (step === undefined) continue;
        expect(Number(step)).toBeGreaterThanOrEqual(1);
        expect(Number(step)).toBeLessThan(limits[unit]);
      }
    }
  });
});

describe("custom interval field", () => {
  it.each([
    ["", Number.NaN],
    ["  ", Number.NaN],
    ["45", 45],
    ["007", 7],
    ["1.5", Number.NaN],
    ["1e1", Number.NaN],
    ["-3", Number.NaN],
  ])("parses %j as %s", (text, value) => {
    expect(parseCustomInterval(text)).toBe(value);
  });

  it("names the range of the unit when the value is outside it", () => {
    expect(getCustomIntervalError("minutes", 60)).toBeUndefined();
    expect(getCustomIntervalError("minutes", 61)).toBe("Enter 1 to 60 minutes");
    expect(getCustomIntervalError("hours", Number.NaN)).toBe(
      "Enter 1 to 24 hours",
    );
    expect(getCustomIntervalError("days", 31)).toBe("Enter 1 to 30 days");
  });
});
