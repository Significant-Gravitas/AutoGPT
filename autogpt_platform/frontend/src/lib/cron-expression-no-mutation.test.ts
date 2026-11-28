import { expect, it } from "vitest";
import {
  makeCronExpression,
  type CronExpressionParams,
} from "./cron-expression-utils";

it.each<[CronExpressionParams, number[], string]>([
  [
    { frequency: "weekly", minute: 0, hour: 9, days: [5, 1, 3] },
    [5, 1, 3],
    "0 9 * * 1,3,5",
  ],
  [
    { frequency: "monthly", minute: 0, hour: 9, days: [21, 2, 10] },
    [21, 2, 10],
    "0 9 2,10,21 * *",
  ],
  [
    { frequency: "yearly", minute: 0, hour: 9, months: [11, 2, 6] },
    [11, 2, 6],
    "0 9 1 2,6,11 *",
  ],
])(
  "formats a selection without reordering the caller's state",
  (params, original, expected) => {
    expect(makeCronExpression(params)).toBe(expected);
    const selection =
      "days" in params ? params.days : "months" in params ? params.months : [];
    expect(selection).toEqual(original);
  },
);
