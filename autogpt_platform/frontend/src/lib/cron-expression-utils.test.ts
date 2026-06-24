import { describe, expect, it } from "vitest";

import { humanizeCronExpression } from "./cron-expression-utils";

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
