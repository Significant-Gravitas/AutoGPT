import { describe, expect, test } from "vitest";
import { getCadenceLabel } from "./helpers";

describe("getCadenceLabel", () => {
  test.each([
    ["40 7 * * *", "Runs every day at 07:40"],
    ["0 9 * * 1,3", "Runs every Monday, Wednesday at 09:00"],
    ["0 9 1,15 * *", "Runs on day 1, 15 of every month at 09:00"],
    ["0 9 1 * 1", "Runs on a schedule"],
    ["not a cron", "Runs on a schedule"],
  ])("%s -> %s", (cron, label) => {
    expect(getCadenceLabel(cron)).toBe(label);
  });
});
