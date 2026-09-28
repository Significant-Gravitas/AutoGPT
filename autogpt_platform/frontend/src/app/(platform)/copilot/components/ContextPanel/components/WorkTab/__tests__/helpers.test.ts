import { describe, expect, it } from "vitest";
import { isBudgetError } from "../helpers";

describe("isBudgetError", () => {
  it.each([
    "Weekly budget cap reached ($5.00)",
    "Delegation cap reached",
    "Spend limit hit for this chat",
    "The cost cap was exceeded",
    "Cap reached before the draft finished",
  ])("offers raising the budget for %j", (error) => {
    expect(isBudgetError(error)).toBe(true);
  });

  it.each([
    "Rate limit exceeded, try again later",
    "Context length limit reached",
    "Wrote a recap of the meeting",
    "Could not spend time on it",
    "",
  ])("does not for %j", (error) => {
    expect(isBudgetError(error)).toBe(false);
  });

  it("does not without an error", () => {
    expect(isBudgetError(null)).toBe(false);
  });
});
