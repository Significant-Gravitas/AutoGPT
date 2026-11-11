import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  describePurchaseFailure,
  formatExtraWeekPrice,
  isOutOfCredits,
  newPurchaseKey,
} from "../helpers";

describe("an extra week's price", () => {
  it.each([
    [8000, "$80"],
    [1250, "$12.50"],
    [100, "$1"],
    [125000, "$1,250"],
  ])("%s credits reads %s", (credits, expected) => {
    expect(formatExtraWeekPrice(credits)).toBe(expected);
  });
});

describe("a purchase key", () => {
  it("is new each time and fits what the server accepts", () => {
    const first = newPurchaseKey();
    expect(first).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
    expect(newPurchaseKey()).not.toBe(first);
  });
});

describe("a refused purchase", () => {
  it("is out of credits only on a 402", () => {
    expect(isOutOfCredits(new ApiError("no", 402, {}))).toBe(true);
    expect(isOutOfCredits(new ApiError("no", 409, {}))).toBe(false);
    expect(isOutOfCredits(new Error("offline"))).toBe(false);
  });

  it("says the plan can't buy one, or that nothing was charged", () => {
    expect(describePurchaseFailure(new ApiError("no", 409, {}))).toBe(
      "An extra week isn’t available on your plan right now.",
    );
    expect(describePurchaseFailure(new Error("offline"))).toBe(
      "Nothing was charged. Please try again.",
    );
  });
});
