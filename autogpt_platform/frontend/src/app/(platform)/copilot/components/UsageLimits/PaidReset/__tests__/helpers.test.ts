import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  describePayment,
  describePurchaseFailure,
  formatResetPrice,
  grantedMessage,
  isPriceChanged,
  newPurchaseKey,
  payLabel,
  resetPromise,
} from "../helpers";

describe("a reset's price", () => {
  it.each([
    [8000, "$80"],
    [1250, "$12.50"],
    [100, "$1"],
    [125000, "$1,250"],
  ])("%s credits reads %s", (credits, expected) => {
    expect(formatResetPrice(credits)).toBe(expected);
  });
});

describe("a purchase key", () => {
  it("is new each time and fits what the server accepts", () => {
    const first = newPurchaseKey();
    expect(first).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
    expect(newPurchaseKey()).not.toBe(first);
  });
});

describe("what a reset promises", () => {
  it("names the weekly reset it lasts until", () => {
    expect(resetPromise("Tue, Nov 12, 8:31 AM")).toBe(
      "Reset your limits now: a full week of usage, no daily limit, until your normal reset on Tue, Nov 12, 8:31 AM.",
    );
  });

  it("still reads right without a date", () => {
    expect(resetPromise(null)).toBe(
      "Reset your limits now: a full week of usage, no daily limit, until your normal reset.",
    );
    expect(grantedMessage(null)).toBe(
      "A full week of usage with no daily limit until your weekly reset.",
    );
  });

  it("says how long a granted reset lasts", () => {
    expect(grantedMessage("Tue, Nov 12, 8:31 AM")).toBe(
      "A full week of usage with no daily limit until Tue, Nov 12, 8:31 AM.",
    );
  });
});

describe("how a reset is paid", () => {
  it("is all from credits when the balance covers it", () => {
    expect(describePayment("credits", "$80")).toBe(
      "$80 is taken from your AutoGPT credit balance.",
    );
    expect(payLabel("credits", "$80")).toBe("Pay $80 with credits");
  });

  it("is all by card otherwise, never part of each", () => {
    expect(describePayment("card", "$12.50")).toBe(
      "Your credit balance doesn’t cover $12.50, so you pay the full $12.50 by card on the next page.",
    );
    expect(payLabel("card", "$12.50")).toBe("Pay $12.50 by card");
  });
});

describe("a refused purchase", () => {
  it("passes on why, and that nothing was charged", () => {
    const refused = new ApiError("no", 409, {
      detail:
        "Your limits reset shortly anyway, so a reset can't be bought now.",
    });

    expect(describePurchaseFailure(refused)).toBe(
      "Your limits reset shortly anyway, so a reset can't be bought now. Nothing was charged.",
    );
    expect(describePurchaseFailure(new ApiError("no", 409, {}))).toBe(
      "Nothing was charged.",
    );
  });

  it("knows a changed price from any other refusal", () => {
    const changed = new ApiError("no", 409, { detail: "price_changed" });

    expect(isPriceChanged(changed)).toBe(true);
    expect(isPriceChanged(new ApiError("no", 409, { detail: "other" }))).toBe(
      false,
    );
    expect(isPriceChanged(new Error("offline"))).toBe(false);
    expect(describePurchaseFailure(changed)).toBe(
      "The price has changed. Check the new price, then reset your limits.",
    );
  });

  it("says a retry is safe when the answer was lost", () => {
    expect(describePurchaseFailure(new Error("offline"))).toBe(
      "We couldn’t confirm the purchase. Try again: you won’t be charged twice.",
    );
  });
});
