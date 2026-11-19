import { describe, expect, it } from "vitest";
import { tierUpgradeBody, yearlyRenewsLine } from "../upgradeCopy";

describe("what an upgrade says it will charge", () => {
  it("is the prorated difference while upgrades keep the billing period", () => {
    const body = tierUpgradeBody({
      resetsBillingDate: false,
      newPrice: "$320.00",
      cycle: "monthly",
    });
    expect(body).toBe(
      "You'll be charged the prorated difference immediately for the rest of your monthly period. After this period, your plan renews at $320.00 per month. Your current usage carries over into the higher allowance.",
    );
  });

  it("is the full new price less the unused credit once upgrades start a fresh month", () => {
    const body = tierUpgradeBody({
      resetsBillingDate: true,
      newPrice: "$320.00",
      cycle: "monthly",
    });
    expect(body).toBe(
      "You pay the full $320.00 today, minus a credit for the unused part of your current plan. Your billing date moves to today, and your plan renews at $320.00 per month. Your usage limits start fresh.",
    );
    expect(body).not.toContain("carries over");
  });

  it("still makes sense without a known price", () => {
    const body = tierUpgradeBody({
      resetsBillingDate: true,
      newPrice: "",
      cycle: "yearly",
    });
    expect(body).toBe(
      "You pay the new plan’s full price today, minus a credit for the unused part of your current plan. Your billing date moves to today. Your usage limits start fresh.",
    );
  });
});

describe("when a switch to yearly billing renews", () => {
  it("is the current period's end while the period is kept", () => {
    expect(
      yearlyRenewsLine({
        resetsBillingDate: false,
        yearlyPrice: "$510.00",
        periodEndLabel: "Dec 4, 2026",
      }),
    ).toEqual({ label: "Renews", text: "$510.00/year on Dec 4, 2026." });
  });

  it("is a year from today once the switch starts a fresh period", () => {
    expect(
      yearlyRenewsLine({
        resetsBillingDate: true,
        yearlyPrice: "$510.00",
        periodEndLabel: "Dec 4, 2026",
      }),
    ).toEqual({
      label: "Renews",
      text: "at $510.00/year, a year from today. Your usage limits start fresh.",
    });
  });
});
