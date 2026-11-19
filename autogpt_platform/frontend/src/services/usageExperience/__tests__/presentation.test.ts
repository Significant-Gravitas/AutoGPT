import { afterEach, describe, expect, it, vi } from "vitest";
import {
  formatOfferPrice,
  formatResumeTime,
  getUsageExperience,
  getUsageOfferAction,
} from "../helpers";
import { usagePresentation } from "../presentation";

afterEach(() => vi.useRealTimers());

describe("usage messages", () => {
  it("anchors an exhausted Pro trial to its automatic start date", () => {
    const experience = getUsageExperience(
      { tier: "TRIAL" },
      {
        active: true,
        converted: false,
        usage_policy: "lifetime",
        allowance_used_percent: 100,
        offer: { tier: "PRO" },
        ends_at: new Date(2026, 9, 12, 12),
      },
    );
    expect(usagePresentation(experience)).toMatchObject({
      title: "You’ve used your trial allowance",
      noticeTitle: "You’ve used your trial allowance",
      description:
        "Pro starts on its own on Oct 12. Start it now to keep working today.",
      secondary: "Wait until Oct 12",
    });
  });

  it("does not invent an automatic Pro start for a non-Pro trial", () => {
    const experience = getUsageExperience(
      { tier: "TRIAL" },
      {
        active: true,
        allowance_used_percent: 100,
        offer: { tier: "MAX" },
        ends_at: new Date(2026, 9, 12, 12),
      },
    );
    expect(usagePresentation(experience).description).not.toContain(
      "Pro starts on its own",
    );
  });

  it("keeps the trial start honest when its end date is unavailable", () => {
    const experience = getUsageExperience(
      { tier: "TRIAL" },
      { active: true, allowance_used_percent: 100, offer: { tier: "PRO" } },
    );
    expect(usagePresentation(experience)).toMatchObject({
      description:
        "Pro starts on its own when your trial ends. Start it now to keep working today.",
      secondary: "Back to my work",
    });
    expect(getUsageOfferAction(experience, { price: 5013 })).toBe(
      "Start Pro now · $50.13",
    );
    expect(formatOfferPrice(undefined)).toBeNull();
    expect(formatOfferPrice(51000, "GBP")).toBe("£510");
  });

  it("uses local calendar days at midnight and does not invent missing times", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 9, 7, 23, 55));
    expect(formatResumeTime(new Date(2026, 9, 8, 0, 5))).toBe(
      "tomorrow at 12:05 AM",
    );
    expect(formatResumeTime(new Date(2026, 9, 11, 17))).toBe(
      "Sunday at 5:00 PM",
    );
    expect(formatResumeTime("invalid")).toBeNull();
    expect(formatResumeTime(null)).toBeNull();
  });

  it("leads with the user's local resume time for the highest plans", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 9, 7, 10));
    const experience = getUsageExperience({
      tier: "MAX",
      daily: { percent_used: 100, resets_at: new Date(2026, 9, 7, 17) },
    });
    expect(usagePresentation(experience)).toMatchObject({
      title: "You can continue today at 5:00 PM",
      noticeTitle: "You can continue today at 5:00 PM",
      description: "Your chats, workflows and results are saved.",
    });
  });
  it("states the amount and start date of a failed first trial charge", () => {
    const experience = getUsageExperience(
      { tier: "NO_TIER" },
      {
        active: false,
        converted: false,
        status: "past_due",
        offer: { tier: "PRO" },
        ends_at: new Date(2026, 9, 12, 12),
        payment_due: {
          amount_due: 5413,
          currency: "usd",
          payment_url: "https://invoice.stripe.com/i/acct_1/test_inv_1",
        },
      },
      "NO_TIER",
    );
    expect(usagePresentation(experience)).toMatchObject({
      title: "Your payment didn’t go through",
      description:
        "Pro was due to start on Oct 12, but the $54.13 charge didn’t go through. Pay now and Pro starts right away. Your chats, workflows and results are saved.",
    });
  });
  it("keeps the billing route when the open invoice is unavailable", () => {
    const experience = getUsageExperience(
      { tier: "NO_TIER" },
      { active: false, converted: false, status: "past_due" },
      "NO_TIER",
    );
    expect(usagePresentation(experience).description).toBe(
      "Review your billing details to continue. Your chats, workflows and results are saved.",
    );
  });
});
