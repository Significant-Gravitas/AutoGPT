import { describe, expect, it } from "vitest";
import {
  getFeaturedUsagePercent,
  getUsageExperience,
  getUsageOfferAction,
  getUsageStatus,
  formatUsageDate,
} from "../helpers";

const usage = {
  tier: "PRO",
  daily: { percent_used: 100, resets_at: "2026-10-08T00:00:00Z" },
  weekly: { percent_used: 100, resets_at: "2026-10-12T00:00:00Z" },
};
const trial = {
  active: true,
  converted: false,
  allowance_used_percent: 100,
  ends_at: "2026-10-12T12:00:00Z",
  usage_policy: "lifetime",
};

describe("usage experience", () => {
  it("waits for the later blocking window when both limits are exhausted", () => {
    expect(getUsageExperience(usage)).toMatchObject({
      blocked: true,
      window: "weekly",
      resetsAt: usage.weekly.resets_at,
      targetTier: "MAX",
    });
  });
  it("uses the durable lifetime percentage for an active trial without promising a reset", () => {
    expect(
      getUsageExperience({ ...usage, tier: "TRIAL" }, trial),
    ).toMatchObject({
      isLifetimeTrial: true,
      blocked: true,
      window: "trial",
      resetsAt: null,
      trialPercent: 100,
      targetTier: "PRO",
    });
  });
  it("does not apply historical trial consumption after paid activation", () => {
    expect(
      getUsageExperience(
        { ...usage, daily: null, weekly: null },
        { ...trial, converted: true },
      ),
    ).toMatchObject({
      isActiveTrial: false,
      blocked: false,
      trialPercent: null,
    });
  });
  it("retains recurring windows for accepted legacy trial offers", () => {
    expect(
      getUsageExperience(
        { ...usage, tier: "TRIAL" },
        { ...trial, usage_policy: "rolling", allowance_used_percent: 20 },
      ),
    ).toMatchObject({
      isLifetimeTrial: false,
      window: "weekly",
      resetsAt: usage.weekly.resets_at,
    });
  });
  it("does not show a stale trial lifetime meter after expiry", () => {
    expect(
      getUsageExperience(
        { ...usage, tier: "NO_TIER" },
        { ...trial, active: false },
      ),
    ).toMatchObject({ isActiveTrial: false, trialPercent: null });
  });
  it("handles generated Date timestamps alongside serialized API timestamps", () => {
    const daily = new Date(usage.daily.resets_at);
    const weekly = new Date(usage.weekly.resets_at);
    expect(
      getUsageExperience({
        ...usage,
        daily: { ...usage.daily, resets_at: daily },
        weekly: { ...usage.weekly, resets_at: weekly },
      }),
    ).toMatchObject({ window: "weekly", resetsAt: weekly });
    expect(formatUsageDate(weekly)).toBe(
      formatUsageDate(usage.weekly.resets_at),
    );
    expect(
      getUsageExperience(
        { tier: "NO_TIER" },
        {
          active: false,
          converted: false,
          status: "trialing",
          ends_at: new Date("2000-01-01T00:00:00Z"),
        },
      ),
    ).toMatchObject({ inactiveTrialStatus: "expired", resetsAt: null });
  });
  it.each(["canceled", "expired"])(
    "never suggests a reset for an inactive %s trial",
    (status) => {
      expect(
        getUsageExperience(
          { ...usage, tier: "NO_TIER" },
          { ...trial, active: false, status },
        ),
      ).toMatchObject({
        blocked: true,
        inactiveTrialStatus: status,
        resetsAt: null,
        trialPercent: null,
        freshProUsage: false,
      });
    },
  );
  it("does not describe automatic Pro activation for a cancel-pending trial", () => {
    expect(
      getUsageExperience(
        { ...usage, tier: "TRIAL" },
        { ...trial, offer: { tier: "PRO" }, cancel_at_period_end: true },
      ),
    ).toMatchObject({ isActiveTrial: true, freshProUsage: false });
  });
  it("does not promise the Pro activation reset for a legacy non-Pro trial", () => {
    expect(
      getUsageExperience(
        { ...usage, tier: "TRIAL" },
        { ...trial, offer: { tier: "MAX" } },
      ),
    ).toMatchObject({ freshProUsage: false });
  });
  it.each(["MAX", "BUSINESS", "ENTERPRISE"])(
    "does not offer a self-service upgrade to %s",
    (tier) => {
      expect(getUsageExperience({ ...usage, tier })).toMatchObject({
        isTopTier: true,
        targetTier: null,
      });
    },
  );
  it("asks a cancel-pending trial to subscribe rather than start Pro", () => {
    const experience = getUsageExperience(
      { ...usage, tier: "TRIAL" },
      { ...trial, offer: { tier: "PRO" }, cancel_at_period_end: true },
    );
    expect(experience.isCancelPending).toBe(true);
    expect(getUsageOfferAction(experience, {})).toBe("Subscribe now");
  });
  it("keeps Start Pro for a trial that ended without a pending cancellation", () => {
    const experience = getUsageExperience(
      { ...usage, tier: "NO_TIER" },
      { ...trial, active: false, status: "expired" },
    );
    expect(experience.isCancelPending).toBe(false);
    expect(getUsageOfferAction(experience, {})).toBe("Start Pro");
  });
  it("carries the open invoice only for a failed first charge", () => {
    const payment_due = {
      amount_due: 5000,
      currency: "usd",
      payment_url: "https://invoice.stripe.com/i/acct_1/test_inv_1",
    };
    expect(
      getUsageExperience(
        { ...usage, tier: "NO_TIER" },
        {
          ...trial,
          active: false,
          status: "past_due",
          offer: { tier: "PRO" },
          payment_due,
        },
      ).payment,
    ).toEqual({ ...payment_due, plan: "PRO", startsOn: trial.ends_at });
    expect(
      getUsageExperience(
        { ...usage, tier: "NO_TIER" },
        { ...trial, active: false, status: "expired", payment_due },
      ).payment,
    ).toBeNull();
  });
  it.each([
    [0, "plenty"],
    [79.9, "plenty"],
    [80, "low"],
    [99.9, "low"],
    [100, "reached"],
  ] as const)("reports %s%% as %s", (percent, status) => {
    expect(getUsageStatus(percent)).toBe(status);
  });
  it("features the same window the usage card shows", () => {
    const lifetime = getUsageExperience(
      { ...usage, tier: "TRIAL" },
      { ...trial, allowance_used_percent: 64 },
    );
    expect(getFeaturedUsagePercent(lifetime, usage)).toBe(64);
    const weekly = {
      ...usage,
      daily: { ...usage.daily, percent_used: 40 },
      weekly: { ...usage.weekly, percent_used: 100 },
    };
    expect(getFeaturedUsagePercent(getUsageExperience(weekly), weekly)).toBe(
      100,
    );
    const daily = {
      ...usage,
      daily: { ...usage.daily, percent_used: 30 },
      weekly: { ...usage.weekly, percent_used: 70 },
    };
    expect(getFeaturedUsagePercent(getUsageExperience(daily), daily)).toBe(30);
  });
});
