import { describe, expect, it } from "vitest";
import { getUsageExperience, formatUsageDate } from "../helpers";

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
});
