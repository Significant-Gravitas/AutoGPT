import { describe, expect, it } from "vitest";
import {
  budgetLowHint,
  budgetOutTitle,
  budgetResetSentence,
  formatCountdown,
  getResetWait,
  getUsageBudget,
} from "../budget";
import { getUsageExperience } from "../helpers";

// Local times, so the copy reads the same in any time zone.
const NOW = new Date(2026, 9, 11, 20, 48); // Sunday 8:48 PM
const TONIGHT_1AM = new Date(2026, 9, 12, 1, 0); // Monday 1:00 AM
const NEXT_MONDAY_1AM = new Date(2026, 9, 19, 1, 0);

function usage(over: Record<string, unknown> = {}) {
  return {
    tier: "MAX",
    daily: { percent_used: 100, resets_at: TONIGHT_1AM },
    weekly: { percent_used: 100, resets_at: NEXT_MONDAY_1AM },
    ...over,
  } as Parameters<typeof getUsageBudget>[0];
}

describe("the budget verdict", () => {
  it("waits for the window the server says decides", () => {
    const budget = getUsageBudget(
      usage({
        state: "exhausted",
        limiting_window: "weekly",
        resets_at: NEXT_MONDAY_1AM,
      }),
    );
    expect(getResetWait(budget)).toEqual({
      window: "weekly",
      resetsAt: NEXT_MONDAY_1AM,
    });
    expect(budget.percentUsed).toBe(100);
  });

  it.each([
    ["a spent trial allowance", "trial"],
    ["a plan that allows nothing", "weekly"],
  ])("has nothing to wait for with %s", (_, limitingWindow) => {
    const budget = getUsageBudget(
      usage({
        state: "exhausted",
        limiting_window: limitingWindow,
        resets_at: null,
      }),
    );
    expect(budget.state).toBe("exhausted");
    expect(getResetWait(budget)).toBeNull();
  });

  it("waits for a trial's own weekly cap, which does refill", () => {
    const trialUsage = usage({
      tier: "TRIAL",
      daily: { percent_used: 40, resets_at: TONIGHT_1AM },
      state: "exhausted",
      limiting_window: "weekly",
      resets_at: NEXT_MONDAY_1AM,
    });
    expect(getResetWait(getUsageBudget(trialUsage))?.window).toBe("weekly");
    const experience = getUsageExperience(trialUsage, {
      active: true,
      converted: false,
      usage_policy: "lifetime",
      allowance_used_percent: 60,
    });
    expect(experience.blocked).toBe(true);
    expect(experience.window).toBe("weekly");
  });

  it("shows no reset time for a window that allows nothing", () => {
    const experience = getUsageExperience(
      usage({
        tier: "NO_TIER",
        state: "exhausted",
        limiting_window: "weekly",
        resets_at: null,
      }),
    );
    expect(experience.blocked).toBe(true);
    expect(experience.resetsAt).toBeNull();
  });

  it("estimates a low budget from the meters when the server sends no verdict", () => {
    const budget = getUsageBudget(
      usage({
        daily: { percent_used: 86, resets_at: TONIGHT_1AM },
        weekly: { percent_used: 61, resets_at: NEXT_MONDAY_1AM },
      }),
    );
    expect(budget).toEqual({
      state: "low",
      window: "daily",
      resetsAt: TONIGHT_1AM,
      percentUsed: 86,
    });
  });

  it("never promises a reset it had to estimate", () => {
    const budget = getUsageBudget(usage());
    expect(budget.state).toBe("exhausted");
    expect(getResetWait(budget)).toBeNull();
  });

  it("is ok below 80% of every window", () => {
    expect(
      getUsageBudget(
        usage({
          daily: { percent_used: 79, resets_at: TONIGHT_1AM },
          weekly: { percent_used: 10, resets_at: NEXT_MONDAY_1AM },
        }),
      ).state,
    ).toBe("ok");
  });
});

describe("budget copy", () => {
  it("titles the window that is used up", () => {
    expect(budgetOutTitle("weekly")).toBe("You’re out of budget for this week");
    expect(budgetOutTitle("daily")).toBe("You’re out of budget for today");
  });

  it("says when a weekly budget resets, by day", () => {
    expect(
      budgetResetSentence(
        { window: "weekly", resetsAt: new Date(2026, 9, 13, 21, 0) },
        NOW,
      ),
    ).toBe(
      "It resets Tuesday at 9:00 PM, in 2 days. You can still browse, edit workflows and read results.",
    );
  });

  it("says when today's budget resets, by the clock", () => {
    expect(
      budgetResetSentence({ window: "daily", resetsAt: TONIGHT_1AM }, NOW),
    ).toBe(
      "It resets at 1:00 AM, in 4 h 12 min. You can still browse, edit workflows and read results.",
    );
  });

  it.each([
    [new Date(2026, 9, 11, 20, 48, 30), "in under a minute"],
    [new Date(2026, 9, 11, 21, 23), "in 35 min"],
    [new Date(2026, 9, 12, 0, 48), "in 4 h"],
    [new Date(2026, 9, 12, 22, 0), "in 1 day"],
  ])("counts down to %s as %s", (resetsAt, expected) => {
    expect(formatCountdown(resetsAt, NOW)).toBe(expected);
  });

  it("names the budget that is running low", () => {
    expect(budgetLowHint("daily", 86)).toBe(
      "About 14% of today’s budget left. A long task may pause partway.",
    );
    expect(budgetLowHint("weekly", 80.4)).toBe(
      "About 20% of this week’s budget left. A long task may pause partway.",
    );
    expect(budgetLowHint("trial", 99.9)).toBe(
      "About 1% of your trial allowance left. A long task may pause partway.",
    );
  });
});
