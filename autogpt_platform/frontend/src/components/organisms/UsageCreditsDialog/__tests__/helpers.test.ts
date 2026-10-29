import { describe, expect, it } from "vitest";
import { getChatAllowance, getUsageCreditsAnswers } from "../helpers";

function answer(
  args: Parameters<typeof getUsageCreditsAnswers>[0],
  question: string,
) {
  return getUsageCreditsAnswers(args).find((item) => item.question === question)
    ?.answer;
}

const CHAT = "What does chat usage cover?";
const CREDITS = "What are automation credits for?";
const CARD = "When is my card charged?";
const MORE = "How do I get more?";

describe("getChatAllowance", () => {
  const paid = { tier: "PRO", isActiveTrial: false, isLifetimeTrial: false };

  it("reads the plan's own windows, like the billing page does", () => {
    expect(getChatAllowance(paid, { daily: {}, weekly: {} })).toBe(
      "daily-weekly",
    );
    expect(getChatAllowance(paid, { daily: {}, weekly: null })).toBe("daily");
    expect(getChatAllowance(paid, { daily: null, weekly: {} })).toBe("weekly");
    expect(getChatAllowance(paid, { daily: null, weekly: null })).toBe(
      "unlimited",
    );
  });

  it("gives a lifetime trial one allowance and a missing plan none", () => {
    expect(
      getChatAllowance(
        { tier: "TRIAL", isActiveTrial: true, isLifetimeTrial: true },
        { daily: null, weekly: null },
      ),
    ).toBe("lifetime-trial");
    expect(
      getChatAllowance(
        { tier: "NO_TIER", isActiveTrial: false, isLifetimeTrial: false },
        { daily: null, weekly: null },
      ),
    ).toBe("no-plan");
  });

  it("says nothing about limits until usage has loaded", () => {
    expect(getChatAllowance(paid, undefined)).toBe("unknown");
  });

  // Sentry: isActiveTrial stays false until the trial query answers, so a
  // trial account with no daily or weekly windows read as "unlimited".
  it("never calls a trial account unlimited before its trial has loaded", () => {
    expect(
      getChatAllowance(
        { tier: "TRIAL", isActiveTrial: false, isLifetimeTrial: false },
        { daily: null, weekly: null },
      ),
    ).toBe("unknown");
  });
});

describe("getUsageCreditsAnswers", () => {
  const paidCapped = {
    paymentEnabled: true,
    allowance: "daily-weekly" as const,
    isTrial: false,
  };

  it("leads the card answer with the reassurance, then every charge the card pays", () => {
    // SECRT-3133 review: renewals and upgrades were missing, and a renewal
    // the card fails is paid from credits (subscription_payment_failure.py).
    expect(answer(paidCapped, CARD)).toBe(
      "A chat or a step never charges your card directly. Your card is charged when your plan starts and each time it renews, when you upgrade, and when you buy credits (including auto-refill). If a renewal payment fails, it's taken from your automation credits when they cover it.",
    );
  });

  it("says an Otto step in a workflow uses chat usage before credits", () => {
    // copilot/credit_billing.py: AutoPilot block turns spend the allowance
    // first and fall back to the credit balance only once it is out.
    expect(answer(paidCapped, CREDITS)).toContain(
      "A workflow step where Otto does the work uses your chat usage first, and your credits only once chat usage runs out.",
    );
  });

  it("matches the billing page for a plan without daily or weekly limits", () => {
    const uncapped = { ...paidCapped, allowance: "unlimited" as const };
    expect(answer(uncapped, CHAT)).toContain(
      "It comes with your plan, which has no daily or weekly limits.",
    );
    expect(answer(uncapped, CHAT)).not.toContain("resets");
    expect(answer(uncapped, MORE)).not.toContain("bigger plan");
  });

  it("names only the windows a capped plan has", () => {
    expect(answer({ ...paidCapped, allowance: "daily" }, CHAT)).toContain(
      "It comes with your plan and resets daily.",
    );
    expect(answer(paidCapped, CHAT)).toContain(
      "It comes with your plan and resets daily and weekly.",
    );
  });

  it("gives a lifetime trial one allowance for the whole trial", () => {
    expect(
      answer(
        { paymentEnabled: true, allowance: "lifetime-trial", isTrial: true },
        CHAT,
      ),
    ).toContain(
      "It comes with your trial, as one allowance for the whole trial.",
    );
  });

  it("never points at buying or the card when payments are off", () => {
    const answers = getUsageCreditsAnswers({
      ...paidCapped,
      paymentEnabled: false,
    });
    const text = answers.map((item) => item.answer).join(" ");

    expect(answers.map((item) => item.question)).not.toContain(CARD);
    expect(text).not.toMatch(/Add credits|buy credits|auto-refill/);
    expect(answer({ ...paidCapped, paymentEnabled: false }, MORE)).toBe(
      "For more automation credits, finish the setup steps in the wallet.",
    );
  });

  // Sentry: upgrades are refused with payments off, and an account without a
  // plan has nothing to make bigger.
  it("only suggests a bigger plan when one can be bought", () => {
    expect(answer({ ...paidCapped, paymentEnabled: false }, MORE)).not.toMatch(
      /plan/,
    );
    expect(answer({ ...paidCapped, allowance: "no-plan" as const }, MORE)).toBe(
      "For chat usage, choose a plan. For more automation credits, select Add credits in the wallet.",
    );
  });

  it("points at Add credits only when the wallet shows it", () => {
    expect(answer(paidCapped, MORE)).toBe(
      "For more chat usage, choose a bigger plan. For more automation credits, select Add credits in the wallet.",
    );
  });
});
