import { trialOffer } from "@/tests/integrations/trial-fixtures";
import { afterEach, describe, expect, test, vi } from "vitest";
import { formatDay, getTrialMilestones } from "../helpers";

const NOW = new Date("2026-10-23T12:00:00Z");
const DAY_MS = 24 * 60 * 60 * 1000;

function daysFromNow(days: number) {
  return new Date(NOW.getTime() + days * DAY_MS);
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("getTrialMilestones", () => {
  test("charges when the trial ends and reminds three days before", () => {
    const [today, reminder, charge] = getTrialMilestones(
      { ...trialOffer, duration_days: 14 },
      NOW,
    );

    expect(today.when).toBe("Today");
    expect(charge.date).toEqual(daysFromNow(14));
    expect(charge.when).toBe(formatDay(daysFromNow(14)));
    expect(reminder.date).toEqual(daysFromNow(11));
    expect(reminder.when).toBe(formatDay(daysFromNow(11)));
  });

  test("promises the day before the charge as the last day to cancel", () => {
    const charge = getTrialMilestones(trialOffer, NOW)[2];

    expect(charge.title).toBe("Your first payment: $20");
    expect(charge.detail).toBe(
      `Then $20 a month, plus any applicable tax. Cancel by ${formatDay(daysFromNow(6))} to pay nothing.`,
    );
  });

  test("keeps the last day to cancel before the charge day across a clock change", () => {
    vi.stubEnv("TZ", "America/New_York");
    // 23:30 on 1 Nov 2026, the day New York's clocks go back an hour.
    const chargeAt = new Date("2026-11-02T04:30:00Z");
    const now = new Date(chargeAt.getTime() - 7 * DAY_MS);

    const charge = getTrialMilestones(trialOffer, now)[2];

    expect(charge.when).toBe(formatDay(new Date("2026-11-01T12:00:00Z")));
    expect(charge.detail).toContain(
      `Cancel by ${formatDay(new Date("2026-10-31T12:00:00Z"))} to pay nothing.`,
    );
  });

  test("sends the reminder today when the trial is too short to wait", () => {
    const reminder = getTrialMilestones(
      { ...trialOffer, duration_days: 3 },
      NOW,
    )[1];

    expect(reminder.when).toBe("Today");
    expect(reminder.date).toBeUndefined();
  });
});

describe("the trial's first two rows", () => {
  test("never prices the trial's allowance from the onboarding credit", () => {
    for (const offer of [
      trialOffer,
      { ...trialOffer, onboarding_credit_amount: 2000 },
      { ...trialOffer, currency: "jpy", onboarding_credit_amount: 3000 },
    ]) {
      const [today] = getTrialMilestones(offer, NOW);
      expect(today.detail).toBe(
        "Full access from right now, within a trial usage allowance.",
      );
    }
  });

  test("says what the reminder email actually carries", () => {
    const reminder = getTrialMilestones(trialOffer, NOW)[1];

    expect(reminder.detail).toBe(
      "3 days before your trial ends, with your end date and how to cancel.",
    );
  });
});
