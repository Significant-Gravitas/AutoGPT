import { trialOffer } from "@/tests/integrations/trial-fixtures";
import { describe, expect, test } from "vitest";
import {
  formatDay,
  getTrialMilestones,
} from "../components/TrialWelcome/helpers";

const NOW = new Date("2026-10-23T12:00:00Z");
const DAY_MS = 24 * 60 * 60 * 1000;

function daysFromNow(days: number) {
  return new Date(NOW.getTime() + days * DAY_MS);
}

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

  test("sends the reminder today when the trial is too short to wait", () => {
    const reminder = getTrialMilestones(
      { ...trialOffer, duration_days: 3 },
      NOW,
    )[1];

    expect(reminder.when).toBe("Today");
    expect(reminder.date).toBeUndefined();
  });
});
