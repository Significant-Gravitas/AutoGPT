import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { formatPlanAmount } from "@/components/organisms/TrialCard/helpers";
import {
  Calendar03Icon,
  Mail01Icon,
  SquareUnlock02Icon,
} from "@hugeicons/core-free-icons";
import type { Milestone } from "../Timeline";

const DAY_MS = 24 * 60 * 60 * 1000;

// Stripe sends `customer.subscription.trial_will_end` three days before the
// trial ends, and the trial reminder email goes out on it.
export const REMINDER_DAYS_BEFORE_CHARGE = 3;

export function getTrialMilestones(
  offer: TrialOfferResponse,
  now = new Date(),
): Milestone[] {
  const chargeDate = new Date(now.getTime() + offer.duration_days * DAY_MS);
  const reminderDate = new Date(
    chargeDate.getTime() - REMINDER_DAYS_BEFORE_CHARGE * DAY_MS,
  );
  // A trial this short gets its reminder as soon as it starts.
  const remindsToday = reminderDate <= now;
  // Stripe charges at the exact moment the trial ends, so the safe date to
  // promise is the day before. Step back a calendar day, not 24 hours, so a
  // daylight-saving change can't land it on the charge day itself.
  const lastFreeDay = new Date(chargeDate);
  lastFreeDay.setDate(lastFreeDay.getDate() - 1);
  const amount = formatPlanAmount(offer);
  return [
    {
      key: "today",
      icon: SquareUnlock02Icon,
      when: "Today",
      title: "Your free trial starts",
      detail: "Full access from right now, within a trial usage allowance.",
    },
    {
      key: "reminder",
      icon: Mail01Icon,
      when: remindsToday ? "Today" : formatDay(reminderDate),
      date: remindsToday ? undefined : reminderDate,
      title: "We email you a reminder",
      detail: remindsToday
        ? "With your trial end date, so nothing comes as a surprise."
        : `${REMINDER_DAYS_BEFORE_CHARGE} days before your trial ends, so nothing comes as a surprise.`,
    },
    {
      key: "charge",
      icon: Calendar03Icon,
      when: formatDay(chargeDate),
      date: chargeDate,
      title: `Your first payment: ${amount}`,
      detail: `Then ${formatCadencePrice(offer)}, plus any applicable tax. Cancel by ${formatDay(lastFreeDay)} to pay nothing.`,
    },
  ];
}

export function formatCadencePrice(offer: TrialOfferResponse) {
  const cadence = offer.billing_cycle === "yearly" ? "year" : "month";
  return `${formatPlanAmount(offer)} a ${cadence}`;
}

export function formatDay(date: Date) {
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

// The Experts named beside Otto; the rest are counted, to match the avatars.
export const MAX_NAMED_EXPERTS = 3;

export function formatReadyLine(names: string[]) {
  const named = names.slice(0, MAX_NAMED_EXPERTS);
  const others = names.length - named.length;
  const everyone = [
    "Otto",
    ...named,
    ...(others > 0 ? [`${others} more`] : []),
  ];
  // The sentence around it is English, so the list is too.
  const list = new Intl.ListFormat("en", {
    style: "long",
    type: "conjunction",
  }).format(everyone);
  return `${list} ${everyone.length === 1 ? "is" : "are"} ready to start`;
}
