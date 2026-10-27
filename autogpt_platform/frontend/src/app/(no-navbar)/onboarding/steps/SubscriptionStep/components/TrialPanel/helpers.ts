import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import {
  formatPlanAmount,
  formatWholeMoney,
} from "@/components/organisms/TrialCard/helpers";
import {
  Calendar03Icon,
  Mail01Icon,
  SquareUnlock02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export interface Milestone {
  key: string;
  icon: IconSvgElement;
  when: string;
  date?: Date;
  title: string;
  detail: string;
}

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
      detail: trialAccessDetail(offer),
    },
    {
      key: "reminder",
      icon: Mail01Icon,
      when: remindsToday ? "Today" : formatDay(reminderDate),
      date: remindsToday ? undefined : reminderDate,
      title: "We email you a reminder",
      // The trial-ending email names the end date and points at billing
      // settings to cancel; it promises nothing more than that.
      detail: remindsToday
        ? "With your end date and how to cancel."
        : `${REMINDER_DAYS_BEFORE_CHARGE} days before your trial ends, with your end date and how to cancel.`,
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

function trialAccessDetail(offer: TrialOfferResponse) {
  const allowance = formatAllowance(offer);
  if (!allowance)
    return "Full access from right now, within a trial usage allowance.";
  return `Full access from right now, with ${allowance} of usage included.`;
}

// The credit granted with the trial, in the offer's currency. Credits are
// minor units, like `unit_amount`; zero-decimal currencies have none.
export function formatAllowance(offer: TrialOfferResponse) {
  const credit = offer.onboarding_credit_amount ?? 0;
  if (credit <= 0) return null;
  const decimals =
    new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: offer.currency,
    }).resolvedOptions().maximumFractionDigits ?? 2;
  return formatWholeMoney(credit / 10 ** decimals, offer.currency);
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

interface PaidArgs {
  price: string;
  cadence: "month" | "year";
  now?: Date;
}

export function getPaidMilestones({
  price,
  cadence,
  now = new Date(),
}: PaidArgs): Milestone[] {
  const renews = getRenewalDate(now, cadence);
  return [
    {
      key: "today",
      icon: SquareUnlock02Icon,
      when: "Today",
      title: "Your plan starts",
      detail: `You pay ${price}, plus any applicable tax. Full access starts straight away.`,
    },
    {
      key: "renew",
      icon: Calendar03Icon,
      when: formatDay(renews),
      date: renews,
      title: "Your plan renews",
      detail: `${price} a ${cadence} until you cancel in Settings.`,
    },
  ];
}

// Stripe renews on the same day of the next period, or on its last day when
// that month is shorter (31 Jan renews on 28 Feb, 29 Feb on 28 Feb).
export function getRenewalDate(now: Date, cadence: "month" | "year") {
  const months = cadence === "year" ? 12 : 1;
  const target = new Date(now.getFullYear(), now.getMonth() + months, 1);
  const lastDay = new Date(
    target.getFullYear(),
    target.getMonth() + 1,
    0,
  ).getDate();
  target.setDate(Math.min(now.getDate(), lastDay));
  target.setHours(now.getHours(), now.getMinutes(), now.getSeconds());
  return target;
}

// The visible date is local, so the machine-readable one must be too.
export function toLocalISODate(date: Date) {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}
