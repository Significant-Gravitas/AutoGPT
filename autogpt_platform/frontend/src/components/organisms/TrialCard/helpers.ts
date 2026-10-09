import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";

export const trialPlanLabels: Record<TrialOfferResponse["tier"], string> = {
  BASIC: "Basic",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
};

export function formatTrialPrice(offer: TrialOfferResponse) {
  const amount = currencyFormatter(offer.currency).format(
    getTrialChargeAmount(offer),
  );
  return `${amount} / ${offer.billing_cycle === "yearly" ? "year" : "month"}`;
}

// What Stripe charges per billing cycle once the trial ends, in major units.
// Zero-decimal currencies (JPY) carry no minor unit.
export function getTrialChargeAmount(offer: TrialOfferResponse) {
  const decimals =
    currencyFormatter(offer.currency).resolvedOptions().maximumFractionDigits ??
    2;
  return offer.unit_amount / 10 ** decimals;
}

function currencyFormatter(currency: string) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency });
}

export function formatTrialEnd(value: Date | string | null | undefined) {
  if (!value) return "the end of your trial";
  return new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function formatTrialEndDate(value: Date | string) {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function formatTrialEndTime(value: Date | string) {
  return new Date(value).toLocaleTimeString(undefined, { timeStyle: "short" });
}

const DAY_MS = 24 * 60 * 60 * 1000;

export function getTrialDaysLeft(
  endsAt: Date | string | null | undefined,
  now = Date.now(),
) {
  if (!endsAt) return 0;
  return Math.max(0, Math.ceil((new Date(endsAt).getTime() - now) / DAY_MS));
}

export function formatTrialDays(days: number) {
  return `${days} ${days === 1 ? "day" : "days"}`;
}

// Under a day the count would always read "1 day", so name the clock time.
export function describeTrialTimeLeft(endsAt: Date | string, now = new Date()) {
  const end = new Date(endsAt);
  if (end.getTime() - now.getTime() >= DAY_MS)
    return {
      kind: "days" as const,
      days: getTrialDaysLeft(end, now.getTime()),
    };
  const tomorrow = new Date(now);
  tomorrow.setDate(now.getDate() + 1);
  return {
    kind:
      end.toDateString() === tomorrow.toDateString()
        ? ("tomorrow" as const)
        : ("today" as const),
    time: formatTrialEndTime(end),
  };
}
