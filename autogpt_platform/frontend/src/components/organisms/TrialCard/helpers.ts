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
