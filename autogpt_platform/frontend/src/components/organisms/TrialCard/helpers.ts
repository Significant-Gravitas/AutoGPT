import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";

export function formatTrialPrice(offer: TrialOfferResponse) {
  const formatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: offer.currency,
  });
  const decimals = formatter.resolvedOptions().maximumFractionDigits ?? 2;
  const amount = formatter.format(offer.unit_amount / 10 ** decimals);
  return `${amount} / ${offer.billing_cycle === "yearly" ? "year" : "month"}`;
}

export function formatTrialEnd(value: Date | string | null | undefined) {
  if (!value) return "the end of your trial";
  return new Date(value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
