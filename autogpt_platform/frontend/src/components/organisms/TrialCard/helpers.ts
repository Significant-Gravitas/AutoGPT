import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { TrialRejectionReason } from "@/app/api/__generated__/models/trialRejectionReason";

export const trialPlanLabels: Record<TrialOfferResponse["tier"], string> = {
  BASIC: "Basic",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Team",
};

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

const rejectionCopy: Record<
  TrialRejectionReason,
  { title: string; detail: string }
> = {
  intro_offer_already_used: {
    title: "This introductory offer has already been used",
    detail:
      "This card or account has already redeemed an introductory offer. Each card and account can use one introductory offer.",
  },
  card_verification_failed: {
    title: "We couldn’t verify your card for this trial",
    detail:
      "Your card could not be verified for trial eligibility, so this trial was not activated.",
  },
  country_not_eligible: {
    title: "This trial isn’t available in your country yet",
    detail:
      "Your card was issued in a country where we don’t offer this trial yet, so the trial was not activated. Paid plans are still available.",
  },
};

export function trialRejectionCopy(reason: TrialRejectionReason) {
  return rejectionCopy[reason] ?? rejectionCopy.card_verification_failed;
}
