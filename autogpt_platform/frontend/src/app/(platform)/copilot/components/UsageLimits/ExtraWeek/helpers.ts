import { ApiError } from "@/lib/autogpt-server-api/helpers";

const WHOLE_DOLLARS = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});
const DOLLARS_AND_CENTS = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
});

export const EXTRA_WEEK_TERMS = [
  "Used after your weekly limit runs out.",
  "Your daily limit doesn’t apply to it.",
  "Lasts 30 days from today, through your usage resets.",
];

/** Credits are cents: 8000 reads "$80", 1250 reads "$12.50". */
export function formatExtraWeekPrice(credits: number) {
  const formatter = credits % 100 === 0 ? WHOLE_DOLLARS : DOLLARS_AND_CENTS;
  return formatter.format(credits / 100);
}

/** One key per purchase: sent again on a retry, it buys nothing more. */
export function newPurchaseKey() {
  return crypto.randomUUID();
}

export function isOutOfCredits(error: unknown) {
  return error instanceof ApiError && error.status === 402;
}

export function describePurchaseFailure(error: unknown) {
  if (error instanceof ApiError && error.status === 409)
    return "An extra week isn’t available on your plan right now.";
  return "Nothing was charged. Please try again.";
}
