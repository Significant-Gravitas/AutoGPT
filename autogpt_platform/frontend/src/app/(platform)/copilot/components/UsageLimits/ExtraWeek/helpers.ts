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

/** The price is no longer the one the offer showed. */
export function isPriceChanged(error: unknown) {
  return (
    error instanceof ApiError &&
    error.status === 409 &&
    error.response?.detail === "price_changed"
  );
}

export function describePurchaseFailure(error: unknown) {
  if (isPriceChanged(error))
    return "The price has changed. Check the new price, then add your extra week.";
  if (error instanceof ApiError && error.status === 409)
    return "An extra week isn’t available on your plan right now. Nothing was charged.";
  // A server or network failure doesn't say whether the purchase went
  // through. Trying again repeats the same purchase, so it can't be charged
  // twice.
  return "We couldn’t confirm the purchase. Try again: you won’t be charged twice.";
}
