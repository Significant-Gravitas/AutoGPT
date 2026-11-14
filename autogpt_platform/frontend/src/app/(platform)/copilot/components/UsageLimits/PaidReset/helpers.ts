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

/** How long to wait between asking whether a card payment has gone through. */
export const CONFIRM_WAIT_MS = 2000;
export const CONFIRM_ATTEMPTS = 4;

/** Credits are cents: 8000 reads "$80", 1250 reads "$12.50". */
export function formatResetPrice(credits: number) {
  const formatter = credits % 100 === 0 ? WHOLE_DOLLARS : DOLLARS_AND_CENTS;
  return formatter.format(credits / 100);
}

/** One key per purchase: sent again on a retry, it buys nothing more. */
export function newPurchaseKey() {
  return crypto.randomUUID();
}

/** What a reset gives, with the weekly reset it lasts until when known. */
export function resetPromise(resetsAt: string | null) {
  const until = resetsAt ? ` on ${resetsAt}` : "";
  return `Reset your limits now: a full week of usage, no daily limit, until your normal reset${until}.`;
}

/** How this purchase is paid: all from credits, or all by card. */
export function describePayment(payWith: "credits" | "card", price: string) {
  if (payWith === "credits")
    return `${price} is taken from your AutoGPT credit balance.`;
  return `Your credit balance doesn’t cover ${price}, so you pay the full ${price} by card on the next page.`;
}

export function payLabel(payWith: "credits" | "card", price: string) {
  return payWith === "credits"
    ? `Pay ${price} with credits`
    : `Pay ${price} by card`;
}

export function grantedMessage(resetsAt: string | null) {
  const until = resetsAt ? ` until ${resetsAt}` : " until your weekly reset";
  return `A full week of usage with no daily limit${until}.`;
}

/** What a reset does to the week in progress, said before anything is paid. */
export const RESET_REPLACES =
  "This replaces what’s left of this week’s usage with a full fresh week.";

/** The price, or the way it would be paid, is no longer what the offer
 *  showed: the offer on screen is stale. */
export function isOfferChanged(error: unknown) {
  if (!(error instanceof ApiError) || error.status !== 409) return false;
  const detail = error.response?.detail;
  return detail === "price_changed" || detail === "payment_changed";
}

export function describePurchaseFailure(error: unknown) {
  if (error instanceof ApiError && error.status === 409) {
    if (error.response?.detail === "price_changed")
      return "The price has changed. Check the new price, then reset your limits.";
    if (error.response?.detail === "payment_changed")
      return "Your credit balance now covers this, so it would be paid with credits. Check the offer, then reset your limits. Nothing was charged.";
  }
  if (error instanceof ApiError && error.status === 409) {
    const detail = error.response?.detail;
    const reason = typeof detail === "string" ? `${detail} ` : "";
    return `${reason}Nothing was charged.`;
  }
  // A server or network failure doesn't say whether the purchase went
  // through. Trying again repeats the same purchase, so it can't be charged
  // twice.
  return "We couldn’t confirm the purchase. Try again: you won’t be charged twice.";
}
