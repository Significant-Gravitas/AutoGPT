import { v4 as uuidv4 } from "uuid";

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

/** One key per purchase: sent again on a retry, it buys nothing more.
 *  `crypto.randomUUID` is missing on plain-HTTP origins and older browsers. */
export function newPurchaseKey() {
  return uuidv4({});
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

/** Why the server refused the purchase (a 409), as its code and the words
 *  it sends for a person; null for any other failure. */
export function purchaseRefusal(error: unknown) {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const detail = error.response?.detail;
  if (typeof detail !== "object" || detail === null) return { message: "" };
  const { code, message } = detail as { code?: unknown; message?: unknown };
  return {
    code: typeof code === "string" ? code : undefined,
    message: typeof message === "string" ? message : "",
  };
}

/** The server refused: nothing was charged, and whatever the dialog shows
 *  (price, way of paying, that a limit is reached) is out of date. */
export function isOfferStale(error: unknown) {
  return purchaseRefusal(error) !== null;
}

export function describePurchaseFailure(error: unknown) {
  const refusal = purchaseRefusal(error);
  if (refusal) {
    if (refusal.code === "price_changed")
      return "The price has changed. Check the new price, then reset your limits. Nothing was charged.";
    if (refusal.code === "payment_changed")
      return "The way this would be paid has changed. Check the offer, then reset your limits. Nothing was charged.";
    return `${refusal.message ? `${refusal.message} ` : ""}Nothing was charged.`;
  }
  // A server or network failure doesn't say whether the purchase went
  // through. Trying again repeats the same purchase, so it can't be charged
  // twice.
  return "We couldn’t confirm the purchase. Try again: you won’t be charged twice.";
}
