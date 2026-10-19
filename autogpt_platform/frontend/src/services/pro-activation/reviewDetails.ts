import type { ActivationTerms } from "@/app/api/__generated__/models/activationTerms";
import { formatAmount, intervalLabel } from "./helpers";

export function shortDate(value: Date | string) {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function ordinal(day: number) {
  const suffix =
    day % 100 >= 11 && day % 100 <= 13
      ? "th"
      : ({ 1: "st", 2: "nd", 3: "rd" }[day % 10] ?? "th");
  return `${day}${suffix}`;
}

export function billingCadence(terms: ActivationTerms) {
  return (terms.billing_interval_count ?? 1) === 1
    ? terms.billing_interval === "year"
      ? "yearly"
      : "monthly"
    : `every ${intervalLabel(terms.billing_interval, terms.billing_interval_count ?? 1)}`;
}

export function trialEndNotice(terms: ActivationTerms) {
  const plan = terms.plan === "MAX" ? "Max" : "Pro";
  const trialEnd = terms.trial_ends_at;
  const start = terms.period_starts_at;
  if (!trialEnd || !start)
    return `Starting ${plan} ends your free trial early. Your paid billing period begins when you confirm.`;
  const billingDay =
    terms.billing_interval === "year"
      ? shortDate(start)
      : `the ${ordinal(new Date(start).getDate())}`;
  return `Your free trial ends now instead of ${shortDate(trialEnd)}, and your billing date moves to ${billingDay}.`;
}

export function nextPaymentDescription(terms: ActivationTerms) {
  const amount = formatAmount(terms.renewal_unit_amount, terms.currency);
  const discountedRenewal = terms.renewal_discounts?.some(
    (discount) => discount.duration !== "once",
  );
  const price = discountedRenewal
    ? `Next payment based on ${amount} before renewal discounts`
    : `Next payment of ${amount}`;
  const tax =
    terms.renewal_tax?.price_tax_behavior === "inclusive"
      ? " including tax"
      : " plus tax";
  return `${price}${tax}, then every ${intervalLabel(terms.billing_interval, terms.billing_interval_count ?? 1)}.`;
}

export function invoiceAdjustment(terms: ActivationTerms) {
  if (
    terms.tax_amount == null ||
    terms.renewal_tax?.price_tax_behavior === "unspecified"
  )
    return null;
  const additionalTax =
    terms.renewal_tax?.price_tax_behavior === "inclusive"
      ? 0
      : terms.tax_amount;
  return terms.amount_due - terms.renewal_unit_amount - additionalTax;
}
