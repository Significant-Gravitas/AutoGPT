import { formatTierLabel } from "@/app/(platform)/copilot/components/usageHelpers";
import {
  formatOfferPrice,
  formatTrialDate,
  type UsageExperience,
} from "./helpers";

export function formatPaymentDue(payment: UsageExperience["payment"]) {
  if (!payment) return null;
  const amount = formatOfferPrice(payment.amount_due, payment.currency);
  if (!amount) return null;
  const plan = formatTierLabel(payment.plan) ?? "Your plan";
  const date = formatTrialDate(payment.startsOn);
  return {
    url: payment.payment_url,
    actionLabel: `Pay ${amount} now`,
    description: `${plan} was due to start${date ? ` on ${date}` : ""}, but the ${amount} charge didn’t go through. Pay now and ${plan === "Your plan" ? "your plan" : plan} starts right away. Your chats, workflows and results are saved.`,
  };
}

interface RenewalBill {
  amount_due: number;
  currency: string;
  payment_url: string;
  plan?: string | null;
  due_since?: number | null;
}

export function formatRenewalPaymentDue(bill: RenewalBill | null) {
  if (!bill) return null;
  const amount = formatOfferPrice(bill.amount_due, bill.currency);
  if (!amount) return null;
  const plan = formatTierLabel(bill.plan ?? null) ?? "Your plan";
  const date = bill.due_since
    ? formatTrialDate(new Date(bill.due_since * 1000))
    : null;
  return {
    url: bill.payment_url,
    actionLabel: `Pay ${amount} now`,
    description: `${plan} was due to renew${date ? ` on ${date}` : ""}, but the ${amount} charge didn’t go through. Pay now and ${plan === "Your plan" ? "your plan" : plan} starts again right away, with a full billing period from the day you pay. Your chats, workflows and results are saved.`,
  };
}
