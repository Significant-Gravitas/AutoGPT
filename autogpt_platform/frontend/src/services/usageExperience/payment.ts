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
