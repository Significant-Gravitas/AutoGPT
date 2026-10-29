import { formatTierLabel } from "@/app/(platform)/copilot/components/usageHelpers";
import {
  formatResumeTime,
  formatTrialDate,
  formatTrialStartDescription,
  type UsageExperience,
} from "./helpers";
import { formatPaymentDue } from "./payment";

export function usagePresentation(
  experience: UsageExperience,
  usageMultiplier?: number,
) {
  const { tier, window, isTopTier, isActiveTrial } = experience;
  if (experience.inactiveTrialStatus) {
    const canceled = experience.inactiveTrialStatus === "canceled";
    const payment = experience.inactiveTrialStatus === "payment_failed";
    const title = payment
      ? "Your payment didn’t go through"
      : canceled
        ? "Your trial was canceled"
        : "Your trial has ended";
    return {
      eyebrow: payment
        ? "Payment needs attention"
        : canceled
          ? "Free trial · Canceled"
          : "Free trial · Ended",
      title,
      noticeTitle: title,
      description: payment
        ? (formatPaymentDue(experience.payment)?.description ??
          "Review your billing details to continue. Your chats, workflows and results are saved.")
        : canceled
          ? "Your trial won’t turn into a paid subscription. Your chats, workflows and results are saved whenever you’re ready to choose a plan."
          : "Your chats, workflows and results are saved. Choose a plan to keep going.",
      supportLabel: "Contact support",
      secondary: "Back to my work",
      trialSpent: false,
    };
  }
  const trialSpent = isActiveTrial && window === "trial";
  const windowLabel = window === "weekly" ? "Weekly" : "Daily";
  const trialDate = formatTrialDate(experience.trialEndsAt);
  const resumeTime = formatResumeTime(experience.resetsAt);
  const resume = resumeTime
    ? `You can continue ${resumeTime}`
    : "You can continue when your allowance refreshes";
  const title = trialSpent
    ? "You’ve used your trial allowance"
    : isTopTier
      ? resume
      : window === "weekly"
        ? "You’ve hit this week’s limit"
        : "You’ve hit today’s limit";
  return {
    eyebrow: `${formatTierLabel(tier) ?? "Your plan"} · ${trialSpent ? "Trial allowance used" : `${windowLabel.toLowerCase()} usage reached`}`,
    title,
    noticeTitle: title,
    description: trialSpent
      ? experience.freshProUsage
        ? formatTrialStartDescription(experience.trialEndsAt)
        : "Choose your next plan to continue. Your chats, workflows and results are saved."
      : isTopTier
        ? "Your chats, workflows and results are saved."
        : !experience.resetsAt
          ? "Choose a plan to continue. Your chats, workflows and results are saved."
          : tier === "PRO"
            ? `${resume}, or upgrade to Max for ${usageMultiplier ? `${usageMultiplier} times the usage` : "more usage"}.`
            : `${resume}, or ${tier === "BASIC" ? "upgrade to Pro" : "start Pro"}.`,
    supportLabel:
      tier === "ENTERPRISE"
        ? "Contact your account team"
        : tier === "BUSINESS"
          ? "Contact support"
          : "Contact us",
    secondary: trialSpent
      ? experience.freshProUsage && trialDate
        ? `Wait until ${trialDate}`
        : "Back to my work"
      : experience.resetsAt
        ? "Wait for reset"
        : "Back to my work",
    trialSpent,
  };
}
