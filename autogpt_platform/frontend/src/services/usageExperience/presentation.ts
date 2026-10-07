import { formatTierLabel } from "@/app/(platform)/copilot/components/usageHelpers";
import { formatUsageDate, type UsageExperience } from "./helpers";

export function usagePresentation(experience: UsageExperience) {
  const { tier, window, isTopTier, isActiveTrial } = experience;
  if (experience.inactiveTrialStatus) {
    const canceled = ["canceled", "cancelled"].includes(
      experience.inactiveTrialStatus,
    );
    const payment = experience.inactiveTrialStatus === "payment_failed";
    return {
      eyebrow: payment
        ? "Payment needs attention"
        : canceled
          ? "Free trial · Canceled"
          : "Free trial · Ended",
      title: payment
        ? "Let’s get you back to work."
        : "Keep what you’ve started.",
      noticeTitle: payment
        ? "Your payment needs attention"
        : canceled
          ? "Your trial was canceled"
          : "Your trial has ended",
      description: payment
        ? "Your payment couldn’t be completed. Review your billing details to continue."
        : canceled
          ? "Your trial won’t turn into a paid subscription. Your chats, agents, and results are saved whenever you’re ready to choose a plan."
          : "Your trial has ended. Your chats, agents, and results are saved. Choose a plan to keep going.",
      resetLabel: null,
      supportLabel: "Contact support",
      secondary: "Back to my work",
      trialSpent: false,
    };
  }
  const trialSpent = isActiveTrial && window === "trial";
  const windowLabel = window === "weekly" ? "Weekly" : "Daily";
  return {
    eyebrow: `${formatTierLabel(tier) ?? "Your plan"} · ${trialSpent ? "Trial allowance used" : `${windowLabel.toLowerCase()} usage reached`}`,
    title: trialSpent
      ? "Keep your momentum."
      : isTopTier
        ? "A little pause. Everything is saved."
        : "More room for what’s next.",
    description: trialSpent
      ? experience.freshProUsage
        ? "You’ve used your trial allowance. Start Pro with fresh daily and weekly usage, and pick up right where you left off."
        : "You’ve used your trial allowance. Choose your next plan and pick up where you left off."
      : `You’ve reached your ${windowLabel.toLowerCase()} usage limit. Your chats, agents, and results are still here.`,
    noticeTitle: trialSpent
      ? "Ready for your next idea?"
      : `${windowLabel} usage reached`,
    resetLabel: experience.resetsAt
      ? formatUsageDate(experience.resetsAt)
      : null,
    supportLabel:
      tier === "ENTERPRISE"
        ? "Contact your account team"
        : tier === "BUSINESS"
          ? "Contact support"
          : "Contact us",
    secondary: trialSpent ? "Back to my work" : "Wait for reset",
    trialSpent,
  };
}
