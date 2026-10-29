export type ChatAllowance =
  | "unknown"
  | "no-plan"
  | "lifetime-trial"
  | "unlimited"
  | "daily"
  | "weekly"
  | "daily-weekly";

interface PlanState {
  tier: string | null;
  isActiveTrial: boolean;
  isLifetimeTrial: boolean;
}

interface UsageWindows {
  daily?: unknown | null;
  weekly?: unknown | null;
}

// Read the same way as the billing page's usage card, so the two never
// describe one plan differently.
export function getChatAllowance(
  plan: PlanState,
  usage: UsageWindows | undefined,
): ChatAllowance {
  if (!usage) return "unknown";
  if (plan.tier === "NO_TIER") return "no-plan";
  if (plan.isLifetimeTrial) return "lifetime-trial";
  if (usage.daily && usage.weekly) return "daily-weekly";
  if (usage.daily) return "daily";
  if (usage.weekly) return "weekly";
  if (plan.isActiveTrial) return "lifetime-trial";
  // A trial account whose trial status hasn't loaded, or failed to, is not
  // an uncapped plan: say nothing about its limits.
  return plan.tier === "TRIAL" ? "unknown" : "unlimited";
}

interface Args {
  paymentEnabled: boolean;
  allowance: ChatAllowance;
  isTrial: boolean;
}

export function getUsageCreditsAnswers({
  paymentEnabled,
  allowance,
  isTrial,
}: Args) {
  return [
    {
      question: "What does chat usage cover?",
      answer: `Your chats with Otto and your experts, including the research and writing they do for you. ${describeAllowance(allowance, isTrial)}`,
    },
    {
      question: "What are automation credits for?",
      // An Otto (AutoPilot) step bills the allowance first and the credit
      // balance only once it is out: copilot/credit_billing.py.
      answer: `Workflow runs and paid steps, like making images. A workflow step where Otto does the work uses your chat usage first, and your credits only once chat usage runs out. ${
        paymentEnabled
          ? "You can buy credits, earn them by finishing setup steps or turn on auto-refill to top them up."
          : "You can earn them by finishing setup steps."
      }`,
    },
    // With payments off there is no checkout, upgrade or top-up to charge it.
    ...(paymentEnabled
      ? [
          {
            question: "When is my card charged?",
            // A failed renewal is paid from the balance when it covers the
            // invoice: subscription_payment_failure.py, pay_invoice_from_wallet.
            answer:
              "A chat or a step never charges your card directly. Your card is charged when your plan starts and each time it renews, when you upgrade, and when you buy credits (including auto-refill). If a renewal payment fails, it's taken from your automation credits when they cover it.",
          },
        ]
      : []),
    {
      question: "How do I get more?",
      answer: [
        describeMoreChat(allowance, paymentEnabled),
        paymentEnabled
          ? "For more automation credits, select Add credits in the wallet."
          : "For more automation credits, finish the setup steps in the wallet.",
      ]
        .filter(Boolean)
        .join(" "),
    },
  ];
}

function describeAllowance(allowance: ChatAllowance, isTrial: boolean) {
  const owner = isTrial ? "your trial" : "your plan";
  switch (allowance) {
    case "no-plan":
      return "It comes with a plan.";
    case "lifetime-trial":
      return "It comes with your trial, as one allowance for the whole trial.";
    case "unlimited":
      return "It comes with your plan, which has no daily or weekly limits.";
    case "daily-weekly":
      return `It comes with ${owner} and resets daily and weekly.`;
    case "daily":
      return `It comes with ${owner} and resets daily.`;
    case "weekly":
      return `It comes with ${owner} and resets weekly.`;
    case "unknown":
      return "It comes with your plan.";
  }
}

// Plans are bought and upgraded only with payments on, and an uncapped plan
// has no more to get.
function describeMoreChat(allowance: ChatAllowance, paymentEnabled: boolean) {
  if (!paymentEnabled || allowance === "unlimited") return null;
  if (allowance === "no-plan") return "For chat usage, choose a plan.";
  return "For more chat usage, choose a bigger plan.";
}
