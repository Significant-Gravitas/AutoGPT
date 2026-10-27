import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { getChatAllowance, getUsageCreditsAnswers } from "./helpers";

export function useUsageCreditsDialog() {
  // The same flag the wallet reads to show Add credits, and the same plan
  // state the billing page's usage card reads.
  const { experience, usage, isBillingEnabled } = useUsageExperience();
  return {
    answers: getUsageCreditsAnswers({
      paymentEnabled: isBillingEnabled === true,
      allowance: getChatAllowance(experience, usage),
      isTrial: experience.isActiveTrial,
    }),
  };
}
