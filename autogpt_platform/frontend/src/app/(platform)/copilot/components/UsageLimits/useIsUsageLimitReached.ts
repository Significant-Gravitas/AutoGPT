import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { spendsPlatformBudget } from "../usageHelpers";

/** The connection the chat runs on decides: only a platform chat waits for
 *  the AutoPilot budget. */
export function useIsUsageLimitReached(authProvider?: string | null) {
  const state = useUsageExperience();
  if (!spendsPlatformBudget(authProvider)) return false;
  return state.experience.blocked || state.isError;
}
