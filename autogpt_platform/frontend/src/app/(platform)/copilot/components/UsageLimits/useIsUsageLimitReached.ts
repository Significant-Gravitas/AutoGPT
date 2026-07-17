import { isUsageExhausted } from "../usageHelpers";
import { useCopilotUsage } from "./useCopilotUsage";

export function useIsUsageLimitReached() {
  const { data: usage } = useCopilotUsage();
  return isUsageExhausted(usage);
}
