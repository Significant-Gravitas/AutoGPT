import { useCopilotUsage } from "./useCopilotUsage";

export function useIsUsageLimitReached() {
  const { data: usage } = useCopilotUsage();
  const daily = usage?.daily?.percent_used ?? 0;
  const weekly = usage?.weekly?.percent_used ?? 0;
  return daily >= 100 || weekly >= 100;
}
