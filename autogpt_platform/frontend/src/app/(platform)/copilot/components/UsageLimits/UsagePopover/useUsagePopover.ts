import { useCopilotUsage } from "../useCopilotUsage";

export function useUsagePopover() {
  const { data: usage, isSuccess } = useCopilotUsage();
  return { usage, isSuccess };
}
