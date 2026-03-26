import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Button } from "@/components/ui/button";
import { ChartBar } from "@phosphor-icons/react";
import { UsagePanelContent } from "./UsagePanelContent";

export { UsagePanelContent, formatResetTime } from "./UsagePanelContent";

export function UsageLimits() {
  const { data: usage, isLoading } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsageStatus,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });

  if (isLoading || !usage?.daily || !usage?.weekly) return null;
  if (usage.daily.limit <= 0 && usage.weekly.limit <= 0) return null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Usage limits">
          <ChartBar className="!size-5" weight="light" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-3">
        <UsagePanelContent usage={usage} />
      </PopoverContent>
    </Popover>
  );
}
