import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Button } from "@/components/ui/button";
import { ChartBar } from "@phosphor-icons/react";
import { useUsageLimits } from "./useUsageLimits";
import { UsagePanelContent } from "./UsagePanelContent";

export { UsagePanelContent, formatResetTime } from "./UsagePanelContent";

export function UsageLimits() {
  const { data: usage, isLoading } = useUsageLimits();

  if (isLoading || !usage) return null;
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
