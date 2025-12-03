import { ShieldIcon, ShieldCheckIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Graph } from "@/lib/autogpt-server-api/types";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";

interface FloatingSafeModeToggleProps {
  graph: GraphModel | LibraryAgent | Graph;
  className?: string;
  fullWidth?: boolean;
}

export function FloatingSafeModeToggle({
  graph,
  className,
  fullWidth = false,
}: FloatingSafeModeToggleProps) {
  const {
    currentSafeMode,
    isPending,
    shouldShowToggle,
    isStateUndetermined,
    handleToggle,
  } = useAgentSafeMode(graph);

  if (!shouldShowToggle || isStateUndetermined) {
    return null;
  }

  return (
    <div className={cn("fixed z-50", className)}>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant="primary"
            size="small"
            onClick={handleToggle}
            disabled={isPending}
            loading={isPending}
            className={cn(fullWidth ? "w-full" : "")}
          >
            {currentSafeMode! ? (
              <>
                <ShieldCheckIcon className="h-4 w-4" />
                Safe Mode: ON
              </>
            ) : (
              <>
                <ShieldIcon className="h-4 w-4" />
                Safe Mode: OFF
              </>
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-center">
            <div className="font-medium">
              Safe Mode: {currentSafeMode! ? "ON" : "OFF"}
            </div>
            <div className="mt-1 text-xs text-muted-foreground">
              {currentSafeMode!
                ? "HITL blocks require manual review"
                : "HITL blocks proceed automatically"}
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
