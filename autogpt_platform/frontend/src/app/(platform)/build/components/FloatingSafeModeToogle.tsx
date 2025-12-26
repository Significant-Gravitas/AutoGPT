import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Graph } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { ShieldCheckIcon, ShieldIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

interface Props {
  graph: GraphModel | LibraryAgent | Graph;
  className?: string;
  fullWidth?: boolean;
}

export function FloatingSafeModeToggle({
  graph,
  className,
  fullWidth = false,
}: Props) {
  const {
    currentSafeMode,
    isPending,
    shouldShowToggle,
    isStateUndetermined,
    handleToggle,
  } = useAgentSafeMode(graph);

  if (!shouldShowToggle || isStateUndetermined || isPending) {
    return null;
  }

  return (
    <div className={cn("fixed z-50", className)}>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant={currentSafeMode! ? "primary" : "outline"}
            key={graph.id}
            size="small"
            title={
              currentSafeMode!
                ? "Safe Mode: ON. Human in the loop blocks require manual review"
                : "Safe Mode: OFF. Human in the loop blocks proceed automatically"
            }
            onClick={handleToggle}
            className={cn(fullWidth ? "w-full" : "")}
          >
            {currentSafeMode! ? (
              <>
                <ShieldCheckIcon weight="bold" size={16} />
                <Text variant="body" className="text-zinc-200">
                  Safe Mode: ON
                </Text>
              </>
            ) : (
              <>
                <ShieldIcon weight="bold" size={16} />
                <Text variant="body" className="text-zinc-600">
                  Safe Mode: OFF
                </Text>
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
                ? "Human in the loop blocks require manual review"
                : "Human in the loop blocks proceed automatically"}
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
