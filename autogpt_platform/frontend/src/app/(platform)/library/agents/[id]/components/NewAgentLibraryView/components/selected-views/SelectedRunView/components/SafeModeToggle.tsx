import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Graph } from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { ShieldCheckIcon, ShieldIcon } from "@phosphor-icons/react";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

interface Props {
  graph: GraphModel | LibraryAgent | Graph;
  className?: string;
}

interface SafeModeIconButtonProps {
  isEnabled: boolean;
  label: string;
  tooltipEnabled: string;
  tooltipDisabled: string;
  onToggle: () => void;
  isPending: boolean;
}

function SafeModeIconButton({
  isEnabled,
  label,
  tooltipEnabled,
  tooltipDisabled,
  onToggle,
  isPending,
}: SafeModeIconButtonProps) {
  return (
    <Tooltip delayDuration={100}>
      <TooltipTrigger asChild>
        <Button
          variant="icon"
          size="icon"
          aria-label={`${label}: ${isEnabled ? "ON" : "OFF"}. ${isEnabled ? tooltipEnabled : tooltipDisabled}`}
          onClick={onToggle}
          disabled={isPending}
          className={cn(isPending ? "opacity-0" : "opacity-100")}
        >
          {isEnabled ? (
            <ShieldCheckIcon weight="bold" size={16} />
          ) : (
            <ShieldIcon weight="bold" size={16} />
          )}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        <div className="text-center">
          <div className="font-medium">
            {label}: {isEnabled ? "ON" : "OFF"}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {isEnabled ? tooltipEnabled : tooltipDisabled}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

export function SafeModeToggle({ graph, className }: Props) {
  const {
    currentHITLSafeMode,
    showHITLToggle,
    handleHITLToggle,
    currentSensitiveActionSafeMode,
    showSensitiveActionToggle,
    handleSensitiveActionToggle,
    isPending,
    shouldShowToggle,
  } = useAgentSafeMode(graph);

  if (!shouldShowToggle) {
    return null;
  }

  return (
    <div className={cn("flex gap-1", className)}>
      {showHITLToggle && (
        <SafeModeIconButton
          isEnabled={currentHITLSafeMode}
          label="Human-in-the-loop"
          tooltipEnabled="The agent will pause at human-in-the-loop blocks and wait for your approval"
          tooltipDisabled="Human-in-the-loop blocks will proceed automatically"
          onToggle={handleHITLToggle}
          isPending={isPending}
        />
      )}
      {showSensitiveActionToggle && (
        <SafeModeIconButton
          isEnabled={currentSensitiveActionSafeMode}
          label="Sensitive actions"
          tooltipEnabled="The agent will pause at sensitive action blocks and wait for your approval"
          tooltipDisabled="Sensitive action blocks will proceed automatically"
          onToggle={handleSensitiveActionToggle}
          isPending={isPending}
        />
      )}
    </div>
  );
}
