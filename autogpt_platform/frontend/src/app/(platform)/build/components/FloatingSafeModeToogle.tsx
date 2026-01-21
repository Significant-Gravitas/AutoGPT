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

interface SafeModeButtonProps {
  isEnabled: boolean;
  label: string;
  tooltipEnabled: string;
  tooltipDisabled: string;
  onToggle: () => void;
  isPending: boolean;
  fullWidth?: boolean;
}

function SafeModeButton({
  isEnabled,
  label,
  tooltipEnabled,
  tooltipDisabled,
  onToggle,
  isPending,
  fullWidth = false,
}: SafeModeButtonProps) {
  return (
    <Tooltip delayDuration={100}>
      <TooltipTrigger asChild>
        <Button
          variant={isEnabled ? "primary" : "outline"}
          size="small"
          onClick={onToggle}
          disabled={isPending}
          className={cn("justify-start", fullWidth ? "w-full" : "")}
        >
          {isEnabled ? (
            <>
              <ShieldCheckIcon weight="bold" size={16} />
              <Text variant="body" className="text-zinc-200">
                {label}: ON
              </Text>
            </>
          ) : (
            <>
              <ShieldIcon weight="bold" size={16} />
              <Text variant="body" className="text-zinc-600">
                {label}: OFF
              </Text>
            </>
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

export function FloatingSafeModeToggle({
  graph,
  className,
  fullWidth = false,
}: Props) {
  const {
    currentHITLSafeMode,
    showHITLToggle,
    isHITLStateUndetermined,
    handleHITLToggle,
    currentSensitiveActionSafeMode,
    showSensitiveActionToggle,
    handleSensitiveActionToggle,
    isPending,
    shouldShowToggle,
  } = useAgentSafeMode(graph);

  if (!shouldShowToggle || isPending) {
    return null;
  }

  const showHITL = showHITLToggle && !isHITLStateUndetermined;
  const showSensitive = showSensitiveActionToggle;

  if (!showHITL && !showSensitive) {
    return null;
  }

  return (
    <div className={cn("fixed z-50 flex flex-col gap-2", className)}>
      {showHITL && (
        <SafeModeButton
          isEnabled={currentHITLSafeMode}
          label="Human in the loop block approval"
          tooltipEnabled="The agent will pause at human-in-the-loop blocks and wait for your approval"
          tooltipDisabled="Human in the loop blocks will proceed automatically"
          onToggle={handleHITLToggle}
          isPending={isPending}
          fullWidth={fullWidth}
        />
      )}
      {showSensitive && (
        <SafeModeButton
          isEnabled={currentSensitiveActionSafeMode}
          label="Sensitive actions blocks approval"
          tooltipEnabled="The agent will pause at sensitive action blocks and wait for your approval"
          tooltipDisabled="Sensitive action blocks will proceed automatically"
          onToggle={handleSensitiveActionToggle}
          isPending={isPending}
          fullWidth={fullWidth}
        />
      )}
    </div>
  );
}
