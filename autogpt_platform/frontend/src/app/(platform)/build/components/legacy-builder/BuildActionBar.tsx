import React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/__legacy__/ui/button";
import { LogOut } from "lucide-react";
import { ClockIcon, WarningIcon } from "@phosphor-icons/react";
import { IconPlay, IconSquare } from "@/components/__legacy__/ui/icons";

interface Props {
  onClickAgentOutputs?: () => void;
  onClickRunAgent?: () => void;
  onClickStopRun: () => void;
  onClickScheduleButton?: () => void;
  isRunning: boolean;
  isDisabled: boolean;
  className?: string;
  resolutionModeActive?: boolean;
}

export const BuildActionBar: React.FC<Props> = ({
  onClickAgentOutputs,
  onClickRunAgent,
  onClickStopRun,
  onClickScheduleButton,
  isRunning,
  isDisabled,
  className,
  resolutionModeActive = false,
}) => {
  const buttonClasses =
    "flex items-center gap-2 text-sm font-medium md:text-lg";

  // Show resolution mode message instead of action buttons
  if (resolutionModeActive) {
    return (
      <div
        className={cn(
          "flex w-fit select-none items-center justify-center p-4",
          className,
        )}
      >
        <div className="flex items-center gap-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 dark:border-amber-700 dark:bg-amber-900/30">
          <WarningIcon className="size-5 text-amber-600 dark:text-amber-400" />
          <span className="text-sm font-medium text-amber-800 dark:text-amber-200">
            Remove incompatible connections to continue
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex w-fit select-none items-center justify-center p-4",
        className,
      )}
    >
      <div className="flex gap-1 md:gap-4">
        {onClickAgentOutputs && (
          <Button
            className={buttonClasses}
            variant="outline"
            size="primary"
            onClick={onClickAgentOutputs}
            title="View agent outputs"
          >
            <LogOut className="hidden size-5 md:flex" /> Agent Outputs
          </Button>
        )}

        {!isRunning ? (
          <Button
            className={cn(
              buttonClasses,
              onClickRunAgent && isDisabled
                ? "cursor-default opacity-50 hover:bg-accent"
                : "",
            )}
            variant="accent"
            size="primary"
            onClick={onClickRunAgent}
            disabled={!onClickRunAgent}
            title="Run the agent"
            aria-label="Run the agent"
            data-testid="primary-action-run-agent"
          >
            <IconPlay /> Run
          </Button>
        ) : (
          <Button
            className={buttonClasses}
            variant="destructive"
            size="primary"
            onClick={onClickStopRun}
            title="Stop the agent"
            data-id="primary-action-stop-agent"
          >
            <IconSquare /> Stop
          </Button>
        )}

        {onClickScheduleButton && (
          <Button
            className={buttonClasses}
            variant="outline"
            size="primary"
            onClick={onClickScheduleButton}
            title="Set up a run schedule for the agent"
            data-id="primary-action-schedule-agent"
          >
            <ClockIcon className="hidden h-5 w-5 md:flex" />
            Schedule Run
          </Button>
        )}
      </div>
    </div>
  );
};
