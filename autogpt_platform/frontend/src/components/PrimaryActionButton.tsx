import React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { FaSpinner } from "react-icons/fa";
import { Clock, LogOut } from "lucide-react";
import { IconPlay, IconSquare } from "@/components/ui/icons";

interface PrimaryActionBarProps {
  onClickAgentOutputs: () => void;
  onClickRunAgent?: () => void;
  onClickStopRun: () => void;
  onClickScheduleButton?: () => void;
  isRunning: boolean;
  isDisabled: boolean;
  isScheduling: boolean;
  className?: string;
}

const PrimaryActionBar: React.FC<PrimaryActionBarProps> = ({
  onClickAgentOutputs,
  onClickRunAgent,
  onClickStopRun,
  onClickScheduleButton,
  isRunning,
  isDisabled,
  isScheduling,
  className,
}) => {
  const buttonClasses =
    "flex items-center gap-2 text-sm font-medium md:text-lg";
  return (
    <div
      className={cn(
        "flex w-fit select-none items-center justify-center p-4",
        className,
      )}
    >
      <div className="flex gap-1 md:gap-4">
        <Button
          className={buttonClasses}
          variant="outline"
          size="primary"
          onClick={onClickAgentOutputs}
          title="View agent outputs"
        >
          <LogOut className="hidden size-5 md:flex" /> Agent Outputs
        </Button>

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
            data-id="primary-action-run-agent"
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
            disabled={isScheduling}
            title="Set up a run schedule for the agent"
            data-id="primary-action-schedule-agent"
          >
            {isScheduling ? (
              <FaSpinner className="animate-spin" />
            ) : (
              <Clock className="hidden h-5 w-5 md:flex" />
            )}
            Schedule Run
          </Button>
        )}
      </div>
    </div>
  );
};

export default PrimaryActionBar;
