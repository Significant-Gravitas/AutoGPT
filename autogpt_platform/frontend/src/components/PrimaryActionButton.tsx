import React, { useState } from "react";
import { Button } from "./ui/button";
import { Clock, LogOut, ChevronLeft } from "lucide-react";
import { IconPlay, IconSquare } from "@/components/ui/icons";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FaSpinner } from "react-icons/fa";

interface PrimaryActionBarProps {
  onClickAgentOutputs: () => void;
  onClickRunAgent: () => void;
  onClickScheduleButton: () => void;
  isRunning: boolean;
  isDisabled: boolean;
  isScheduling: boolean;
  requestStopRun: () => void;
  runAgentTooltip: string;
}

const PrimaryActionBar: React.FC<PrimaryActionBarProps> = ({
  onClickAgentOutputs,
  onClickRunAgent,
  onClickScheduleButton,
  isRunning,
  isDisabled,
  isScheduling,
  requestStopRun,
  runAgentTooltip,
}) => {
  const runButtonLabel = !isRunning ? "Run" : "Stop";

  const runButtonIcon = !isRunning ? <IconPlay /> : <IconSquare />;

  const runButtonOnClick = !isRunning ? onClickRunAgent : requestStopRun;

  return (
    <div className="absolute bottom-0 left-1/2 z-50 flex w-fit -translate-x-1/2 transform select-none items-center justify-center p-4">
      <div className={`flex gap-1 md:gap-4`}>
        <Tooltip key="ViewOutputs" delayDuration={500}>
          <TooltipTrigger asChild>
            <Button
              className="flex items-center gap-2"
              onClick={onClickAgentOutputs}
              size="primary"
              variant="outline"
            >
              <LogOut className="hidden h-5 w-5 md:flex" />
              <span className="text-sm font-medium md:text-lg">
                Agent Outputs{" "}
              </span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>View agent outputs</p>
          </TooltipContent>
        </Tooltip>
        <Tooltip key="RunAgent" delayDuration={500}>
          <TooltipTrigger asChild>
            <Button
              className="flex items-center gap-2"
              onClick={runButtonOnClick}
              size="primary"
              style={{
                background: isRunning ? "#DF4444" : "#7544DF",
                opacity: isDisabled ? 0.5 : 1,
              }}
              data-id="primary-action-run-agent"
            >
              {runButtonIcon}
              <span className="text-sm font-medium md:text-lg">
                {runButtonLabel}
              </span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{runAgentTooltip}</p>
          </TooltipContent>
        </Tooltip>
        <Tooltip key="ScheduleAgent" delayDuration={500}>
          <TooltipTrigger asChild>
            <Button
              className="flex items-center gap-2"
              onClick={onClickScheduleButton}
              size="primary"
              disabled={isScheduling}
              variant="outline"
              data-id="primary-action-schedule-agent"
            >
              {isScheduling ? (
                <FaSpinner className="animate-spin" />
              ) : (
                <Clock className="hidden h-5 w-5 md:flex" />
              )}
              <span className="text-sm font-medium md:text-lg">
                Schedule Run
              </span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Schedule this Agent</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
};

export default PrimaryActionBar;
