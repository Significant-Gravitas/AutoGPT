import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { FlaskIcon } from "@phosphor-icons/react";
import { RunVariant } from "../../useAgentRunModal";

interface Props {
  defaultRunType: RunVariant;
  onRun: () => void;
  onSimulate?: () => void;
  isExecuting?: boolean;
  isSettingUpTrigger?: boolean;
  isRunReady?: boolean;
  scheduleButton?: React.ReactNode;
}

export function RunActions({
  defaultRunType,
  onRun,
  onSimulate,
  isExecuting = false,
  isSettingUpTrigger = false,
  isRunReady = true,
  scheduleButton,
}: Props) {
  const isTrigger =
    defaultRunType === "automatic-trigger" ||
    defaultRunType === "manual-trigger";

  return (
    <div className="flex justify-end gap-3">
      {!isTrigger && onSimulate && (
        <Button
          variant="ghost"
          onClick={onSimulate}
          disabled={isExecuting || isSettingUpTrigger}
          loading={isExecuting}
          className="gap-1.5 text-amber-600 hover:bg-amber-50 hover:text-amber-700"
        >
          <FlaskIcon size={16} weight="fill" />
          Simulate
        </Button>
      )}
      {scheduleButton}
      <Button
        variant="primary"
        onClick={onRun}
        disabled={!isRunReady || isExecuting || isSettingUpTrigger}
        loading={isExecuting || isSettingUpTrigger}
      >
        {isTrigger ? "Set up Trigger" : "Start Task"}
      </Button>
    </div>
  );
}
