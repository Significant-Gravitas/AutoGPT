import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { RunVariant } from "../../useAgentRunModal";
import { FlaskConicalIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
          <Icon icon={FlaskConicalIcon} size={16} />
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
