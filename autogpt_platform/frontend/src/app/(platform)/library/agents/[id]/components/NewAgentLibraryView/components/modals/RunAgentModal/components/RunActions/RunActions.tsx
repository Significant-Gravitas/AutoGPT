import { Button } from "@/components/atoms/Button/Button";
import { RunVariant } from "../../useAgentRunModal";

interface Props {
  defaultRunType: RunVariant;
  onRun: () => void;
  isExecuting?: boolean;
  isSettingUpTrigger?: boolean;
  isRunReady?: boolean;
}

export function RunActions({
  defaultRunType,
  onRun,
  isExecuting = false,
  isSettingUpTrigger = false,
  isRunReady = true,
}: Props) {
  return (
    <div className="flex justify-end gap-3">
      <Button
        variant="primary"
        onClick={onRun}
        disabled={!isRunReady || isExecuting || isSettingUpTrigger}
        loading={isExecuting || isSettingUpTrigger}
      >
        {defaultRunType === "automatic-trigger" ||
        defaultRunType === "manual-trigger"
          ? "Set up Trigger"
          : "Start Task"}
      </Button>
    </div>
  );
}
