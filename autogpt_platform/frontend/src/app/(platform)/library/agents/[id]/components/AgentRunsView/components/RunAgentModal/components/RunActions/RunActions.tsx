import { Button } from "@/components/atoms/Button/Button";
import { RunVariant } from "../../useAgentRunModal";

interface Props {
  defaultRunType: RunVariant;
  onRun: () => void;
  isExecuting?: boolean;
  isSettingUpTrigger?: boolean;
  allRequiredInputsAreSet?: boolean;
}

export function RunActions({
  defaultRunType,
  onRun,
  isExecuting = false,
  isSettingUpTrigger = false,
  allRequiredInputsAreSet = true,
}: Props) {
  return (
    <div className="flex justify-end gap-3">
      <Button
        variant="primary"
        onClick={onRun}
        disabled={!allRequiredInputsAreSet || isExecuting || isSettingUpTrigger}
        loading={isExecuting || isSettingUpTrigger}
      >
        {defaultRunType === "automatic-trigger"
          ? "Set up Trigger"
          : "Run Agent"}
      </Button>
    </div>
  );
}
