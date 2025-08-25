import { Button } from "@/components/atoms/Button/Button";
import { RunVariant } from "../../useAgentRunModal";

interface Props {
  hasExternalTrigger: boolean;
  defaultRunType: RunVariant;
  onShowSchedule: () => void;
  onRun: () => void;
  isExecuting?: boolean;
  isSettingUpTrigger?: boolean;
  allRequiredInputsAreSet?: boolean;
}

export function RunActions({
  hasExternalTrigger,
  defaultRunType,
  onShowSchedule,
  onRun,
  isExecuting = false,
  isSettingUpTrigger = false,
  allRequiredInputsAreSet = true,
}: Props) {
  return (
    <div className="flex justify-end gap-3">
      {!hasExternalTrigger && (
        <Button variant="secondary" onClick={onShowSchedule}>
          Schedule Run
        </Button>
      )}
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
