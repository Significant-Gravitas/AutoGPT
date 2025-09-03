import { Button } from "@/components/atoms/Button/Button";

interface Props {
  onSchedule: () => void;
  isCreatingSchedule?: boolean;
  allRequiredInputsAreSet?: boolean;
}

export function ScheduleActions({
  onSchedule,
  isCreatingSchedule = false,
  allRequiredInputsAreSet = true,
}: Props) {
  return (
    <div className="flex justify-end gap-3">
      <Button
        variant="primary"
        onClick={onSchedule}
        disabled={!allRequiredInputsAreSet || isCreatingSchedule}
        loading={isCreatingSchedule}
      >
        Schedule Agent
      </Button>
    </div>
  );
}
