import { Button } from "@/components/atoms/Button/Button";

interface Props {
  onGoBack: () => void;
  onSchedule: () => void;
  isCreatingSchedule?: boolean;
  allRequiredInputsAreSet?: boolean;
}

export function ScheduleActions({
  onGoBack,
  onSchedule,
  isCreatingSchedule = false,
  allRequiredInputsAreSet = true,
}: Props) {
  return (
    <div className="flex justify-end gap-3">
      <Button variant="ghost" onClick={onGoBack}>
        Go Back
      </Button>
      <Button
        variant="primary"
        onClick={onSchedule}
        disabled={!allRequiredInputsAreSet || isCreatingSchedule}
        loading={isCreatingSchedule}
      >
        Create Schedule
      </Button>
    </div>
  );
}
