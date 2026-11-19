import { Button } from "@/components/atoms/Button/Button";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { WizardPlaceholder } from "../WizardPlaceholder";
import { useLoadingSetup } from "./useLoadingSetup";

interface Props {
  totalSteps: number;
  waitingFor: string[];
  onSlow: () => void;
}

export function LoadingSetup({ totalSteps, waitingFor, onSlow }: Props) {
  const { isSlow, noticeRef, reload } = useLoadingSetup({
    waitingFor,
    onSlow,
  });

  if (isSlow)
    return (
      <div
        ref={noticeRef}
        tabIndex={-1}
        className="flex w-full max-w-md flex-col items-center gap-4 px-4 outline-none"
      >
        <Alert variant="warning">
          <AlertTitle>Your setup is taking longer than usual</AlertTitle>
          <AlertDescription>
            Reload to continue; your progress is saved.
          </AlertDescription>
        </Alert>
        <Button
          type="button"
          size="small"
          leadingIcon={Refresh01Icon}
          onClick={reload}
        >
          Reload
        </Button>
      </div>
    );

  return (
    <WizardPlaceholder message="Loading your setup…" totalSteps={totalSteps} />
  );
}
