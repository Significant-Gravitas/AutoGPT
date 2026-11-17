import type { CancelTrialRequest } from "@/app/api/__generated__/models/cancelTrialRequest";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { CancelTrialFeedbackDialog } from "@/components/organisms/FeedbackPrompts/CancelTrialFeedbackDialog";
import { formatTrialEnd } from "./helpers";
import { useCancelTrialPrompt } from "./useCancelTrialPrompt";

interface Props {
  trial: TrialStatusResponse;
  isCanceling: boolean;
  onCancel: (request: CancelTrialRequest) => void;
}

export function CancelTrialDialog({ trial, isCanceling, onCancel }: Props) {
  const prompt = useCancelTrialPrompt();

  return (
    <CancelTrialFeedbackDialog
      trigger={
        <Button
          variant="outline"
          size="small"
          loading={isCanceling}
          disabled={isCanceling}
        >
          Cancel trial
        </Button>
      }
      endDate={formatTrialEnd(trial.ends_at)}
      isCanceling={isCanceling}
      onOpen={prompt.handleOpen}
      onReason={prompt.handleReason}
      onConfirm={(answer) => onCancel(prompt.toCancelRequest(answer))}
    />
  );
}
