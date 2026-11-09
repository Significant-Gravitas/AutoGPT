import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { CancelTrialFeedbackDialog } from "@/components/organisms/FeedbackPrompts/CancelTrialFeedbackDialog";
import { formatTrialEnd } from "./helpers";
import { useCancelTrialOffer } from "./useCancelTrialOffer";

interface Props {
  trial: TrialStatusResponse;
  isCanceling: boolean;
  onCancel: () => void;
}

export function CancelTrialDialog({ trial, isCanceling, onCancel }: Props) {
  const offer = useCancelTrialOffer();

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
      starters={offer.starters}
      onOpen={offer.handleOpen}
      onReason={(reason) => offer.save({ reason })}
      onNote={(note) => offer.save({ note })}
      onConfirm={onCancel}
      onPickStarter={offer.startTask}
      onTryOtto={offer.tryOtto}
    />
  );
}
