import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Text } from "@/components/atoms/Text/Text";
import { CancelTrialDialog } from "./CancelTrialDialog";
import { formatTrialEnd, formatTrialPrice } from "./helpers";
import { TrialRejection } from "./TrialRejection";

interface Props {
  trial: TrialStatusResponse;
  isCanceling: boolean;
  onCancel: () => void;
}

export function TrialStatus({ trial, isCanceling, onCancel }: Props) {
  if (!trial.offer) return null;
  if (trial.status === "canceled" && trial.rejection_reason)
    return <TrialRejection reason={trial.rejection_reason} />;
  const end = formatTrialEnd(trial.ends_at);
  return (
    <div className="flex flex-col gap-3">
      <Text variant="h4">
        {trial.active ? "Your trial" : "Your trial has ended"}
      </Text>
      <Text variant="body" unmask={false}>
        {trial.status === "canceled"
          ? "Cancellation confirmed. Trial access has ended and your trial will not convert to a paid plan."
          : trial.cancel_at_period_end
            ? `Cancellation confirmed. Your trial will not convert to a paid plan. Trial access ends ${end}.`
            : trial.active
              ? `Your trial ends ${end}. Your saved card will then be charged ${formatTrialPrice(trial.offer)}, plus applicable tax.`
              : "Paid access requires a successful payment. Review your payment method and plan below."}
      </Text>
      {trial.active ? (
        <Text variant="small">Canceling ends trial access immediately.</Text>
      ) : null}
      {trial.active && !trial.cancel_at_period_end ? (
        <CancelTrialDialog isCanceling={isCanceling} onCancel={onCancel} />
      ) : null}
    </div>
  );
}
