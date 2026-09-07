import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatTrialEnd, formatTrialPrice } from "./helpers";

interface Props {
  trial: TrialStatusResponse;
  isCanceling: boolean;
  onCancel: () => void;
}

export function TrialStatus({ trial, isCanceling, onCancel }: Props) {
  if (!trial.offer) return null;
  const end = formatTrialEnd(trial.ends_at);
  return (
    <div className="flex flex-col gap-3">
      <Text variant="h4">
        {trial.active ? "Your trial" : "Your trial has ended"}
      </Text>
      <Text variant="body">
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
        <Button
          variant="outline"
          onClick={onCancel}
          loading={isCanceling}
          disabled={isCanceling}
        >
          Cancel trial
        </Button>
      ) : null}
    </div>
  );
}
