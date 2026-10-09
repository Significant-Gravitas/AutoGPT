import { useState } from "react";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { formatTrialEnd } from "./helpers";

interface Props {
  trial: TrialStatusResponse;
  isCanceling: boolean;
  onCancel: () => void;
}

export function CancelTrialDialog({ trial, isCanceling, onCancel }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const keepsAccess = Boolean(trial.cancel_keeps_access);

  function confirmCancellation() {
    setIsOpen(false);
    onCancel();
  }

  return (
    <Dialog
      title={keepsAccess ? "Cancel your trial?" : "End your trial now?"}
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button
          variant="outline"
          size="small"
          loading={isCanceling}
          disabled={isCanceling}
        >
          Cancel trial
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        {keepsAccess ? (
          <div className="flex flex-col gap-3">
            <Text variant="body" className="!text-zinc-800">
              Your trial won&apos;t convert to a paid plan and your card
              won&apos;t be charged.
            </Text>
            <Text variant="body" unmask={false} className="!text-zinc-800">
              You keep full access until{" "}
              <strong className="font-semibold">
                {formatTrialEnd(trial.ends_at)}
              </strong>
              , and you can resume your trial any time before then.
            </Text>
          </div>
        ) : (
          <Text variant="body">
            Your trial access will end immediately and your trial will not
            convert to a paid plan. You cannot restart this trial.
          </Text>
        )}
        <Dialog.Footer>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Keep trial
          </Button>
          <Button
            variant={keepsAccess ? "primary" : "destructive"}
            onClick={confirmCancellation}
            disabled={isCanceling}
          >
            {keepsAccess ? "Cancel trial" : "End trial now"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
