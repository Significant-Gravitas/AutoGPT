import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { TrialTimeLeft } from "./components/TrialTimeLeft";
import { WorthDoingPanel } from "./components/WorthDoingPanel";

interface Props {
  trial: TrialStatusResponse;
  isOpen: boolean;
  isResuming: boolean;
  onResume: () => void;
  onSubscribe: () => void;
  onClose: () => void;
}

export function TrialCanceledDialog({
  trial,
  isOpen,
  isResuming,
  onResume,
  onSubscribe,
  onClose,
}: Props) {
  if (!trial.offer) return null;
  return (
    <Dialog
      title="Cancellation confirmed"
      styling={{ maxWidth: "29rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" unmask={false} className="!text-zinc-800">
            Your card won&apos;t be charged.{" "}
            <TrialTimeLeft endsAt={trial.ends_at} /> Nothing changes until then.
          </Text>
          <WorthDoingPanel offer={trial.offer} endsAt={trial.ends_at} />
        </div>
        <Dialog.Footer>
          <Button
            variant="outline"
            onClick={onResume}
            loading={isResuming}
            disabled={isResuming}
          >
            Resume trial
          </Button>
          <Button variant="primary" onClick={onSubscribe}>
            Subscribe now
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
