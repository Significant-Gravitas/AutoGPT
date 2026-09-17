import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isCanceling: boolean;
  onCancel: () => void;
}

export function CancelTrialDialog({ isCanceling, onCancel }: Props) {
  const [isOpen, setIsOpen] = useState(false);

  function confirmCancellation() {
    setIsOpen(false);
    onCancel();
  }

  return (
    <Dialog
      title="End your trial now?"
      styling={{ maxWidth: "440px" }}
      controlled={{ isOpen, set: setIsOpen }}
    >
      <Dialog.Trigger>
        <Button variant="outline" loading={isCanceling} disabled={isCanceling}>
          Cancel trial
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <Text variant="body">
          Your trial access will end immediately and your trial will not convert
          to a paid plan. You cannot restart this trial.
        </Text>
        <Dialog.Footer>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Keep trial
          </Button>
          <Button
            variant="destructive"
            onClick={confirmCancellation}
            disabled={isCanceling}
          >
            End trial now
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
