import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  isOpen: boolean;
  planLabel: string;
  price: string;
  isSaving: boolean;
  onConfirm: () => void;
  onClose: () => void;
}

export function ConfirmTrialPlanDialog({
  isOpen,
  planLabel,
  price,
  isSaving,
  onConfirm,
  onClose,
}: Props) {
  return (
    <Dialog
      title={`Start ${planLabel} today?`}
      styling={{ maxWidth: "28rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <Text variant="body">
          Your trial ends now and your saved card is charged {price}, plus
          applicable tax.
        </Text>
        <Dialog.Footer>
          <Button variant="outline" onClick={onClose} disabled={isSaving}>
            Keep my trial
          </Button>
          <Button
            variant="primary"
            onClick={onConfirm}
            disabled={isSaving}
            loading={isSaving}
          >
            Subscribe to {planLabel}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
