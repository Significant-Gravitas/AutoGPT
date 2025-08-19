import { Dialog } from "@/components/molecules/Dialog/Dialog";

interface Props {
  open: boolean;
  onClose: () => void;
  triggerSlot: React.ReactNode;
}

export function RunAgentModal({ open, onClose, triggerSlot }: Props) {
  return (
    <Dialog
      controlled={{
        isOpen: open,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Trigger>{triggerSlot}</Dialog.Trigger>
      <Dialog.Content>Run Agent</Dialog.Content>
    </Dialog>
  );
}
