import { Dialog } from "@/components/molecules/Dialog/Dialog";

type Props = {
  open: boolean;
  onClose: () => void;
  providerName: string;
};

export function OAuthFlowWaitingModal({ open, onClose, providerName }: Props) {
  return (
    <Dialog
      title={`Waiting on ${providerName} sign-in process...`}
      controlled={{
        isOpen: open,
        set: (isOpen) => {
          if (!isOpen) onClose();
        },
      }}
      onClose={onClose}
    >
      <Dialog.Content>
        <p className="text-sm text-zinc-600">
          Complete the sign-in process in the pop-up window.
          <br />
          Closing this dialog will cancel the sign-in process.
        </p>
      </Dialog.Content>
    </Dialog>
  );
}
