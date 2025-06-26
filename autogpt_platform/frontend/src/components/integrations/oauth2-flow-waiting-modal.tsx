import { FC } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export const OAuth2FlowWaitingModal: FC<{
  open: boolean;
  onClose: () => void;
  providerName: string;
}> = ({ open, onClose, providerName }) => {
  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            Waiting on {providerName} sign-in process...
          </DialogTitle>
          <DialogDescription>
            Complete the sign-in process in the pop-up window.
            <br />
            Closing this dialog will cancel the sign-in process.
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
};
