import { Dialog } from "../molecules/Dialog/Dialog";
import { WaitlistErrorContent } from "./WaitlistErrorContent";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function EmailNotAllowedModal({ isOpen, onClose }: Props) {
  return (
    <Dialog
      controlled={{ isOpen, set: onClose }}
      styling={{ maxWidth: "35rem" }}
    >
      <Dialog.Content>
        <div className="py-4">
          <WaitlistErrorContent onClose={onClose} />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
