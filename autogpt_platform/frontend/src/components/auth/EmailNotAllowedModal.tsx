import { Button } from "../atoms/Button/Button";
import { Text } from "../atoms/Text/Text";
import { Dialog } from "../molecules/Dialog/Dialog";

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
        <div className="flex flex-col items-center gap-8 py-4">
          <Text variant="h3">Access Restricted</Text>
          <Text variant="large-medium" className="text-center">
            We&apos;re currently in a limited access phase. Your email address
            isn&apos;t on our current allowlist for early access. If you believe
            this is an error or would like to request access, please contact us.
          </Text>
          <div className="flex justify-end pt-4">
            <Button variant="primary" onClick={onClose}>
              I understand
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
