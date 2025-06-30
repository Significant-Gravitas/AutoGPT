import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "../atoms/Button/Button";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function EmailNotAllowedModal({ isOpen, onClose }: Props) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Access Restricted</DialogTitle>
          <DialogDescription className="pt-2">
            We&apos;re currently in a limited access phase. Your email address
            isn&apos;t on our current allowlist for early access.
            <br />
            <br />
            If you believe this is an error or would like to request access,
            please contact our support team.
          </DialogDescription>
        </DialogHeader>
        <div className="flex justify-end pt-4">
          <Button onClick={onClose}>I understand</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
