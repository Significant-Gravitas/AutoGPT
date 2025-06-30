import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { GoogleLogoIcon } from "@phosphor-icons/react/dist/ssr";

interface GoogleLoadingModalProps {
  isOpen: boolean;
}

export function GoogleLoadingModal({ isOpen }: GoogleLoadingModalProps) {
  return (
    <Dialog open={isOpen}>
      <DialogContent className="sm:max-w-md [&>button]:hidden">
        <DialogHeader>
          <div className="mb-2 flex items-center justify-center gap-3">
            <GoogleLogoIcon size={24} />
            <DialogTitle>Signing in with Google</DialogTitle>
          </div>
          <DialogDescription className="text-center">
            You&apos;re being redirected to Google to complete the sign-in
            process.
            <br />
            <br />
            Please don&apos;t close this tab or navigate away from this page.
          </DialogDescription>
        </DialogHeader>
        <div className="flex justify-center pt-4">
          <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-gray-900"></div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
