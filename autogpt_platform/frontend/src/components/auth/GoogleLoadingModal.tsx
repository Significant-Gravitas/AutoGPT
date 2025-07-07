import { CircleNotchIcon } from "@phosphor-icons/react/dist/ssr";
import Image from "next/image";
import { Text } from "../atoms/Text/Text";
import { Dialog } from "../molecules/Dialog/Dialog";

interface GoogleLoadingModalProps {
  isOpen: boolean;
}

export function GoogleLoadingModal({ isOpen }: GoogleLoadingModalProps) {
  return (
    <Dialog forceOpen={isOpen} styling={{ maxWidth: "32rem" }}>
      <Dialog.Content>
        <div className="flex flex-col items-center gap-8 py-4">
          <div className="mb-2 flex items-center justify-center gap-3">
            <Image src="/google-logo.svg" alt="Google" width={20} height={20} />
            <Text variant="h3">Signing in with Google</Text>
          </div>
          <CircleNotchIcon
            className="h-10 w-10 animate-spin"
            weight="regular"
          />
          <Text variant="large-medium" className="text-center">
            You&apos;re being redirected to Google to complete the sign-in
            process.
            <br /> Please don&apos;t close this tab or navigate away from this
            page.
          </Text>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
