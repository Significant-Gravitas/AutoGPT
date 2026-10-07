import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

type Props = {
  open: boolean;
  onClose: () => void;
  providerName: string;
  /**
   * Set when the browser blocked the OAuth popup window and the helper
   * fell back to opening the login URL in a new tab. Changes the copy
   * to direct the user there instead of to a popup that doesn't exist.
   */
  popupBlocked?: boolean;
};

export function OAuthFlowWaitingModal({
  open,
  onClose,
  providerName,
  popupBlocked = false,
}: Props) {
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
        {popupBlocked ? (
          <Text variant="body" tone="secondary">
            Your browser blocked the sign-in window, so we opened it in a new
            tab. Switch to that tab and complete sign-in there.
            <br />
            If you don&apos;t see the tab, allow popups for this site, close
            this dialog, and try connecting again.
          </Text>
        ) : (
          <Text variant="body" tone="secondary">
            Complete the sign-in process in the sign-in window.
            <br />
            Closing this dialog will cancel the sign-in process.
          </Text>
        )}
      </Dialog.Content>
    </Dialog>
  );
}
