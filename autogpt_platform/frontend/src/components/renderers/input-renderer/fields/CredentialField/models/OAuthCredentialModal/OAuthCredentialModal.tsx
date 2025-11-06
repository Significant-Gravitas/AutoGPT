import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { providerIcons, toDisplayName } from "../../helpers";
import { useOAuthCredentialModal } from "./useOAuthCredentialModal";
import { Text } from "@/components/atoms/Text/Text";

type OAuthCredentialModalProps = {
  provider: string;
};

export const OAuthCredentialModal = ({
  provider,
}: OAuthCredentialModalProps) => {
  const Icon = providerIcons[provider];
  const { handleOAuthLogin, loading, error, onClose, open, setOpen } =
    useOAuthCredentialModal({
      provider,
    });
  return (
    <>
      <Dialog
        title={`Waiting on ${toDisplayName(provider)} sign-in process...`}
        controlled={{
          isOpen: open,
          set: (isOpen) => {
            if (!isOpen) setOpen(false);
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

      <Button
        type="button"
        className="w-fit"
        size="small"
        onClick={() => {
          handleOAuthLogin();
        }}
        disabled={loading}
      >
        {Icon && <Icon className="size-4" />}
        <Text variant="small" className="!text-white opacity-100">
          Add OAuth2
        </Text>
      </Button>
      {error && (
        <div className="mt-2 flex w-fit items-center rounded-full bg-red-50 p-1 px-3 ring-1 ring-red-600">
          <Text variant="small" className="!text-red-600">
            {error as string}
          </Text>
        </div>
      )}
    </>
  );
};
