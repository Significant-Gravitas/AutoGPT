import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { DeviceAuthConnectButton } from "@/components/contextual/DeviceAuth/DeviceAuthConnectButton";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";

type Props = {
  open: boolean;
  onClose: () => void;
  provider: string;
  providerName: string;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
};

/**
 * Builder-side host for the device-code flow.
 *
 * The integrations panel shows the same flow as a tab; here it is a modal
 * opened from "Add account", because that is where a block's credentials
 * input lives. Without this the input fell through to the OAuth branch and
 * the backend rejected it — the provider is device-code only.
 */
export function DeviceAuthCredentialsModal({
  open,
  onClose,
  provider,
  providerName,
  onCredentialsCreate,
}: Props) {
  return (
    <Dialog
      title={`Connect ${providerName}`}
      controlled={{
        isOpen: open,
        set: (isOpen) => {
          if (!isOpen) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "25rem" }}
    >
      <Dialog.Content>
        <DeviceAuthConnectButton
          provider={provider}
          providerName={providerName}
          onSuccess={(credentials?: CredentialsMetaResponse) => {
            // The poll hands back the credential it just stored, so the node
            // can be wired up immediately instead of leaving the user to pick
            // it out of the list by hand.
            if (credentials) {
              onCredentialsCreate({
                id: credentials.id,
                type: credentials.type,
                provider: credentials.provider,
                title: credentials.title ?? undefined,
              });
            }
            onClose();
          }}
        />
      </Dialog.Content>
    </Dialog>
  );
}
