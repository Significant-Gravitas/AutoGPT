import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { DeviceAuthConnectButton } from "@/components/contextual/DeviceAuth/DeviceAuthConnectButton";
import {
  MICROSOFT_PROVIDER,
  type MicrosoftSetupReason,
} from "@/lib/microsoft-setup";
import { MicrosoftAdminSetup } from "../MicrosoftAdminSetup";

interface Props {
  browserAvailable: boolean;
  scopes?: string[];
  loading: boolean;
  pending: boolean;
  recovery: MicrosoftSetupReason | null;
  onSignIn: () => void;
  onDeviceConnected: (credential?: CredentialsMetaResponse) => void;
}

export function MicrosoftSignIn({
  browserAvailable,
  scopes,
  loading,
  pending,
  recovery,
  onSignIn,
  onDeviceConnected,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body">
        Personal Microsoft accounts connect mail, calendar and files. Work or
        school accounts can also add Teams chats, Microsoft 365 Copilot and Work
        IQ.
      </Text>
      {browserAvailable ? (
        <Button
          variant="primary"
          loading={pending}
          disabled={loading || pending}
          onClick={onSignIn}
        >
          Sign in with Microsoft
        </Button>
      ) : (
        <DeviceAuthConnectButton
          provider={MICROSOFT_PROVIDER}
          providerName="Microsoft"
          scopes={scopes}
          onSuccess={onDeviceConnected}
        />
      )}
      {recovery && <MicrosoftAdminSetup reason={recovery} onRetry={onSignIn} />}
    </div>
  );
}
