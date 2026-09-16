"use client";

import { Button } from "@/components/atoms/Button/Button";
import { ConnectMethodView } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/ConnectMethodView/ConnectMethodView";
import {
  AuthType,
  type AuthMethod,
  type ConnectableProvider,
} from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import { getConnectableCredentialTypes } from "@/hooks/useCredentials";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import { useConnectCredentialDialog } from "./useConnectCredentialDialog";

const KNOWN_AUTH_METHODS: ReadonlySet<AuthMethod> = new Set(
  Object.values(AuthType),
);

interface Props {
  schema: BlockIOCredentialsSubSchema;
  provider: string;
  displayName: string;
  /** Existing account to upgrade in place rather than signing in afresh. */
  credentialID?: string;
  open: boolean;
  onClose: () => void;
  /** Fires only on a completed sign-in, unlike onClose. */
  onConnected?: () => void;
}

/** The onboarding connect flow (logo pair, "Connect AutoGPT to X",
 *  method cards with the API-key form inlined) scoped to a single
 *  provider — used by CredentialsInput's "Add credential" action so the
 *  copilot and the onboarding funnel ask for credentials the same way. */
export function ConnectCredentialDialog({
  schema,
  provider,
  displayName,
  credentialID,
  open,
  onClose,
  onConnected,
}: Props) {
  const {
    selectedMethod,
    setSelectedMethod,
    apiKeyForm,
    handleApiKeySubmit,
    showContinue,
    isContinueDisabled,
    isConnecting,
    handleContinue,
    reset,
  } = useConnectCredentialDialog({
    provider,
    onConnected: () => {
      onConnected?.();
      onClose();
    },
    scopes: schema.credentials_scopes,
    credentialID,
  });

  function handleClose() {
    reset();
    onClose();
  }

  // Device auth completes inside ConnectMethodView, so it never passes through
  // the hook's success path and has to report the connection itself.
  function handleDeviceAuthSuccess() {
    reset();
    onConnected?.();
    onClose();
  }

  const connectable: ConnectableProvider = {
    id: provider,
    name: displayName,
    description: null,
    supportedAuthTypes: getConnectableCredentialTypes(
      schema.credentials_types ?? [],
    ).filter((t): t is AuthMethod => KNOWN_AUTH_METHODS.has(t as AuthMethod)),
  };

  return (
    <Dialog
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) handleClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-5 pb-2">
          <ConnectMethodView
            provider={connectable}
            selectedMethod={selectedMethod}
            onSelectMethod={setSelectedMethod}
            apiKeyForm={apiKeyForm}
            onApiKeySubmit={handleApiKeySubmit}
            onDeviceAuthSuccess={handleDeviceAuthSuccess}
          />
          <div className="flex items-center justify-end gap-3">
            <Button variant="secondary" size="small" onClick={handleClose}>
              Cancel
            </Button>
            {showContinue && (
              <Button
                variant="primary"
                size="small"
                disabled={isContinueDisabled}
                loading={isConnecting}
                onClick={handleContinue}
              >
                {isConnecting ? "Connecting…" : "Continue"}
              </Button>
            )}
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
