"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
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
import { useState } from "react";
import { ExistingCredentialsView } from "./components/ExistingCredentialsView/ExistingCredentialsView";
import type { ExistingCredentialsOffer } from "./helpers";
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
  /** Accounts to offer before the connect methods. With none, or once the
   *  user picks Add new, the dialog is the plain connect flow. */
  existing?: ExistingCredentialsOffer;
  open: boolean;
  onClose: () => void;
  /** Fires only on a completed sign-in, unlike onClose, with the credential
   *  the flow produced when it reports one. Using an existing account goes
   *  through `existing.onUse` instead. */
  onConnected?: (credential?: CredentialsMetaResponse) => void;
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
  existing,
  open,
  onClose,
  onConnected,
}: Props) {
  const [addingNew, setAddingNew] = useState(false);
  const [chosenId, setChosenId] = useState<string | null>(null);
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
    onConnected: handleConnected,
    scopes: schema.credentials_scopes,
    // Add new asks for a second account, not a re-auth: keeping the upgrade
    // target would sign the user back into the very account they are trying
    // to add another alongside.
    credentialID: addingNew ? undefined : credentialID,
  });

  const offered = existing?.credentials ?? [];
  const showExisting = offered.length > 0 && !addingNew;
  const chosen = offered.find((c) => c.id === chosenId) ?? offered[0];

  function resetAll() {
    reset();
    setAddingNew(false);
    setChosenId(null);
  }

  function handleClose() {
    resetAll();
    onClose();
  }

  // The hook has already reset by the time it calls this.
  function handleConnected(credential?: CredentialsMetaResponse) {
    setAddingNew(false);
    setChosenId(null);
    onConnected?.(credential);
    onClose();
  }

  // Device auth completes inside ConnectMethodView, bypassing the hook, so
  // this is the only place its reset can happen.
  function handleDeviceAuthSuccess(credential?: CredentialsMetaResponse) {
    reset();
    handleConnected(credential);
  }

  async function handleUseExisting() {
    if (!existing || !chosen) return;
    if (await existing.onUse(chosen)) handleClose();
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
          {showExisting && chosen ? (
            <ExistingCredentialsView
              provider={provider}
              displayName={displayName}
              credentials={offered}
              selectedId={chosen.id}
              onSelect={setChosenId}
            />
          ) : (
            <ConnectMethodView
              provider={connectable}
              selectedMethod={selectedMethod}
              onSelectMethod={setSelectedMethod}
              apiKeyForm={apiKeyForm}
              onApiKeySubmit={handleApiKeySubmit}
              onDeviceAuthSuccess={handleDeviceAuthSuccess}
            />
          )}
          {showExisting && existing?.error && (
            <span role="alert" className="text-center text-xs text-red-600">
              {existing.error}
            </span>
          )}
          <div className="flex items-center justify-end gap-3">
            <Button variant="secondary" size="small" onClick={handleClose}>
              Cancel
            </Button>
            {showExisting ? (
              <>
                <Button
                  variant="outline"
                  size="small"
                  disabled={existing?.isPending}
                  onClick={() => setAddingNew(true)}
                >
                  Add new
                </Button>
                <Button
                  variant="primary"
                  size="small"
                  loading={existing?.isPending}
                  onClick={handleUseExisting}
                >
                  {existing?.isPending ? "Granting…" : "Use existing"}
                </Button>
              </>
            ) : (
              showContinue && (
                <Button
                  variant="primary"
                  size="small"
                  disabled={isContinueDisabled}
                  loading={isConnecting}
                  onClick={handleContinue}
                >
                  {isConnecting ? "Connecting…" : "Continue"}
                </Button>
              )
            )}
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
