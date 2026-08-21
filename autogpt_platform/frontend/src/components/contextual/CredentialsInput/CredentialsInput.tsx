"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useState } from "react";
import { APIKeyCredentialsModal } from "./components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { ConnectCredentialDialog } from "./components/ConnectCredentialDialog/ConnectCredentialDialog";
import { CredentialsFlatView } from "./components/CredentialsFlatView/CredentialsFlatView";
import { CredentialTypeSelector } from "./components/CredentialTypeSelector/CredentialTypeSelector";
import { DeleteConfirmationModal } from "./components/DeleteConfirmationModal/DeleteConfirmationModal";
import { DeviceAuthCredentialsModal } from "./components/DeviceAuthCredentialsModal/DeviceAuthCredentialsModal";
import { HostScopedCredentialsModal } from "./components/HotScopedCredentialsModal/HotScopedCredentialsModal";
import { OAuthFlowWaitingModal } from "./components/OAuthWaitingModal/OAuthWaitingModal";
import { PasswordCredentialsModal } from "./components/PasswordCredentialsModal/PasswordCredentialsModal";
import { getRemovedCredentialMessage, isSystemCredential } from "./helpers";
import {
  CredentialsInputState,
  useCredentialsInput,
} from "./useCredentialsInput";

function isLoaded(
  data: CredentialsInputState,
): data is Extract<CredentialsInputState, { isLoading: false }> {
  return data.isLoading === false;
}

type Props = {
  schema: BlockIOCredentialsSubSchema;
  className?: string;
  selectedCredentials?: CredentialsMetaInput;
  siblingInputs?: Record<string, any>;
  onSelectCredentials: (newValue?: CredentialsMetaInput) => void;
  onLoaded?: (loaded: boolean) => void;
  readOnly?: boolean;
  isOptional?: boolean;
  showTitle?: boolean;
  variant?: "default" | "node";
};

export function CredentialsInput({
  schema,
  className,
  selectedCredentials: selectedCredential,
  onSelectCredentials: onSelectCredential,
  siblingInputs,
  onLoaded,
  readOnly = false,
  isOptional = false,
  showTitle = true,
  variant = "default",
}: Props) {
  const hookData = useCredentialsInput({
    schema,
    selectedCredential,
    onSelectCredential,
    siblingInputs,
    onLoaded,
    readOnly,
    isOptional,
  });
  const [isConnectDialogOpen, setConnectDialogOpen] = useState(false);
  // The unified connect dialog ships with the new tool UI; off keeps the
  // legacy per-type action flow everywhere.
  const usesConnectDialog =
    useGetFlag(Flag.NEW_TOOL_UI) && variant === "default";

  if (!isLoaded(hookData)) {
    return null;
  }

  const {
    provider,
    providerName,
    supportsApiKey,
    supportsOAuth2,
    supportsDeviceCode,
    supportsUserPassword,
    supportsHostScoped,
    hasMultipleCredentialTypes,
    supportedTypes,
    userCredentials,
    systemCredentials,
    oAuthError,
    removedCredentialTitle,
    isAPICredentialsModalOpen,
    isUserPasswordCredentialsModalOpen,
    isHostScopedCredentialsModalOpen,
    isDeviceAuthModalOpen,
    isCredentialTypeSelectorOpen,
    isOAuth2FlowInProgress,
    oAuthPopupBlocked,
    cancelOAuthFlow,
    actionButtonText,
    setAPICredentialsModalOpen,
    setUserPasswordCredentialsModalOpen,
    setHostScopedCredentialsModalOpen,
    setDeviceAuthModalOpen,
    setCredentialTypeSelectorOpen,
    handleActionButtonClick,
    handleCredentialSelect,
    handleOAuthLogin,
    handleDeleteCredential,
    handleDeleteConfirm,
    credentialToDelete,
    deleteWarningMessage,
    setCredentialToDelete,
    isDeletingCredential,
    handleCredentialChange,
  } = hookData;

  const displayName = toDisplayName(provider);
  const selectedCredentialIsSystem =
    selectedCredential && isSystemCredential(selectedCredential);

  const allCredentials = [...userCredentials, ...systemCredentials];

  if (readOnly && selectedCredentialIsSystem) {
    return null;
  }

  return (
    <div className={cn("mb-6", className)}>
      <CredentialsFlatView
        schema={schema}
        provider={provider}
        displayName={displayName}
        credentials={allCredentials}
        selectedCredential={selectedCredential}
        onSelectCredential={handleCredentialSelect}
        onClearCredential={() => handleCredentialChange(undefined)}
        onAddCredential={
          usesConnectDialog
            ? () => setConnectDialogOpen(true)
            : handleActionButtonClick
        }
        onDeleteCredential={readOnly ? undefined : handleDeleteCredential}
        actionButtonText={actionButtonText}
        isOptional={isOptional}
        showTitle={showTitle}
        readOnly={readOnly}
        variant={variant}
      />

      {!readOnly && (
        <>
          {usesConnectDialog && (
            <ConnectCredentialDialog
              schema={schema}
              provider={provider}
              displayName={displayName}
              open={isConnectDialogOpen}
              onClose={() => setConnectDialogOpen(false)}
            />
          )}
          {hasMultipleCredentialTypes && (
            <CredentialTypeSelector
              schema={schema}
              open={isCredentialTypeSelectorOpen}
              onClose={() => setCredentialTypeSelectorOpen(false)}
              provider={provider}
              providerName={providerName}
              supportedTypes={supportedTypes}
              onCredentialsCreate={(creds) => {
                handleCredentialChange(creds);
              }}
              onOAuthLogin={handleOAuthLogin}
              onOpenPasswordModal={() =>
                setUserPasswordCredentialsModalOpen(true)
              }
              onOpenHostScopedModal={() =>
                setHostScopedCredentialsModalOpen(true)
              }
              siblingInputs={siblingInputs}
            />
          )}
          {supportsApiKey && !hasMultipleCredentialTypes && (
            <APIKeyCredentialsModal
              schema={schema}
              open={isAPICredentialsModalOpen}
              onClose={() => setAPICredentialsModalOpen(false)}
              onCredentialsCreate={(credsMeta) => {
                handleCredentialChange(credsMeta);
                setAPICredentialsModalOpen(false);
              }}
              siblingInputs={siblingInputs}
            />
          )}
          {supportsOAuth2 && (
            <OAuthFlowWaitingModal
              open={isOAuth2FlowInProgress}
              onClose={cancelOAuthFlow}
              providerName={provider === "codex" ? "ChatGPT" : providerName}
              popupBlocked={oAuthPopupBlocked}
            />
          )}
          {supportsDeviceCode && (
            <DeviceAuthCredentialsModal
              open={isDeviceAuthModalOpen}
              onClose={() => setDeviceAuthModalOpen(false)}
              provider={provider}
              providerName={providerName}
              onCredentialsCreate={(creds) => {
                handleCredentialChange(creds);
                setDeviceAuthModalOpen(false);
              }}
            />
          )}
          {supportsUserPassword && (
            <PasswordCredentialsModal
              schema={schema}
              open={isUserPasswordCredentialsModalOpen}
              onClose={() => setUserPasswordCredentialsModalOpen(false)}
              onCredentialsCreate={(creds) => {
                handleCredentialChange(creds);
                setUserPasswordCredentialsModalOpen(false);
              }}
              siblingInputs={siblingInputs}
            />
          )}
          {supportsHostScoped && (
            <HostScopedCredentialsModal
              schema={schema}
              open={isHostScopedCredentialsModalOpen}
              onClose={() => setHostScopedCredentialsModalOpen(false)}
              onCredentialsCreate={(creds) => {
                handleCredentialChange(creds);
                setHostScopedCredentialsModalOpen(false);
              }}
              siblingInputs={siblingInputs}
            />
          )}

          {removedCredentialTitle && (
            <Alert variant="warning" aria-live="polite" className="mt-2">
              <AlertDescription>
                <Text variant="body" unmask={false}>
                  {getRemovedCredentialMessage(
                    removedCredentialTitle,
                    selectedCredential,
                    displayName,
                  )}
                </Text>
              </AlertDescription>
            </Alert>
          )}

          {oAuthError && (
            <Text variant="body" className="mt-2 text-red-500">
              Error: {oAuthError}
            </Text>
          )}

          <DeleteConfirmationModal
            credentialToDelete={credentialToDelete}
            warningMessage={deleteWarningMessage}
            isDeleting={isDeletingCredential}
            onClose={() => setCredentialToDelete(null)}
            onConfirm={() => handleDeleteConfirm(false)}
            onForceConfirm={() => handleDeleteConfirm(true)}
          />
        </>
      )}
    </div>
  );
}
