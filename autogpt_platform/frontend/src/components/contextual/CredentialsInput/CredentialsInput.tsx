"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { APIKeyCredentialsModal } from "./components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { CredentialsFlatView } from "./components/CredentialsFlatView/CredentialsFlatView";
import { HostScopedCredentialsModal } from "./components/HotScopedCredentialsModal/HotScopedCredentialsModal";
import { OAuthFlowWaitingModal } from "./components/OAuthWaitingModal/OAuthWaitingModal";
import { PasswordCredentialsModal } from "./components/PasswordCredentialsModal/PasswordCredentialsModal";
import { isSystemCredential } from "./helpers";
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

  if (!isLoaded(hookData)) {
    return null;
  }

  const {
    provider,
    providerName,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    userCredentials,
    systemCredentials,
    oAuthError,
    isAPICredentialsModalOpen,
    isUserPasswordCredentialsModalOpen,
    isHostScopedCredentialsModalOpen,
    isOAuth2FlowInProgress,
    oAuthPopupController,
    actionButtonText,
    setAPICredentialsModalOpen,
    setUserPasswordCredentialsModalOpen,
    setHostScopedCredentialsModalOpen,
    handleActionButtonClick,
    handleCredentialSelect,
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
        onClearCredential={() => onSelectCredential(undefined)}
        onAddCredential={handleActionButtonClick}
        actionButtonText={actionButtonText}
        isOptional={isOptional}
        showTitle={showTitle}
        readOnly={readOnly}
        variant={variant}
      />

      {!readOnly && (
        <>
          {supportsApiKey && (
            <APIKeyCredentialsModal
              schema={schema}
              open={isAPICredentialsModalOpen}
              onClose={() => setAPICredentialsModalOpen(false)}
              onCredentialsCreate={(credsMeta) => {
                onSelectCredential(credsMeta);
                setAPICredentialsModalOpen(false);
              }}
              siblingInputs={siblingInputs}
            />
          )}
          {supportsOAuth2 && (
            <OAuthFlowWaitingModal
              open={isOAuth2FlowInProgress}
              onClose={() => oAuthPopupController?.abort("canceled")}
              providerName={providerName}
            />
          )}
          {supportsUserPassword && (
            <PasswordCredentialsModal
              schema={schema}
              open={isUserPasswordCredentialsModalOpen}
              onClose={() => setUserPasswordCredentialsModalOpen(false)}
              onCredentialsCreate={(creds) => {
                onSelectCredential(creds);
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
                onSelectCredential(creds);
                setHostScopedCredentialsModalOpen(false);
              }}
              siblingInputs={siblingInputs}
            />
          )}

          {oAuthError && (
            <Text variant="body" className="mt-2 text-red-500">
              Error: {oAuthError}
            </Text>
          )}
        </>
      )}
    </div>
  );
}
