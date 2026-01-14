import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { APIKeyCredentialsModal } from "./components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { CredentialRow } from "./components/CredentialRow/CredentialRow";
import { CredentialsSelect } from "./components/CredentialsSelect/CredentialsSelect";
import { DeleteConfirmationModal } from "./components/DeleteConfirmationModal/DeleteConfirmationModal";
import { HostScopedCredentialsModal } from "./components/HotScopedCredentialsModal/HotScopedCredentialsModal";
import { OAuthFlowWaitingModal } from "./components/OAuthWaitingModal/OAuthWaitingModal";
import { PasswordCredentialsModal } from "./components/PasswordCredentialsModal/PasswordCredentialsModal";
import { getCredentialDisplayName } from "./helpers";
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
    credentialsToShow,
    oAuthError,
    isAPICredentialsModalOpen,
    isUserPasswordCredentialsModalOpen,
    isHostScopedCredentialsModalOpen,
    isOAuth2FlowInProgress,
    oAuthPopupController,
    credentialToDelete,
    deleteCredentialsMutation,
    actionButtonText,
    setAPICredentialsModalOpen,
    setUserPasswordCredentialsModalOpen,
    setHostScopedCredentialsModalOpen,
    setCredentialToDelete,
    handleActionButtonClick,
    handleCredentialSelect,
    handleDeleteCredential,
    handleDeleteConfirm,
  } = hookData;

  const displayName = toDisplayName(provider);
  const hasCredentialsToShow = credentialsToShow.length > 0;

  return (
    <div className={cn("mb-6", className)}>
      {showTitle && (
        <div className="mb-2 flex items-center gap-2">
          <Text variant="large-medium">
            {displayName} credentials
            {isOptional && (
              <span className="ml-1 text-sm font-normal text-gray-500">
                (optional)
              </span>
            )}
          </Text>
          {schema.description && (
            <InformationTooltip description={schema.description} />
          )}
        </div>
      )}

      {hasCredentialsToShow ? (
        <>
          {(credentialsToShow.length > 1 || isOptional) && !readOnly ? (
            <CredentialsSelect
              credentials={credentialsToShow}
              provider={provider}
              displayName={displayName}
              selectedCredentials={selectedCredential}
              onSelectCredential={handleCredentialSelect}
              onClearCredential={() => onSelectCredential(undefined)}
              readOnly={readOnly}
              allowNone={isOptional}
              variant={variant}
            />
          ) : (
            <div className="mb-4 space-y-2">
              {credentialsToShow.map((credential) => {
                return (
                  <CredentialRow
                    key={credential.id}
                    credential={credential}
                    provider={provider}
                    displayName={displayName}
                    onSelect={() => handleCredentialSelect(credential.id)}
                    onDelete={() =>
                      handleDeleteCredential({
                        id: credential.id,
                        title: getCredentialDisplayName(
                          credential,
                          displayName,
                        ),
                      })
                    }
                    readOnly={readOnly}
                  />
                );
              })}
            </div>
          )}
          {!readOnly && (
            <Button
              variant="secondary"
              size="small"
              onClick={handleActionButtonClick}
              className="w-fit"
              type="button"
            >
              {actionButtonText}
            </Button>
          )}
        </>
      ) : (
        !readOnly && (
          <Button
            variant="secondary"
            size="small"
            onClick={handleActionButtonClick}
            className="w-fit"
            type="button"
          >
            {actionButtonText}
          </Button>
        )
      )}

      {!readOnly && (
        <>
          {supportsApiKey ? (
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
          ) : null}
          {supportsOAuth2 ? (
            <OAuthFlowWaitingModal
              open={isOAuth2FlowInProgress}
              onClose={() => oAuthPopupController?.abort("canceled")}
              providerName={providerName}
            />
          ) : null}
          {supportsUserPassword ? (
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
          ) : null}
          {supportsHostScoped ? (
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
          ) : null}

          {oAuthError ? (
            <Text variant="body" className="mt-2 text-red-500">
              Error: {oAuthError}
            </Text>
          ) : null}

          <DeleteConfirmationModal
            credentialToDelete={credentialToDelete}
            isDeleting={deleteCredentialsMutation.isPending}
            onClose={() => setCredentialToDelete(null)}
            onConfirm={handleDeleteConfirm}
          />
        </>
      )}
    </div>
  );
}
