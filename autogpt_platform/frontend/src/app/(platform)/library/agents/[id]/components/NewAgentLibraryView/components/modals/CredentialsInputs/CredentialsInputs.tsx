import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { SlidersHorizontalIcon } from "lucide-react";
import { APIKeyCredentialsModal } from "./components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { CredentialRow } from "./components/CredentialRow/CredentialRow";
import { CredentialsSelect } from "./components/CredentialsSelect/CredentialsSelect";
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
  collapseSystemCredentials?: boolean;
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
  collapseSystemCredentials = false,
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
    isSystemProvider,
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

  // When collapseSystemCredentials is true AND provider is a system provider,
  // collapse ALL credentials (both user and system) under the accordion.
  // This keeps the provider section clean when platform credits are available.
  const shouldCollapseAll = collapseSystemCredentials && isSystemProvider;

  // Determine which credentials to show in main section
  const credentialsToShow = shouldCollapseAll
    ? [] // All credentials go to accordion when provider has system creds
    : collapseSystemCredentials
      ? userCredentials // No system creds, show user creds normally
      : allCredentials; // Show all when not collapsing

  const hasCredentialsToShow = credentialsToShow.length > 0;

  // Credentials to show in accordion
  const credentialsToCollapse = shouldCollapseAll
    ? allCredentials // All credentials collapsed when provider has system creds
    : collapseSystemCredentials
      ? systemCredentials // Only system creds collapsed
      : [];

  const hasCredentialsToCollapse = credentialsToCollapse.length > 0;

  // If required and no credential selected, keep accordion open
  const shouldOpenAccordionByDefault =
    shouldCollapseAll && !isOptional && !selectedCredential;

  if (readOnly && selectedCredentialIsSystem) {
    return null;
  }

  return (
    <div className={cn("mb-6", className)}>
      {showTitle && !shouldCollapseAll && (
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
              {credentialsToShow.map((credential) => (
                <CredentialRow
                  key={credential.id}
                  credential={credential}
                  provider={provider}
                  displayName={displayName}
                  onSelect={() => handleCredentialSelect(credential.id)}
                  readOnly={readOnly}
                />
              ))}
            </div>
          )}
          {!readOnly && !shouldCollapseAll && (
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
        !readOnly &&
        !shouldCollapseAll && (
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

      {shouldCollapseAll && !readOnly && (
        <Accordion
          type="single"
          collapsible
          defaultValue={
            shouldOpenAccordionByDefault ? "system-credentials" : undefined
          }
        >
          <AccordionItem value="system-credentials" className="border-none">
            <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
              <div className="flex items-center gap-1">
                <SlidersHorizontalIcon className="size-4" /> System credentials
              </div>
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4 px-1 pt-2">
                <div className="flex items-center gap-2">
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
                {credentialsToCollapse.length > 0 && (
                  <CredentialsSelect
                    credentials={credentialsToCollapse}
                    provider={provider}
                    displayName={displayName}
                    selectedCredentials={selectedCredential}
                    onSelectCredential={handleCredentialSelect}
                    onClearCredential={() => onSelectCredential(undefined)}
                    readOnly={readOnly}
                    allowNone={isOptional}
                    variant={variant}
                  />
                )}
                <Button
                  variant="secondary"
                  size="small"
                  onClick={handleActionButtonClick}
                  className="w-fit"
                  type="button"
                >
                  {actionButtonText}
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {hasCredentialsToCollapse && !shouldCollapseAll && !readOnly && (
        <Accordion type="single" collapsible className="mt-4">
          <AccordionItem value="system-credentials" className="border-none">
            <AccordionTrigger className="py-2 text-sm text-muted-foreground hover:no-underline">
              System credentials
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-2 pt-2">
                {credentialsToCollapse.map((credential) => (
                  <CredentialRow
                    key={credential.id}
                    credential={credential}
                    provider={provider}
                    displayName={displayName}
                    onSelect={() => handleCredentialSelect(credential.id)}
                    readOnly={readOnly}
                  />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
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
        </>
      )}
    </div>
  );
}
