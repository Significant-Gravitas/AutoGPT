import { useEffect, useRef } from "react";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { KeyIcon, CheckIcon, WarningIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { useChatCredentialsSetup } from "./useChatCredentialsSetup";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";

export interface CredentialInfo {
  provider: string;
  providerName: string;
  credentialType: "api_key" | "oauth2" | "user_password" | "host_scoped";
  title: string;
  scopes?: string[];
}

interface Props {
  credentials: CredentialInfo[];
  agentName?: string;
  message: string;
  onAllCredentialsComplete: () => void;
  onCancel: () => void;
  className?: string;
}

function createSchemaFromCredentialInfo(
  credential: CredentialInfo,
): BlockIOCredentialsSubSchema {
  return {
    type: "object",
    properties: {},
    credentials_provider: [credential.provider],
    credentials_types: [credential.credentialType],
    credentials_scopes: credential.scopes,
    discriminator: undefined,
    discriminator_mapping: undefined,
    discriminator_values: undefined,
  };
}

export function ChatCredentialsSetup({
  credentials,
  agentName: _agentName,
  message,
  onAllCredentialsComplete,
  onCancel: _onCancel,
  className,
}: Props) {
  const { selectedCredentials, isAllComplete, handleCredentialSelect } =
    useChatCredentialsSetup(credentials);

  // Track if we've already called completion to prevent double calls
  const hasCalledCompleteRef = useRef(false);

  // Reset the completion flag when credentials change (new credential setup flow)
  useEffect(
    function resetCompletionFlag() {
      hasCalledCompleteRef.current = false;
    },
    [credentials],
  );

  // Auto-call completion when all credentials are configured
  useEffect(
    function autoCompleteWhenReady() {
      if (isAllComplete && !hasCalledCompleteRef.current) {
        hasCalledCompleteRef.current = true;
        onAllCredentialsComplete();
      }
    },
    [isAllComplete, onAllCredentialsComplete],
  );

  return (
    <Card
      className={cn(
        "mx-4 my-2 overflow-hidden border-orange-200 bg-orange-50 dark:border-orange-900 dark:bg-orange-950",
        className,
      )}
    >
      <div className="flex items-start gap-4 p-6">
        <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-orange-500">
          <KeyIcon size={24} weight="bold" className="text-white" />
        </div>
        <div className="flex-1">
          <Text
            variant="h3"
            className="mb-2 text-orange-900 dark:text-orange-100"
          >
            Credentials Required
          </Text>
          <Text
            variant="body"
            className="mb-4 text-orange-700 dark:text-orange-300"
          >
            {message}
          </Text>

          <div className="space-y-3">
            {credentials.map((cred, index) => {
              const schema = createSchemaFromCredentialInfo(cred);
              const isSelected = !!selectedCredentials[cred.provider];

              return (
                <div
                  key={`${cred.provider}-${index}`}
                  className={cn(
                    "relative rounded-lg border border-orange-200 bg-white p-4 dark:border-orange-800 dark:bg-orange-900/20",
                    isSelected &&
                      "border-green-500 bg-green-50 dark:border-green-700 dark:bg-green-950/30",
                  )}
                >
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {isSelected ? (
                        <CheckIcon
                          size={20}
                          className="text-green-500"
                          weight="bold"
                        />
                      ) : (
                        <WarningIcon
                          size={20}
                          className="text-orange-500"
                          weight="bold"
                        />
                      )}
                      <Text
                        variant="body"
                        className="font-semibold text-orange-900 dark:text-orange-100"
                      >
                        {cred.providerName}
                      </Text>
                    </div>
                  </div>

                  <CredentialsInput
                    schema={schema}
                    selectedCredentials={selectedCredentials[cred.provider]}
                    onSelectCredentials={(credMeta) =>
                      handleCredentialSelect(cred.provider, credMeta)
                    }
                    hideIfSingleCredentialAvailable={false}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </Card>
  );
}
