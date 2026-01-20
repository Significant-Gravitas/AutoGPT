import { Text } from "@/components/atoms/Text/Text";
import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { cn } from "@/lib/utils";
import { CheckIcon, RobotIcon, WarningIcon } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";
import { useChatCredentialsSetup } from "./useChatCredentialsSetup";

export interface CredentialInfo {
  provider: string;
  providerName: string;
  credentialTypes: Array<
    "api_key" | "oauth2" | "user_password" | "host_scoped"
  >;
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
    credentials_types: credential.credentialTypes,
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
    <div className="group relative flex w-full justify-start gap-3 px-4 py-3">
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex-shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500">
            <RobotIcon className="h-4 w-4 text-indigo-50" />
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="group relative min-w-20 overflow-hidden rounded-xl border border-slate-100 bg-slate-50/20 px-6 py-2.5 text-sm leading-relaxed backdrop-blur-xl">
            <div className="absolute inset-0 bg-gradient-to-br from-slate-200/20 via-slate-300/10 to-transparent" />
            <div className="relative z-10 space-y-3 text-slate-900">
              <div>
                <Text variant="h4" className="mb-1 text-slate-900">
                  Credentials Required
                </Text>
                <Text variant="small" className="text-slate-600">
                  {message}
                </Text>
              </div>

              <div className="space-y-3">
                {credentials.map((cred, index) => {
                  const schema = createSchemaFromCredentialInfo(cred);
                  const isSelected = !!selectedCredentials[cred.provider];

                  return (
                    <div
                      key={`${cred.provider}-${index}`}
                      className={cn(
                        "relative rounded-lg border p-3",
                        isSelected
                          ? "border-green-500 bg-green-50/50"
                          : "border-slate-200 bg-white/50",
                      )}
                    >
                      <div className="mb-2 flex items-center gap-2">
                        {isSelected ? (
                          <CheckIcon
                            size={16}
                            className="text-green-500"
                            weight="bold"
                          />
                        ) : (
                          <WarningIcon
                            size={16}
                            className="text-slate-500"
                            weight="bold"
                          />
                        )}
                        <Text
                          variant="small"
                          className="font-semibold text-slate-900"
                        >
                          {cred.providerName}
                        </Text>
                      </div>

                      <CredentialsInput
                        schema={schema}
                        selectedCredentials={selectedCredentials[cred.provider]}
                        onSelectCredentials={(credMeta) =>
                          handleCredentialSelect(cred.provider, credMeta)
                        }
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
