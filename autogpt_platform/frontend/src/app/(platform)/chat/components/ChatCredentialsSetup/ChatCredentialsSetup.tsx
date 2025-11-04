import { useEffect } from "react";
import { Card } from "@/components/atoms/Card/Card";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Key, Check, Warning } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { useChatCredentialsSetup } from "./useChatCredentialsSetup";
import { APIKeyCredentialsModal } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { OAuthFlowWaitingModal } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OAuthWaitingModal/OAuthWaitingModal";
import { PasswordCredentialsModal } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/PasswordCredentialsModal/PasswordCredentialsModal";
import { HostScopedCredentialsModal } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/HotScopedCredentialsModal/HotScopedCredentialsModal";

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

export function ChatCredentialsSetup({
  credentials,
  agentName,
  message,
  onAllCredentialsComplete,
  onCancel,
  className,
}: Props) {
  const {
    credentialStatuses,
    isAllComplete,
    activeModal,
    handleSetupClick,
    handleModalClose,
    handleCredentialCreated,
  } = useChatCredentialsSetup(credentials);

  // Auto-call completion when all credentials are configured
  useEffect(
    function autoCompleteWhenReady() {
      if (isAllComplete) {
        onAllCredentialsComplete();
      }
    },
    [isAllComplete, onAllCredentialsComplete]
  );

  return (
    <>
      <Card
        className={cn(
          "mx-4 my-2 overflow-hidden border-orange-200 bg-orange-50 dark:border-orange-900 dark:bg-orange-950",
          className
        )}
      >
        <div className="flex items-start gap-4 p-6">
          <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-orange-500">
            <Key size={24} weight="bold" className="text-white" />
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
              {credentials.map((cred, index) => (
                <CredentialRow
                  key={`${cred.provider}-${index}`}
                  credential={cred}
                  status={credentialStatuses[index]}
                  onSetup={() => handleSetupClick(index, cred)}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="border-t border-orange-200 px-6 py-4 dark:border-orange-900">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </Card>

      {/* Modals - reuse existing components */}
      {activeModal?.type === "api_key" && activeModal.schema && (
        <APIKeyCredentialsModal
          schema={activeModal.schema}
          open={true}
          onClose={handleModalClose}
          onCredentialsCreate={handleCredentialCreated}
        />
      )}

      {activeModal?.type === "oauth2" && (
        <OAuthFlowWaitingModal
          open={true}
          onClose={handleModalClose}
          providerName={activeModal.providerName || ""}
        />
      )}

      {activeModal?.type === "user_password" && activeModal.schema && (
        <PasswordCredentialsModal
          schema={activeModal.schema}
          open={true}
          onClose={handleModalClose}
          onCredentialsCreate={handleCredentialCreated}
        />
      )}

      {activeModal?.type === "host_scoped" && activeModal.schema && (
        <HostScopedCredentialsModal
          schema={activeModal.schema}
          open={true}
          onClose={handleModalClose}
          onCredentialsCreate={handleCredentialCreated}
        />
      )}
    </>
  );
}

interface CredentialRowProps {
  credential: CredentialInfo;
  status?: {
    isConfigured: boolean;
    credentialId?: string;
  };
  onSetup: () => void;
}

function CredentialRow({ credential, status, onSetup }: CredentialRowProps) {
  const isConfigured = status?.isConfigured || false;

  function getCredentialTypeLabel(type: string): string {
    switch (type) {
      case "api_key":
        return "API Key";
      case "oauth2":
        return "OAuth";
      case "user_password":
        return "Username & Password";
      case "host_scoped":
        return "Custom Headers";
      default:
        return "Credentials";
    }
  }

  return (
    <div className="flex items-center justify-between rounded-lg border border-orange-200 bg-white p-3 dark:border-orange-800 dark:bg-orange-900/20">
      <div className="flex items-center gap-3">
        {isConfigured ? (
          <Check size={20} className="text-green-500" weight="bold" />
        ) : (
          <Warning size={20} className="text-orange-500" weight="bold" />
        )}
        <div>
          <Text variant="body" className="font-semibold text-orange-900 dark:text-orange-100">
            {credential.providerName}
          </Text>
          <Text variant="small" className="text-orange-700 dark:text-orange-300">
            {getCredentialTypeLabel(credential.credentialType)}
          </Text>
        </div>
      </div>

      {!isConfigured && (
        <Button size="small" onClick={onSetup} variant="primary">
          Setup
        </Button>
      )}
    </div>
  );
}
