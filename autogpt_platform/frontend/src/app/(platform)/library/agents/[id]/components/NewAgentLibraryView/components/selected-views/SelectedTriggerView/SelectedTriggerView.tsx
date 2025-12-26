"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Input } from "@/components/atoms/Input/Input";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  getAgentCredentialsFields,
  getAgentInputFields,
} from "../../modals/AgentInputsReadOnly/helpers";
import { CredentialsInput } from "../../modals/CredentialsInputs/CredentialsInputs";
import { RunAgentInputs } from "../../modals/RunAgentInputs/RunAgentInputs";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { WebhookTriggerCard } from "../SelectedTemplateView/components/WebhookTriggerCard";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { SelectedTriggerActions } from "./components/SelectedTriggerActions";
import { useSelectedTriggerView } from "./useSelectedTriggerView";

interface Props {
  agent: LibraryAgent;
  triggerId: string;
  onClearSelectedRun?: () => void;
  onSwitchToRunsTab?: () => void;
  banner?: React.ReactNode;
}

export function SelectedTriggerView({
  agent,
  triggerId,
  onClearSelectedRun,
  onSwitchToRunsTab,
  banner,
}: Props) {
  const {
    trigger,
    isLoading,
    error,
    name,
    setName,
    description,
    setDescription,
    inputs,
    setInputValue,
    credentials,
    setCredentialValue,
    handleSaveChanges,
    isSaving,
  } = useSelectedTriggerView({
    triggerId,
    graphId: agent.graph_id,
  });

  const agentInputFields = getAgentInputFields(agent);
  const agentCredentialsFields = getAgentCredentialsFields(agent);
  const inputFields = Object.entries(agentInputFields);
  const credentialFields = Object.entries(agentCredentialsFields);

  if (error) {
    return (
      <ErrorCard
        responseError={
          error
            ? {
                message: String(
                  (error as unknown as { message?: string })?.message ||
                    "Failed to load trigger",
                ),
              }
            : undefined
        }
        httpError={
          (error as any)?.status
            ? {
                status: (error as any).status,
                statusText: (error as any).statusText,
              }
            : undefined
        }
        context="trigger"
      />
    );
  }

  if (isLoading && !trigger) {
    return <LoadingSelectedContent agent={agent} />;
  }

  if (!trigger) {
    return null;
  }

  const hasWebhook = !!trigger.webhook_id && trigger.webhook;

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout agent={agent} banner={banner}>
          <div className="flex flex-col gap-4">
            <RunDetailHeader agent={agent} run={undefined} />

            <RunDetailCard title="Trigger Details">
              <div className="flex flex-col gap-2">
                <Input
                  id="trigger-name"
                  label="Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Enter trigger name"
                />

                <Input
                  id="trigger-description"
                  label="Description"
                  type="textarea"
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Enter trigger description"
                />
              </div>
            </RunDetailCard>

            {hasWebhook && agent.trigger_setup_info && (
              <WebhookTriggerCard
                template={trigger}
                triggerSetupInfo={agent.trigger_setup_info}
              />
            )}

            {inputFields.length > 0 && (
              <RunDetailCard title="Your Input">
                <div className="flex flex-col gap-4">
                  {inputFields.map(([key, inputSubSchema]) => (
                    <RunAgentInputs
                      key={key}
                      schema={inputSubSchema}
                      value={inputs[key] ?? inputSubSchema.default}
                      placeholder={inputSubSchema.description}
                      onChange={(value) => setInputValue(key, value)}
                    />
                  ))}
                </div>
              </RunDetailCard>
            )}

            {credentialFields.length > 0 && (
              <RunDetailCard title="Task Credentials">
                <div className="flex flex-col gap-6">
                  {credentialFields.map(([key, inputSubSchema]) => (
                    <CredentialsInput
                      key={key}
                      schema={
                        { ...inputSubSchema, discriminator: undefined } as any
                      }
                      selectedCredentials={
                        credentials[key] ?? inputSubSchema.default
                      }
                      onSelectCredentials={(value) =>
                        setCredentialValue(key, value!)
                      }
                      siblingInputs={inputs}
                    />
                  ))}
                </div>
              </RunDetailCard>
            )}
          </div>
        </SelectedViewLayout>
      </div>
      {trigger ? (
        <div className="-mt-2 max-w-[3.75rem] flex-shrink-0">
          <SelectedTriggerActions
            agent={agent}
            triggerId={trigger.id}
            onDeleted={onClearSelectedRun}
            onSaveChanges={handleSaveChanges}
            isSaving={isSaving}
            onSwitchToRunsTab={onSwitchToRunsTab}
          />
        </div>
      ) : null}
    </div>
  );
}
