"use client";

import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Input } from "@/components/atoms/Input/Input";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  getAgentCredentialsFields,
  getAgentInputFields,
} from "../../modals/AgentInputsReadOnly/helpers";
import { CredentialsInput } from "../../modals/CredentialsInputs/CredentialsInputs";
import { RunAgentInputs } from "../../modals/RunAgentInputs/RunAgentInputs";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { SelectedTemplateActions } from "./components/SelectedTemplateActions";
import { WebhookTriggerCard } from "./components/WebhookTriggerCard";
import { useSelectedTemplateView } from "./useSelectedTemplateView";

interface Props {
  agent: LibraryAgent;
  templateId: string;
  onClearSelectedRun?: () => void;
  onRunCreated?: (execution: GraphExecutionMeta) => void;
  onSwitchToRunsTab?: () => void;
}

export function SelectedTemplateView({
  agent,
  templateId,
  onClearSelectedRun,
  onRunCreated,
  onSwitchToRunsTab,
}: Props) {
  const {
    template,
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
    handleStartTask,
    isSaving,
    isStarting,
  } = useSelectedTemplateView({
    templateId,
    graphId: agent.graph_id,
    onRunCreated,
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
                    "Failed to load template",
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
        context="template"
      />
    );
  }

  if (isLoading && !template) {
    return <LoadingSelectedContent agentName={agent.name} agentId={agent.id} />;
  }

  if (!template) {
    return null;
  }

  const templateOrTrigger = agent.trigger_setup_info ? "Trigger" : "Template";
  const hasWebhook = !!template.webhook_id && template.webhook;

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout agentName={agent.name} agentId={agent.id}>
          <div className="flex flex-col gap-4">
            <RunDetailHeader agent={agent} run={undefined} />

            {hasWebhook && agent.trigger_setup_info && (
              <WebhookTriggerCard
                template={template}
                triggerSetupInfo={agent.trigger_setup_info}
              />
            )}

            <RunDetailCard title={`${templateOrTrigger} Details`}>
              <div className="flex flex-col gap-2">
                <Input
                  id="template-name"
                  label="Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={`Enter ${templateOrTrigger.toLowerCase()} name`}
                />

                <Input
                  id="template-description"
                  label="Description"
                  type="textarea"
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder={`Enter ${templateOrTrigger.toLowerCase()} description`}
                />
              </div>
            </RunDetailCard>

            {inputFields.length > 0 && (
              <RunDetailCard title="Your Input">
                <div className="flex flex-col gap-4">
                  {inputFields.map(([key, inputSubSchema]) => (
                    <div
                      key={key}
                      className="flex w-full flex-col gap-0 space-y-2"
                    >
                      <label className="flex items-center gap-1 text-sm font-medium">
                        {inputSubSchema.title || key}
                        {inputSubSchema.description && (
                          <InformationTooltip
                            description={inputSubSchema.description}
                          />
                        )}
                      </label>
                      <RunAgentInputs
                        schema={inputSubSchema}
                        value={inputs[key] ?? inputSubSchema.default}
                        placeholder={inputSubSchema.description}
                        onChange={(value) => setInputValue(key, value)}
                      />
                    </div>
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
      {template ? (
        <div className="-mt-2 max-w-[3.75rem] flex-shrink-0">
          <SelectedTemplateActions
            agent={agent}
            templateId={template.id}
            onDeleted={onClearSelectedRun}
            onSaveChanges={handleSaveChanges}
            onStartTask={handleStartTask}
            isSaving={isSaving}
            isStarting={isStarting}
            onSwitchToRunsTab={onSwitchToRunsTab}
          />
        </div>
      ) : null}
    </div>
  );
}
