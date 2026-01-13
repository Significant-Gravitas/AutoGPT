import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import { Input } from "@/components/atoms/Input/Input";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext, useMemo } from "react";
import { RunAgentInputs } from "../../../RunAgentInputs/RunAgentInputs";
import { useRunAgentModalContext } from "../../context";
import { ModalSection } from "../ModalSection/ModalSection";
import { WebhookTriggerBanner } from "../WebhookTriggerBanner/WebhookTriggerBanner";

export function ModalRunSection() {
  const {
    agent,
    defaultRunType,
    presetName,
    setPresetName,
    presetDescription,
    setPresetDescription,
    inputValues,
    setInputValue,
    agentInputFields,
    inputCredentials,
    setInputCredentialsValue,
    agentCredentialsInputFields,
  } = useRunAgentModalContext();

  const allProviders = useContext(CredentialsProvidersContext);

  const inputFields = Object.entries(agentInputFields || {});

  // Sort credential fields: user credentials first, system credentials at the bottom
  const sortedCredentialFields = useMemo(() => {
    if (!allProviders || !agentCredentialsInputFields) return [];

    const entries = Object.entries(agentCredentialsInputFields);

    return entries.sort(([_keyA, schemaA], [_keyB, schemaB]) => {
      const providerNamesA = schemaA.credentials_provider || [];
      const providerNamesB = schemaB.credentials_provider || [];

      // Check if A has system credentials
      const aHasSystemCred = providerNamesA.some((providerName: string) => {
        const providerData = allProviders[providerName];
        if (!providerData) return false;
        return providerData.savedCredentials.some(
          (cred: { is_system?: boolean }) => cred.is_system === true,
        );
      });

      // Check if B has system credentials
      const bHasSystemCred = providerNamesB.some((providerName: string) => {
        const providerData = allProviders[providerName];
        if (!providerData) return false;
        return providerData.savedCredentials.some(
          (cred: { is_system?: boolean }) => cred.is_system === true,
        );
      });

      // User credentials first, system credentials at the bottom
      if (aHasSystemCred && !bHasSystemCred) return 1;
      if (!aHasSystemCred && bHasSystemCred) return -1;
      return 0;
    });
  }, [agentCredentialsInputFields, allProviders]);

  const requiredCredentials = new Set(
    (agent.credentials_input_schema?.required as string[]) || [],
  );

  return (
    <div className="flex flex-col gap-4">
      {defaultRunType === "automatic-trigger" ||
      defaultRunType === "manual-trigger" ? (
        <ModalSection
          title="Task Trigger"
          subtitle="Set up a trigger for the agent to run this task automatically"
        >
          <WebhookTriggerBanner />
          <div className="flex flex-col gap-4">
            <div className="flex flex-col space-y-2">
              <label className="flex items-center gap-1 text-sm font-medium">
                Trigger Name
                <InformationTooltip description="Name of the trigger you are setting up" />
              </label>
              <Input
                id="trigger_name"
                label="Trigger Name"
                size="small"
                hideLabel
                value={presetName}
                placeholder="Enter trigger name"
                onChange={(e) => setPresetName(e.target.value)}
              />
            </div>
            <div className="flex flex-col space-y-2">
              <label className="flex items-center gap-1 text-sm font-medium">
                Trigger Description
                <InformationTooltip description="Description of the trigger you are setting up" />
              </label>
              <Input
                id="trigger_description"
                label="Trigger Description"
                size="small"
                hideLabel
                value={presetDescription}
                placeholder="Enter trigger description"
                onChange={(e) => setPresetDescription(e.target.value)}
              />
            </div>
          </div>
        </ModalSection>
      ) : null}

      {inputFields.length > 0 ? (
        <ModalSection
          title="Task Inputs"
          subtitle="Enter the information you want to provide to the agent for this task"
        >
          {inputFields.map(([key, inputSubSchema]) => (
            <RunAgentInputs
              key={key}
              schema={inputSubSchema}
              value={inputValues[key] ?? inputSubSchema.default}
              placeholder={inputSubSchema.description}
              onChange={(value) => setInputValue(key, value)}
              data-testid={`agent-input-${key}`}
            />
          ))}
        </ModalSection>
      ) : null}

      {sortedCredentialFields.length > 0 ? (
        <ModalSection
          title="Task Credentials"
          subtitle="These are the credentials the agent will use to perform this task"
        >
          <div className="space-y-6">
            {sortedCredentialFields.map(([key, inputSubSchema]) => {
              const selectedCred = inputCredentials?.[key];

              return (
                <CredentialsInput
                  key={key}
                  schema={
                    { ...inputSubSchema, discriminator: undefined } as any
                  }
                  selectedCredentials={selectedCred}
                  onSelectCredentials={(value) => {
                    setInputCredentialsValue(key, value);
                  }}
                  siblingInputs={inputValues}
                  isOptional={!requiredCredentials.has(key)}
                  collapseSystemCredentials
                />
              );
            })}
          </div>
        </ModalSection>
      ) : null}
    </div>
  );
}
