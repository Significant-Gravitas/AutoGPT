import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import { Input } from "@/components/atoms/Input/Input";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { RunAgentInputs } from "../../../RunAgentInputs/RunAgentInputs";
import { useRunAgentModalContext } from "../../context";
import { ModalSection } from "../ModalSection/ModalSection";
import { WebhookTriggerBanner } from "../WebhookTriggerBanner/WebhookTriggerBanner";

export function ModalRunSection() {
  const {
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

  const inputFields = Object.entries(agentInputFields || {});
  const credentialFields = Object.entries(agentCredentialsInputFields || {});

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

      {credentialFields.length > 0 ? (
        <ModalSection
          title="Task Credentials"
          subtitle="These are the credentials the agent will use to perform this task"
        >
          <div className="space-y-6">
            {Object.entries(agentCredentialsInputFields || {}).map(
              ([key, inputSubSchema]) => (
                <CredentialsInput
                  key={key}
                  schema={
                    { ...inputSubSchema, discriminator: undefined } as any
                  }
                  selectedCredentials={
                    (inputCredentials && inputCredentials[key]) ??
                    inputSubSchema.default
                  }
                  onSelectCredentials={(value) =>
                    setInputCredentialsValue(key, value)
                  }
                  siblingInputs={inputValues}
                />
              ),
            )}
          </div>
        </ModalSection>
      ) : null}
    </div>
  );
}
