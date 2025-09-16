import { WebhookTriggerBanner } from "../WebhookTriggerBanner/WebhookTriggerBanner";
import { Input } from "@/components/atoms/Input/Input";
import SchemaTooltip from "@/components/SchemaTooltip";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { useRunAgentModalContext } from "../../context";
import { RunAgentInputs } from "../../../RunAgentInputs/RunAgentInputs";
import { InfoIcon } from "@phosphor-icons/react";

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

  return (
    <div className="my-4">
      {defaultRunType === "automatic-trigger" && <WebhookTriggerBanner />}

      {/* Preset/Trigger fields */}
      {defaultRunType === "automatic-trigger" && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col space-y-2">
            <label className="flex items-center gap-1 text-sm font-medium">
              Trigger Name
              <SchemaTooltip description="Name of the trigger you are setting up" />
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
              <SchemaTooltip description="Description of the trigger you are setting up" />
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
      )}

      {/* Instructions */}
      {agent.instructions && (
        <div className="mb-4 flex items-start gap-2 rounded-md border border-blue-200 bg-blue-50 p-3">
          <InfoIcon className="mt-0.5 h-4 w-4 text-blue-600" />
          <div>
            <h4 className="text-sm font-medium text-blue-900">
              How to use this agent
            </h4>
            <p className="mt-1 whitespace-pre-wrap text-sm text-blue-800">
              {agent.instructions}
            </p>
          </div>
        </div>
      )}

      {/* Credentials inputs */}
      {Object.entries(agentCredentialsInputFields || {}).map(
        ([key, inputSubSchema]) => (
          <CredentialsInput
            key={key}
            schema={{ ...inputSubSchema, discriminator: undefined } as any}
            selectedCredentials={
              (inputCredentials && inputCredentials[key]) ??
              inputSubSchema.default
            }
            onSelectCredentials={(value) =>
              setInputCredentialsValue(key, value)
            }
            siblingInputs={inputValues}
            hideIfSingleCredentialAvailable={!agent.has_external_trigger}
          />
        ),
      )}

      {/* Regular inputs */}
      {Object.entries(agentInputFields || {}).map(([key, inputSubSchema]) => (
        <div key={key} className="flex flex-col gap-0 space-y-2">
          <label className="flex items-center gap-1 text-sm font-medium">
            {inputSubSchema.title || key}
            <SchemaTooltip description={inputSubSchema.description} />
          </label>

          <RunAgentInputs
            schema={inputSubSchema}
            value={inputValues[key] ?? inputSubSchema.default}
            placeholder={inputSubSchema.description}
            onChange={(value) => setInputValue(key, value)}
            data-testid={`agent-input-${key}`}
          />
        </div>
      ))}
    </div>
  );
}
