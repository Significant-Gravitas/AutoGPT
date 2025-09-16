import { WebhookTriggerBanner } from "../WebhookTriggerBanner/WebhookTriggerBanner";
import { Input } from "@/components/atoms/Input/Input";
import SchemaTooltip from "@/components/SchemaTooltip";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { useRunAgentModalContext } from "../../context";
import { RunAgentInputs } from "../../../RunAgentInputs/RunAgentInputs";

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
        <div className="mb-4 rounded-lg border bg-blue-50 p-3">
          <div className="flex items-start gap-2">
            <div className="text-blue-600">
              <svg
                className="mt-0.5 h-4 w-4"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div>
              <h4 className="text-sm font-medium text-blue-900">How to use this agent</h4>
              <p className="mt-1 text-sm text-blue-800 whitespace-pre-wrap">{agent.instructions}</p>
            </div>
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
