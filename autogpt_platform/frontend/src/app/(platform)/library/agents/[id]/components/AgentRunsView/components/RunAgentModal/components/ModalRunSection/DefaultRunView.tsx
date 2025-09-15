import { WebhookTriggerBanner } from "../WebhookTriggerBanner/WebhookTriggerBanner";
import { Input } from "@/components/atoms/Input/Input";
import SchemaTooltip from "@/components/SchemaTooltip";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { useRunAgentModalContext } from "../../context";
import { RunAgentInputs } from "../../../RunAgentInputs/RunAgentInputs";
import { Text } from "@/components/atoms/Text/Text";

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

      {/* Selected Credentials Preview */}
      {Object.keys(inputCredentials).length > 0 && (
        <div className="mt-6 flex flex-col gap-6">
          {Object.entries(agentCredentialsInputFields || {}).map(
            ([key, _sub]) => {
              const credential = inputCredentials[key];
              if (!credential) return null;

              const getProviderDisplayName = (provider: string) => {
                const providerMap: Record<string, string> = {
                  linear: "Linear",
                  github: "GitHub",
                  openai: "OpenAI",
                  google: "Google",
                  http: "HTTP",
                  slack: "Slack",
                  notion: "Notion",
                  discord: "Discord",
                };
                return providerMap[provider.toLowerCase()] || provider;
              };

              const getTypeDisplayName = (type: string) => {
                const typeMap: Record<string, string> = {
                  api_key: "API key",
                  oauth2: "OAuth2",
                  user_password: "Username/Password",
                  host_scoped: "Host-Scoped",
                };
                return typeMap[type] || type;
              };

              return (
                <div key={key} className="flex flex-col gap-4">
                  <Text variant="body-medium" as="h3">
                    {getProviderDisplayName(credential.provider)} credentials
                  </Text>
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center justify-between text-sm">
                      <Text
                        variant="body"
                        as="span"
                        className="!text-neutral-600"
                      >
                        Name
                      </Text>
                      <Text
                        variant="body"
                        as="span"
                        className="!text-neutral-600"
                      >
                        {getTypeDisplayName(credential.type)}
                      </Text>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <Text
                        variant="body"
                        as="span"
                        className="!text-neutral-900"
                      >
                        {credential.title || "Untitled"}
                      </Text>
                      <span className="font-mono text-neutral-400">
                        {"*".repeat(25)}
                      </span>
                    </div>
                  </div>
                </div>
              );
            },
          )}
        </div>
      )}
    </div>
  );
}
