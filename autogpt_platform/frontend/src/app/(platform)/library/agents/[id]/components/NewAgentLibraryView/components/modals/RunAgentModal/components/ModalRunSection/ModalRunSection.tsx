import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/CredentialsInputs/CredentialsInputs";
import { Input } from "@/components/atoms/Input/Input";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext, useMemo } from "react";
import {
  NONE_CREDENTIAL_MARKER,
  useAgentCredentialPreferencesStore,
} from "../../../../../stores/agentCredentialPreferencesStore";
import {
  filterSystemCredentials,
  isSystemCredential,
} from "../../../CredentialsInputs/helpers";
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
  const store = useAgentCredentialPreferencesStore();

  const inputFields = Object.entries(agentInputFields || {});

  // Only show credential fields that have user credentials (NOT system credentials)
  // System credentials should only be shown in settings, not in run modal
  const credentialFields = useMemo(() => {
    if (!allProviders || !agentCredentialsInputFields) return [];

    return Object.entries(agentCredentialsInputFields).filter(
      ([_key, schema]) => {
        const providerNames = schema.credentials_provider || [];
        const supportedTypes = schema.credentials_types || [];

        // Check if any provider has user credentials (NOT system credentials)
        for (const providerName of providerNames) {
          const providerData = allProviders[providerName];
          if (!providerData) continue;

          const userCreds = filterSystemCredentials(
            providerData.savedCredentials,
          );
          const matchingUserCreds = userCreds.filter((cred: { type: string }) =>
            supportedTypes.includes(cred.type),
          );

          // If there are user credentials available, show this field
          if (matchingUserCreds.length > 0) {
            return true;
          }
        }

        // Hide the field if only system credentials exist (or no credentials at all)
        return false;
      },
    );
  }, [agentCredentialsInputFields, allProviders]);

  // Get the list of required credentials from the schema
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

      {credentialFields.length > 0 ? (
        <ModalSection
          title="Task Credentials"
          subtitle="These are the credentials the agent will use to perform this task"
        >
          <div className="space-y-6">
            {credentialFields
              .map(([key, inputSubSchema]) => {
                const selectedCred = inputCredentials?.[key];

                // Check if the selected credential is a system credential
                // First check the credential object itself, then look it up in providers
                let isSystemCredSelected = false;
                if (selectedCred) {
                  // Check if credential object has is_system or title indicates system credential
                  isSystemCredSelected = isSystemCredential(
                    selectedCred as { title?: string; is_system?: boolean },
                  );

                  // If not detected by title/is_system, check by looking up in providers
                  if (
                    !isSystemCredSelected &&
                    selectedCred.id &&
                    allProviders
                  ) {
                    const providerNames =
                      inputSubSchema.credentials_provider || [];
                    for (const providerName of providerNames) {
                      const providerData = allProviders[providerName];
                      if (!providerData) continue;
                      const systemCreds = providerData.savedCredentials.filter(
                        (cred: any) => cred.is_system === true,
                      );
                      if (
                        systemCreds.some(
                          (cred: any) => cred.id === selectedCred.id,
                        )
                      ) {
                        isSystemCredSelected = true;
                        break;
                      }
                    }
                  }
                }

                // If a system credential is selected, check if there are user credentials available
                // If not, hide this field entirely (it will still be used for execution)
                if (isSystemCredSelected) {
                  const providerNames =
                    inputSubSchema.credentials_provider || [];
                  const supportedTypes = inputSubSchema.credentials_types || [];
                  const hasUserCreds = providerNames.some(
                    (providerName: string) => {
                      const providerData = allProviders?.[providerName];
                      if (!providerData) return false;
                      const userCreds = filterSystemCredentials(
                        providerData.savedCredentials,
                      );
                      return userCreds.some((cred: { type: string }) =>
                        supportedTypes.includes(cred.type),
                      );
                    },
                  );

                  // If no user credentials available, hide the field completely
                  if (!hasUserCreds) {
                    return null;
                  }
                }

                // If a system credential is selected but user creds exist, don't show it in the UI
                // (it will still be used for execution, but user can select a user credential instead)
                const credToDisplay = isSystemCredSelected
                  ? undefined
                  : selectedCred;

                return (
                  <CredentialsInput
                    key={key}
                    schema={
                      { ...inputSubSchema, discriminator: undefined } as any
                    }
                    selectedCredentials={credToDisplay}
                    onSelectCredentials={(value) => {
                      // When user selects a credential, update the state and save to preferences
                      setInputCredentialsValue(key, value);
                      // Save to preferences store
                      if (value === undefined) {
                        store.setCredentialPreference(
                          agent.id.toString(),
                          key,
                          NONE_CREDENTIAL_MARKER,
                        );
                      } else if (value === null) {
                        store.setCredentialPreference(
                          agent.id.toString(),
                          key,
                          null,
                        );
                      } else {
                        store.setCredentialPreference(
                          agent.id.toString(),
                          key,
                          value,
                        );
                      }
                    }}
                    siblingInputs={inputValues}
                    isOptional={!requiredCredentials.has(key)}
                  />
                );
              })
              .filter(Boolean)}
          </div>
        </ModalSection>
      ) : null}
    </div>
  );
}
