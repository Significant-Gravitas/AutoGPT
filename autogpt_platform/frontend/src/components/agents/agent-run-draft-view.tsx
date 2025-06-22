"use client";
import React, { useCallback, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  CredentialsMetaInput,
  GraphExecutionID,
  LibraryAgent,
  LibraryAgentPreset,
  LibraryAgentPresetUpdatable,
} from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { IconPlay, IconSave, IconSquare } from "@/components/ui/icons";
import { CredentialsInput } from "@/components/integrations/credentials-input";
import { TypeBasedInput } from "@/components/type-based-input";
import { useToastOnFail } from "@/components/ui/use-toast";
import ActionButtonGroup from "@/components/agptui/action-button-group";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import SchemaTooltip from "@/components/SchemaTooltip";
import { useToast } from "@/components/ui/use-toast";
import { isEmpty } from "lodash";
import { Input } from "@/components/ui/input";

export default function AgentRunDraftView({
  agent,
  agentPreset,
  onRun,
  onCreatePreset,
  onUpdatePreset,
  agentActions,
}: {
  agent: LibraryAgent;
  agentActions: ButtonAction[];
  onRun: (runID: GraphExecutionID) => void;
} & (
  | {
      onCreatePreset: (preset: LibraryAgentPreset) => void;
      agentPreset?: never;
      onUpdatePreset?: never;
    }
  | {
      onCreatePreset?: never;
      agentPreset: LibraryAgentPreset;
      onUpdatePreset: (preset: LibraryAgentPreset) => void;
    }
)): React.ReactNode {
  const api = useBackendAPI();
  const { toast } = useToast();
  const toastOnFail = useToastOnFail();

  const [inputValues, setInputValues] = useState<Record<string, any>>(
    agentPreset?.inputs ?? {},
  );
  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >(agentPreset?.credentials ?? {});
  const [presetName, setPresetName] = useState<string>(agentPreset?.name ?? "");
  const [presetDescription, setPresetDescription] = useState<string>(
    agentPreset?.description ?? "",
  );
  const [changedPresetAttributes, setChangedPresetAttributes] = useState<
    Set<keyof LibraryAgentPresetUpdatable>
  >(new Set());
  const { state: onboardingState, completeStep: completeOnboardingStep } =
    useOnboarding();

  const agentInputSchema = useMemo(
    () =>
      agent.has_external_trigger
        ? agent.trigger_setup_info.config_schema
        : agent.input_schema,
    [agent],
  );
  const agentInputFields = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(agentInputSchema.properties).filter(
          ([_, subSchema]) => !subSchema.hidden,
        ),
      ),
    [agentInputSchema],
  );
  const agentCredentialsInputFields = useMemo(
    () => agent.credentials_input_schema.properties,
    [agent],
  );

  const [allRequiredInputsAreSet, missingInputs] = useMemo(() => {
    const nonEmptyInputs = new Set(
      Object.keys(inputValues).filter((k) => !isEmpty(inputValues[k])),
    );
    const requiredInputs = new Set(
      agentInputSchema.required as string[] | undefined,
    );
    return [
      nonEmptyInputs.isSupersetOf(requiredInputs),
      [...requiredInputs.difference(nonEmptyInputs)],
    ];
  }, [agentInputSchema.required, inputValues]);
  const [allCredentialsAreSet, missingCredentials] = useMemo(() => {
    const availableCredentials = new Set(Object.keys(inputCredentials));
    const allCredentials = new Set(Object.keys(agentCredentialsInputFields));
    return [
      availableCredentials.isSupersetOf(allCredentials),
      [...allCredentials.difference(availableCredentials)],
    ];
  }, [agentCredentialsInputFields, inputCredentials]);
  const notifyMissingInputs = useCallback(
    (needPresetName: boolean = true) => {
      const allMissingFields = (
        needPresetName && !presetName
          ? [agent.has_external_trigger ? "trigger_name" : "preset_name"]
          : []
      )
        .concat(missingInputs)
        .concat(missingCredentials);
      toast({
        title: "⚠️ Not all required inputs are set",
        description: `Please set ${allMissingFields.map((k) => `\`${k}\``).join(", ")}`,
      });
    },
    [missingInputs, missingCredentials],
  );

  const doRun = useCallback(() => {
    // Manually running webhook-triggered agents is not supported
    if (agent.has_external_trigger) return;

    if (!agentPreset || changedPresetAttributes) {
      if (!allRequiredInputsAreSet || !allCredentialsAreSet) {
        notifyMissingInputs(false);
        return;
      }
      // TODO: on executing preset with changes, ask for confirmation and offer save+run
      api
        .executeGraph(
          agent.graph_id,
          agent.graph_version,
          inputValues,
          inputCredentials,
        )
        .then((newRun) => onRun(newRun.graph_exec_id))
        .catch(toastOnFail("execute agent"));
    } else {
      api
        .executeLibraryAgentPreset(agentPreset.id)
        .then((newRun) => onRun(newRun.id))
        .catch(toastOnFail("execute agent preset"));
    }
    // Mark run agent onboarding step as completed
    if (onboardingState?.completedSteps.includes("MARKETPLACE_ADD_AGENT")) {
      completeOnboardingStep("MARKETPLACE_RUN_AGENT");
    }
  }, [
    api,
    agent,
    inputValues,
    inputCredentials,
    onRun,
    toastOnFail,
    onboardingState,
    completeOnboardingStep,
  ]);

  const doCreatePreset = useCallback(() => {
    if (!onCreatePreset) return;

    if (!presetName || !allRequiredInputsAreSet || !allCredentialsAreSet) {
      notifyMissingInputs();
      return;
    }

    api
      .createLibraryAgentPreset({
        name: presetName,
        description: presetDescription,
        graph_id: agent.graph_id,
        graph_version: agent.graph_version,
        inputs: inputValues,
        credentials: inputCredentials,
      })
      .then((newPreset) => {
        onCreatePreset(newPreset);
        setChangedPresetAttributes(new Set()); // reset change tracker
      })
      .catch(toastOnFail("save agent preset"));
  }, [
    api,
    agent,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onCreatePreset,
    toast,
    toastOnFail,
    onboardingState,
    completeOnboardingStep,
  ]);

  const doUpdatePreset = useCallback(() => {
    if (!agentPreset || !changedPresetAttributes) return;

    if (!presetName || !allRequiredInputsAreSet || !allCredentialsAreSet) {
      notifyMissingInputs();
      return;
    }

    const updatePreset: LibraryAgentPresetUpdatable = {};
    if (changedPresetAttributes.has("name")) updatePreset["name"] = presetName;
    if (changedPresetAttributes.has("description"))
      updatePreset["description"] = presetDescription;
    if (
      changedPresetAttributes.has("inputs") ||
      changedPresetAttributes.has("credentials")
    ) {
      updatePreset["inputs"] = inputValues;
      updatePreset["credentials"] = inputCredentials;
    }
    api
      .updateLibraryAgentPreset(agentPreset.id, updatePreset)
      .then((updatedPreset) => {
        onUpdatePreset(updatedPreset);
        setChangedPresetAttributes(new Set()); // reset change tracker
      })
      .catch(toastOnFail("update agent preset"));
  }, [
    api,
    agent,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onUpdatePreset,
    toast,
    toastOnFail,
    onboardingState,
    completeOnboardingStep,
  ]);

  const doSetupTrigger = useCallback(() => {
    // Setting up a trigger for non-webhook-triggered agents is not supported
    if (!agent.has_external_trigger || !onCreatePreset) return;

    if (!presetName || !allRequiredInputsAreSet || !allCredentialsAreSet) {
      notifyMissingInputs();
      return;
    }

    const credentialsInputName =
      agent.trigger_setup_info.credentials_input_name;

    if (!credentialsInputName) {
      // FIXME: implement support for manual-setup webhooks
      toast({
        variant: "destructive",
        title: "🚧 Feature under construction",
        description: "Setting up non-auto-setup triggers is not yet supported.",
      });
      return;
    }

    api
      .setupAgentTrigger(agent.id, {
        name: presetName,
        description: presetDescription,
        trigger_config: inputValues,
        agent_credentials: inputCredentials,
      })
      .then((newPreset) => {
        onCreatePreset(newPreset);
        setChangedPresetAttributes(new Set()); // reset change tracker
      })
      .catch(toastOnFail("set up agent trigger"));

    // Mark run agent onboarding step as completed(?)
    if (onboardingState?.completedSteps.includes("MARKETPLACE_ADD_AGENT")) {
      completeOnboardingStep("MARKETPLACE_RUN_AGENT");
    }
  }, [
    api,
    agent,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onCreatePreset,
    toast,
    toastOnFail,
    onboardingState,
    completeOnboardingStep,
  ]);

  const runActions: ButtonAction[] = useMemo(
    () => [
      !agent.has_external_trigger
        ? {
            label: (
              <>
                <IconPlay className="mr-2 size-5" />
                Run
              </>
            ),
            variant: "accent",
            callback: doRun,
          }
        : !agentPreset?.webhook_id
          ? {
              label: (
                <>
                  <IconPlay className="mr-2 size-5" />
                  Set up trigger
                </>
              ),
              variant: "accent",
              callback: doSetupTrigger,
              disabled: !(
                presetName &&
                allRequiredInputsAreSet &&
                allCredentialsAreSet
              ),
            }
          : agentPreset.is_active
            ? {
                label: (
                  <>
                    <IconSquare className="mr-2 size-5" />
                    Disable trigger
                  </>
                ),
                variant: "destructive",
                callback: () =>
                  api.updateLibraryAgentPreset(agentPreset.id, {
                    is_active: false,
                  }),
              }
            : {
                label: (
                  <>
                    <IconPlay className="mr-2 size-5" />
                    Enable trigger
                  </>
                ),
                variant: "accent",
                callback: () =>
                  api.updateLibraryAgentPreset(agentPreset.id, {
                    is_active: true,
                  }),
              },
      !agentPreset
        ? {
            label: (
              <>
                <IconSave className="mr-2 size-5" /> Save as a preset
              </>
            ),
            callback: doCreatePreset,
            disabled: !(
              presetName &&
              allRequiredInputsAreSet &&
              allCredentialsAreSet
            ),
          }
        : {
            label: (
              <>
                <IconSave className="mr-2 size-5" /> Save changes
              </>
            ),
            callback: doUpdatePreset,
            disabled: !(
              changedPresetAttributes &&
              presetName &&
              allRequiredInputsAreSet &&
              allCredentialsAreSet
            ),
          },
    ],
    [
      agent.has_external_trigger,
      agentPreset,
      api,
      doRun,
      doSetupTrigger,
      doUpdatePreset,
      changedPresetAttributes,
    ],
  );

  return (
    <div className="agpt-div flex gap-6">
      <div className="flex flex-1 flex-col gap-4">
        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {(agentPreset || agent.has_external_trigger) && (
              <>
                {/* Preset name and description */}
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center gap-1 text-sm font-medium">
                    {agent.has_external_trigger ? "Trigger" : "Preset"} Name
                    <SchemaTooltip
                      description={`Name of the ${agent.has_external_trigger ? "trigger" : "preset"} you are setting up`}
                    />
                  </label>
                  <Input
                    value={presetName}
                    placeholder={`Enter ${agent.has_external_trigger ? "trigger" : "preset"} name`}
                    onChange={(e) => {
                      setPresetName(e.target.value);
                      setChangedPresetAttributes((prev) => prev.add("name"));
                    }}
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center gap-1 text-sm font-medium">
                    {agent.has_external_trigger ? "Trigger" : "Preset"}{" "}
                    Description
                    <SchemaTooltip
                      description={`Description of the ${agent.has_external_trigger ? "trigger" : "preset"} you are setting up`}
                    />
                  </label>
                  <Input
                    value={presetDescription}
                    placeholder={`Enter ${agent.has_external_trigger ? "trigger" : "preset"} description`}
                    onChange={(e) => {
                      setPresetDescription(e.target.value);
                      setChangedPresetAttributes((prev) =>
                        prev.add("description"),
                      );
                    }}
                  />
                </div>
              </>
            )}

            {/* Credentials inputs */}
            {Object.entries(agentCredentialsInputFields).map(
              ([key, inputSubSchema]) => (
                <CredentialsInput
                  key={key}
                  schema={{ ...inputSubSchema, discriminator: undefined }}
                  selectedCredentials={
                    inputCredentials[key] ?? inputSubSchema.default
                  }
                  onSelectCredentials={(value) => {
                    setInputCredentials((obj) => {
                      const newObj = { ...obj };
                      if (value === undefined) {
                        delete newObj[key];
                        return newObj;
                      }
                      return {
                        ...obj,
                        [key]: value,
                      };
                    });
                    setChangedPresetAttributes((prev) =>
                      prev.add("credentials"),
                    );
                  }}
                  hideIfSingleCredentialAvailable={
                    !agentPreset && !agent.has_external_trigger
                  }
                />
              ),
            )}

            {/* Regular inputs */}
            {Object.entries(agentInputFields).map(([key, inputSubSchema]) => (
              <div key={key} className="flex flex-col space-y-2">
                <label className="flex items-center gap-1 text-sm font-medium">
                  {inputSubSchema.title || key}
                  <SchemaTooltip description={inputSubSchema.description} />
                </label>

                <TypeBasedInput
                  schema={inputSubSchema}
                  value={inputValues[key] ?? inputSubSchema.default}
                  placeholder={inputSubSchema.description}
                  onChange={(value) => {
                    setInputValues((obj) => ({
                      ...obj,
                      [key]: value,
                    }));
                    setChangedPresetAttributes((prev) => prev.add("inputs"));
                  }}
                />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Actions */}
      <aside className="w-48 xl:w-56">
        <div className="flex flex-col gap-8">
          <ActionButtonGroup title="Run actions" actions={runActions} />

          <ActionButtonGroup title="Agent actions" actions={agentActions} />
        </div>
      </aside>
    </div>
  );
}
