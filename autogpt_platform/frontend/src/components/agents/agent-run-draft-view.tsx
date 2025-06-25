"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  CredentialsMetaInput,
  GraphExecutionID,
  LibraryAgent,
  LibraryAgentPreset,
  LibraryAgentPresetID,
  LibraryAgentPresetUpdatable,
} from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { IconCross, IconPlay, IconSave } from "@/components/ui/icons";
import { CredentialsInput } from "@/components/integrations/credentials-input";
import { TypeBasedInput } from "@/components/type-based-input";
import { useToastOnFail } from "@/components/ui/use-toast";
import ActionButtonGroup from "@/components/agptui/action-button-group";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { Trash2Icon } from "lucide-react";
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
  doDeletePreset,
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
      doDeletePreset?: never;
    }
  | {
      onCreatePreset?: never;
      agentPreset: LibraryAgentPreset;
      onUpdatePreset: (preset: LibraryAgentPreset) => void;
      doDeletePreset: (presetID: LibraryAgentPresetID) => void;
    }
)): React.ReactNode {
  const api = useBackendAPI();
  const { toast } = useToast();
  const toastOnFail = useToastOnFail();

  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [presetName, setPresetName] = useState<string>("");
  const [presetDescription, setPresetDescription] = useState<string>("");
  const [changedPresetAttributes, setChangedPresetAttributes] = useState<
    Set<keyof LibraryAgentPresetUpdatable>
  >(new Set());
  const { state: onboardingState, completeStep: completeOnboardingStep } =
    useOnboarding();

  // Update values if agentPreset parameter is changed
  useEffect(() => {
    setInputValues(agentPreset?.inputs ?? {});
    setInputCredentials(agentPreset?.credentials ?? {});
    setPresetName(agentPreset?.name ?? "");
    setPresetDescription(agentPreset?.description ?? "");
    setChangedPresetAttributes(new Set());
  }, [agentPreset]);

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

    if (!agentPreset || changedPresetAttributes.size > 0) {
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
    if (!agentPreset || changedPresetAttributes.size == 0) return;

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

  const doSetPresetActive = useCallback(
    async (active: boolean) => {
      if (!agentPreset) return;
      const updatedPreset = await api.updateLibraryAgentPreset(agentPreset.id, {
        is_active: active,
      });
      onUpdatePreset(updatedPreset);
    },
    [agentPreset, api, onUpdatePreset],
  );

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
      // "Regular" agent: [run] + [save as preset] buttons
      ...(!agent.has_external_trigger
        ? ([
            {
              label: (
                <>
                  <IconPlay className="mr-2 size-4" /> Run
                </>
              ),
              variant: "accent",
              callback: doRun,
            },
            // {
            //   label: (
            //     <>
            //       <IconSave className="mr-2 size-4" /> Save as a preset
            //     </>
            //   ),
            //   callback: doCreatePreset,
            //   disabled: !(
            //     presetName &&
            //     allRequiredInputsAreSet &&
            //     allCredentialsAreSet
            //   ),
            // },
          ] satisfies ButtonAction[])
        : []),
      // Triggered agent: [setup] button
      ...(agent.has_external_trigger && !agentPreset?.webhook_id
        ? ([
            {
              label: (
                <>
                  <IconPlay className="mr-2 size-4" /> Set up trigger
                </>
              ),
              variant: "accent",
              callback: doSetupTrigger,
              disabled: !(
                presetName &&
                allRequiredInputsAreSet &&
                allCredentialsAreSet
              ),
            },
          ] satisfies ButtonAction[])
        : []),
      // Existing agent trigger: [enable]/[disable] button
      ...(agentPreset?.webhook_id
        ? ([
            agentPreset.is_active
              ? {
                  label: (
                    <>
                      <IconCross className="mr-2.5 size-3.5" /> Disable trigger
                    </>
                  ),
                  variant: "destructive",
                  callback: () => doSetPresetActive(false),
                }
              : {
                  label: (
                    <>
                      <IconPlay className="mr-2 size-4" /> Enable trigger
                    </>
                  ),
                  variant: "accent",
                  callback: () => doSetPresetActive(true),
                },
          ] satisfies ButtonAction[])
        : []),
      // Existing agent preset/trigger: [save] and [delete] buttons
      ...(agentPreset
        ? ([
            {
              label: (
                <>
                  <IconSave className="mr-2 size-4" /> Save changes
                </>
              ),
              callback: doUpdatePreset,
              disabled: !(
                changedPresetAttributes.size > 0 &&
                presetName &&
                allRequiredInputsAreSet &&
                allCredentialsAreSet
              ),
            },
            {
              label: (
                <>
                  <Trash2Icon className="mr-2 size-4" />
                  Delete {agent.has_external_trigger ? "trigger" : "preset"}
                </>
              ),
              callback: () => doDeletePreset(agentPreset.id),
            },
          ] satisfies ButtonAction[])
        : []),
    ],
    [
      agent.has_external_trigger,
      agentPreset,
      api,
      doRun,
      doSetupTrigger,
      doCreatePreset,
      doUpdatePreset,
      doDeletePreset,
      changedPresetAttributes,
      presetName,
      allRequiredInputsAreSet,
      allCredentialsAreSet,
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
          <ActionButtonGroup
            title={`${agent.has_external_trigger ? "Trigger" : agentPreset ? "Preset" : "Run"} actions`}
            actions={runActions}
          />

          <ActionButtonGroup title="Agent actions" actions={agentActions} />
        </div>
      </aside>
    </div>
  );
}
