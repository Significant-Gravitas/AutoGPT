"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import {
  CredentialsMetaInput,
  GraphExecutionID,
  GraphMeta,
  LibraryAgentPreset,
  LibraryAgentPresetID,
  LibraryAgentPresetUpdatable,
  Schedule,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

import ActionButtonGroup from "@/components/__legacy__/action-button-group";
import type { ButtonAction } from "@/components/__legacy__/types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/__legacy__/ui/card";
import {
  IconCross,
  IconPlay,
  IconSave,
} from "@/components/__legacy__/ui/icons";
import { CalendarClockIcon, Trash2Icon } from "lucide-react";
import { ClockIcon, InfoIcon } from "@phosphor-icons/react";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { ScheduleTaskDialog } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/cron-scheduler-dialog";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import { RunAgentInputs } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/RunAgentInputs/RunAgentInputs";
import { cn, isEmpty } from "@/lib/utils";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { CopyIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/__legacy__/ui/input";
import {
  useToast,
  useToastOnFail,
} from "@/components/molecules/Toast/use-toast";

import { AgentStatus, AgentStatusChip } from "./agent-status-chip";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";

export function AgentRunDraftView({
  graph,
  agentPreset,
  doRun: _doRun,
  onRun,
  onCreatePreset,
  onUpdatePreset,
  doDeletePreset,
  doCreateSchedule: _doCreateSchedule,
  onCreateSchedule,
  agentActions,
  runCount,
  className,
  recommendedScheduleCron,
}: {
  graph: GraphMeta;
  agentActions?: ButtonAction[];
  recommendedScheduleCron?: string | null;
  doRun?: (
    inputs: Record<string, any>,
    credentialsInputs: Record<string, CredentialsMetaInput>,
  ) => Promise<void>;
  onRun?: (runID: GraphExecutionID) => void;
  doCreateSchedule?: (
    cronExpression: string,
    scheduleName: string,
    inputs: Record<string, any>,
    credentialsInputs: Record<string, CredentialsMetaInput>,
  ) => Promise<void>;
  onCreateSchedule?: (schedule: Schedule) => void;
  runCount: number;
  className?: string;
} & (
  | {
      onCreatePreset?: (preset: LibraryAgentPreset) => void;
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
  const { completeStep: completeOnboardingStep } = useOnboarding();
  const [cronScheduleDialogOpen, setCronScheduleDialogOpen] = useState(false);

  // Update values if agentPreset parameter is changed
  useEffect(() => {
    setInputValues(agentPreset?.inputs ?? {});
    setInputCredentials(agentPreset?.credentials ?? {});
    setPresetName(agentPreset?.name ?? "");
    setPresetDescription(agentPreset?.description ?? "");
    setChangedPresetAttributes(new Set());
  }, [agentPreset]);

  const agentInputSchema = useMemo(
    () => graph.trigger_setup_info?.config_schema ?? graph.input_schema,
    [graph],
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
    () => graph.credentials_input_schema.properties,
    [graph],
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
          ? [graph.has_external_trigger ? "trigger_name" : "preset_name"]
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

  const doRun = useCallback(async () => {
    // Manually running webhook-triggered agents is not supported
    if (graph.has_external_trigger) return;

    if (!agentPreset || changedPresetAttributes.size > 0) {
      if (!allRequiredInputsAreSet || !allCredentialsAreSet) {
        notifyMissingInputs(false);
        return;
      }
      if (_doRun) {
        await _doRun(inputValues, inputCredentials);
        return;
      }
      // TODO: on executing preset with changes, ask for confirmation and offer save+run
      const newRun = await api
        .executeGraph(graph.id, graph.version, inputValues, inputCredentials)
        .catch(toastOnFail("execute agent"));

      if (newRun && onRun) onRun(newRun.id);
    } else {
      await api
        .executeLibraryAgentPreset(agentPreset.id)
        .then((newRun) => onRun && onRun(newRun.id))
        .catch(toastOnFail("execute agent preset"));
    }
    // Mark run agent onboarding step as completed
    completeOnboardingStep("MARKETPLACE_RUN_AGENT");
    if (runCount > 0) {
      completeOnboardingStep("RE_RUN_AGENT");
    }
  }, [
    api,
    graph,
    inputValues,
    inputCredentials,
    onRun,
    toastOnFail,
    completeOnboardingStep,
  ]);

  const doCreatePreset = useCallback(async () => {
    if (!onCreatePreset) return;

    if (!presetName || !allRequiredInputsAreSet || !allCredentialsAreSet) {
      notifyMissingInputs();
      return;
    }

    await api
      .createLibraryAgentPreset({
        name: presetName,
        description: presetDescription,
        graph_id: graph.id,
        graph_version: graph.version,
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
    graph,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onCreatePreset,
    toast,
    toastOnFail,
    completeOnboardingStep,
  ]);

  const doUpdatePreset = useCallback(async () => {
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
    await api
      .updateLibraryAgentPreset(agentPreset.id, updatePreset)
      .then((updatedPreset) => {
        onUpdatePreset(updatedPreset);
        setChangedPresetAttributes(new Set()); // reset change tracker
      })
      .catch(toastOnFail("update agent preset"));
  }, [
    api,
    graph,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onUpdatePreset,
    toast,
    toastOnFail,
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

  const doSetupTrigger = useCallback(async () => {
    // Setting up a trigger for non-webhook-triggered agents is not supported
    if (!graph.trigger_setup_info || !onCreatePreset) return;

    if (!presetName || !allRequiredInputsAreSet || !allCredentialsAreSet) {
      notifyMissingInputs();
      return;
    }

    await api
      .setupAgentTrigger({
        name: presetName,
        description: presetDescription,
        graph_id: graph.id,
        graph_version: graph.version,
        trigger_config: inputValues,
        agent_credentials: inputCredentials,
      })
      .then((newPreset) => {
        onCreatePreset(newPreset);
        setChangedPresetAttributes(new Set()); // reset change tracker
      })
      .catch(toastOnFail("set up agent trigger"));
  }, [
    api,
    graph,
    presetName,
    presetDescription,
    inputValues,
    inputCredentials,
    onCreatePreset,
    toast,
    toastOnFail,
    completeOnboardingStep,
  ]);

  const openScheduleDialog = useCallback(() => {
    // Scheduling is not supported for webhook-triggered agents
    if (graph.has_external_trigger) return;

    if (!allRequiredInputsAreSet || !allCredentialsAreSet) {
      notifyMissingInputs(false);
      return;
    }

    setCronScheduleDialogOpen(true);
  }, [
    graph,
    allRequiredInputsAreSet,
    allCredentialsAreSet,
    notifyMissingInputs,
  ]);

  const doSetupSchedule = useCallback(
    async (cronExpression: string, scheduleName: string) => {
      // Scheduling is not supported for webhook-triggered agents
      if (graph.has_external_trigger) return;

      if (_doCreateSchedule) {
        await _doCreateSchedule(
          cronExpression,
          scheduleName || graph.name,
          inputValues,
          inputCredentials,
        );
        return;
      }
      const schedule = await api
        .createGraphExecutionSchedule({
          graph_id: graph.id,
          graph_version: graph.version,
          name: scheduleName || graph.name,
          cron: cronExpression,
          inputs: inputValues,
          credentials: inputCredentials,
        })
        .catch(toastOnFail("set up agent run schedule"));

      if (schedule && onCreateSchedule) onCreateSchedule(schedule);
    },
    [api, graph, inputValues, inputCredentials, onCreateSchedule, toastOnFail],
  );

  const runActions: ButtonAction[] = useMemo(
    () => [
      // "Regular" agent: [run] + [save as preset] buttons
      ...(!graph.has_external_trigger
        ? ([
            {
              label: (
                <>
                  <CalendarClockIcon className="mr-2 size-4" /> Schedule run
                </>
              ),
              variant: "accent",
              callback: openScheduleDialog,
              extraProps: { "data-testid": "agent-schedule-button" },
            },
            {
              label: (
                <>
                  <IconPlay className="mr-2 size-4" /> Manual run
                </>
              ),
              callback: doRun,
              extraProps: { "data-testid": "agent-run-button" },
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
      ...(graph.has_external_trigger && !agentPreset?.webhook_id
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
                  Delete {graph.has_external_trigger ? "trigger" : "preset"}
                </>
              ),
              callback: () => doDeletePreset(agentPreset.id),
            },
          ] satisfies ButtonAction[])
        : []),
    ],
    [
      graph.has_external_trigger,
      agentPreset,
      doRun,
      doSetupTrigger,
      doCreatePreset,
      doUpdatePreset,
      doDeletePreset,
      openScheduleDialog,
      changedPresetAttributes,
      presetName,
      allRequiredInputsAreSet,
      allCredentialsAreSet,
    ],
  );

  const triggerStatus: AgentStatus | null = !agentPreset
    ? null
    : !agentPreset.webhook
      ? "broken"
      : agentPreset.is_active
        ? "active"
        : "inactive";

  return (
    <div className={cn("agpt-div flex gap-6", className)}>
      <div className="flex min-w-0 flex-1 flex-col gap-4">
        {graph.trigger_setup_info && agentPreset && (
          <Card className="agpt-box">
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle className="font-poppins text-lg">
                Trigger status
              </CardTitle>
              {triggerStatus && <AgentStatusChip status={triggerStatus} />}
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {!agentPreset.webhook_id ? (
                /* Shouldn't happen, but technically possible */
                <p className="text-sm text-destructive">
                  This trigger is not attached to a webhook. Use &quot;Set up
                  trigger&quot; to fix this.
                </p>
              ) : !graph.trigger_setup_info.credentials_input_name ? (
                /* Expose webhook URL if not auto-setup */
                <div className="text-sm">
                  <p>
                    This trigger is ready to be used. Use the Webhook URL below
                    to set up the trigger connection with the service of your
                    choosing.
                  </p>
                  <div className="nodrag mt-5 flex flex-col gap-1">
                    Webhook URL:
                    <div className="flex gap-2 rounded-md bg-gray-50 p-2">
                      <code className="select-all text-sm">
                        {agentPreset.webhook.url}
                      </code>
                      <Button
                        variant="outline"
                        size="icon"
                        className="size-7 flex-none p-1"
                        onClick={() =>
                          agentPreset.webhook &&
                          navigator.clipboard.writeText(agentPreset.webhook.url)
                        }
                        title="Copy webhook URL"
                      >
                        <CopyIcon className="size-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  This agent trigger is{" "}
                  {agentPreset.is_active
                    ? "ready. When a trigger is received, it will run with the provided settings."
                    : "disabled. It will not respond to triggers until you enable it."}
                </p>
              )}
            </CardContent>
          </Card>
        )}

        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {/* Schedule recommendation tip */}
            {recommendedScheduleCron && !graph.has_external_trigger && (
              <div className="flex items-center gap-2 rounded-md border border-violet-200 bg-violet-50 p-3">
                <ClockIcon className="h-4 w-4 text-violet-600" />
                <p className="text-sm text-violet-800">
                  <strong>Tip:</strong> For best results, run this agent{" "}
                  {humanizeCronExpression(
                    recommendedScheduleCron,
                  ).toLowerCase()}
                </p>
              </div>
            )}

            {/* Setup Instructions */}
            {graph.instructions && (
              <div className="flex items-start gap-2 rounded-md border border-violet-200 bg-violet-50 p-3">
                <InfoIcon className="mt-0.5 h-4 w-4 flex-shrink-0 text-violet-600" />
                <div className="text-sm text-violet-800">
                  <strong>Setup Instructions:</strong>{" "}
                  <span className="whitespace-pre-wrap">
                    {graph.instructions}
                  </span>
                </div>
              </div>
            )}

            {(agentPreset || graph.has_external_trigger) && (
              <>
                {/* Preset name and description */}
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center gap-1 text-sm font-medium">
                    {graph.has_external_trigger ? "Trigger" : "Preset"} Name
                    <InformationTooltip
                      description={`Name of the ${graph.has_external_trigger ? "trigger" : "preset"} you are setting up`}
                    />
                  </label>
                  <Input
                    value={presetName}
                    placeholder={`Enter ${graph.has_external_trigger ? "trigger" : "preset"} name`}
                    onChange={(e) => {
                      setPresetName(e.target.value);
                      setChangedPresetAttributes((prev) => prev.add("name"));
                    }}
                  />
                </div>
                <div className="flex flex-col space-y-2">
                  <label className="flex items-center gap-1 text-sm font-medium">
                    {graph.has_external_trigger ? "Trigger" : "Preset"}{" "}
                    Description
                    <InformationTooltip
                      description={`Description of the ${graph.has_external_trigger ? "trigger" : "preset"} you are setting up`}
                    />
                  </label>
                  <Input
                    value={presetDescription}
                    placeholder={`Enter ${graph.has_external_trigger ? "trigger" : "preset"} description`}
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
                    !agentPreset && !graph.has_external_trigger
                  }
                />
              ),
            )}

            {/* Regular inputs */}
            {Object.entries(agentInputFields).map(([key, inputSubSchema]) => (
              <div key={key} className="flex flex-col space-y-2">
                <label className="flex items-center gap-1 text-sm font-medium">
                  {inputSubSchema.title || key}
                  <InformationTooltip
                    description={inputSubSchema.description}
                  />
                </label>

                <RunAgentInputs
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
                  data-testid={`agent-input-${key}`}
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
            title={`${graph.has_external_trigger ? "Trigger" : agentPreset ? "Preset" : "Run"} actions`}
            actions={runActions}
          />
          <ScheduleTaskDialog
            open={cronScheduleDialogOpen}
            setOpen={setCronScheduleDialogOpen}
            onSubmit={doSetupSchedule}
            defaultScheduleName={graph.name}
            defaultCronExpression={recommendedScheduleCron || undefined}
          />

          {agentActions && agentActions.length > 0 && (
            <ActionButtonGroup title="Agent actions" actions={agentActions} />
          )}
        </div>
      </aside>
    </div>
  );
}
