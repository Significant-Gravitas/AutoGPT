"use client";
import React, {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  CredentialsMetaInput,
  CredentialsType,
  Graph,
  GraphExecutionID,
  LibraryAgentPreset,
  LibraryAgentPresetID,
  LibraryAgentPresetUpdatable,
  Schedule,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

import { RunAgentInputs } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentInputs/RunAgentInputs";
import { ScheduleTaskDialog } from "@/app/(platform)/library/agents/[id]/components/OldAgentLibraryView/components/cron-scheduler-dialog";
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
import { Input } from "@/components/__legacy__/ui/input";
import { Button } from "@/components/atoms/Button/Button";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import {
  findSavedCredentialByProviderAndType,
  findSavedUserCredentialByProviderAndType,
} from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/helpers";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import {
  useToast,
  useToastOnFail,
} from "@/components/molecules/Toast/use-toast";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { cn, isEmpty } from "@/lib/utils";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { ClockIcon, CopyIcon, InfoIcon } from "@phosphor-icons/react";
import { CalendarClockIcon, Trash2Icon } from "lucide-react";

import { analytics } from "@/services/analytics";
import { AgentStatus, AgentStatusChip } from "./agent-status-chip";

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
  className,
  recommendedScheduleCron,
}: {
  graph: Graph;
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
  const allProviders = useContext(CredentialsProvidersContext);

  const [inputValues, setInputValues] = useState<Record<string, any>>({});
  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [presetName, setPresetName] = useState<string>("");
  const [presetDescription, setPresetDescription] = useState<string>("");
  const [changedPresetAttributes, setChangedPresetAttributes] = useState<
    Set<keyof LibraryAgentPresetUpdatable>
  >(new Set());
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
  const credentialFields = useMemo(
    function getCredentialFields() {
      return Object.entries(agentCredentialsInputFields);
    },
    [agentCredentialsInputFields],
  );
  const requiredCredentials = useMemo(
    function getRequiredCredentials() {
      return new Set(
        (graph.credentials_input_schema?.required as string[]) || [],
      );
    },
    [graph.credentials_input_schema?.required],
  );

  useEffect(
    function initializeDefaultCredentials() {
      if (!allProviders) return;
      if (!graph.credentials_input_schema?.properties) return;
      if (requiredCredentials.size === 0) return;

      setInputCredentials(function updateCredentials(currentCreds) {
        const next = { ...currentCreds };
        let didAdd = false;

        for (const key of requiredCredentials) {
          if (next[key]) continue;
          const schema = graph.credentials_input_schema.properties[key];
          if (!schema) continue;

          const providerNames = schema.credentials_provider || [];
          const credentialTypes = schema.credentials_types || [];
          const requiredScopes = schema.credentials_scopes;

          const userCredential = findSavedUserCredentialByProviderAndType(
            providerNames,
            credentialTypes,
            requiredScopes,
            allProviders,
          );

          const savedCredential =
            userCredential ||
            findSavedCredentialByProviderAndType(
              providerNames,
              credentialTypes,
              requiredScopes,
              allProviders,
            );

          if (!savedCredential) continue;

          next[key] = {
            id: savedCredential.id,
            provider: savedCredential.provider,
            type: savedCredential.type as CredentialsType,
            title: savedCredential.title,
          };
          didAdd = true;
        }

        if (!didAdd) return currentCreds;
        return next;
      });
    },
    [
      allProviders,
      graph.credentials_input_schema?.properties,
      requiredCredentials,
    ],
  );

  const [allRequiredInputsAreSet, missingInputs] = useMemo(() => {
    const nonEmptyInputs = new Set(
      Object.keys(inputValues).filter((k) => !isEmpty(inputValues[k])),
    );
    const requiredInputs = new Set(
      agentInputSchema.required as string[] | undefined,
    );
    // Backwards-compatible implementation of isSupersetOf and difference
    const isSuperset = Array.from(requiredInputs).every((item) =>
      nonEmptyInputs.has(item),
    );
    const difference = Array.from(requiredInputs).filter(
      (item) => !nonEmptyInputs.has(item),
    );
    return [isSuperset, difference];
  }, [agentInputSchema.required, inputValues]);
  const [allCredentialsAreSet, missingCredentials] = useMemo(
    function getCredentialStatus() {
      const missing = Array.from(requiredCredentials).filter((key) => {
        const cred = inputCredentials[key];
        return !cred || !cred.id;
      });
      return [missing.length === 0, missing];
    },
    [requiredCredentials, inputCredentials],
  );
  function addChangedCredentials(prev: Set<keyof LibraryAgentPresetUpdatable>) {
    const next = new Set(prev);
    next.add("credentials");
    return next;
  }

  function handleCredentialChange(key: string, value?: CredentialsMetaInput) {
    setInputCredentials(function updateInputCredentials(currentCreds) {
      const next = { ...currentCreds };
      if (value === undefined) {
        delete next[key];
        return next;
      }
      next[key] = value;
      return next;
    });
    setChangedPresetAttributes(addChangedCredentials);
  }

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
        .executeGraph(
          graph.id,
          graph.version,
          inputValues,
          inputCredentials,
          "library",
        )
        .catch(toastOnFail("execute agent"));

      if (newRun && onRun) onRun(newRun.id);
    } else {
      await api
        .executeLibraryAgentPreset(agentPreset.id)
        .then((newRun) => onRun && onRun(newRun.id))
        .catch(toastOnFail("execute agent preset"));
    }

    analytics.sendDatafastEvent("run_agent", {
      name: graph.name,
      id: graph.id,
    });
  }, [api, graph, inputValues, inputCredentials, onRun, toastOnFail]);

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

      analytics.sendDatafastEvent("schedule_agent", {
        name: graph.name,
        id: graph.id,
        cronExpression: cronExpression,
      });

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

            {/* Regular inputs */}
            {Object.entries(agentInputFields).map(([key, inputSubSchema]) => (
              <RunAgentInputs
                key={key}
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
            ))}

            {/* Credentials inputs */}
            {credentialFields.length > 0 && (
              <CredentialsGroupedView
                credentialFields={credentialFields}
                requiredCredentials={requiredCredentials}
                inputCredentials={inputCredentials}
                inputValues={inputValues}
                onCredentialChange={handleCredentialChange}
              />
            )}
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
