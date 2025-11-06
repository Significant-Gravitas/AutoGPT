"use client";
import moment from "moment";
import React, { useCallback, useMemo } from "react";

import {
  Graph,
  GraphExecution,
  GraphExecutionID,
  GraphExecutionMeta,
  LibraryAgent,
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
  IconRefresh,
  IconSquare,
  IconCircleAlert,
} from "@/components/__legacy__/ui/icons";
import { Input } from "@/components/__legacy__/ui/input";
import LoadingBox from "@/components/__legacy__/ui/loading";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useToastOnFail } from "@/components/molecules/Toast/use-toast";

import { AgentRunStatus, agentRunStatusMap } from "./agent-run-status-chip";
import useCredits from "@/hooks/useCredits";
import { AgentRunOutputView } from "./agent-run-output-view";
import { analytics } from "@/services/analytics";

export function AgentRunDetailsView({
  agent,
  graph,
  run,
  agentActions,
  onRun,
  doDeleteRun,
  doCreatePresetFromRun,
}: {
  agent: LibraryAgent;
  graph: Graph;
  run: GraphExecution | GraphExecutionMeta;
  agentActions: ButtonAction[];
  onRun: (runID: GraphExecutionID) => void;
  doDeleteRun: () => void;
  doCreatePresetFromRun: () => void;
}): React.ReactNode {
  const api = useBackendAPI();
  const { formatCredits } = useCredits();

  const runStatus: AgentRunStatus = useMemo(
    () => agentRunStatusMap[run.status],
    [run],
  );

  const toastOnFail = useToastOnFail();

  const infoStats: { label: string; value: React.ReactNode }[] = useMemo(() => {
    if (!run) return [];
    return [
      {
        label: "Status",
        value: runStatus.charAt(0).toUpperCase() + runStatus.slice(1),
      },
      {
        label: "Started",
        value: `${moment(run.started_at).fromNow()}, ${moment(run.started_at).format("HH:mm")}`,
      },
      ...(run.stats
        ? [
            {
              label: "Duration",
              value: moment.duration(run.stats.duration, "seconds").humanize(),
            },
            { label: "Steps", value: run.stats.node_exec_count },
            { label: "Cost", value: formatCredits(run.stats.cost) },
          ]
        : []),
    ];
  }, [run, runStatus, formatCredits]);

  const agentRunInputs:
    | Record<
        string,
        {
          title?: string;
          /* type: BlockIOSubType; */
          value: string | number | undefined;
        }
      >
    | undefined = useMemo(() => {
    if (!run.inputs) return undefined;
    // TODO: show (link to) preset - https://github.com/Significant-Gravitas/AutoGPT/issues/9168

    // Add type info from agent input schema
    return Object.fromEntries(
      Object.entries(run.inputs).map(([k, v]) => [
        k,
        {
          title: graph.input_schema.properties[k]?.title,
          // type: graph.input_schema.properties[k].type, // TODO: implement typed graph inputs
          value: typeof v == "object" ? JSON.stringify(v, undefined, 2) : v,
        },
      ]),
    );
  }, [graph, run]);

  const runAgain = useCallback(() => {
    if (
      !run.inputs ||
      !(graph.credentials_input_schema?.required ?? []).every(
        (k) => k in (run.credential_inputs ?? {}),
      )
    )
      return;

    if (run.preset_id) {
      return api
        .executeLibraryAgentPreset(
          run.preset_id,
          run.inputs!,
          run.credential_inputs!,
        )
        .then(({ id }) => {
          analytics.sendDatafastEvent("run_agent", {
            name: graph.name,
            id: graph.id,
          });
          onRun(id);
        })
        .catch(toastOnFail("execute agent preset"));
    }

    return api
      .executeGraph(
        graph.id,
        graph.version,
        run.inputs!,
        run.credential_inputs!,
      )
      .then(({ id }) => {
        analytics.sendDatafastEvent("run_agent", {
          name: graph.name,
          id: graph.id,
        });
        onRun(id);
      })
      .catch(toastOnFail("execute agent"));
  }, [api, graph, run, onRun, toastOnFail]);

  const stopRun = useCallback(
    () => api.stopGraphExecution(graph.id, run.id),
    [api, graph.id, run.id],
  );

  const agentRunOutputs:
    | Record<
        string,
        {
          title?: string;
          /* type: BlockIOSubType; */
          values: Array<React.ReactNode>;
        }
      >
    | null
    | undefined = useMemo(() => {
    if (!("outputs" in run)) return undefined;
    if (!["running", "success", "failed", "stopped"].includes(runStatus))
      return null;

    // Add type info from agent input schema
    return Object.fromEntries(
      Object.entries(run.outputs).map(([k, vv]) => [
        k,
        {
          title: graph.output_schema.properties[k].title,
          /* type: agent.output_schema.properties[k].type */
          values: vv.map((v) =>
            typeof v == "object" ? JSON.stringify(v, undefined, 2) : v,
          ),
        },
      ]),
    );
  }, [graph, run, runStatus]);

  const runActions: ButtonAction[] = useMemo(
    () => [
      ...(["running", "queued"].includes(runStatus)
        ? ([
            {
              label: (
                <>
                  <IconSquare className="mr-2 size-4" />
                  Stop run
                </>
              ),
              variant: "secondary",
              callback: stopRun,
            },
          ] satisfies ButtonAction[])
        : []),
      ...(["success", "failed", "stopped"].includes(runStatus) &&
      !graph.has_external_trigger &&
      (graph.credentials_input_schema?.required ?? []).every(
        (k) => k in (run.credential_inputs ?? {}),
      )
        ? [
            {
              label: (
                <>
                  <IconRefresh className="mr-2 size-4" />
                  Run again
                </>
              ),
              callback: runAgain,
              dataTestId: "run-again-button",
            },
          ]
        : []),
      ...(agent.can_access_graph
        ? [
            {
              label: "Open run in builder",
              href: `/build?flowID=${run.graph_id}&flowVersion=${run.graph_version}&flowExecutionID=${run.id}`,
            },
          ]
        : []),
      { label: "Create preset from run", callback: doCreatePresetFromRun },
      { label: "Delete run", variant: "secondary", callback: doDeleteRun },
    ],
    [
      runStatus,
      runAgain,
      stopRun,
      doDeleteRun,
      doCreatePresetFromRun,
      graph.has_external_trigger,
      graph.credentials_input_schema?.required,
      agent.can_access_graph,
      run.graph_id,
      run.graph_version,
      run.id,
    ],
  );

  return (
    <div className="agpt-div flex gap-6">
      <div className="flex flex-1 flex-col gap-4">
        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Info</CardTitle>
          </CardHeader>

          <CardContent>
            <div className="flex justify-stretch gap-4">
              {infoStats.map(({ label, value }) => (
                <div key={label} className="flex-1">
                  <p className="text-sm font-medium text-black">{label}</p>
                  <p className="text-sm text-neutral-600">{value}</p>
                </div>
              ))}
            </div>
            {run.status === "FAILED" && (
              <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20">
                <p className="text-sm text-red-800 dark:text-red-200">
                  <strong>Error:</strong>{" "}
                  {run.stats?.error ||
                    "The execution failed due to an internal error. You can re-run the agent to retry."}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Smart Agent Execution Summary */}
        {run.stats?.activity_status && (
          <Card className="agpt-box">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 font-poppins text-lg">
                Task Summary
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <IconCircleAlert className="size-4 cursor-help text-neutral-500 hover:text-neutral-700" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">
                        This AI-generated summary describes how the agent
                        handled your task. It’s an experimental feature and may
                        occasionally be inaccurate.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed text-neutral-700">
                {run.stats.activity_status}
              </p>
            </CardContent>
          </Card>
        )}

        {agentRunOutputs !== null && (
          <AgentRunOutputView agentRunOutputs={agentRunOutputs} />
        )}

        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {agentRunInputs !== undefined ? (
              Object.entries(agentRunInputs).map(([key, { title, value }]) => (
                <div key={key} className="flex flex-col gap-1.5">
                  <label className="text-sm font-medium">{title || key}</label>
                  <Input value={value} className="rounded-full" disabled />
                </div>
              ))
            ) : (
              <LoadingBox spinnerSize={12} className="h-24" />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Run / Agent Actions */}
      <aside className="w-48 xl:w-56">
        <div className="flex flex-col gap-8">
          <ActionButtonGroup title="Run actions" actions={runActions} />

          <ActionButtonGroup title="Agent actions" actions={agentActions} />
        </div>
      </aside>
    </div>
  );
}
