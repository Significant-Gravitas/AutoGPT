"use client";
import React, { useCallback, useMemo } from "react";

import {
  GraphExecutionID,
  GraphMeta,
  Schedule,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

import type { ButtonAction } from "@/components/agptui/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AgentRunStatus } from "@/components/agents/agent-run-status-chip";
import { Button } from "@/components/agptui/Button";
import { Input } from "@/components/ui/input";

export default function AgentScheduleDetailsView({
  graph,
  schedule,
  onForcedRun,
  agentActions,
}: {
  graph: GraphMeta;
  schedule: Schedule;
  onForcedRun: (runID: GraphExecutionID) => void;
  agentActions: ButtonAction[];
}): React.ReactNode {
  const api = useBackendAPI();

  const selectedRunStatus: AgentRunStatus = "scheduled";

  const infoStats: { label: string; value: React.ReactNode }[] = useMemo(() => {
    return [
      {
        label: "Status",
        value:
          selectedRunStatus.charAt(0).toUpperCase() +
          selectedRunStatus.slice(1),
      },
      {
        label: "Scheduled for",
        value: schedule.next_run_time.toLocaleString(),
      },
    ];
  }, [schedule, selectedRunStatus]);

  const agentRunInputs: Record<
    string,
    { title?: string; /* type: BlockIOSubType; */ value: any }
  > = useMemo(() => {
    // TODO: show (link to) preset - https://github.com/Significant-Gravitas/AutoGPT/issues/9168

    // Add type info from agent input schema
    return Object.fromEntries(
      Object.entries(schedule.input_data).map(([k, v]) => [
        k,
        {
          title: graph.input_schema.properties[k].title,
          /* TODO: type: agent.input_schema.properties[k].type */
          value: v,
        },
      ]),
    );
  }, [graph, schedule]);

  const runNow = useCallback(
    () =>
      api
        .executeGraph(graph.id, graph.version, schedule.input_data)
        .then((run) => onForcedRun(run.graph_exec_id)),
    [api, graph, schedule, onForcedRun],
  );

  const runActions: { label: string; callback: () => void }[] = useMemo(
    () => [{ label: "Run now", callback: () => runNow() }],
    [runNow],
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
          </CardContent>
        </Card>

        <Card className="agpt-box">
          <CardHeader>
            <CardTitle className="font-poppins text-lg">Input</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {agentRunInputs !== undefined ? (
              Object.entries(agentRunInputs).map(([key, { title, value }]) => (
                <div key={key} className="flex flex-col gap-1.5">
                  <label className="text-sm font-medium">{title || key}</label>
                  <Input
                    defaultValue={value}
                    className="rounded-full"
                    disabled
                  />
                </div>
              ))
            ) : (
              <p>Loading...</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Run / Agent Actions */}
      <aside className="w-48 xl:w-56">
        <div className="flex flex-col gap-8">
          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">Run actions</h3>
            {runActions.map((action, i) => (
              <Button key={i} variant="outline" onClick={action.callback}>
                {action.label}
              </Button>
            ))}
          </div>

          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-medium">Agent actions</h3>
            {agentActions.map((action, i) => (
              <Button
                key={i}
                variant={action.variant ?? "outline"}
                onClick={action.callback}
              >
                {action.label}
              </Button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
