"use client";
import React, { useCallback, useMemo } from "react";
import moment from "moment";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  BlockIOSubType,
  GraphExecution,
  GraphExecutionMeta,
  GraphMeta,
} from "@/lib/autogpt-server-api";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/agptui/Button";
import { Input } from "@/components/ui/input";

import {
  AgentRunStatus,
  agentRunStatusMap,
} from "@/components/agents/agent-run-status-chip";

export default function AgentRunDetailsView({
  agent,
  run,
  agentActions,
}: {
  agent: GraphMeta;
  run: GraphExecution | GraphExecutionMeta;
  agentActions: { label: string; callback: () => void }[];
}): React.ReactNode {
  const api = useBackendAPI();

  const selectedRunStatus: AgentRunStatus = useMemo(
    () => agentRunStatusMap[run.status],
    [run],
  );

  const infoStats: { label: string; value: React.ReactNode }[] = useMemo(() => {
    if (!run) return [];
    return [
      {
        label: "Status",
        value:
          selectedRunStatus.charAt(0).toUpperCase() +
          selectedRunStatus.slice(1),
      },
      {
        label: "Started",
        value: `${moment(run.started_at).fromNow()}, ${moment(run.started_at).format("HH:mm")}`,
      },
      {
        label: "Duration",
        value: `${moment.duration(run.duration, "seconds").humanize()}`,
      },
      // { label: "Cost", value: selectedRun.cost },  // TODO: implement cost - https://github.com/Significant-Gravitas/AutoGPT/issues/9181
    ];
  }, [run, selectedRunStatus]);

  const agentRunInputs:
    | Record<string, { title?: string; /* type: BlockIOSubType; */ value: any }>
    | undefined = useMemo(() => {
    if (!("inputs" in run)) return undefined;
    // TODO: show (link to) preset - https://github.com/Significant-Gravitas/AutoGPT/issues/9168

    // Add type info from agent input schema
    return Object.fromEntries(
      Object.entries(run.inputs).map(([k, v]) => [
        k,
        {
          title: agent.input_schema.properties[k].title,
          // type: agent.input_schema.properties[k].type, // TODO: implement typed graph inputs
          value: v,
        },
      ]),
    );
  }, [agent, run]);

  const runAgain = useCallback(
    () =>
      agentRunInputs &&
      api.executeGraph(
        agent.id,
        agent.version,
        Object.fromEntries(
          Object.entries(agentRunInputs).map(([k, v]) => [k, v.value]),
        ),
      ),
    [api, agent, agentRunInputs],
  );

  const agentRunOutputs:
    | Record<
        string,
        { title?: string; /* type: BlockIOSubType; */ values: Array<any> }
      >
    | null
    | undefined = useMemo(() => {
    if (!("outputs" in run)) return undefined;
    if (!["running", "success", "failed"].includes(selectedRunStatus))
      return null;

    // Add type info from agent input schema
    return Object.fromEntries(
      Object.entries(run.outputs).map(([k, v]) => [
        k,
        {
          title: agent.output_schema.properties[k].title,
          /* type: agent.output_schema.properties[k].type */
          values: v,
        },
      ]),
    );
  }, [agent, run, selectedRunStatus]);

  const runActions: { label: string; callback: () => void }[] = useMemo(
    () => [{ label: "Run again", callback: () => runAgain() }],
    [runAgain],
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

        {agentRunOutputs !== null && (
          <Card className="agpt-box">
            <CardHeader>
              <CardTitle className="font-poppins text-lg">Output</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {agentRunOutputs !== undefined ? (
                Object.entries(agentRunOutputs).map(
                  ([key, { title, values }]) => (
                    <div key={key} className="flex flex-col gap-1.5">
                      <label className="text-sm font-medium">
                        {title || key}
                      </label>
                      {values.map((value, i) => (
                        <pre key={i}>{value}</pre>
                      ))}
                      {/* TODO: pretty type-dependent rendering */}
                    </div>
                  ),
                )
              ) : (
                <p>Loading...</p>
              )}
            </CardContent>
          </Card>
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
              <Button key={i} variant="outline" onClick={action.callback}>
                {action.label}
              </Button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
