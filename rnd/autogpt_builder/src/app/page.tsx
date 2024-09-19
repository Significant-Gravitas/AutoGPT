"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import AutoGPTServerAPI, {
  GraphMeta,
  ExecutionMeta,
  NodeExecutionResult,
} from "@/lib/autogpt-server-api";

import { Card } from "@/components/ui/card";
import { FlowRun } from "@/lib/types";
import {
  AgentFlowList,
  FlowInfo,
  FlowRunInfo,
  FlowRunsList,
  FlowRunsStats,
} from "@/components/monitor";

const getTime = (time: Date | null | undefined | string) => {
  if (typeof time === "string") {
    return Date.parse(time);
  }
  return time ? time.getTime() : 0;
};

const Monitor = () => {
  const [agents, setAgents] = useState<GraphMeta[]>([]);
  const [agentRuns, setAgentRuns] = useState<FlowRun[]>([]);
  const [selectedFlow, setSelectedFlow] = useState<GraphMeta | null>(null);
  const [selectedRun, setSelectedRun] = useState<FlowRun | null>(null);

  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const fetchAgents = useCallback(() => {
    api.listGraphs().then((agent) => {
      setAgents(agent);
      const flowRuns = agent.flatMap((graph) =>
        graph.executions != null
          ? graph.executions.map((execution) => ({
              id: execution.execution_id,
              graphID: graph.id,
              graphVersion: graph.version,
              status: execution.status.toLowerCase() as
                | "running"
                | "waiting"
                | "success"
                | "failed",
              startTime: getTime(execution.started_at),
              endTime: getTime(execution.ended_at),
              duration:
                (getTime(execution.ended_at) - getTime(execution.started_at)) /
                1000,
              totalRunTime:
                (getTime(execution.ended_at) - getTime(execution.started_at)) /
                1000,
              nodeExecutionResults: [],
            }))
          : [],
      );
      setAgentRuns(flowRuns);
    });
  }, [api]);

  useEffect(() => {
    fetchAgents();
  });

  useEffect(() => {
    const intervalId = setInterval(() => fetchAgents(), 5000);
    return () => clearInterval(intervalId);
  }, [fetchAgents, agents]);

  const column1 = "md:col-span-2 xl:col-span-3 xxl:col-span-2";
  const column2 = "md:col-span-3 lg:col-span-2 xl:col-span-3 space-y-4";
  const column3 = "col-span-full xl:col-span-4 xxl:col-span-5";

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-5 lg:grid-cols-4 xl:grid-cols-10">
      <AgentFlowList
        className={column1}
        flows={agents}
        flowRuns={agentRuns}
        selectedFlow={selectedFlow}
        onSelectFlow={(f) => {
          setSelectedRun(null);
          setSelectedFlow(f.id == selectedFlow?.id ? null : f);
        }}
      />
      <FlowRunsList
        className={column2}
        flows={agents}
        runs={[
          ...(selectedFlow
            ? agentRuns.filter((v) => v.graphID == selectedFlow.id)
            : agentRuns),
        ].sort((a, b) => Number(a.startTime) - Number(b.startTime))}
        selectedRun={selectedRun}
        onSelectRun={(r) => setSelectedRun(r.id == selectedRun?.id ? null : r)}
      />
      {(selectedRun && (
        <FlowRunInfo
          flow={
            selectedFlow || agents.find((f) => f.id == selectedRun.graphID)!
          }
          flowRun={selectedRun}
          className={column3}
        />
      )) ||
        (selectedFlow && (
          <FlowInfo
            flow={selectedFlow}
            flowRuns={agentRuns.filter((r) => r.graphID == selectedFlow.id)}
            className={column3}
          />
        )) || (
          <Card className={`p-6 ${column3}`}>
            <FlowRunsStats flows={agents} flowRuns={agentRuns} />
          </Card>
        )}
    </div>
  );
};

function flowRunFromNodeExecutionResults(
  nodeExecutionResults: NodeExecutionResult[],
): FlowRun {
  // Determine overall status
  let status: "running" | "waiting" | "success" | "failed" = "success";
  for (const execution of nodeExecutionResults) {
    if (execution.status === "FAILED") {
      status = "failed";
      break;
    } else if (["QUEUED", "RUNNING"].includes(execution.status)) {
      status = "running";
      break;
    } else if (execution.status === "INCOMPLETE") {
      status = "waiting";
    }
  }

  // Determine aggregate startTime, endTime, and totalRunTime
  const now = Date.now();
  const startTime = Math.min(
    ...nodeExecutionResults.map((ner) => getTime(ner.add_time)),
    now,
  );
  const endTime = ["success", "failed"].includes(status)
    ? Math.max(
        ...nodeExecutionResults.map((ner) => getTime(ner.end_time) || 0),
        startTime,
      )
    : now;
  const duration = (endTime - startTime) / 1000; // Convert to seconds
  const totalRunTime =
    nodeExecutionResults.reduce(
      (cum, node) =>
        cum +
        ((getTime(node.end_time) ?? now) - (getTime(node.start_time) ?? now)),
      0,
    ) / 1000;

  return {
    id: nodeExecutionResults[0].graph_exec_id,
    graphID: nodeExecutionResults[0].graph_id,
    graphVersion: nodeExecutionResults[0].graph_version,
    status,
    startTime,
    endTime,
    duration,
    totalRunTime,
    nodeExecutionResults: nodeExecutionResults,
  };
}

export default Monitor;
