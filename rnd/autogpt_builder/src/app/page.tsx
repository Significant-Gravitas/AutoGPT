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

const Monitor = () => {
  const [flows, setFlows] = useState<GraphMeta[]>([]);
  const [flowRuns, setFlowRuns] = useState<FlowRun[]>([]);
  const [selectedFlow, setSelectedFlow] = useState<GraphMeta | null>(null);
  const [selectedRun, setSelectedRun] = useState<FlowRun | null>(null);

  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const fetchAgents = useCallback(() => {
    api.listGraphs(true).then((agent) => {
      setFlows(agent);
      const flowRuns = agent.flatMap((graph) =>
        graph.executions != null
          ? graph.executions.map((execution) =>
              flowRunFromExecutionMeta(graph, execution),
            )
          : [],
      );
      setFlowRuns(flowRuns);
    });
  }, [api]);

  useEffect(() => {
    fetchAgents();
  }, [api, fetchAgents]);

  useEffect(() => {
    const intervalId = setInterval(() => fetchAgents(), 5000);
    return () => clearInterval(intervalId);
  }, [fetchAgents, flows]);

  const column1 = "md:col-span-2 xl:col-span-3 xxl:col-span-2";
  const column2 = "md:col-span-3 lg:col-span-2 xl:col-span-3 space-y-4";
  const column3 = "col-span-full xl:col-span-4 xxl:col-span-5";

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-5 lg:grid-cols-4 xl:grid-cols-10">
      <AgentFlowList
        className={column1}
        flows={flows}
        flowRuns={flowRuns}
        selectedFlow={selectedFlow}
        onSelectFlow={(f) => {
          setSelectedRun(null);
          setSelectedFlow(f.id == selectedFlow?.id ? null : f);
        }}
      />
      <FlowRunsList
        className={column2}
        flows={flows}
        runs={[
          ...(selectedFlow
            ? flowRuns.filter((v) => v.graphID == selectedFlow.id)
            : flowRuns),
        ].sort((a, b) => Number(a.startTime) - Number(b.startTime))}
        selectedRun={selectedRun}
        onSelectRun={(r) => setSelectedRun(r.id == selectedRun?.id ? null : r)}
      />
      {(selectedRun && (
        <FlowRunInfo
          flow={selectedFlow || flows.find((f) => f.id == selectedRun.graphID)!}
          flowRun={selectedRun}
          className={column3}
        />
      )) ||
        (selectedFlow && (
          <FlowInfo
            flow={selectedFlow}
            flowRuns={flowRuns.filter((r) => r.graphID == selectedFlow.id)}
            className={column3}
          />
        )) || (
          <Card className={`p-6 ${column3}`}>
            <FlowRunsStats flows={flows} flowRuns={flowRuns} />
          </Card>
        )}
    </div>
  );
};

function flowRunFromExecutionMeta(
  graphMeta: GraphMeta,
  executionMeta: ExecutionMeta,
): FlowRun {
  let status: "running" | "waiting" | "success" | "failed" = "success";
  if (executionMeta.status === "FAILED") {
    status = "failed";
  } else if (["QUEUED", "RUNNING"].includes(executionMeta.status)) {
    status = "running";
  } else if (executionMeta.status === "INCOMPLETE") {
    status = "waiting";
  }
  return {
    id: executionMeta.execution_id,
    graphID: graphMeta.id,
    graphVersion: graphMeta.version,
    status,
    startTime: new Date(executionMeta.started_at).getTime(),
    endTime: executionMeta.ended_at
      ? new Date(executionMeta.ended_at).getTime()
      : undefined,
    duration: executionMeta.duration,
    totalRunTime: executionMeta.total_run_time,
  } as FlowRun;
}

export default Monitor;
