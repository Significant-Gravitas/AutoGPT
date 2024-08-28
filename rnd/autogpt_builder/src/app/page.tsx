"use client";
import React, { useEffect, useState } from "react";

import AutoGPTServerAPI, {
  GraphMeta,
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

  const api = new AutoGPTServerAPI();

  useEffect(() => fetchFlowsAndRuns(), []);
  useEffect(() => {
    const intervalId = setInterval(
      () => flows.map((f) => refreshFlowRuns(f.id)),
      5000,
    );
    return () => clearInterval(intervalId);
  }, []);

  function fetchFlowsAndRuns() {
    api.listGraphs().then((flows) => {
      setFlows(flows);
      flows.map((flow) => refreshFlowRuns(flow.id));
    });
  }

  function refreshFlowRuns(flowID: string) {
    // Fetch flow run IDs
    api.listGraphRunIDs(flowID).then((runIDs) =>
      runIDs.map((runID) => {
        let run;
        if (
          (run = flowRuns.find((fr) => fr.id == runID)) &&
          !["waiting", "running"].includes(run.status)
        ) {
          return;
        }

        // Fetch flow run
        api.getGraphExecutionInfo(flowID, runID).then((execInfo) =>
          setFlowRuns((flowRuns) => {
            if (execInfo.length == 0) return flowRuns;

            const flowRunIndex = flowRuns.findIndex((fr) => fr.id == runID);
            const flowRun = flowRunFromNodeExecutionResults(execInfo);
            if (flowRunIndex > -1) {
              flowRuns.splice(flowRunIndex, 1, flowRun);
            } else {
              flowRuns.push(flowRun);
            }
            return [...flowRuns];
          }),
        );
      }),
    );
  }

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
        runs={(selectedFlow
          ? flowRuns.filter((v) => v.graphID == selectedFlow.id)
          : flowRuns
        ).toSorted((a, b) => Number(a.startTime) - Number(b.startTime))}
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
    ...nodeExecutionResults.map((ner) => ner.add_time.getTime()),
    now,
  );
  const endTime = ["success", "failed"].includes(status)
    ? Math.max(
        ...nodeExecutionResults.map((ner) => ner.end_time?.getTime() || 0),
        startTime,
      )
    : now;
  const duration = (endTime - startTime) / 1000; // Convert to seconds
  const totalRunTime =
    nodeExecutionResults.reduce(
      (cum, node) =>
        cum +
        ((node.end_time?.getTime() ?? now) -
          (node.start_time?.getTime() ?? now)),
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
