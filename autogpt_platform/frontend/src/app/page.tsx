"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import AutoGPTServerAPI, {
  GraphMetaWithRuns,
  ExecutionMeta,
  Schedule,
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
import { SchedulesTable } from "@/components/monitor/scheduleTable";

const Monitor = () => {
  const [flows, setFlows] = useState<GraphMetaWithRuns[]>([]);
  const [flowRuns, setFlowRuns] = useState<FlowRun[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [selectedFlow, setSelectedFlow] = useState<GraphMetaWithRuns | null>(
    null,
  );
  const [selectedRun, setSelectedRun] = useState<FlowRun | null>(null);
  const [sortColumn, setSortColumn] = useState<keyof Schedule>("id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const fetchSchedules = useCallback(async () => {
    setSchedules(await api.listSchedules());
  }, [api]);

  const removeSchedule = useCallback(
    async (scheduleId: string) => {
      const removedSchedule = await api.deleteSchedule(scheduleId);
      setSchedules(schedules.filter((s) => s.id !== removedSchedule.id));
    },
    [schedules, api],
  );

  const fetchAgents = useCallback(() => {
    api.listGraphsWithRuns().then((agents) => {
      // Group graphs by ID
      const graphsById = agents.reduce(
        (acc, graph) => {
          if (!acc[graph.id]) {
            // Initialize with the first version we see
            acc[graph.id] = {
              ...graph,
              executions: [],
            };
          }

          // Combine executions from all versions
          if (graph.executions) {
            acc[graph.id].executions = [
              ...(acc[graph.id].executions || []),
              ...graph.executions,
            ];
          }

          // Always keep the latest version's metadata
          if (!acc[graph.id].version || graph.version > acc[graph.id].version) {
            // Update metadata while preserving combined executions
            const executions = acc[graph.id].executions;
            acc[graph.id] = {
              ...graph,
              executions,
            };
          }

          return acc;
        },
        {} as Record<string, GraphMetaWithRuns>,
      );

      // Convert back to array
      const combinedGraphs = Object.values(graphsById);
      setFlows(combinedGraphs);

      // Create flow runs with version information
      const flowRuns = agents.flatMap((graph) =>
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
  }, [fetchAgents]);

  useEffect(() => {
    fetchSchedules();
  }, [fetchSchedules]);

  useEffect(() => {
    const intervalId = setInterval(() => fetchAgents(), 5000);
    return () => clearInterval(intervalId);
  }, [fetchAgents, flows]);

  const column1 = "md:col-span-2 xl:col-span-3 xxl:col-span-2";
  const column2 = "md:col-span-3 lg:col-span-2 xl:col-span-3";
  const column3 = "col-span-full xl:col-span-4 xxl:col-span-5";

  const handleSort = (column: keyof Schedule) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-5 lg:grid-cols-4 xl:grid-cols-10">
      <AgentFlowList
        className={column1}
        flows={flows}
        flowRuns={flowRuns}
        selectedFlow={selectedFlow}
        onSelectFlow={(f) => {
          setSelectedRun(null);
          setSelectedFlow(
            f.id == selectedFlow?.id ? null : (f as GraphMetaWithRuns),
          );
        }}
      />
      <FlowRunsList
        className={column2}
        flows={flows}
        runs={[
          ...(selectedFlow
            ? flowRuns.filter((v) => v.graphID == selectedFlow.id)
            : flowRuns),
        ].sort((a, b) => Number(b.startTime) - Number(a.startTime))}
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
            refresh={() => {
              fetchAgents();
              setSelectedFlow(null);
              setSelectedRun(null);
            }}
          />
        )) || (
          <Card className={`p-6 ${column3}`}>
            <FlowRunsStats flows={flows} flowRuns={flowRuns} />
          </Card>
        )}
      <div className="col-span-full xl:col-span-6">
        <SchedulesTable
          schedules={schedules} // all schedules
          agents={flows} // for filtering purpose
          onRemoveSchedule={removeSchedule}
          sortColumn={sortColumn}
          sortDirection={sortDirection}
          onSort={handleSort}
        />
      </div>
    </div>
  );
};

function flowRunFromExecutionMeta(
  graphMeta: GraphMetaWithRuns,
  executionMeta: ExecutionMeta,
): FlowRun {
  return {
    id: executionMeta.execution_id,
    graphID: graphMeta.id,
    graphVersion: graphMeta.version,
    status: executionMeta.status,
    startTime: executionMeta.started_at,
    endTime: executionMeta.ended_at,
    duration: executionMeta.duration,
    totalRunTime: executionMeta.total_run_time,
  } as FlowRun;
}

export default Monitor;
