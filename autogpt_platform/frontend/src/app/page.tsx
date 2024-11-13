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
  const [sortColumn, setSortColumn] = useState<keyof Schedule>("agentGraphId");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const fetchSchedules = useCallback(async () => {
    const schedulesData: Schedule[] = [];
    for (const flow of flows) {
      const flowSchedules = await api.getSchedules(flow.id);
      Object.entries(flowSchedules).forEach(([id, schedule]) => {
        schedulesData.push({
          id,
          schedule,
          isEnabled: true,
          graphId: flow.id,
          lastRun: new Date(
            Date.now() - Math.random() * 86400000,
          ).toISOString(),
          nextRun: new Date(
            Date.now() + Math.random() * 86400000,
          ).toISOString(),
          runCount: Math.floor(Math.random() * 100),
          status:
            Math.random() > 0.8
              ? "error"
              : Math.random() > 0.5
                ? "active"
                : "paused",
        });
      });
    }
    setSchedules(schedulesData);
  }, [api, flows]);

  const toggleSchedule = useCallback(
    async (scheduleId: string, enabled: boolean) => {
      await api.updateSchedule(scheduleId, { is_enabled: enabled });
      setSchedules((prevSchedules) =>
        prevSchedules.map((schedule) =>
          schedule.id === scheduleId
            ? { ...schedule, isEnabled: enabled }
            : schedule,
        ),
      );
    },
    [api],
  );

  const fetchAgents = useCallback(() => {
    api.listGraphsWithRuns().then((agent) => {
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
    fetchSchedules();
  }, [fetchSchedules, flows]);

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
      <div className={column2}>
        <FlowRunsList
          flows={flows}
          runs={[
            ...(selectedFlow
              ? flowRuns.filter((v) => v.graphID == selectedFlow.id)
              : flowRuns),
          ].sort((a, b) => Number(a.startTime) - Number(b.startTime))}
          selectedRun={selectedRun}
          onSelectRun={(r) =>
            setSelectedRun(r.id == selectedRun?.id ? null : r)
          }
        />
      </div>
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
      <div className="col-span-full md:col-span-3 lg:col-span-2 xl:col-span-6">
        <SchedulesTable
          schedules={schedules}
          onToggleSchedule={toggleSchedule}
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
