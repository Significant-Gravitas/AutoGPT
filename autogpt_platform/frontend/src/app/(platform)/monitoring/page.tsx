"use client";
import React, { useCallback, useEffect, useState } from "react";

import { GraphExecutionMeta, LibraryAgent } from "@/lib/autogpt-server-api";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  useGetV1ListExecutionSchedulesForAUser,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";

import { Card } from "@/components/ui/card";
import { SchedulesTable } from "@/app/(platform)/monitoring/components/SchedulesTable";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import AgentFlowList from "./components/AgentFlowList";
import FlowRunsList from "./components/FlowRunsList";
import FlowRunInfo from "./components/FlowRunInfo";
import FlowInfo from "./components/FlowInfo";
import FlowRunsStatus from "./components/FlowRunsStatus";

const Monitor = () => {
  const [flows, setFlows] = useState<LibraryAgent[]>([]);
  const [executions, setExecutions] = useState<GraphExecutionMeta[]>([]);
  const [selectedFlow, setSelectedFlow] = useState<LibraryAgent | null>(null);
  const [selectedRun, setSelectedRun] = useState<GraphExecutionMeta | null>(
    null,
  );
  const [sortColumn, setSortColumn] =
    useState<keyof GraphExecutionJobInfo>("id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const api = useBackendAPI();

  // Use generated API hooks for schedules
  const { data: schedulesResponse, refetch: refetchSchedules } =
    useGetV1ListExecutionSchedulesForAUser();
  const deleteScheduleMutation = useDeleteV1DeleteExecutionSchedule();

  const schedules =
    schedulesResponse?.status === 200 ? schedulesResponse.data : [];

  const removeSchedule = useCallback(
    async (scheduleId: string) => {
      await deleteScheduleMutation.mutateAsync({ scheduleId });
      refetchSchedules();
    },
    [deleteScheduleMutation, refetchSchedules],
  );

  const fetchAgents = useCallback(() => {
    api.listLibraryAgents().then((response) => {
      setFlows(response.agents);
    });
    api.getExecutions().then((executions) => {
      setExecutions(executions);
    });
  }, [api]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  useEffect(() => {
    const intervalId = setInterval(() => fetchAgents(), 5000);
    return () => clearInterval(intervalId);
  }, [fetchAgents, flows]);

  const column1 = "md:col-span-2 xl:col-span-3 xxl:col-span-2";
  const column2 = "md:col-span-3 lg:col-span-2 xl:col-span-3";
  const column3 = "col-span-full xl:col-span-4 xxl:col-span-5";

  const handleSort = (column: keyof GraphExecutionJobInfo) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  return (
    <div
      className="grid grid-cols-1 gap-4 p-4 md:grid-cols-5 lg:grid-cols-4 xl:grid-cols-10"
      data-testid="monitor-page"
    >
      <AgentFlowList
        className={column1}
        flows={flows}
        executions={executions}
        selectedFlow={selectedFlow}
        onSelectFlow={(f) => {
          setSelectedRun(null);
          setSelectedFlow(f.id == selectedFlow?.id ? null : f);
        }}
      />
      <FlowRunsList
        className={column2}
        flows={flows}
        executions={[
          ...(selectedFlow
            ? executions.filter((v) => v.graph_id == selectedFlow.graph_id)
            : executions),
        ].sort((a, b) => b.started_at.getTime() - a.started_at.getTime())}
        selectedRun={selectedRun}
        onSelectRun={(r) => setSelectedRun(r.id == selectedRun?.id ? null : r)}
      />
      {(selectedRun && (
        <FlowRunInfo
          agent={
            selectedFlow ||
            flows.find((f) => f.graph_id == selectedRun.graph_id)!
          }
          execution={selectedRun}
          className={column3}
        />
      )) ||
        (selectedFlow && (
          <FlowInfo
            flow={selectedFlow}
            executions={executions.filter(
              (e) => e.graph_id == selectedFlow.graph_id,
            )}
            className={column3}
            refresh={() => {
              fetchAgents();
              setSelectedFlow(null);
              setSelectedRun(null);
            }}
          />
        )) || (
          <Card className={`p-6 ${column3}`}>
            <FlowRunsStatus flows={flows} executions={executions} />
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

export default Monitor;
