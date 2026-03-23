"use client";
import React, { useCallback, useState } from "react";

import { useQuery } from "@tanstack/react-query";
import { GraphExecutionMeta, LibraryAgent } from "@/lib/autogpt-server-api";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  useGetV1ListExecutionSchedulesForAUser,
  useDeleteV1DeleteExecutionSchedule,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

import { Card } from "@/components/__legacy__/ui/card";
import { SchedulesTable } from "@/app/(platform)/monitoring/components/SchedulesTable";
import AgentFlowList from "./components/AgentFlowList";
import FlowRunsList from "./components/FlowRunsList";
import FlowRunInfo from "./components/FlowRunInfo";
import FlowInfo from "./components/FlowInfo";
import FlowRunsStatus from "./components/FlowRunsStatus";

const POLL_INTERVAL_MS = 5_000;

export default function Monitor() {
  const [selectedFlow, setSelectedFlow] = useState<LibraryAgent | null>(null);
  const [selectedRun, setSelectedRun] = useState<GraphExecutionMeta | null>(
    null,
  );
  const [sortColumn, setSortColumn] =
    useState<keyof GraphExecutionJobInfo>("id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const api = useBackendAPI();

  // Agents — polled every 5 s via React Query's built-in refetchInterval.
  // React Query deduplicates concurrent requests, pauses on window blur, and
  // handles errors without crashing the component.
  const {
    data: agentsData,
    isLoading: agentsLoading,
    error: agentsError,
  } = useQuery({
    queryKey: ["monitoring", "agents"],
    queryFn: () => api.listLibraryAgents(),
    refetchInterval: POLL_INTERVAL_MS,
    retry: 2,
  });

  const {
    data: executionsData,
    isLoading: executionsLoading,
    error: executionsError,
  } = useQuery({
    queryKey: ["monitoring", "executions"],
    queryFn: () => api.getExecutions(),
    refetchInterval: POLL_INTERVAL_MS,
    retry: 2,
  });

  const flows: LibraryAgent[] = agentsData?.agents ?? [];
  const executions: GraphExecutionMeta[] = executionsData ?? [];

  // Use generated API hooks for schedules
  const { data: schedulesResponse, refetch: refetchSchedules } =
    useGetV1ListExecutionSchedulesForAUser();
  const deleteScheduleMutation = useDeleteV1DeleteExecutionSchedule();

  const schedules = okData(schedulesResponse) ?? [];

  const removeSchedule = useCallback(
    async (scheduleId: string) => {
      await deleteScheduleMutation.mutateAsync({ scheduleId });
      refetchSchedules();
    },
    [deleteScheduleMutation, refetchSchedules],
  );

  const column1 = "md:col-span-2 xl:col-span-3 xxl:col-span-2";
  const column2 = "md:col-span-3 lg:col-span-2 xl:col-span-3";
  const column3 = "col-span-full xl:col-span-4 xxl:col-span-5";

  function handleSort(column: keyof GraphExecutionJobInfo) {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  }

  const isLoading = agentsLoading || executionsLoading;
  const fetchError = agentsError ?? executionsError;

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
        isLoading={isLoading}
        error={fetchError}
      />
      <FlowRunsList
        className={column2}
        flows={flows}
        executions={[
          ...(selectedFlow
            ? executions.filter((v) => v.graph_id == selectedFlow.graph_id)
            : executions),
        ].sort((a, b) => {
          const aTime = a.started_at?.getTime() ?? 0;
          const bTime = b.started_at?.getTime() ?? 0;
          return bTime - aTime;
        })}
        selectedRun={selectedRun}
        onSelectRun={(r) => setSelectedRun(r.id == selectedRun?.id ? null : r)}
        isLoading={isLoading}
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
          schedules={schedules}
          agents={flows}
          onRemoveSchedule={removeSchedule}
          sortColumn={sortColumn}
          sortDirection={sortDirection}
          onSort={handleSort}
        />
      </div>
    </div>
  );
}
