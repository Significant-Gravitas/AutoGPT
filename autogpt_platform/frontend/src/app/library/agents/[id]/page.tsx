"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  GraphExecution,
  GraphExecutionMeta,
  GraphMeta,
  Schedule,
} from "@/lib/autogpt-server-api";

import AgentRunDraftView from "@/components/agents/agent-run-draft-view";
import AgentRunDetailsView from "@/components/agents/agent-run-details-view";
import AgentRunsSelectorList from "@/components/agents/agent-runs-selector-list";
import AgentScheduleDetailsView from "@/components/agents/agent-schedule-details-view";

export default function AgentRunsPage(): React.ReactElement {
  const { id: agentID }: { id: string } = useParams();
  const router = useRouter();
  const api = useBackendAPI();

  const [agent, setAgent] = useState<GraphMeta | null>(null);
  const [agentRuns, setAgentRuns] = useState<GraphExecutionMeta[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [selectedView, selectView] = useState<{
    type: "run" | "schedule";
    id?: string;
  }>({ type: "run" });
  const [selectedRun, setSelectedRun] = useState<
    GraphExecution | GraphExecutionMeta | null
  >(null);
  const [selectedSchedule, setSelectedSchedule] = useState<Schedule | null>(
    null,
  );
  const [isFirstLoad, setIsFirstLoad] = useState<boolean>(true);

  const openRunDraftView = useCallback(() => {
    selectView({ type: "run" });
  }, []);

  const selectRun = useCallback((id: string) => {
    selectView({ type: "run", id });
  }, []);

  const selectSchedule = useCallback((schedule: Schedule) => {
    selectView({ type: "schedule", id: schedule.id });
    setSelectedSchedule(schedule);
  }, []);

  const fetchAgents = useCallback(() => {
    api.getGraph(agentID).then(setAgent);
    api.getGraphExecutions(agentID).then((agentRuns) => {
      const sortedRuns = agentRuns.toSorted(
        (a, b) => b.started_at - a.started_at,
      );
      setAgentRuns(sortedRuns);

      if (!selectedView.id && isFirstLoad && sortedRuns.length > 0) {
        // only for first load or first execution
        setIsFirstLoad(false);
        selectView({ type: "run", id: sortedRuns[0].execution_id });
        setSelectedRun(sortedRuns[0]);
      }
    });
    if (selectedView.type == "run" && selectedView.id) {
      api.getGraphExecutionInfo(agentID, selectedView.id).then(setSelectedRun);
    }
  }, [api, agentID, selectedView, isFirstLoad]);

  useEffect(() => {
    fetchAgents();
  }, []);

  // load selectedRun based on selectedView
  useEffect(() => {
    if (selectedView.type != "run" || !selectedView.id) return;

    // pull partial data from "cache" while waiting for the rest to load
    if (selectedView.id !== selectedRun?.execution_id) {
      setSelectedRun(
        agentRuns.find((r) => r.execution_id == selectedView.id) ?? null,
      );
    }

    api.getGraphExecutionInfo(agentID, selectedView.id).then(setSelectedRun);
  }, [api, selectedView, agentRuns, agentID]);

  const fetchSchedules = useCallback(async () => {
    // TODO: filter in backend - https://github.com/Significant-Gravitas/AutoGPT/issues/9183
    setSchedules(
      (await api.listSchedules()).filter((s) => s.graph_id == agentID),
    );
  }, [api, agentID]);

  useEffect(() => {
    fetchSchedules();
  }, [fetchSchedules]);

  const removeSchedule = useCallback(
    async (scheduleId: string) => {
      const removedSchedule = await api.deleteSchedule(scheduleId);
      setSchedules(schedules.filter((s) => s.id !== removedSchedule.id));
    },
    [schedules, api],
  );

  /* TODO: use websockets instead of polling - https://github.com/Significant-Gravitas/AutoGPT/issues/8782 */
  useEffect(() => {
    const intervalId = setInterval(() => fetchAgents(), 5000);
    return () => clearInterval(intervalId);
  }, [fetchAgents, agent]);

  const agentActions: { label: string; callback: () => void }[] = useMemo(
    () => [
      {
        label: "Open in builder",
        callback: () => agent && router.push(`/build?flowID=${agent.id}`),
      },
    ],
    [agent, router],
  );

  if (!agent) {
    /* TODO: implement loading indicators / skeleton page */
    return <span>Loading...</span>;
  }

  return (
    <div className="container justify-stretch p-0 lg:flex">
      {/* Sidebar w/ list of runs */}
      {/* TODO: render this below header in sm and md layouts */}
      <AgentRunsSelectorList
        className="agpt-div w-full border-b lg:w-auto lg:border-b-0 lg:border-r"
        agent={agent}
        agentRuns={agentRuns}
        schedules={schedules}
        selectedView={selectedView}
        onSelectRun={selectRun}
        onSelectSchedule={selectSchedule}
        onDraftNewRun={openRunDraftView}
      />

      <div className="flex-1">
        {/* Header */}
        <div className="agpt-div w-full border-b">
          <h1 className="font-poppins text-3xl font-medium">
            {
              agent.name /* TODO: use dynamic/custom run title - https://github.com/Significant-Gravitas/AutoGPT/issues/9184 */
            }
          </h1>
        </div>

        {/* Run / Schedule views */}
        {(selectedView.type == "run" ? (
          selectedView.id ? (
            selectedRun && (
              <AgentRunDetailsView
                agent={agent}
                run={selectedRun}
                agentActions={agentActions}
              />
            )
          ) : (
            <AgentRunDraftView
              agent={agent}
              onRun={(runID) => selectRun(runID)}
              agentActions={agentActions}
            />
          )
        ) : selectedView.type == "schedule" ? (
          selectedSchedule && (
            <AgentScheduleDetailsView
              agent={agent}
              schedule={selectedSchedule}
              onForcedRun={(runID) => selectRun(runID)}
              agentActions={agentActions}
            />
          )
        ) : null) || <p>Loading...</p>}
      </div>
    </div>
  );
}
