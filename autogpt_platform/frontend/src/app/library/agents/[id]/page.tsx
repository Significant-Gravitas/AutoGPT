"use client";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { exportAsJSONFile } from "@/lib/utils";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  GraphExecution,
  GraphExecutionID,
  GraphExecutionMeta,
  Graph,
  GraphID,
  LibraryAgent,
  LibraryAgentID,
  Schedule,
  ScheduleID,
} from "@/lib/autogpt-server-api";

import type { ButtonAction } from "@/components/agptui/types";
import DeleteConfirmDialog from "@/components/agptui/delete-confirm-dialog";
import AgentRunDraftView from "@/components/agents/agent-run-draft-view";
import AgentRunDetailsView from "@/components/agents/agent-run-details-view";
import AgentRunsSelectorList from "@/components/agents/agent-runs-selector-list";
import AgentScheduleDetailsView from "@/components/agents/agent-schedule-details-view";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";

export default function AgentRunsPage(): React.ReactElement {
  const { id: agentID }: { id: LibraryAgentID } = useParams();
  const router = useRouter();
  const api = useBackendAPI();

  // ============================ STATE =============================

  const [graph, setGraph] = useState<Graph | null>(null);
  const [agent, setAgent] = useState<LibraryAgent | null>(null);
  const [agentRuns, setAgentRuns] = useState<GraphExecutionMeta[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [selectedView, selectView] = useState<
    | { type: "run"; id?: GraphExecutionID }
    | { type: "schedule"; id: ScheduleID }
  >({ type: "run" });
  const [selectedRun, setSelectedRun] = useState<
    GraphExecution | GraphExecutionMeta | null
  >(null);
  const [selectedSchedule, setSelectedSchedule] = useState<Schedule | null>(
    null,
  );
  const [isFirstLoad, setIsFirstLoad] = useState<boolean>(true);
  const [agentDeleteDialogOpen, setAgentDeleteDialogOpen] =
    useState<boolean>(false);
  const [confirmingDeleteAgentRun, setConfirmingDeleteAgentRun] =
    useState<GraphExecutionMeta | null>(null);
  const { state, updateState } = useOnboarding();

  const openRunDraftView = useCallback(() => {
    selectView({ type: "run" });
  }, []);

  const selectRun = useCallback((id: GraphExecutionID) => {
    selectView({ type: "run", id });
  }, []);

  const selectSchedule = useCallback((schedule: Schedule) => {
    selectView({ type: "schedule", id: schedule.id });
    setSelectedSchedule(schedule);
  }, []);

  const [graphVersions, setGraphVersions] = useState<Record<number, Graph>>({});
  const getGraphVersion = useCallback(
    async (graphID: GraphID, version: number) => {
      if (graphVersions[version]) return graphVersions[version];

      const graphVersion = await api.getGraph(graphID, version);
      setGraphVersions((prev) => ({
        ...prev,
        [version]: graphVersion,
      }));
      return graphVersion;
    },
    [api, graphVersions],
  );

  // Reward user for viewing results of their onboarding agent
  useEffect(() => {
    if (!state || !selectedRun || state.completedSteps.includes("GET_RESULTS"))
      return;

    if (selectedRun.id === state.onboardingAgentExecutionId) {
      updateState({
        completedSteps: [...state.completedSteps, "GET_RESULTS"],
      });
    }
  }, [selectedRun, state]);

  const fetchAgents = useCallback(() => {
    api.getLibraryAgent(agentID).then((agent) => {
      setAgent(agent);

      getGraphVersion(agent.graph_id, agent.graph_version).then(
        (_graph) =>
          (graph && graph.version == _graph.version) || setGraph(_graph),
      );
      api.getGraphExecutions(agent.graph_id).then((agentRuns) => {
        setAgentRuns(agentRuns);

        // Preload the corresponding graph versions
        new Set(agentRuns.map((run) => run.graph_version)).forEach((version) =>
          getGraphVersion(agent.graph_id, version),
        );

        if (!selectedView.id && isFirstLoad && agentRuns.length > 0) {
          // only for first load or first execution
          setIsFirstLoad(false);

          const latestRun = agentRuns.reduce((latest, current) => {
            if (latest.started_at && !current.started_at) return current;
            else if (!latest.started_at) return latest;
            return latest.started_at > current.started_at ? latest : current;
          }, agentRuns[0]);
          selectView({ type: "run", id: latestRun.id });
        }
      });
    });
    if (selectedView.type == "run" && selectedView.id && agent) {
      api
        .getGraphExecutionInfo(agent.graph_id, selectedView.id)
        .then(setSelectedRun);
    }
  }, [api, agentID, getGraphVersion, graph, selectedView, isFirstLoad, agent]);

  useEffect(() => {
    fetchAgents();
  }, []);

  // Subscribe to websocket updates for agent runs
  useEffect(() => {
    if (!agent) return;

    // Subscribe to all executions for this agent
    api.subscribeToGraphExecutions(agent.graph_id);
  }, [api, agent]);

  // Handle execution updates
  useEffect(() => {
    const detachExecUpdateHandler = api.onWebSocketMessage(
      "graph_execution_event",
      (data) => {
        if (data.graph_id != agent?.graph_id) return;

        setAgentRuns((prev) => {
          const index = prev.findIndex((run) => run.id === data.id);
          if (index === -1) {
            return [...prev, data];
          }
          const newRuns = [...prev];
          newRuns[index] = { ...newRuns[index], ...data };
          return newRuns;
        });
        if (data.id === selectedView.id) {
          setSelectedRun((prev) => ({ ...prev, ...data }));
        }
      },
    );

    return () => {
      detachExecUpdateHandler();
    };
  }, [api, agent?.graph_id, selectedView.id]);

  // load selectedRun based on selectedView
  useEffect(() => {
    if (selectedView.type != "run" || !selectedView.id || !agent) return;

    const newSelectedRun = agentRuns.find((run) => run.id == selectedView.id);
    if (selectedView.id !== selectedRun?.id) {
      // Pull partial data from "cache" while waiting for the rest to load
      setSelectedRun(newSelectedRun ?? null);

      // Ensure corresponding graph version is available before rendering I/O
      api
        .getGraphExecutionInfo(agent.graph_id, selectedView.id)
        .then(async (run) => {
          await getGraphVersion(run.graph_id, run.graph_version);
          setSelectedRun(run);
        });
    }
  }, [api, selectedView, agent, agentRuns, selectedRun?.id, getGraphVersion]);

  const fetchSchedules = useCallback(async () => {
    if (!agent) return;

    // TODO: filter in backend - https://github.com/Significant-Gravitas/AutoGPT/issues/9183
    setSchedules(
      (await api.listSchedules()).filter((s) => s.graph_id == agent.graph_id),
    );
  }, [api, agent]);

  useEffect(() => {
    fetchSchedules();
  }, [fetchSchedules]);

  // =========================== ACTIONS ============================

  const deleteRun = useCallback(
    async (run: GraphExecutionMeta) => {
      if (run.status == "RUNNING" || run.status == "QUEUED") {
        await api.stopGraphExecution(run.graph_id, run.id);
      }
      await api.deleteGraphExecution(run.id);

      setConfirmingDeleteAgentRun(null);
      if (selectedView.type == "run" && selectedView.id == run.id) {
        openRunDraftView();
      }
      setAgentRuns(agentRuns.filter((r) => r.id !== run.id));
    },
    [agentRuns, api, selectedView, openRunDraftView],
  );

  const deleteSchedule = useCallback(
    async (scheduleID: ScheduleID) => {
      const removedSchedule = await api.deleteSchedule(scheduleID);
      setSchedules(schedules.filter((s) => s.id !== removedSchedule.id));
    },
    [schedules, api],
  );

  const downloadGraph = useCallback(
    async () =>
      agent &&
      // Export sanitized graph from backend
      api
        .getGraph(agent.graph_id, agent.graph_version, true)
        .then((graph) =>
          exportAsJSONFile(graph, `${graph.name}_v${graph.version}.json`),
        ),
    [api, agent],
  );

  const agentActions: ButtonAction[] = useMemo(
    () => [
      ...(agent?.can_access_graph
        ? [
            {
              label: "Open graph in builder",
              href: `/build?flowID=${agent.graph_id}&flowVersion=${agent.graph_version}`,
            },
            { label: "Export agent to file", callback: downloadGraph },
          ]
        : []),
      {
        label: "Delete agent",
        variant: "destructive",
        callback: () => setAgentDeleteDialogOpen(true),
      },
    ],
    [agent, downloadGraph],
  );

  if (!agent || !graph) {
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
        allowDraftNewRun={!graph.has_webhook_trigger}
        onSelectRun={selectRun}
        onSelectSchedule={selectSchedule}
        onSelectDraftNewRun={openRunDraftView}
        onDeleteRun={setConfirmingDeleteAgentRun}
        onDeleteSchedule={(id) => deleteSchedule(id)}
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
        {(selectedView.type == "run" && selectedView.id ? (
          selectedRun && (
            <AgentRunDetailsView
              agent={agent}
              graph={graphVersions[selectedRun.graph_version] ?? graph}
              run={selectedRun}
              agentActions={agentActions}
              onRun={(runID) => selectRun(runID)}
              deleteRun={() => setConfirmingDeleteAgentRun(selectedRun)}
            />
          )
        ) : selectedView.type == "run" ? (
          <AgentRunDraftView
            graph={graph}
            onRun={(runID) => selectRun(runID)}
            agentActions={agentActions}
          />
        ) : selectedView.type == "schedule" ? (
          selectedSchedule && (
            <AgentScheduleDetailsView
              graph={graph}
              schedule={selectedSchedule}
              onForcedRun={(runID) => selectRun(runID)}
              agentActions={agentActions}
            />
          )
        ) : null) || <p>Loading...</p>}

        <DeleteConfirmDialog
          entityType="agent"
          open={agentDeleteDialogOpen}
          onOpenChange={setAgentDeleteDialogOpen}
          onDoDelete={() =>
            agent &&
            api
              .updateLibraryAgent(agent.id, { is_deleted: true })
              .then(() => router.push("/library"))
          }
        />

        <DeleteConfirmDialog
          entityType="agent run"
          open={!!confirmingDeleteAgentRun}
          onOpenChange={(open) => !open && setConfirmingDeleteAgentRun(null)}
          onDoDelete={() =>
            confirmingDeleteAgentRun && deleteRun(confirmingDeleteAgentRun)
          }
        />
      </div>
    </div>
  );
}
