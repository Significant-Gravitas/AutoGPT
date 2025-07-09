"use client";
import { useParams, useRouter } from "next/navigation";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Graph,
  GraphExecution,
  GraphExecutionID,
  GraphExecutionMeta,
  GraphID,
  LibraryAgent,
  LibraryAgentID,
  LibraryAgentPreset,
  LibraryAgentPresetID,
  Schedule,
  ScheduleID,
} from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { exportAsJSONFile } from "@/lib/utils";

import AgentRunDetailsView from "@/components/agents/agent-run-details-view";
import AgentRunDraftView from "@/components/agents/agent-run-draft-view";
import AgentRunsSelectorList from "@/components/agents/agent-runs-selector-list";
import AgentScheduleDetailsView from "@/components/agents/agent-schedule-details-view";
import DeleteConfirmDialog from "@/components/agptui/delete-confirm-dialog";
import type { ButtonAction } from "@/components/agptui/types";
import { useOnboarding } from "@/components/onboarding/onboarding-provider";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import LoadingBox, { LoadingSpinner } from "@/components/ui/loading";
import { useToast } from "@/components/molecules/Toast/use-toast";

export default function AgentRunsPage(): React.ReactElement {
  const { id: agentID }: { id: LibraryAgentID } = useParams();
  const { toast } = useToast();
  const router = useRouter();
  const api = useBackendAPI();

  // ============================ STATE =============================

  const [graph, setGraph] = useState<Graph | null>(null); // Graph version corresponding to LibraryAgent
  const [agent, setAgent] = useState<LibraryAgent | null>(null);
  const [agentRuns, setAgentRuns] = useState<GraphExecutionMeta[]>([]);
  const [agentPresets, setAgentPresets] = useState<LibraryAgentPreset[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [selectedView, selectView] = useState<
    | { type: "run"; id?: GraphExecutionID }
    | { type: "preset"; id: LibraryAgentPresetID }
    | { type: "schedule"; id: ScheduleID }
  >({ type: "run" });
  const [selectedRun, setSelectedRun] = useState<
    GraphExecution | GraphExecutionMeta | null
  >(null);
  const selectedSchedule =
    selectedView.type == "schedule"
      ? schedules.find((s) => s.id == selectedView.id)
      : null;
  const [isFirstLoad, setIsFirstLoad] = useState<boolean>(true);
  const [agentDeleteDialogOpen, setAgentDeleteDialogOpen] =
    useState<boolean>(false);
  const [confirmingDeleteAgentRun, setConfirmingDeleteAgentRun] =
    useState<GraphExecutionMeta | null>(null);
  const [confirmingDeleteAgentPreset, setConfirmingDeleteAgentPreset] =
    useState<LibraryAgentPresetID | null>(null);
  const {
    state: onboardingState,
    updateState: updateOnboardingState,
    incrementRuns,
  } = useOnboarding();
  const [copyAgentDialogOpen, setCopyAgentDialogOpen] = useState(false);

  // Set page title with agent name
  useEffect(() => {
    if (agent) {
      document.title = `${agent.name} - Library - AutoGPT Platform`;
    }
  }, [agent]);

  const openRunDraftView = useCallback(() => {
    selectView({ type: "run" });
  }, []);

  const selectRun = useCallback((id: GraphExecutionID) => {
    selectView({ type: "run", id });
  }, []);

  const selectPreset = useCallback((id: LibraryAgentPresetID) => {
    selectView({ type: "preset", id });
  }, []);

  const selectSchedule = useCallback((id: ScheduleID) => {
    selectView({ type: "schedule", id });
  }, []);

  const graphVersions = useRef<Record<number, Graph>>({});
  const loadingGraphVersions = useRef<Record<number, Promise<Graph>>>({});
  const getGraphVersion = useCallback(
    async (graphID: GraphID, version: number) => {
      if (version in graphVersions.current)
        return graphVersions.current[version];
      if (version in loadingGraphVersions.current)
        return loadingGraphVersions.current[version];

      const pendingGraph = api.getGraph(graphID, version).then((graph) => {
        graphVersions.current[version] = graph;
        return graph;
      });
      // Cache promise as well to avoid duplicate requests
      loadingGraphVersions.current[version] = pendingGraph;
      return pendingGraph;
    },
    [api, graphVersions, loadingGraphVersions],
  );

  // Reward user for viewing results of their onboarding agent
  useEffect(() => {
    if (
      !onboardingState ||
      !selectedRun ||
      onboardingState.completedSteps.includes("GET_RESULTS")
    )
      return;

    if (selectedRun.id === onboardingState.onboardingAgentExecutionId) {
      updateOnboardingState({
        completedSteps: [...onboardingState.completedSteps, "GET_RESULTS"],
      });
    }
  }, [selectedRun, onboardingState, updateOnboardingState]);

  const lastRefresh = useRef<number>(0);
  const refreshPageData = useCallback(() => {
    if (Date.now() - lastRefresh.current < 2e3) return; // 2 second debounce
    lastRefresh.current = Date.now();

    api.getLibraryAgent(agentID).then((agent) => {
      setAgent(agent);

      getGraphVersion(agent.graph_id, agent.graph_version).then(
        (_graph) =>
          (graph && graph.version == _graph.version) || setGraph(_graph),
      );
      Promise.all([
        api.getGraphExecutions(agent.graph_id),
        api.listLibraryAgentPresets({
          graph_id: agent.graph_id,
          page_size: 100,
        }),
      ]).then(([runs, presets]) => {
        setAgentRuns(runs);
        setAgentPresets(presets.presets);

        // Preload the corresponding graph versions for the latest 10 runs
        new Set(runs.slice(0, 10).map((run) => run.graph_version)).forEach(
          (version) => getGraphVersion(agent.graph_id, version),
        );
      });
    });
  }, [api, agentID, getGraphVersion, graph]);

  // On first load: select the latest run
  useEffect(() => {
    // Only for first load or first execution
    if (selectedView.id || !isFirstLoad) return;
    if (agentRuns.length == 0 && agentPresets.length == 0) return;

    setIsFirstLoad(false);
    if (agentRuns.length > 0) {
      // select latest run
      const latestRun = agentRuns.reduce((latest, current) => {
        if (latest.started_at && !current.started_at) return current;
        else if (!latest.started_at) return latest;
        return latest.started_at > current.started_at ? latest : current;
      }, agentRuns[0]);
      selectRun(latestRun.id);
    } else {
      // select top preset
      const latestPreset = agentPresets.toSorted(
        (a, b) => b.updated_at.getTime() - a.updated_at.getTime(),
      )[0];
      selectPreset(latestPreset.id);
    }
  }, [
    isFirstLoad,
    selectedView.id,
    agentRuns,
    agentPresets,
    selectRun,
    selectPreset,
  ]);

  // Initial load
  useEffect(() => {
    refreshPageData();

    // Show a toast when the WebSocket connection disconnects
    let connectionToast: ReturnType<typeof toast> | null = null;
    const cancelDisconnectHandler = api.onWebSocketDisconnect(() => {
      connectionToast ??= toast({
        title: "Connection to server was lost",
        variant: "destructive",
        description: (
          <div className="flex items-center">
            Trying to reconnect...
            <LoadingSpinner className="ml-1.5 size-3.5" />
          </div>
        ),
        duration: Infinity, // show until connection is re-established
        dismissable: false,
      });
    });
    const cancelConnectHandler = api.onWebSocketConnect(() => {
      if (connectionToast)
        connectionToast.update({
          id: connectionToast.id,
          title: "✅ Connection re-established",
          variant: "default",
          description: (
            <div className="flex items-center">
              Refreshing data...
              <LoadingSpinner className="ml-1.5 size-3.5" />
            </div>
          ),
          duration: 2000,
          dismissable: true,
        });
      connectionToast = null;
    });
    return () => {
      cancelDisconnectHandler();
      cancelConnectHandler();
    };
  }, []);

  // Subscribe to WebSocket updates for agent runs
  useEffect(() => {
    if (!agent?.graph_id) return;

    return api.onWebSocketConnect(() => {
      refreshPageData(); // Sync up on (re)connect

      // Subscribe to all executions for this agent
      api.subscribeToGraphExecutions(agent.graph_id);
    });
  }, [api, agent?.graph_id, refreshPageData]);

  // Handle execution updates
  useEffect(() => {
    const detachExecUpdateHandler = api.onWebSocketMessage(
      "graph_execution_event",
      (data) => {
        if (data.graph_id != agent?.graph_id) return;

        if (data.status == "COMPLETED") {
          incrementRuns();
        }

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
  }, [api, agent?.graph_id, selectedView.id, incrementRuns]);

  // Pre-load selectedRun based on selectedView
  useEffect(() => {
    if (selectedView.type != "run" || !selectedView.id) return;

    const newSelectedRun = agentRuns.find((run) => run.id == selectedView.id);
    if (selectedView.id !== selectedRun?.id) {
      // Pull partial data from "cache" while waiting for the rest to load
      setSelectedRun(newSelectedRun ?? null);
    }
  }, [api, selectedView, agentRuns, selectedRun?.id]);

  // Load selectedRun based on selectedView; refresh on agent refresh
  useEffect(() => {
    if (selectedView.type != "run" || !selectedView.id || !agent) return;

    api
      .getGraphExecutionInfo(agent.graph_id, selectedView.id)
      .then(async (run) => {
        // Ensure corresponding graph version is available before rendering I/O
        await getGraphVersion(run.graph_id, run.graph_version);
        setSelectedRun(run);
      });
  }, [api, selectedView, agent, getGraphVersion]);

  const fetchSchedules = useCallback(async () => {
    if (!agent) return;

    setSchedules(await api.listGraphExecutionSchedules(agent.graph_id));
  }, [api, agent?.graph_id]);

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
      setAgentRuns((runs) => runs.filter((r) => r.id !== run.id));
    },
    [api, selectedView, openRunDraftView],
  );

  const deletePreset = useCallback(
    async (presetID: LibraryAgentPresetID) => {
      await api.deleteLibraryAgentPreset(presetID);

      setConfirmingDeleteAgentPreset(null);
      if (selectedView.type == "preset" && selectedView.id == presetID) {
        openRunDraftView();
      }
      setAgentPresets((presets) => presets.filter((p) => p.id !== presetID));
    },
    [api, selectedView, openRunDraftView],
  );

  const deleteSchedule = useCallback(
    async (scheduleID: ScheduleID) => {
      const removedSchedule =
        await api.deleteGraphExecutionSchedule(scheduleID);

      setSchedules((schedules) => {
        const newSchedules = schedules.filter(
          (s) => s.id !== removedSchedule.id,
        );
        if (
          selectedView.type == "schedule" &&
          selectedView.id == removedSchedule.id
        ) {
          if (newSchedules.length > 0) {
            // Select next schedule if available
            selectSchedule(newSchedules[0].id);
          } else {
            // Reset to draft view if current schedule was deleted
            openRunDraftView();
          }
        }
        return newSchedules;
      });
      openRunDraftView();
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

  const copyAgent = useCallback(async () => {
    setCopyAgentDialogOpen(false);
    api
      .forkLibraryAgent(agentID)
      .then((newAgent) => {
        router.push(`/library/agents/${newAgent.id}`);
      })
      .catch((error) => {
        console.error("Error copying agent:", error);
        toast({
          title: "Error copying agent",
          description: `An error occurred while copying the agent: ${error.message}`,
          variant: "destructive",
        });
      });
  }, [agentID, api, router, toast]);

  const agentActions: ButtonAction[] = useMemo(
    () => [
      {
        label: "Customize agent",
        href: `/build?flowID=${agent?.graph_id}&flowVersion=${agent?.graph_version}`,
        disabled: !agent?.can_access_graph,
      },
      { label: "Export agent to file", callback: downloadGraph },
      ...(!agent?.can_access_graph
        ? [
            {
              label: "Edit a copy",
              callback: () => setCopyAgentDialogOpen(true),
            },
          ]
        : []),
      {
        label: "Delete agent",
        callback: () => setAgentDeleteDialogOpen(true),
      },
    ],
    [agent, downloadGraph],
  );

  const runGraph =
    graphVersions.current[selectedRun?.graph_version ?? 0] ?? graph;

  const onCreateSchedule = useCallback(
    (schedule: Schedule) => {
      setSchedules((prev) => [...prev, schedule]);
      selectSchedule(schedule.id);
    },
    [selectView],
  );

  const onCreatePreset = useCallback(
    (preset: LibraryAgentPreset) => {
      setAgentPresets((prev) => [...prev, preset]);
      selectPreset(preset.id);
    },
    [selectPreset],
  );

  const onUpdatePreset = useCallback(
    (updated: LibraryAgentPreset) => {
      setAgentPresets((prev) =>
        prev.map((p) => (p.id === updated.id ? updated : p)),
      );
      selectPreset(updated.id);
    },
    [selectPreset],
  );

  if (!agent || !graph) {
    return <LoadingBox className="h-[90vh]" />;
  }

  return (
    <div className="container justify-stretch p-0 lg:flex">
      {/* Sidebar w/ list of runs */}
      {/* TODO: render this below header in sm and md layouts */}
      <AgentRunsSelectorList
        className="agpt-div w-full border-b lg:w-auto lg:border-b-0 lg:border-r"
        agent={agent}
        agentRuns={agentRuns}
        agentPresets={agentPresets}
        schedules={schedules}
        selectedView={selectedView}
        onSelectRun={selectRun}
        onSelectPreset={selectPreset}
        onSelectSchedule={selectSchedule}
        onSelectDraftNewRun={openRunDraftView}
        doDeleteRun={setConfirmingDeleteAgentRun}
        doDeletePreset={setConfirmingDeleteAgentPreset}
        doDeleteSchedule={deleteSchedule}
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
          selectedRun && runGraph ? (
            <AgentRunDetailsView
              agent={agent}
              graph={runGraph}
              run={selectedRun}
              agentActions={agentActions}
              onRun={selectRun}
              deleteRun={() => setConfirmingDeleteAgentRun(selectedRun)}
            />
          ) : null
        ) : selectedView.type == "run" ? (
          /* Draft new runs / Create new presets */
          <AgentRunDraftView
            graph={graph}
            triggerSetupInfo={agent.trigger_setup_info}
            onRun={selectRun}
            onCreateSchedule={onCreateSchedule}
            onCreatePreset={onCreatePreset}
            agentActions={agentActions}
          />
        ) : selectedView.type == "preset" ? (
          /* Edit & update presets */
          <AgentRunDraftView
            graph={graph}
            triggerSetupInfo={agent.trigger_setup_info}
            agentPreset={
              agentPresets.find((preset) => preset.id == selectedView.id)!
            }
            onRun={selectRun}
            onCreateSchedule={onCreateSchedule}
            onUpdatePreset={onUpdatePreset}
            doDeletePreset={setConfirmingDeleteAgentPreset}
            agentActions={agentActions}
          />
        ) : selectedView.type == "schedule" ? (
          selectedSchedule &&
          graph && (
            <AgentScheduleDetailsView
              graph={graph}
              schedule={selectedSchedule}
              // agent={agent}
              agentActions={agentActions}
              onForcedRun={selectRun}
              doDeleteSchedule={deleteSchedule}
            />
          )
        ) : null) || <LoadingBox className="h-[70vh]" />}

        <DeleteConfirmDialog
          entityType="agent"
          open={agentDeleteDialogOpen}
          onOpenChange={setAgentDeleteDialogOpen}
          onDoDelete={() =>
            agent &&
            api.deleteLibraryAgent(agent.id).then(() => router.push("/library"))
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
        <DeleteConfirmDialog
          entityType={agent.has_external_trigger ? "trigger" : "agent preset"}
          open={!!confirmingDeleteAgentPreset}
          onOpenChange={(open) => !open && setConfirmingDeleteAgentPreset(null)}
          onDoDelete={() =>
            confirmingDeleteAgentPreset &&
            deletePreset(confirmingDeleteAgentPreset)
          }
        />
        {/* Copy agent confirmation dialog */}
        <Dialog
          onOpenChange={setCopyAgentDialogOpen}
          open={copyAgentDialogOpen}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>You&apos;re making an editable copy</DialogTitle>
              <DialogDescription className="pt-2">
                The original Marketplace agent stays the same and cannot be
                edited. We&apos;ll save a new version of this agent to your
                Library. From there, you can customize it however you&apos;d
                like by clicking &quot;Customize agent&quot; — this will open
                the builder where you can see and modify the inner workings.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter className="justify-end">
              <Button
                type="button"
                variant="outline"
                onClick={() => setCopyAgentDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button type="button" onClick={copyAgent}>
                Continue
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
