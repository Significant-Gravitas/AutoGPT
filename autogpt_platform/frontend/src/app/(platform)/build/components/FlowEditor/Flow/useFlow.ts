import { useCallback, useEffect, useMemo, useState } from "react";
import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import {
  useGetV1GetExecutionDetails,
  useGetV1GetSpecificGraph,
  useGetV1ListUserGraphs,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useNodeStore } from "../../../stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { convertNodesPlusBlockInfoIntoCustomNodes } from "../../helper";
import { useEdgeStore } from "../../../stores/edgeStore";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import { useGraphStore } from "../../../stores/graphStore";
import { useReactFlow } from "@xyflow/react";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useHistoryStore } from "../../../stores/historyStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { useIsReadOnlyGraph } from "../../../hooks/useIsReadOnlyGraph";
import {
  getGraphLoadErrorToast,
  retryUnlessClientError,
} from "../../../helpers/graphLoadError";
import {
  showsAnotherSavedGraph,
  showsSavedGraph,
} from "../../../helpers/waitForLoadedGraph";
import { useIsExecutionSubscribed } from "./useFlowRealtime";
import {
  isSnapshotStatusCurrent,
  orderSnapshotExecutions,
} from "./executionUpdateOrder";

const POLL_INTERVAL_MS = 1_000;
// While subscribed, a slow poll still catches an event the backend failed to
// deliver (e.g. a dropped pub/sub message), as the 1 s poll used to.
const RECONCILE_INTERVAL_MS = 10_000;

export const useFlow = () => {
  const { isReadOnly } = useIsReadOnlyGraph();
  const { toast } = useToast();
  const router = useRouter();
  const [isLockedState, setIsLockedState] = useState(false);
  const [hasAutoFramed, setHasAutoFramed] = useState(false);
  const [isInitialLoadComplete, setIsInitialLoadComplete] = useState(false);
  const addNodes = useNodeStore(useShallow((state) => state.addNodes));
  const addLinks = useEdgeStore(useShallow((state) => state.addLinks));
  const applyNodeExecutionEvent = useNodeStore(
    useShallow((state) => state.applyNodeExecutionEvent),
  );
  const setGraphSchemas = useGraphStore(
    useShallow((state) => state.setGraphSchemas),
  );
  const setGraphExecutionStatus = useGraphStore(
    useShallow((state) => state.setGraphExecutionStatus),
  );
  const setAvailableSubGraphs = useGraphStore(
    useShallow((state) => state.setAvailableSubGraphs),
  );
  const setLoadedGraph = useGraphStore((state) => state.setLoadedGraph);
  const requestLoad = useGraphStore((state) => state.requestLoad);
  const resolveLatestVersion = useGraphStore(
    (state) => state.resolveLatestVersion,
  );
  const setLoadFailed = useGraphStore((state) => state.setLoadFailed);
  const { screenToFlowPosition, fitView } = useReactFlow();
  const addBlock = useNodeStore(useShallow((state) => state.addBlock));
  const setBlockMenuOpen = useControlPanelStore(
    useShallow((state) => state.setBlockMenuOpen),
  );
  const [{ flowID, flowVersion, flowExecutionID }, setQueryStates] =
    useQueryStates({
      flowID: parseAsString,
      flowVersion: parseAsInteger,
      flowExecutionID: parseAsString,
    });

  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );
  const isExecutionSubscribed = useIsExecutionSubscribed(flowExecutionID);

  const { data: executionDetails, dataUpdatedAt: executionDetailsUpdatedAt } =
    useGetV1GetExecutionDetails(flowID || "", flowExecutionID || "", {
      query: {
        select: (res) => res.data as GraphExecution | GraphExecutionWithNodes,
        enabled: !!flowID && !!flowExecutionID,
        // Poll every second while the graph is running until the server
        // acknowledges the WebSocket subscription (race condition on fast
        // executions like dry-runs); useFlowRealtime refetches once on that
        // acknowledgement. Then only reconcile slowly, every second again if
        // the WebSocket drops, and stop once the execution reaches a terminal
        // state and isGraphRunning becomes false.
        refetchInterval: isGraphRunning
          ? isExecutionSubscribed
            ? RECONCILE_INTERVAL_MS
            : POLL_INTERVAL_MS
          : false,
      },
    });

  // Fetch all available graphs for sub-agent update detection
  const { data: availableGraphs } = useGetV1ListUserGraphs({
    query: { select: okData },
  });

  const {
    data: graph,
    isLoading: isGraphLoading,
    isError: isGraphError,
    error: graphError,
  } = useGetV1GetSpecificGraph(
    flowID ?? "",
    flowVersion !== null ? { version: flowVersion } : {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID,
        retry: retryUnlessClientError,
      },
    },
  );

  // An error with no graph to show. A refetch that fails (window focus, a
  // network blip) keeps the graph it had, and the builder keeps working.
  const graphLoadFailed = isGraphError && !graph;

  // The graph couldn't load — bail out of the builder instead of falling into
  // the fully-editable "new agent" state, which would look like a wiped agent
  // and risk an empty save over the real flowID.
  useEffect(() => {
    if (!flowID || !graphLoadFailed) return;
    const { title, description } = getGraphLoadErrorToast(graphError);
    toast({ title, description, variant: "destructive" });
    router.push("/library");
  }, [flowID, graphLoadFailed, graphError, toast, router]);

  const nodes = graph?.nodes;
  const blockIds = nodes
    ? Array.from(new Set(nodes.map((node) => node.block_id)))
    : undefined;

  const {
    data: blocks,
    isLoading: isBlocksLoading,
    isError: isBlocksError,
  } = useGetV2GetSpecificBlocks(
    { block_ids: blockIds ?? [] },
    {
      query: {
        select: (res) => res.data as BlockInfo[],
        enabled: !!flowID && !!blockIds && blockIds.length > 0,
      },
    },
  );
  const blocksLoadFailed = isBlocksError && !blocks;

  const customNodes = useMemo(() => {
    if (!nodes || !blocks) return [];

    return nodes.map((node) => {
      const customNode = convertNodesPlusBlockInfoIntoCustomNodes(
        node,
        blocks?.find((block) => block.id === node.block_id) ??
          ({} as BlockInfo),
      );
      return customNode;
    });
  }, [nodes, blocks]);

  // load graph schemas
  useEffect(() => {
    if (graph) {
      setQueryStates({
        flowVersion: graph.version ?? 1,
      });
      setGraphSchemas(
        graph.input_schema as Record<string, any> | null,
        graph.credentials_input_schema as Record<string, any> | null,
        graph.output_schema as Record<string, any> | null,
      );
    }
  }, [graph]);

  // Update available sub-graphs in store for sub-agent update detection
  useEffect(() => {
    if (availableGraphs) {
      setAvailableSubGraphs(availableGraphs);
    }
  }, [availableGraphs, setAvailableSubGraphs]);

  // A save waiting for the canvas gives up when the builder moves to another
  // workflow or version, or closes (the reset below). Runs before the fill so
  // the loaded graph is recorded against the current request.
  useEffect(() => {
    requestLoad(flowID, flowVersion);
  }, [flowID, flowVersion, requestLoad]);

  // Records the version a URL without one resolved to, which the schemas
  // effect above writes into the URL. That write keeps a waiting save; picking
  // any other version cancels it.
  useEffect(() => {
    if (graph?.id && flowVersion === null) {
      resolveLatestVersion(graph.id, graph.version ?? 1);
    }
  }, [graph, flowVersion, resolveLatestVersion]);

  // Nodes and links are replaced together, once the graph's block info is in,
  // so the canvas never pairs one graph's links with another's nodes. Only a
  // workflow or version the canvas isn't showing replaces it: fetching the one
  // it shows again (window focus, an invalidation) keeps the user's edits.
  useEffect(() => {
    if (customNodes.length === 0 || !graph || showsSavedGraph(graph)) return;
    useNodeStore.getState().setNodes([]);
    useNodeStore.getState().clearResolutionState();
    addNodes(customNodes);
    useEdgeStore.getState().setEdges([]);
    addLinks(graph.links ?? []);
  }, [customNodes, graph, addNodes, addLinks]);

  // Saves wait for this (waitForLoadedGraph), so it runs after the fill. A
  // graph with no nodes has nothing to wait for. When the canvas shows another
  // saved graph, it is emptied first, undo included, so a save can't copy that
  // graph's blocks into this one; an unsaved canvas keeps its blocks for it.
  // It reruns when only the URL's version changes too: React Query can hand
  // back an equal graph as the same object, and the load is then recorded
  // against the new request (a URL without a version showing its latest).
  useEffect(() => {
    if (!graph || (customNodes.length === 0 && graph.nodes?.length)) return;
    if (!graph.nodes?.length && showsAnotherSavedGraph(graph)) {
      useNodeStore.getState().setNodes([]);
      useNodeStore.getState().clearResolutionState();
      useEdgeStore.getState().setEdges([]);
      useHistoryStore.getState().initializeHistory();
    }
    setLoadedGraph(graph);
  }, [graph, customNodes, flowVersion, setLoadedGraph]);

  // A save waiting for the canvas fails at once if the workflow or its blocks
  // can't load. A refetch that fails keeps the data it had, so fails no save.
  useEffect(() => {
    setLoadFailed(graphLoadFailed || blocksLoadFailed);
  }, [graphLoadFailed, blocksLoadFailed, setLoadFailed]);

  useEffect(() => {
    if (customNodes.length > 0 && graph?.links) {
      customNodes.forEach((node) => {
        useNodeStore.getState().syncHardcodedValuesWithHandleIds(node.id);
      });
    }
  }, [customNodes, graph?.links]);

  // update node statuses, execution results and edge beads. Executions the
  // store already has are no-ops, so a poll that changed nothing writes
  // nothing. Re-runs when the nodes are rebuilt (customNodes), which drops them.
  // Websocket deliveries that arrived after this snapshot's fetch started may
  // be newer than it, so they replace the snapshot's copies and go last.
  useEffect(() => {
    if (
      executionDetails &&
      "node_executions" in executionDetails &&
      executionDetails.node_executions
    ) {
      orderSnapshotExecutions(
        executionDetails.node_executions,
        executionDetailsUpdatedAt,
      ).forEach((nodeExecution) => {
        applyNodeExecutionEvent(nodeExecution.node_id, nodeExecution);
      });
    }
  }, [executionDetails, applyNodeExecutionEvent, customNodes]);

  // update graph execution status, unless the websocket sent one after this
  // snapshot's fetch started
  useEffect(() => {
    if (
      executionDetails &&
      isSnapshotStatusCurrent(executionDetailsUpdatedAt)
    ) {
      setGraphExecutionStatus(executionDetails.status as AgentExecutionStatus);
    }
  }, [executionDetails]);

  useEffect(() => {
    if (customNodes.length > 0 && graph?.links) {
      const timer = setTimeout(() => {
        useHistoryStore.getState().initializeHistory();
        // Mark initial load as complete after history is initialized
        setIsInitialLoadComplete(true);
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [customNodes, graph?.links]);

  // Also mark as complete for new flows (no flowID) after a short delay
  useEffect(() => {
    if (!flowID && !isGraphLoading && !isBlocksLoading) {
      const timer = setTimeout(() => {
        setIsInitialLoadComplete(true);
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [flowID, isGraphLoading, isBlocksLoading]);

  useEffect(() => {
    return () => {
      useNodeStore.getState().setNodes([]);
      useNodeStore.getState().clearResolutionState();
      useEdgeStore.getState().setEdges([]);
      useGraphStore.getState().reset();
      useEdgeStore.getState().resetEdgeBeads();
    };
  }, []);

  const linkCount = graph?.links?.length ?? 0;

  useEffect(() => {
    if (isGraphLoading || isBlocksLoading) {
      setHasAutoFramed(false);
      return;
    }

    if (hasAutoFramed) {
      return;
    }

    const rafId = requestAnimationFrame(() => {
      fitView({ padding: 0.2, duration: 800, maxZoom: 1 });
      setHasAutoFramed(true);
    });

    return () => cancelAnimationFrame(rafId);
  }, [
    fitView,
    hasAutoFramed,
    customNodes.length,
    isBlocksLoading,
    isGraphLoading,
    linkCount,
  ]);

  useEffect(() => {
    setHasAutoFramed(false);
    setIsInitialLoadComplete(false);
  }, [flowID, flowVersion]);

  // Drag and drop block from block menu
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const onDrop = useCallback(
    async (event: React.DragEvent) => {
      event.preventDefault();
      const blockDataString = event.dataTransfer.getData(
        "application/reactflow",
      );
      if (!blockDataString) return;

      try {
        const blockData = JSON.parse(blockDataString) as BlockInfo;
        const position = screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });
        addBlock(blockData, {}, position);

        await new Promise((resolve) => setTimeout(resolve, 200));
        setBlockMenuOpen(true);
      } catch (error) {
        console.error("Failed to drop block:", error);
        setBlockMenuOpen(true);
      }
    },
    [screenToFlowPosition, addBlock, setBlockMenuOpen],
  );

  // When the graph is read-only, force the canvas locked and ignore unlock
  // attempts so non-owners cannot edit nodes.
  const isLocked = isReadOnly || isLockedState;
  const setIsLocked = useCallback(
    (locked: boolean) => {
      if (isReadOnly) return;
      setIsLockedState(locked);
    },
    [isReadOnly],
  );

  return {
    isFlowContentLoading: isGraphLoading || isBlocksLoading,
    isInitialLoadComplete,
    onDragOver,
    onDrop,
    isLocked,
    setIsLocked,
    isReadOnly,
  };
};
