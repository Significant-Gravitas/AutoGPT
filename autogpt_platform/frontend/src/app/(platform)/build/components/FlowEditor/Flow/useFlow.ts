import { useCallback, useEffect, useMemo, useState } from "react";
import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import {
  useGetV1GetExecutionDetails,
  useGetV1GetSpecificGraph,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useNodeStore } from "../../../stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { convertNodesPlusBlockInfoIntoCustomNodes } from "../../helper";
import { useEdgeStore } from "../../../stores/edgeStore";
import { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { useGraphStore } from "../../../stores/graphStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { useReactFlow } from "@xyflow/react";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { useHistoryStore } from "../../../stores/historyStore";

export const useFlow = () => {
  const [isLocked, setIsLocked] = useState(false);
  const addNodes = useNodeStore(useShallow((state) => state.addNodes));
  const addLinks = useEdgeStore(useShallow((state) => state.addLinks));
  const updateNodeStatus = useNodeStore(
    useShallow((state) => state.updateNodeStatus),
  );
  const updateNodeExecutionResult = useNodeStore(
    useShallow((state) => state.updateNodeExecutionResult),
  );
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );
  const setGraphSchemas = useGraphStore(
    useShallow((state) => state.setGraphSchemas),
  );
  const updateEdgeBeads = useEdgeStore(
    useShallow((state) => state.updateEdgeBeads),
  );
  const { screenToFlowPosition, fitView } = useReactFlow();
  const addBlock = useNodeStore(useShallow((state) => state.addBlock));
  const setBlockMenuOpen = useControlPanelStore(
    useShallow((state) => state.setBlockMenuOpen),
  );
  const [{ flowID, flowVersion, flowExecutionID }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
    flowExecutionID: parseAsString,
  });

  const { data: executionDetails } = useGetV1GetExecutionDetails(
    flowID || "",
    flowExecutionID || "",
    {
      query: {
        select: (res) => res.data as GetV1GetExecutionDetails200,
        enabled: !!flowID && !!flowExecutionID,
      },
    },
  );

  const { data: graph, isLoading: isGraphLoading } = useGetV1GetSpecificGraph(
    flowID ?? "",
    flowVersion !== null ? { version: flowVersion } : {},
    {
      query: {
        select: (res) => res.data as GraphModel,
        enabled: !!flowID,
      },
    },
  );

  const nodes = graph?.nodes;
  const blockIds = nodes
    ? Array.from(new Set(nodes.map((node) => node.block_id)))
    : undefined;

  const { data: blocks, isLoading: isBlocksLoading } =
    useGetV2GetSpecificBlocks(
      { block_ids: blockIds ?? [] },
      {
        query: {
          select: (res) => res.data as BlockInfo[],
          enabled: !!flowID && !!blockIds,
        },
      },
    );

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
      setGraphSchemas(
        graph.input_schema as Record<string, any> | null,
        graph.credentials_input_schema as Record<string, any> | null,
        graph.output_schema as Record<string, any> | null,
      );
    }
  }, [graph]);

  // adding nodes
  useEffect(() => {
    if (customNodes.length > 0) {
      useNodeStore.getState().setNodes([]);
      addNodes(customNodes);
    }
  }, [customNodes, addNodes]);

  // adding links
  useEffect(() => {
    if (graph?.links) {
      useEdgeStore.getState().setEdges([]);
      addLinks(graph.links);
    }
  }, [graph?.links, addLinks]);

  // update graph running status
  useEffect(() => {
    const isRunning =
      executionDetails?.status === AgentExecutionStatus.RUNNING ||
      executionDetails?.status === AgentExecutionStatus.QUEUED;

    setIsGraphRunning(isRunning);
  }, [executionDetails?.status, customNodes]);

  // update node execution status in nodes
  useEffect(() => {
    if (
      executionDetails &&
      "node_executions" in executionDetails &&
      executionDetails.node_executions
    ) {
      executionDetails.node_executions.forEach((nodeExecution) => {
        updateNodeStatus(nodeExecution.node_id, nodeExecution.status);
      });
    }
  }, [executionDetails, updateNodeStatus, customNodes]);

  // update node execution results in nodes, also update edge beads
  useEffect(() => {
    if (
      executionDetails &&
      "node_executions" in executionDetails &&
      executionDetails.node_executions
    ) {
      executionDetails.node_executions.forEach((nodeExecution) => {
        updateNodeExecutionResult(nodeExecution.node_id, nodeExecution);
        updateEdgeBeads(nodeExecution.node_id, nodeExecution);
      });
    }
  }, [
    executionDetails,
    updateNodeExecutionResult,
    updateEdgeBeads,
    customNodes,
  ]);

  useEffect(() => {
    if (customNodes.length > 0 && graph?.links) {
      const timer = setTimeout(() => {
        useHistoryStore.getState().initializeHistory();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [customNodes, graph?.links]);

  useEffect(() => {
    return () => {
      useNodeStore.getState().setNodes([]);
      useEdgeStore.getState().setEdges([]);
      useGraphStore.getState().reset();
      useEdgeStore.getState().resetEdgeBeads();
      setIsGraphRunning(false);
    };
  }, []);

  useEffect(() => {
    fitView({ padding: 0.2, duration: 800, maxZoom: 2 });
  }, [fitView]);

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

  return {
    isFlowContentLoading: isGraphLoading || isBlocksLoading,
    onDragOver,
    onDrop,
    isLocked,
    setIsLocked,
  };
};
