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
import { useEffect, useMemo } from "react";
import { convertNodesPlusBlockInfoIntoCustomNodes } from "../../helper";
import { useEdgeStore } from "../../../stores/edgeStore";
import { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { useGraphStore } from "../../../stores/graphStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

export const useFlow = () => {
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
  const blockIds = nodes?.map((node) => node.block_id);

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

  useEffect(() => {
    // load graph schemas
    if (graph) {
      setGraphSchemas(
        graph.input_schema as Record<string, any> | null,
        graph.credentials_input_schema as Record<string, any> | null,
      );
    }

    // adding nodes
    if (customNodes.length > 0) {
      useNodeStore.getState().setNodes([]);
      addNodes(customNodes);
    }

    // adding links
    if (graph?.links) {
      useEdgeStore.getState().setConnections([]);
      addLinks(graph.links);
    }

    // update graph running status
    const isRunning =
      executionDetails?.status === AgentExecutionStatus.RUNNING ||
      executionDetails?.status === AgentExecutionStatus.QUEUED;
    setIsGraphRunning(isRunning);

    // update node execution status in nodes
    if (
      executionDetails &&
      "node_executions" in executionDetails &&
      executionDetails.node_executions
    ) {
      executionDetails.node_executions.forEach((nodeExecution) => {
        updateNodeStatus(nodeExecution.node_id, nodeExecution.status);
      });
    }

    // update node execution results in nodes
    if (
      executionDetails &&
      "node_executions" in executionDetails &&
      executionDetails.node_executions
    ) {
      executionDetails.node_executions.forEach((nodeExecution) => {
        updateNodeExecutionResult(nodeExecution.node_id, nodeExecution);
      });
    }
  }, [customNodes, addNodes, graph?.links, executionDetails, updateNodeStatus]);

  useEffect(() => {
    return () => {
      useNodeStore.getState().setNodes([]);
      useEdgeStore.getState().setConnections([]);
      useGraphStore.getState().reset();
      setIsGraphRunning(false);
    };
  }, []);

  return { isFlowContentLoading: isGraphLoading || isBlocksLoading };
};
