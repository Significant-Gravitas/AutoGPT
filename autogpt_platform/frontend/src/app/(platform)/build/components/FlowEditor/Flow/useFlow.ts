import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import { useGetV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useNodeStore } from "../../../stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { useEffect, useMemo } from "react";
import { convertNodesPlusBlockInfoIntoCustomNodes } from "../../helper";
import { useEdgeStore } from "../../../stores/edgeStore";

export const useFlow = () => {
  const addNodes = useNodeStore(useShallow((state) => state.addNodes));
  const addLinks = useEdgeStore(useShallow((state) => state.addLinks));

  const [{ flowID, flowVersion }] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
  });

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
    if (customNodes.length > 0) {
      useNodeStore.getState().setNodes([]);
      addNodes(customNodes);
    }

    if (graph?.links) {
      useEdgeStore.getState().setConnections([]);
      addLinks(graph.links);
    }
  }, [customNodes, addNodes, graph?.links]);

  useEffect(() => {
    return () => {
      useNodeStore.getState().setNodes([]);
      useEdgeStore.getState().setConnections([]);
    };
  }, []);

  return { isFlowContentLoading: isGraphLoading || isBlocksLoading };
};
