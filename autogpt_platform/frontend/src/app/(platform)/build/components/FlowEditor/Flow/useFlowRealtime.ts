// In this hook, I am only keeping websocket related code.

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { parseAsString, useQueryStates } from "nuqs";
import { useEffect } from "react";
import { useNodeStore } from "../../../stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { useGraphStore } from "../../../stores/graphStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV1GetExecutionDetailsQueryKey } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { parseGraphExecutionID, parseGraphID } from "@/lib/graph-ids";

export const useFlowRealtime = () => {
  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const updateNodeExecutionResult = useNodeStore(
    useShallow((state) => state.updateNodeExecutionResult),
  );
  const updateStatus = useNodeStore(
    useShallow((state) => state.updateNodeStatus),
  );
  const setGraphExecutionStatus = useGraphStore(
    useShallow((state) => state.setGraphExecutionStatus),
  );
  const updateEdgeBeads = useEdgeStore(
    useShallow((state) => state.updateEdgeBeads),
  );
  const resetEdgeBeads = useEdgeStore(
    useShallow((state) => state.resetEdgeBeads),
  );

  const [{ flowExecutionID: rawFlowExecutionID, flowID: rawFlowID }] =
    useQueryStates({
      flowExecutionID: parseAsString,
      flowID: parseAsString,
    });

  const flowExecutionID = parseGraphExecutionID(rawFlowExecutionID);
  const flowID = parseGraphID(rawFlowID);

  useEffect(() => {
    const deregisterNodeExecutionEvent = api.onWebSocketMessage(
      "node_execution_event",
      (data) => {
        if (data.graph_exec_id != flowExecutionID) {
          return;
        }
        updateNodeExecutionResult(
          data.node_id,
          data as unknown as NodeExecutionResult,
        );
        updateStatus(data.node_id, data.status);
        updateEdgeBeads(data.node_id, data as unknown as NodeExecutionResult);
      },
    );

    const deregisterGraphExecutionStatusEvent = api.onWebSocketMessage(
      "graph_execution_event",
      (graphExecution) => {
        if (graphExecution.id != flowExecutionID) {
          return;
        }

        setGraphExecutionStatus(graphExecution.status as AgentExecutionStatus);
      },
    );

    const deregisterGraphExecutionSubscription =
      flowID && flowExecutionID
        ? api.onWebSocketConnect(() => {
            // Subscribe to execution updates — both IDs are validated UUIDs
            // by this point (parseGraph* returns null for malformed values,
            // which makes the `flowID && flowExecutionID` guard fail above).
            api
              .subscribeToGraphExecution(flowExecutionID)
              .then(() => {
                console.debug(
                  `Subscribed to updates for execution #${flowExecutionID}`,
                );
                // Refetch execution details to catch any events that were
                // published before the WebSocket subscription was established.
                // This closes the race-condition window for fast-completing
                // executions like dry-runs / simulations.
                void queryClient.invalidateQueries({
                  queryKey: getGetV1GetExecutionDetailsQueryKey(
                    flowID,
                    flowExecutionID,
                  ),
                });
              })
              .catch((error) =>
                console.error(
                  `Failed to subscribe to updates for execution #${flowExecutionID}:`,
                  error,
                ),
              );
          })
        : () => {};

    return () => {
      deregisterNodeExecutionEvent();
      deregisterGraphExecutionSubscription();
      deregisterGraphExecutionStatusEvent();
      resetEdgeBeads();
    };
  }, [api, flowExecutionID, resetEdgeBeads, queryClient, flowID]);

  return {};
};
