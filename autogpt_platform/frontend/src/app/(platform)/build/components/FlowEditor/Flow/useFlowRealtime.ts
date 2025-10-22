// In this hook, I am only keeping websocket related code.

import { GraphExecutionID } from "@/lib/autogpt-server-api";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { parseAsString, useQueryStates } from "nuqs";
import { useEffect } from "react";
import { useNodeStore } from "../../../stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { NodeExecutionResult } from "@/app/api/__generated__/models/nodeExecutionResult";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { useGraphStore } from "../../../stores/graphStore";

export const useFlowRealtime = () => {
  const api = useBackendAPI();
  const updateNodeExecutionResult = useNodeStore(
    useShallow((state) => state.updateNodeExecutionResult),
  );
  const updateStatus = useNodeStore(
    useShallow((state) => state.updateNodeStatus),
  );
  const setIsGraphRunning = useGraphStore(
    useShallow((state) => state.setIsGraphRunning),
  );

  const [{ flowExecutionID, flowID }] = useQueryStates({
    flowExecutionID: parseAsString,
    flowID: parseAsString,
  });

  useEffect(() => {
    const deregisterNodeExecutionEvent = api.onWebSocketMessage(
      "node_execution_event",
      (data) => {
        if (data.graph_exec_id != flowExecutionID) {
          return;
        }
        // TODO: Update the states of nodes
        updateNodeExecutionResult(
          data.node_id,
          data as unknown as NodeExecutionResult,
        );
        updateStatus(data.node_id, data.status);
      },
    );

    const deregisterGraphExecutionStatusEvent = api.onWebSocketMessage(
      "graph_execution_event",
      (graphExecution) => {
        if (graphExecution.id != flowExecutionID) {
          return;
        }

        const isRunning =
          graphExecution.status === AgentExecutionStatus.RUNNING ||
          graphExecution.status === AgentExecutionStatus.QUEUED;

        setIsGraphRunning(isRunning);
      },
    );

    const deregisterGraphExecutionSubscription =
      flowID && flowExecutionID
        ? api.onWebSocketConnect(() => {
            // Subscribe to execution updates
            api
              .subscribeToGraphExecution(flowExecutionID as GraphExecutionID) // TODO: We are currently using a manual type, we need to fix it in future
              .then(() => {
                console.debug(
                  `Subscribed to updates for execution #${flowExecutionID}`,
                );
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
    };
  }, [api, flowExecutionID]);

  return {};
};
