"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import * as Sentry from "@sentry/nextjs";
import { useEffect, useRef } from "react";

type ExecutionEventHandler = (execution: GraphExecution) => void;

interface UseExecutionEventsOptions {
  graphId?: GraphID | string | null;
  graphIds?: (GraphID | string)[];
  enabled?: boolean;
  onExecutionUpdate?: ExecutionEventHandler;
}

/**
 * Generic hook to subscribe to graph execution events via WebSocket.
 * Automatically handles subscription/unsubscription and reconnection.
 *
 * @param options - Configuration options
 * @param options.graphId - The graph ID to subscribe to (single graph)
 * @param options.graphIds - Array of graph IDs to subscribe to (multiple graphs)
 * @param options.enabled - Whether the subscription is enabled (default: true)
 * @param options.onExecutionUpdate - Callback invoked when an execution is updated
 */
export function useExecutionEvents({
  graphId,
  graphIds,
  enabled = true,
  onExecutionUpdate,
}: UseExecutionEventsOptions) {
  const api = useBackendAPI();
  const onExecutionUpdateRef = useRef(onExecutionUpdate);

  useEffect(() => {
    onExecutionUpdateRef.current = onExecutionUpdate;
  }, [onExecutionUpdate]);

  useEffect(() => {
    if (!enabled) return;

    const idsToSubscribe = graphIds || (graphId ? [graphId] : []);
    if (idsToSubscribe.length === 0) return;

    // Normalize IDs to strings for consistent comparison
    const normalizedIds = idsToSubscribe.map((id) => String(id));
    const subscribedIds = new Set<string>();

    const handleExecutionEvent = (execution: GraphExecution) => {
      // Filter by graphIds if provided, using normalized string comparison
      if (normalizedIds.length > 0) {
        const executionGraphId = String(execution.graph_id);
        if (!normalizedIds.includes(executionGraphId)) return;
      }

      onExecutionUpdateRef.current?.(execution);
    };

    const connectHandler = api.onWebSocketConnect(() => {
      normalizedIds.forEach((id) => {
        // Track subscriptions to avoid duplicate subscriptions
        if (subscribedIds.has(id)) return;
        subscribedIds.add(id);

        api
          .subscribeToGraphExecutions(id as GraphID)
          .then(() => {
            console.debug(`Subscribed to execution updates for graph ${id}`);
          })
          .catch((error) => {
            console.error(
              `Failed to subscribe to execution updates for graph ${id}:`,
              error,
            );
            Sentry.captureException(error, {
              tags: { graphId: id },
            });
            subscribedIds.delete(id);
          });
      });
    });

    const messageHandler = api.onWebSocketMessage(
      "graph_execution_event",
      handleExecutionEvent,
    );

    api.connectWebSocket();

    return () => {
      connectHandler();
      messageHandler();
      // Note: Backend automatically cleans up subscriptions on websocket disconnect
      // If IDs change while connected, old subscriptions remain but are filtered client-side
      subscribedIds.clear();
    };
  }, [api, graphId, graphIds, enabled]);
}
