"use client";

import { WebSocketSubscriptionInterrupted } from "@/lib/autogpt-server-api/client";
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
  onSubscribed?: () => void;
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
 * @param options.onSubscribed - Called once the server has confirmed the
 *   subscriptions, on the first connect and every reconnect. Refetch here:
 *   events sent before the subscription was live (or while disconnected)
 *   are never delivered
 */
export function useExecutionEvents({
  graphId,
  graphIds,
  enabled = true,
  onExecutionUpdate,
  onSubscribed,
}: UseExecutionEventsOptions) {
  const api = useBackendAPI();
  const onExecutionUpdateRef = useRef(onExecutionUpdate);
  const onSubscribedRef = useRef(onSubscribed);

  useEffect(() => {
    onExecutionUpdateRef.current = onExecutionUpdate;
    onSubscribedRef.current = onSubscribed;
  }, [onExecutionUpdate, onSubscribed]);

  useEffect(() => {
    if (!enabled) return;

    const idsToSubscribe = graphIds || (graphId ? [graphId] : []);
    if (idsToSubscribe.length === 0) return;

    // Normalize IDs to strings for consistent comparison
    const normalizedIds = idsToSubscribe.map((id) => String(id));
    const handleExecutionEvent = (execution: GraphExecution) => {
      // Filter by graphIds if provided, using normalized string comparison
      if (normalizedIds.length > 0) {
        const executionGraphId = String(execution.graph_id);
        if (!normalizedIds.includes(executionGraphId)) return;
      }

      onExecutionUpdateRef.current?.(execution);
    };

    // The server drops every subscription along with the socket, so they
    // are all re-sent on each (re)connect.
    const connectHandler = api.onWebSocketConnect(() => {
      const subscriptions = normalizedIds.map((id) =>
        api.subscribeToGraphExecutions(id as GraphID).catch((error) => {
          if (error instanceof WebSocketSubscriptionInterrupted) return;
          console.error(
            `Failed to subscribe to execution updates for graph ${id}:`,
            error,
          );
          Sentry.captureException(error, {
            tags: { graphId: id },
          });
        }),
      );
      void Promise.all(subscriptions).then(() => onSubscribedRef.current?.());
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
    };
  }, [api, graphId, graphIds, enabled]);
}
