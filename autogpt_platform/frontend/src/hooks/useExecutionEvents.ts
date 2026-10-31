"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import * as Sentry from "@sentry/nextjs";
import { useEffect, useRef } from "react";

type ExecutionEventHandler = (execution: GraphExecution) => void;

/** For handlers that refetch a list: at most one refetch per 2 s. */
export const EXECUTION_REFETCH_THROTTLE_MS = 2000;

interface UseExecutionEventsOptions {
  graphId?: GraphID | string | null;
  graphIds?: (GraphID | string)[];
  enabled?: boolean;
  /**
   * Call `onExecutionUpdate` at most once per this many milliseconds: the
   * first event right away, then the latest one when the window closes.
   * For handlers that refetch, where only the newest event matters.
   */
  throttleMs?: number;
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
 * @param options.throttleMs - Deliver at most one update per this many ms
 * @param options.onExecutionUpdate - Callback invoked when an execution is updated
 */
export function useExecutionEvents({
  graphId,
  graphIds,
  enabled = true,
  throttleMs,
  onExecutionUpdate,
}: UseExecutionEventsOptions) {
  const api = useBackendAPI();
  const onExecutionUpdateRef = useRef(onExecutionUpdate);

  useEffect(() => {
    onExecutionUpdateRef.current = onExecutionUpdate;
  }, [onExecutionUpdate]);

  // Keyed on the IDs' content, so an equal array built on every render of
  // the caller does not tear the subscription down and send it again.
  const idsKey = (graphIds || (graphId ? [graphId] : []))
    .map((id) => String(id))
    .join(",");

  useEffect(() => {
    if (!enabled) return;
    if (idsKey === "") return;

    const normalizedIds = new Set(idsKey.split(","));
    const subscribedIds = new Set<string>();
    const throttle = createThrottle((execution: GraphExecution) => {
      onExecutionUpdateRef.current?.(execution);
    }, throttleMs);

    const handleExecutionEvent = (execution: GraphExecution) => {
      if (!normalizedIds.has(String(execution.graph_id))) return;
      throttle.call(execution);
    };

    const connectHandler = api.onWebSocketConnect(() => {
      normalizedIds.forEach((id) => {
        // Track subscriptions to avoid duplicate subscriptions
        if (subscribedIds.has(id)) return;
        subscribedIds.add(id);

        api.subscribeToGraphExecutions(id as GraphID).catch((error) => {
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

    // A new connection starts with no subscriptions on the server, so the
    // next connect must send them all again.
    const disconnectHandler = api.onWebSocketDisconnect(() => {
      subscribedIds.clear();
    });

    const messageHandler = api.onWebSocketMessage(
      "graph_execution_event",
      handleExecutionEvent,
    );

    api.connectWebSocket();

    return () => {
      connectHandler();
      disconnectHandler();
      messageHandler();
      throttle.flush();
      // Note: Backend automatically cleans up subscriptions on websocket disconnect
      // If IDs change while connected, old subscriptions remain but are filtered client-side
      subscribedIds.clear();
    };
  }, [api, idsKey, enabled, throttleMs]);
}

function createThrottle<T>(fn: (value: T) => void, waitMs?: number) {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: { value: T } | null = null;

  function closeWindow() {
    timer = null;
    if (!pending) return;
    const { value } = pending;
    pending = null;
    fn(value);
    timer = setTimeout(closeWindow, waitMs);
  }

  return {
    call(value: T) {
      if (!waitMs) return fn(value);
      if (timer) {
        pending = { value };
        return;
      }
      fn(value);
      timer = setTimeout(closeWindow, waitMs);
    },
    /** Delivers a held-back value now, so tearing down never drops one. */
    flush() {
      if (timer) clearTimeout(timer);
      timer = null;
      if (!pending) return;
      const { value } = pending;
      pending = null;
      fn(value);
    },
  };
}
