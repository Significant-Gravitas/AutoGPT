"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import * as Sentry from "@sentry/nextjs";
import {
  hashKey,
  type QueryClient,
  type QueryKey,
} from "@tanstack/react-query";
import { useEffect, useRef } from "react";

type ExecutionEventHandler = (execution: GraphExecution) => void;

/** For handlers that refetch a list: at most one refetch per 2 s. */
export const EXECUTION_REFETCH_THROTTLE_MS = 2000;

// The backend allows 1000 subscriptions per connection, shared by every
// subscriber on the page. Lists of graphs (the whole library) take the first
// 500, which leaves room for the run and builder pages.
export const MAX_GRAPH_SUBSCRIPTIONS = 500;

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
  /**
   * Called whenever live state should be re-read, because updates may have
   * been missed: after the socket reconnects, once the server has these
   * subscriptions again, and after a subscribe that failed settles on a
   * retry, on any connection, the first included. Refetch here.
   */
  onReconnect?: () => void;
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
 * @param options.onReconnect - Callback invoked after a reconnect, to refetch
 */
export function useExecutionEvents({
  graphId,
  graphIds,
  enabled = true,
  throttleMs,
  onExecutionUpdate,
  onReconnect,
}: UseExecutionEventsOptions) {
  const api = useBackendAPI();
  const onExecutionUpdateRef = useRef(onExecutionUpdate);
  const onReconnectRef = useRef(onReconnect);

  useEffect(() => {
    onExecutionUpdateRef.current = onExecutionUpdate;
    onReconnectRef.current = onReconnect;
  }, [onExecutionUpdate, onReconnect]);

  // Keyed on the IDs' content, so an equal array built on every render of
  // the caller does not tear the subscription down and send it again.
  const idsKey = (graphIds || (graphId ? [graphId] : []))
    .slice(0, MAX_GRAPH_SUBSCRIPTIONS)
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

    // Keyed to the reconnect, not to this effect: a new ID list subscribes
    // again without asking for a refetch.
    const reconnectHandler = api.onWebSocketReconnect(() => {
      onReconnectRef.current?.();
    });

    const messageHandler = api.onWebSocketMessage(
      "graph_execution_event",
      handleExecutionEvent,
    );

    api.connectWebSocket();

    return () => {
      connectHandler();
      disconnectHandler();
      reconnectHandler();
      messageHandler();
      throttle.flush();
      // Note: Backend automatically cleans up subscriptions on websocket disconnect
      // If IDs change while connected, old subscriptions remain but are filtered client-side
      subscribedIds.clear();
    };
  }, [api, idsKey, enabled, throttleMs]);
}

// Per client, the queries asked to refetch in the current task.
const refetchingNow = new WeakMap<QueryClient, Set<string>>();

/**
 * Refetches `queryKey` from an `onReconnect`. Every view runs its handler in
 * the same task, so for a query several views share only the first refetches;
 * another call would cancel that request and start its own.
 */
export function refetchAfterReconnect(
  queryClient: QueryClient,
  queryKey: QueryKey,
) {
  const hashes = refetchingNow.get(queryClient) ?? new Set<string>();
  refetchingNow.set(queryClient, hashes);
  const hash = hashKey(queryKey);
  if (hashes.has(hash)) return;
  hashes.add(hash);
  queueMicrotask(() => hashes.delete(hash));
  // A fetch already under way may have read before the reconnect, and with
  // nothing cached yet invalidating would wait for it instead of reading again.
  void queryClient
    .cancelQueries({ queryKey })
    .then(() => queryClient.invalidateQueries({ queryKey }));
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
