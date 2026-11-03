"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import * as Sentry from "@sentry/nextjs";
import { useEffect, useRef } from "react";
import { getTenantEntityKey } from "@/services/org-team/identity";

type ExecutionEventHandler = (execution: GraphExecution) => void;

/** For handlers that refetch a list: at most one refetch per 2 s. */
export const EXECUTION_REFETCH_THROTTLE_MS = 2000;

interface GraphScope {
  graphId: GraphID | string;
  organizationId: string | null;
  teamId: string | null;
}

type ScopeTuple = [
  graphId: string,
  organizationId: string | null,
  teamId: string | null,
];

interface UseExecutionEventsOptions {
  graphId?: GraphID | string | null;
  organizationId?: string | null;
  teamId?: string | null;
  graphScopes?: GraphScope[];
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
 * @param options.organizationId - The single graph's organization
 * @param options.teamId - The single graph's team (null for org-home)
 * @param options.graphScopes - Graphs to subscribe to, each in its own scope
 * @param options.enabled - Whether the subscription is enabled (default: true)
 * @param options.throttleMs - Deliver at most one update per this many ms
 * @param options.onExecutionUpdate - Callback invoked when an execution is updated
 */
export function useExecutionEvents({
  graphId,
  organizationId,
  teamId,
  graphScopes,
  enabled = true,
  throttleMs,
  onExecutionUpdate,
}: UseExecutionEventsOptions) {
  const api = useBackendAPI();
  const onExecutionUpdateRef = useRef(onExecutionUpdate);

  useEffect(() => {
    onExecutionUpdateRef.current = onExecutionUpdate;
  }, [onExecutionUpdate]);

  // Keyed on the scopes' content, so an equal list built on every render of
  // the caller does not tear the subscription down and send it again.
  const scopesKey = getScopesKey(
    graphScopes ||
      (graphId
        ? [
            {
              graphId,
              organizationId: organizationId ?? null,
              teamId: teamId ?? null,
            },
          ]
        : []),
  );

  useEffect(() => {
    if (!enabled) return;

    const scopes = (JSON.parse(scopesKey) as ScopeTuple[]).map(
      ([scopeGraphId, scopeOrganizationId, scopeTeamId]) => ({
        graphId: scopeGraphId,
        organizationId: scopeOrganizationId,
        teamId: scopeTeamId,
        key: getTenantEntityKey(scopeGraphId, scopeOrganizationId, scopeTeamId),
      }),
    );
    if (scopes.length === 0) return;

    const scopeKeys = new Set(scopes.map((scope) => scope.key));
    const subscribedScopes = new Set<string>();
    const throttle = createThrottle((execution: GraphExecution) => {
      onExecutionUpdateRef.current?.(execution);
    }, throttleMs);

    const handleExecutionEvent = (execution: GraphExecution) => {
      const executionKey = getTenantEntityKey(
        String(execution.graph_id),
        execution.organization_id,
        execution.team_id,
      );
      if (!scopeKeys.has(executionKey)) return;
      throttle.call(execution);
    };

    const connectHandler = api.onWebSocketConnect(() => {
      scopes.forEach((scope) => {
        // Track subscriptions to avoid duplicate subscriptions
        if (subscribedScopes.has(scope.key)) return;
        subscribedScopes.add(scope.key);

        api
          .subscribeToGraphExecutions(
            scope.graphId as GraphID,
            scope.organizationId,
            scope.teamId,
          )
          .catch((error) => {
            console.error(
              `Failed to subscribe to execution updates for graph ${scope.graphId}:`,
              error,
            );
            Sentry.captureException(error, {
              tags: { graphId: scope.graphId },
            });
            subscribedScopes.delete(scope.key);
          });
      });
    });

    // A new connection starts with no subscriptions on the server, so the
    // next connect must send them all again.
    const disconnectHandler = api.onWebSocketDisconnect(() => {
      subscribedScopes.clear();
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
      subscribedScopes.clear();
    };
  }, [api, scopesKey, enabled, throttleMs]);
}

/** One string per set of scopes, whatever order the caller lists them in. */
function getScopesKey(scopes: GraphScope[]) {
  const tuples = new Map<string, ScopeTuple>();
  for (const scope of scopes) {
    const tuple: ScopeTuple = [
      String(scope.graphId),
      scope.organizationId,
      scope.teamId,
    ];
    tuples.set(JSON.stringify(tuple), tuple);
  }
  return JSON.stringify(
    [...tuples.entries()]
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([, tuple]) => tuple),
  );
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
