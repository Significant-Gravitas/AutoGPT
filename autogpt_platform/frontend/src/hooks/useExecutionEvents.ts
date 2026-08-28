"use client";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution, GraphID } from "@/lib/autogpt-server-api/types";
import * as Sentry from "@sentry/nextjs";
import { useEffect, useRef } from "react";
import { getTenantEntityKey } from "@/services/org-team/identity";

type ExecutionEventHandler = (execution: GraphExecution) => void;

interface UseExecutionEventsOptions {
  graphId?: GraphID | string | null;
  organizationId?: string | null;
  teamId?: string | null;
  graphScopes?: Array<{
    graphId: GraphID | string;
    organizationId: string | null;
    teamId: string | null;
  }>;
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
  organizationId,
  teamId,
  graphScopes,
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

    const scopesToSubscribe =
      graphScopes ||
      (graphId
        ? [
            {
              graphId,
              organizationId: organizationId ?? null,
              teamId: teamId ?? null,
            },
          ]
        : []);
    if (scopesToSubscribe.length === 0) return;

    const normalizedScopes = scopesToSubscribe.map((scope) => ({
      graphId: String(scope.graphId),
      organizationId: scope.organizationId,
      teamId: scope.teamId,
      key: getTenantEntityKey(
        String(scope.graphId),
        scope.organizationId,
        scope.teamId,
      ),
    }));
    const subscribedScopes = new Set<string>();

    const handleExecutionEvent = (execution: GraphExecution) => {
      if (normalizedScopes.length > 0) {
        const executionKey = getTenantEntityKey(
          String(execution.graph_id),
          execution.organization_id,
          execution.team_id,
        );
        if (!normalizedScopes.some((scope) => scope.key === executionKey)) {
          return;
        }
      }

      onExecutionUpdateRef.current?.(execution);
    };

    const connectHandler = api.onWebSocketConnect(() => {
      normalizedScopes.forEach((scope) => {
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
      subscribedScopes.clear();
    };
  }, [api, graphId, organizationId, teamId, graphScopes, enabled]);
}
