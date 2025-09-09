"use client";

import { useLDClient } from "launchdarkly-react-client-sdk";
import { useCallback } from "react";
import { BehaveAs, getBehaveAs } from "@/lib/utils";

export interface TrackEventData {
  [key: string]: any;
}

export interface NumericTrackEventData extends TrackEventData {
  metricValue?: number;
}

/**
 * Hook for tracking custom events in LaunchDarkly
 * Automatically handles environment checks and provides type-safe event tracking
 */
export function useTrackEvent() {
  const client = useLDClient();
  const isCloud = getBehaveAs() === BehaveAs.CLOUD;

  const track = useCallback(
    (eventKey: string, data?: TrackEventData, metricValue?: number) => {
      if (!isCloud || !client) {
        console.debug(
          `Event tracking skipped (not cloud or no client): ${eventKey}`,
          data,
        );
        return;
      }

      try {
        if (metricValue !== undefined) {
          // Track with numeric value for numeric metrics
          client.track(eventKey, data, metricValue);
        } else {
          // Track without numeric value for binary/count metrics
          client.track(eventKey, data);
        }

        console.debug(`Event tracked: ${eventKey}`, data, metricValue);
      } catch (error) {
        console.error(`Failed to track event ${eventKey}:`, error);
      }
    },
    [client, isCloud],
  );

  return { track };
}

/**
 * Common event keys for tracking user interactions
 * These should match the event keys configured in LaunchDarkly metrics
 */
export const EventKeys = {
  // Agent/Graph events
  AGENT_CREATED: "agent-created",
  AGENT_UPDATED: "agent-updated",
  AGENT_DELETED: "agent-deleted",
  AGENT_RUN_STARTED: "agent-run-started",
  AGENT_RUN_COMPLETED: "agent-run-completed",
  AGENT_RUN_FAILED: "agent-run-failed",
  AGENT_PUBLISHED_TO_STORE: "agent-published-to-store",
  AGENT_IMPORTED_FROM_STORE: "agent-imported-from-store",

  // Block events
  BLOCK_ADDED: "block-added",
  BLOCK_REMOVED: "block-removed",
  BLOCK_CONFIGURED: "block-configured",
  BETA_BLOCK_USED: "beta-block-used",

  // Marketplace events
  STORE_ACCESSED: "store-accessed",
  STORE_AGENT_VIEWED: "store-agent-viewed",
  STORE_AGENT_INSTALLED: "store-agent-installed",
  STORE_SEARCH_PERFORMED: "store-search-performed",
  STORE_FILTER_APPLIED: "store-filter-applied",

  // User engagement events
  ONBOARDING_STARTED: "onboarding-started",
  ONBOARDING_COMPLETED: "onboarding-completed",
  ONBOARDING_SKIPPED: "onboarding-skipped",
  FEATURE_FLAG_VIEWED: "feature-flag-viewed",
  TUTORIAL_STARTED: "tutorial-started",
  TUTORIAL_COMPLETED: "tutorial-completed",

  // Builder events
  BUILDER_OPENED: "builder-opened",
  BUILDER_SAVED: "builder-saved",
  BUILDER_NODE_CONNECTED: "builder-node-connected",
  BUILDER_NODE_DISCONNECTED: "builder-node-disconnected",
  BUILDER_SEARCH_USED: "builder-search-used",
  BUILDER_UNDO_USED: "builder-undo-used",
  BUILDER_REDO_USED: "builder-redo-used",

  // Performance metrics (with numeric values)
  PAGE_LOAD_TIME: "page-load-time",
  API_RESPONSE_TIME: "api-response-time",
  AGENT_EXECUTION_TIME: "agent-execution-time",
  BLOCK_EXECUTION_TIME: "block-execution-time",

  // Error events
  ERROR_OCCURRED: "error-occurred",
  API_ERROR: "api-error",
  VALIDATION_ERROR: "validation-error",

  // Credit/Billing events
  CREDITS_PURCHASED: "credits-purchased",
  CREDITS_DEPLETED: "credits-depleted",
  SUBSCRIPTION_UPGRADED: "subscription-upgraded",
  SUBSCRIPTION_DOWNGRADED: "subscription-downgraded",
} as const;

export type EventKey = (typeof EventKeys)[keyof typeof EventKeys];
