import { useCallback } from "react";
import {
  useTrackEvent,
  EventKeys,
} from "@/services/feature-flags/use-track-event";

export function useAgentTracking() {
  const { track } = useTrackEvent();

  const trackBlockAdded = useCallback(
    (blockType: string, blockId: string, isBeta: boolean = false) => {
      track(EventKeys.BLOCK_ADDED, {
        blockType,
        blockId,
        isBeta,
        timestamp: new Date().toISOString(),
      });

      // Track beta block usage separately
      if (isBeta) {
        track(EventKeys.BETA_BLOCK_USED, {
          blockType,
          blockId,
          timestamp: new Date().toISOString(),
        });
      }
    },
    [track],
  );

  const trackBlockRemoved = useCallback(
    (blockType: string, blockId: string) => {
      track(EventKeys.BLOCK_REMOVED, {
        blockType,
        blockId,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBlockConfigured = useCallback(
    (blockType: string, blockId: string, changes: Record<string, any>) => {
      track(EventKeys.BLOCK_CONFIGURED, {
        blockType,
        blockId,
        changes,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBuilderOpened = useCallback(
    (agentId?: string, agentVersion?: number) => {
      track(EventKeys.BUILDER_OPENED, {
        agentId,
        agentVersion,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBuilderSaved = useCallback(
    (agentId: string, agentVersion: number) => {
      track(EventKeys.BUILDER_SAVED, {
        agentId,
        agentVersion,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBuilderNodeDisconnected = useCallback(
    (sourceNode: string, targetNode: string) => {
      track(EventKeys.BUILDER_NODE_DISCONNECTED, {
        sourceNode,
        targetNode,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBuilderSearchUsed = useCallback(
    (searchQuery: string, resultCount: number) => {
      track(EventKeys.BUILDER_SEARCH_USED, {
        searchQuery,
        resultCount,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBuilderUndoUsed = useCallback(() => {
    track(EventKeys.BUILDER_UNDO_USED, {
      timestamp: new Date().toISOString(),
    });
  }, [track]);

  const trackBuilderRedoUsed = useCallback(() => {
    track(EventKeys.BUILDER_REDO_USED, {
      timestamp: new Date().toISOString(),
    });
  }, [track]);

  const trackBuilderAction = useCallback(
    (action: string, data?: Record<string, any>) => {
      track("builder-action", {
        action,
        ...data,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  return {
    trackBlockAdded,
    trackBlockRemoved,
    trackBlockConfigured,
    trackBuilderOpened,
    trackBuilderSaved,
    trackBuilderNodeDisconnected,
    trackBuilderSearchUsed,
    trackBuilderUndoUsed,
    trackBuilderRedoUsed,
    trackBuilderAction,
  };
}
