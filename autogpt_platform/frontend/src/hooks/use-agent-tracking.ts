import { useCallback } from "react";
import { useTrackEvent } from "@/services/feature-flags/use-track-event";

export function useAgentTracking() {
  const { track } = useTrackEvent();

  const trackBlockAdded = useCallback(
    (blockType: string, blockId: string, isBeta: boolean = false) => {
      track("builder-block-added", {
        blockType,
        blockId,
        isBeta,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

  const trackBlockRemoved = useCallback(
    (blockType: string, blockId: string) => {
      track("builder-block-removed", {
        blockType,
        blockId,
        timestamp: new Date().toISOString(),
      });
    },
    [track],
  );

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
    trackBuilderAction,
  };
}
