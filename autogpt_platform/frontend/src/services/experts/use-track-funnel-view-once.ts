import { useEffect, useRef } from "react";
import { trackFunnel, type FunnelViewEvent } from "./experts-analytics";

export function useTrackFunnelViewOnce(event: FunnelViewEvent, enabled = true) {
  const trackedRef = useRef(false);

  useEffect(() => {
    if (trackedRef.current || !enabled) return;
    trackedRef.current = true;
    trackFunnel(event);
  }, [enabled, event]);
}
