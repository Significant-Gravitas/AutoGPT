import { useEffect, useRef } from "react";
import { trackFunnel } from "./experts-analytics";

type FunnelViewEvent =
  | "experts_section_viewed"
  | "home_viewed"
  | "briefing_opened";

export function useTrackFunnelViewOnce(event: FunnelViewEvent, enabled = true) {
  const trackedRef = useRef(false);

  useEffect(() => {
    if (trackedRef.current || !enabled) return;
    trackedRef.current = true;
    trackFunnel(event);
  }, [enabled, event]);
}
