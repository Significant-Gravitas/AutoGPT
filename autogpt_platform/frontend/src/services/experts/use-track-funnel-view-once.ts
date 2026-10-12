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

/** Keyed on the id, not on the mount: Next.js reuses the expert page's tree
 *  when navigating between two profiles, which a mount-once guard would miss. */
export function useTrackExpertProfileOpened(templateId: string | undefined) {
  const trackedRef = useRef<string | null>(null);

  useEffect(() => {
    if (!templateId || trackedRef.current === templateId) return;
    trackedRef.current = templateId;
    trackFunnel("expert_profile_opened", { template_id: templateId });
  }, [templateId]);
}
