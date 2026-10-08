import { useCommunicationMode } from "@/hooks/useCommunicationMode";
import { useRouter } from "next/navigation";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect } from "react";

export const COMPACT_COPILOT_PATH = "/copilot/compact";
export const TECHNICAL_VIEW_PARAM = "view";

export function useCompactModeRedirect() {
  const { mode } = useCommunicationMode();
  const router = useRouter();
  const [view] = useQueryState(TECHNICAL_VIEW_PARAM, parseAsString);
  const shouldRedirect = mode === "compact" && view !== "technical";

  useEffect(() => {
    if (!shouldRedirect) return;
    router.replace(`${COMPACT_COPILOT_PATH}${window.location.search}`);
  }, [shouldRedirect, router]);

  return { isRedirecting: shouldRedirect };
}
