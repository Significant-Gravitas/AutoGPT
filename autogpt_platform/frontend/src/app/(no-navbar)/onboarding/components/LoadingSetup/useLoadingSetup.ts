import * as Sentry from "@sentry/nextjs";
import { useEffect, useState } from "react";

// The wizard's reads each give up on their own, but the signed-in account
// has no deadline of its own, so the page ends the wait here (#16965).
export const SETUP_LOAD_DEADLINE_MS = 15_000;

export function useLoadingSetup() {
  const [isStalled, setIsStalled] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsStalled(true);
      Sentry.captureMessage("Onboarding setup did not load in time", {
        level: "warning",
        tags: { context: "onboarding-setup" },
      });
    }, SETUP_LOAD_DEADLINE_MS);
    return () => clearTimeout(timer);
  }, []);

  function reload() {
    window.location.reload();
  }

  return { isStalled, reload };
}
