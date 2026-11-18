import { router } from "expo-router";
import { useState } from "react";

import { isSessionID } from "@/lib/api/chat";
import { useMountEffect } from "@/lib/useMountEffect";

import { markConnected } from "./connectedThisRun";
import { allowForExpert } from "./expertAccess";
import { credentialLabel } from "./helpers";
import { completeOAuthReturn } from "./oauth";
import { oauthReturnFromParams } from "./oauthHelpers";

const PAYWALL_MESSAGE =
  "Connecting a service needs an active plan. Finish this on the web.";

function leave(sessionID: string | null) {
  if (router.canGoBack()) router.back();
  else if (isSessionID(sessionID)) router.replace(`/chat/${sessionID}`);
  else router.replace("/");
}

/**
 * Finishes a sign-in that came back as a link: on Android after the app was
 * closed meanwhile, or alongside the browser session that started it (both
 * share one code swap). Then it goes back to the chat it started from.
 */
export function useOAuthReturn(params: Record<string, unknown>) {
  const [error, setError] = useState<string | null>(null);
  const [sessionID, setSessionID] = useState<string | null>(null);

  useMountEffect(() => {
    const returned = oauthReturnFromParams(params);
    if (!returned) {
      leave(null);
      return;
    }
    let isMounted = true;
    void completeOAuthReturn(returned).then(({ outcome, flow }) => {
      // The card and the expert's access are not this screen's: they are
      // settled even when it was left before the swap finished.
      if (outcome.status === "connected" && flow?.cardKey && flow.rowKey) {
        markConnected(flow.cardKey, flow.rowKey, {
          id: outcome.credential.id,
          title: credentialLabel(outcome.credential),
        });
      }
      // Started from an expert's chat panel: the new account is for them.
      // The panel grants it too when it is still open; a second grant is a no-op.
      if (outcome.status === "connected" && flow && !flow.cardKey && flow.expertID) {
        void allowForExpert(flow.expertID, [outcome.credential.id]).catch(() => undefined);
      }
      if (!isMounted) return;
      if (outcome.status === "failed" || outcome.status === "paywall") {
        setSessionID(flow?.sessionID ?? null);
        setError(outcome.status === "paywall" ? PAYWALL_MESSAGE : outcome.message);
        return;
      }
      leave(flow?.sessionID ?? null);
    });
    return () => {
      isMounted = false;
    };
  });

  function goBack() {
    leave(sessionID);
  }

  return { error, goBack };
}
