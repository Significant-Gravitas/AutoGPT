import { router } from "expo-router";
import { useRef, useState } from "react";

import { isSessionID } from "@/lib/api/chat";
import { useMountEffect } from "@/lib/useMountEffect";

import { markConnected } from "./connectedThisRun";
import { allowErrorText, allowForExpert } from "./expertAccess";
import { credentialLabel } from "./helpers";
import { claimOAuthResult, completeOAuthReturn } from "./oauth";
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
interface Grant {
  expertID: string;
  credentialID: string;
}

export function useOAuthReturn(params: Record<string, unknown>) {
  const [error, setError] = useState<string | null>(null);
  const [sessionID, setSessionID] = useState<string | null>(null);
  // An account saved for an expert who couldn't be given it yet.
  const [failedGrant, setFailedGrant] = useState<Grant | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);
  // False once the user left, so a retry that ends later doesn't navigate again.
  const isOnScreen = useRef(true);

  useMountEffect(() => {
    const returned = oauthReturnFromParams(params);
    if (!returned) {
      leave(null);
      return;
    }
    let isMounted = true;
    // A repeat of a link already handled only goes back to the chat.
    const isFirst = claimOAuthResult(returned.state);
    void completeOAuthReturn(returned).then(async ({ outcome, flow }) => {
      const isNew = isFirst && outcome.status === "connected";
      // The card and the expert's access are not this screen's: they are
      // settled even when it was left before the swap finished.
      if (isNew && flow?.cardKey && flow.rowKey) {
        markConnected(flow.cardKey, flow.rowKey, {
          id: outcome.credential.id,
          title: credentialLabel(outcome.credential),
        });
      }
      // Started from an expert's chat panel: the new account is for them.
      let grantError: string | null = null;
      const grant =
        isNew && flow && !flow.cardKey && flow.expertID
          ? { expertID: flow.expertID, credentialID: outcome.credential.id }
          : null;
      if (grant) {
        grantError = await allowForExpert(grant.expertID, [grant.credentialID]).then(
          () => null,
          allowErrorText,
        );
      }
      if (!isMounted) return;
      if (grant && grantError) {
        setSessionID(flow?.sessionID ?? null);
        setFailedGrant(grant);
        setError(grantError);
        return;
      }
      if (outcome.status === "failed" || outcome.status === "paywall") {
        setSessionID(flow?.sessionID ?? null);
        setError(outcome.status === "paywall" ? PAYWALL_MESSAGE : outcome.message);
        return;
      }
      leave(flow?.sessionID ?? null);
    });
    return () => {
      isMounted = false;
      isOnScreen.current = false;
    };
  });

  function goBack() {
    isOnScreen.current = false;
    leave(sessionID);
  }

  /** Gives the saved account to the expert again, without another sign-in. */
  async function retryGrant() {
    if (!failedGrant || isRetrying) return;
    setIsRetrying(true);
    try {
      await allowForExpert(failedGrant.expertID, [failedGrant.credentialID]);
      if (isOnScreen.current) leave(sessionID);
    } catch (caught) {
      setError(allowErrorText(caught));
    } finally {
      setIsRetrying(false);
    }
  }

  return { error, goBack, canRetry: failedGrant !== null, isRetrying, retryGrant };
}
