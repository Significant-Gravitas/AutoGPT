"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV1ListCredentialsQueryKey,
  getV1InitiateOauthFlow,
  postV1ExchangeOauthCodeForTokens,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  OAUTH_ERROR_POPUP_BLOCKED,
  openOAuthPopup,
  preOpenOAuthPopup,
} from "@/lib/oauth-popup";

import { getOAuthErrorMessage } from "./helpers";

interface Args {
  provider: string;
  onSuccess: () => void;
}

export function useOAuthConnect({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);
  const isPendingRef = useRef(false);
  const abortRef = useRef<(() => void) | null>(null);
  const isUnmountedRef = useRef(false);
  const preOpenedWindowRef = useRef<Window | null>(null);

  useEffect(() => {
    isUnmountedRef.current = false;
    return () => {
      isUnmountedRef.current = true;
      abortRef.current?.();
      // Close a window pre-opened by a flow still fetching the login URL —
      // its abort isn't registered yet, so the line above can't reach it.
      if (preOpenedWindowRef.current && !preOpenedWindowRef.current.closed) {
        preOpenedWindowRef.current.close();
      }
      preOpenedWindowRef.current = null;
    };
  }, []);

  async function connect() {
    // Ignore re-entry while a flow is in flight: a rapid double-click fires
    // again before the pending state re-renders the button, and a second
    // concurrent flow would overwrite preOpenedWindowRef/abortRef. State
    // isn't readable synchronously here, so track it in a ref.
    // Re-entrancy contract: BLOCK — this button has a single flow target, so
    // a second call is always a duplicate. Deliberately different from
    // useCredentialsInput.executeOAuthFlow, which SUPERSEDES (its next call
    // may target a different credential). Reconcile the two explicitly if
    // this lifecycle is ever extracted into a shared hook.
    if (isPendingRef.current) return;
    isPendingRef.current = true;
    setIsPending(true);
    // Open the sign-in window synchronously, before the first await — iOS
    // Safari discards the tap's user-gesture context at any async break and
    // then blocks every window.open(), including the new-tab fallback, so
    // nothing would open at all.
    const preOpenedWindow = preOpenOAuthPopup();
    preOpenedWindowRef.current = preOpenedWindow;
    try {
      const initiateResponse = await getV1InitiateOauthFlow(provider);
      // customMutator rejects non-2xx, so this branch is unreachable at
      // runtime — it exists only to narrow the discriminated union so the
      // 200-only LoginResponse shape is accessible below.
      if (initiateResponse.status !== 200) {
        throw new Error("Unexpected OAuth initiate response");
      }
      const { login_url, state_token } = initiateResponse.data;

      // Unmounted while the login URL was being fetched — the cleanup
      // already closed the window; don't adopt it.
      if (isUnmountedRef.current) return;

      const { promise, cleanup, popupBlocked, fallbackBlocked } =
        openOAuthPopup(login_url, {
          stateToken: state_token,
          preOpenedWindow,
          // BroadcastChannel + localStorage listeners are the only delivery
          // path when the flow runs in a tab without window.opener (the iOS
          // fallback) — the callback page already writes to both.
          useCrossOriginListeners: true,
          acceptMessageTypes: ["oauth_popup_result"],
        });
      // Ownership transferred — the helper closes the window on abort now.
      preOpenedWindowRef.current = null;
      abortRef.current = () => cleanup.abort("unmounted");

      // The browser blocked even the synchronous window.open but the new-tab
      // fallback opened — that tab is easy to miss, so tell the user. Skip
      // the hint when the fallback was blocked too: the promise has already
      // rejected and the catch below shows the correct allow-popups-and-retry
      // message — showing both would contradict each other.
      if (popupBlocked && !fallbackBlocked) {
        toast({
          title: "Popup blocked",
          description: OAUTH_ERROR_POPUP_BLOCKED,
        });
      }

      const { code, state } = await promise;
      abortRef.current = null;

      await postV1ExchangeOauthCodeForTokens(provider, {
        code,
        state_token: state,
      });

      toast({ title: "Connected via OAuth", variant: "success" });
      await queryClient.invalidateQueries({
        queryKey: getGetV1ListCredentialsQueryKey(),
      });
      onSuccess();
    } catch (error) {
      // Close the dangling about:blank window only while this flow still
      // owns it — i.e. the error occurred before openOAuthPopup adopted the
      // window. After handoff the ref is nulled and the helper closes the
      // window itself on abort, so it's no longer ours to close.
      if (preOpenedWindowRef.current === preOpenedWindow) {
        preOpenedWindowRef.current = null;
        if (preOpenedWindow && !preOpenedWindow.closed) {
          preOpenedWindow.close();
        }
      }
      if (isUnmountedRef.current) return;
      toast({
        title: "OAuth connection failed",
        description: getOAuthErrorMessage(error),
        variant: "destructive",
      });
    } finally {
      isPendingRef.current = false;
      setIsPending(false);
    }
  }

  return { connect, isPending };
}
