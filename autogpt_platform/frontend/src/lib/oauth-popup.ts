/**
 * Shared utility for OAuth popup flows with cross-origin support.
 *
 * Handles BroadcastChannel, postMessage, and localStorage polling
 * to reliably receive OAuth callback results even when COOP headers
 * sever the window.opener relationship.
 */

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

export const OAUTH_ERROR_WINDOW_CLOSED = "Sign-in window was closed";
export const OAUTH_ERROR_FLOW_CANCELED = "OAuth flow was canceled";
export const OAUTH_ERROR_FLOW_TIMED_OUT = "OAuth flow timed out";
export const OAUTH_ERROR_POPUP_BLOCKED =
  "Popup blocked — sign-in opened in a new tab. If you don't see it, allow popups for this site and retry.";

export type OAuthPopupResult = {
  code: string;
  state: string;
};

export type OAuthPopupOptions = {
  /** State token to validate against incoming messages */
  stateToken: string;
  /**
   * Use BroadcastChannel + localStorage polling on top of postMessage. Needed
   * whenever `window.opener` may not survive (cross-origin OAuth providers
   * stripped by COOP headers, popup-blocked → new-tab fallback, etc.).
   */
  useCrossOriginListeners?: boolean;
  /** BroadcastChannel name (default: "oauth_popup") */
  broadcastChannelName?: string;
  /** localStorage key for cross-origin fallback (default: "oauth_popup_result") */
  localStorageKey?: string;
  /** Message types to accept (default: ["oauth_popup_result", "mcp_oauth_result"]) */
  acceptMessageTypes?: string[];
  /** Timeout in ms (default: 5 minutes) */
  timeout?: number;
};

type Cleanup = {
  /** Abort the OAuth flow and close the popup */
  abort: (reason?: string) => void;
  /** The AbortController signal */
  signal: AbortSignal;
};

/**
 * Opens an OAuth popup and sets up listeners for the callback result.
 *
 * Opens a blank popup synchronously (to avoid popup blockers), then navigates
 * it to the login URL. Returns a promise that resolves with the OAuth code/state.
 *
 * @param loginUrl - The OAuth authorization URL to navigate to
 * @param options - Configuration for message handling
 * @returns Object with `promise` (resolves with OAuth result) and `abort` (cancels flow)
 */
export function openOAuthPopup(
  loginUrl: string,
  options: OAuthPopupOptions,
): {
  promise: Promise<OAuthPopupResult>;
  cleanup: Cleanup;
  /**
   * True iff the browser refused the popup and we fell back to opening the
   * login URL in a new tab. Callers should surface a hint to the user (the
   * tab can be easy to miss) and offer a retry path.
   */
  popupBlocked: boolean;
} {
  const {
    stateToken,
    useCrossOriginListeners = false,
    broadcastChannelName = "oauth_popup",
    localStorageKey = "oauth_popup_result",
    acceptMessageTypes = ["oauth_popup_result", "mcp_oauth_result"],
    timeout = DEFAULT_TIMEOUT_MS,
  } = options;

  const controller = new AbortController();

  // Open popup synchronously (before any async work) to avoid browser popup blockers
  const width = 500;
  const height = 700;
  const left = window.screenX + (window.outerWidth - width) / 2;
  const top = window.screenY + (window.outerHeight - height) / 2;
  const popup = window.open(
    "about:blank",
    "_blank",
    `width=${width},height=${height},left=${left},top=${top},popup=true,scrollbars=yes`,
  );

  let popupBlocked = false;
  if (popup && !popup.closed) {
    popup.location.href = loginUrl;
  } else {
    // Popup was blocked — open in new tab as fallback so the OAuth flow can
    // still complete via postMessage / BroadcastChannel / localStorage poll.
    popupBlocked = true;
    window.open(loginUrl, "_blank");
  }

  // Close popup on abort
  controller.signal.addEventListener("abort", () => {
    if (popup && !popup.closed) popup.close();
  });

  // Scope the localStorage key by stateToken so concurrent OAuth flows do
  // not race for a single shared slot. Each flow only reads/writes its own
  // key, so a poller cannot destructively consume a result intended for a
  // different flow. BroadcastChannel is pub/sub so it doesn't need scoping.
  const scopedLocalStorageKey = `${localStorageKey}_${stateToken}`;

  // Clear any stale localStorage entry for this specific flow only.
  if (useCrossOriginListeners) {
    try {
      localStorage.removeItem(scopedLocalStorageKey);
    } catch {}
  }

  const promise = new Promise<OAuthPopupResult>((resolve, reject) => {
    let handled = false;

    const handleResult = (data: any) => {
      if (handled) return; // Prevent double-handling

      // Validate message type
      const messageType = data?.message_type ?? data?.type;
      if (!messageType || !acceptMessageTypes.includes(messageType)) return;

      // Validate state token
      if (data.state !== stateToken) {
        // State mismatch — this message is for a different listener. Ignore silently.
        return;
      }

      handled = true;

      if (!data.success) {
        reject(new Error(data.message || "OAuth authentication failed"));
      } else {
        resolve({ code: data.code, state: data.state });
      }

      controller.abort("completed");
    };

    // Listener: postMessage (works for same-origin popups)
    window.addEventListener(
      "message",
      (event: MessageEvent) => {
        if (typeof event.data === "object") {
          handleResult(event.data);
        }
      },
      { signal: controller.signal },
    );

    // Cross-origin listeners for MCP OAuth
    if (useCrossOriginListeners) {
      // Listener: BroadcastChannel (works across tabs/popups without opener)
      try {
        const bc = new BroadcastChannel(broadcastChannelName);
        bc.onmessage = (event) => handleResult(event.data);
        controller.signal.addEventListener("abort", () => bc.close());
      } catch {}

      // Listener: localStorage polling (most reliable cross-tab fallback)
      const pollInterval = setInterval(() => {
        try {
          const stored = localStorage.getItem(scopedLocalStorageKey);
          if (stored) {
            const data = JSON.parse(stored);
            localStorage.removeItem(scopedLocalStorageKey);
            handleResult(data);
          }
        } catch {}
      }, 500);
      controller.signal.addEventListener("abort", () =>
        clearInterval(pollInterval),
      );
    }

    // Detect popup closed without completing sign-in.
    //
    // Three timeouts apply to the OAuth flow, only the outermost bounds
    // user time:
    //   1. ``timeout`` (default 5 min) — overall deadline; rejects with
    //      OAUTH_ERROR_FLOW_TIMED_OUT if the user never finishes signing
    //      in.  This is the only timeout that limits how long the user
    //      has to log in.
    //   2. 500 ms polling on ``popup.closed`` — only fires after the
    //      popup window actually goes away (user closed it OR callback
    //      page self-closed).  Doesn't run while the popup is open.
    //   3. POPUP_CLOSE_GRACE_MS (3000 ms) — only starts after step 2
    //      observes a closed popup; gives in-flight result messages a
    //      chance to land before we declare failure.
    //
    // Why the grace at all?  The callback page (see
    // ``frontend/src/app/(platform)/auth/integrations/mcp_callback/route.ts``)
    // does:
    //   bc.postMessage(...); localStorage.setItem(...); setTimeout(close, 1500)
    // BroadcastChannel delivery is async across-origin; the parent's
    // localStorage poll fires every 500 ms.  Without a grace window the
    // ``popup.closed`` rejection can win the race against a successful
    // result that's a few hundred ms behind, surfacing the bogus
    // "Sign-in window was closed" error John screenshotted.
    //
    // On detected close we ALSO do one synchronous final localStorage
    // read — covers the case where the BroadcastChannel listener never
    // fired (storage-partitioning / BCG isolation) and the poll tick
    // hasn't run yet.  The grace then handles any remaining
    // post-message latency.
    //
    // The setTimeout lives in a plain JS closure, not React state — it
    // does NOT stack across re-renders, and the AbortController cleanup
    // tears it down if the caller aborts before grace expires.
    // Skip the close-poll entirely when the popup was blocked.  The
    // ``window.open("about:blank", ...)`` reference can be non-null but
    // already-closed in that branch (we fell back to a separate new-tab
    // ``window.open(loginUrl, "_blank")`` whose handle we deliberately
    // don't keep — the new tab is the user's primary surface and
    // pre-emptively rejecting on its ``closed`` state would short-circuit
    // a successful sign-in arriving via BroadcastChannel/localStorage.
    if (popup && !popupBlocked) {
      // Grace window only applies to cross-origin flows.  Same-origin
      // OAuth resolves via ``window.opener.postMessage`` which the
      // parent receives synchronously on the same event-loop tick the
      // popup posts it — there's no async delivery race to wait for,
      // and adding 3 s of fake spinner after a manual close hurts UX.
      const POPUP_CLOSE_GRACE_MS = useCrossOriginListeners ? 3000 : 0;
      const finalLocalStorageCheck = () => {
        if (!useCrossOriginListeners || handled) return;
        try {
          const stored = localStorage.getItem(scopedLocalStorageKey);
          if (stored) {
            const data = JSON.parse(stored);
            localStorage.removeItem(scopedLocalStorageKey);
            handleResult(data);
          }
        } catch {}
      };

      const closedPollInterval = setInterval(() => {
        if (popup.closed && !handled) {
          clearInterval(closedPollInterval);
          finalLocalStorageCheck();
          if (handled) return;
          if (POPUP_CLOSE_GRACE_MS === 0) {
            // Same-origin path: the ``window.addEventListener("message", …)``
            // at line 149 fires synchronously when the popup posts before
            // closing, so any successful result has already been handled
            // by the time ``popup.closed`` flips.  Reject immediately —
            // no async-delivery race to wait for.
            handled = true;
            reject(new Error(OAUTH_ERROR_WINDOW_CLOSED));
            controller.abort("popup_closed");
            return;
          }
          const graceTimeout = setTimeout(() => {
            if (handled) return;
            finalLocalStorageCheck();
            if (handled) return;
            handled = true;
            reject(new Error(OAUTH_ERROR_WINDOW_CLOSED));
            controller.abort("popup_closed");
          }, POPUP_CLOSE_GRACE_MS);
          controller.signal.addEventListener("abort", () =>
            clearTimeout(graceTimeout),
          );
        }
      }, 500);
      controller.signal.addEventListener("abort", () =>
        clearInterval(closedPollInterval),
      );
    }

    // Reject on abort (e.g. from cancel button in the waiting modal)
    controller.signal.addEventListener("abort", () => {
      if (!handled) {
        handled = true;
        reject(new Error(OAUTH_ERROR_FLOW_CANCELED));
      }
    });

    // Timeout
    const timeoutId = setTimeout(() => {
      if (!handled) {
        handled = true;
        reject(new Error(OAUTH_ERROR_FLOW_TIMED_OUT));
        controller.abort("timeout");
      }
    }, timeout);
    controller.signal.addEventListener("abort", () => clearTimeout(timeoutId));
  });

  return {
    promise,
    cleanup: {
      abort: (reason?: string) => controller.abort(reason || "canceled"),
      signal: controller.signal,
    },
    popupBlocked,
  };
}
