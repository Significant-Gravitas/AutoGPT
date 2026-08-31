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
  "Popup blocked — the sign-in window opened in a new tab instead. If you don't see it, allow popups for this site and retry.";
export const OAUTH_ERROR_POPUP_BLOCKED_NO_TAB =
  "Popup blocked — allow popups for this site and try again.";

export type OAuthPopupResult = {
  code: string;
  state: string;
};

export type OAuthPopupOptions = {
  /** State token to validate against incoming messages */
  stateToken: string;
  /**
   * A window pre-opened via {@link preOpenOAuthPopup} synchronously inside the
   * user-gesture handler, before any `await`. iOS Safari discards the gesture
   * context at the first async break, so a `window.open()` issued after an
   * `await` (e.g. after fetching the login URL) is always blocked — callers
   * that need to do async work before knowing the login URL must pre-open the
   * window and pass it here. `null` means the browser blocked even the
   * synchronous open; the new-tab fallback is used. When omitted, the window
   * is opened inline as before.
   */
  preOpenedWindow?: Window | null;
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
  /** Same-origin endpoint that cancels provider-side pending state on abort */
  cancelUrl?: string | null;
};

type Cleanup = {
  /** Abort the OAuth flow and close the popup */
  abort: (reason?: string) => void;
  /** The AbortController signal */
  signal: AbortSignal;
};

/**
 * Opens the blank OAuth window synchronously and returns its handle (null when
 * the browser blocks it). Must be called directly from a user-gesture handler
 * (click/tap), before any `await` — browsers, most strictly iOS Safari, block
 * `window.open()` once the gesture context is lost across an async break.
 */
export function preOpenOAuthPopup(): Window | null {
  const width = 500;
  const height = 700;
  const left = window.screenX + (window.outerWidth - width) / 2;
  const top = window.screenY + (window.outerHeight - height) / 2;
  return window.open(
    "about:blank",
    "_blank",
    `width=${width},height=${height},left=${left},top=${top},popup=true,scrollbars=yes`,
  );
}

/**
 * Opens an OAuth popup and sets up listeners for the callback result.
 *
 * Opens a blank popup synchronously (to avoid popup blockers) — or adopts the
 * caller's `preOpenedWindow` when given — then navigates it to the login URL.
 * Returns a promise that resolves with the OAuth code/state.
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
  /**
   * True iff the new-tab fallback was refused too — no window exists at all
   * and the promise has already rejected. Callers must NOT show the
   * popupBlocked "opened in a new tab" hint in this case; the rejection
   * already carries the correct allow-popups-and-retry message.
   */
  fallbackBlocked: boolean;
} {
  const {
    stateToken,
    preOpenedWindow,
    useCrossOriginListeners = false,
    broadcastChannelName = "oauth_popup",
    localStorageKey = "oauth_popup_result",
    acceptMessageTypes = ["oauth_popup_result", "mcp_oauth_result"],
    timeout = DEFAULT_TIMEOUT_MS,
    cancelUrl,
  } = options;

  const controller = new AbortController();

  // Adopt the caller's pre-opened window when given (required on iOS Safari,
  // where window.open only works synchronously inside the gesture handler);
  // otherwise open it now — still synchronous from the caller's perspective,
  // so the gesture context is intact when openOAuthPopup itself is called
  // straight from a click handler.
  const popup =
    preOpenedWindow !== undefined ? preOpenedWindow : preOpenOAuthPopup();
  let activeWindow = popup;

  let popupBlocked = false;
  let fallbackBlocked = false;
  if (popup && !popup.closed) {
    popup.location.href = loginUrl;
  } else {
    // Popup was blocked — open in new tab as fallback so the OAuth flow can
    // still complete via postMessage / BroadcastChannel / localStorage poll.
    popupBlocked = true;
    const fallback = window.open(loginUrl, "_blank");
    activeWindow = fallback;
    // The fallback open can be blocked too — iOS Safari blocks every
    // window.open() after an async break, so when the caller pre-opened
    // (and lost) the window, this open has no gesture context either. No
    // window exists at all then, so no result can ever arrive.
    fallbackBlocked = !fallback || fallback.closed;
  }

  let cancelRequested = false;
  function cancelServerAttempt() {
    if (!cancelUrl || cancelRequested) return;
    cancelRequested = true;
    try {
      const url = new URL(cancelUrl, window.location.origin);
      if (url.origin !== window.location.origin) return;
      void fetch(url.toString(), {
        method: "POST",
        credentials: "same-origin",
        keepalive: true,
      }).catch(() => {});
    } catch {}
  }

  // Close popup and cancel provider-side pending state on abort
  controller.signal.addEventListener("abort", () => {
    if (controller.signal.reason !== "completed") cancelServerAttempt();
    if (activeWindow && !activeWindow.closed) activeWindow.close();
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

    // Both the popup and the new-tab fallback were blocked — no window
    // exists, so no result can ever arrive. Fail fast instead of hanging
    // until the timeout while the UI claims a tab opened. Retrying from a
    // fresh tap (the connect button is re-enabled once this rejects) gets a
    // new user-gesture context.
    if (fallbackBlocked) {
      handled = true;
      reject(new Error(OAUTH_ERROR_POPUP_BLOCKED_NO_TAB));
      controller.abort("popup_blocked");
      return;
    }

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

    // Detect popup closed without completing sign-in — same-origin flows
    // only.
    //
    // Cross-origin flows (``useCrossOriginListeners``) must not poll
    // ``popup.closed``: providers that serve
    // ``Cross-Origin-Opener-Policy: same-origin`` (e.g. Stripe's MCP
    // authorize pages) trigger a browsing-context-group swap when the popup
    // navigates to them, severing this handle — ``closed`` then reports
    // ``true`` while the window is still open and the user is minutes away
    // from finishing sign-in.  Rejecting on it (with any grace period)
    // kills the flow and tears down the BroadcastChannel/localStorage
    // listeners before the callback page can post its result, surfacing a
    // bogus "Sign-in window was closed" error on a successful sign-in.
    // Those flows rely on the callback result, the outer ``timeout``, and
    // the caller's explicit abort instead — the same reasoning the
    // popup-blocked new-tab fallback above applies.
    //
    // Same-origin flows resolve via ``window.opener.postMessage``, which
    // the parent receives synchronously on the tick the popup posts before
    // closing — by the time ``closed`` is observed, any successful result
    // has already been handled, so rejecting immediately is safe and spares
    // the user a fake spinner after a manual close.
    if (popup && !popupBlocked && !useCrossOriginListeners) {
      const closedPollInterval = setInterval(() => {
        if (popup.closed && !handled) {
          clearInterval(closedPollInterval);
          handled = true;
          reject(new Error(OAUTH_ERROR_WINDOW_CLOSED));
          controller.abort("popup_closed");
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
    fallbackBlocked,
  };
}
