/**
 * Shared utility for OAuth popup flows with cross-origin support.
 *
 * Handles BroadcastChannel, postMessage, and localStorage polling
 * to reliably receive OAuth callback results even when COOP headers
 * sever the window.opener relationship.
 */

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes

export type OAuthPopupResult = {
  code: string;
  state: string;
};

export type OAuthPopupOptions = {
  /** State token to validate against incoming messages */
  stateToken: string;
  /**
   * Use BroadcastChannel + localStorage polling for cross-origin OAuth (MCP).
   * Standard OAuth only uses postMessage via window.opener.
   */
  useCrossOriginListeners?: boolean;
  /** BroadcastChannel name (default: "mcp_oauth") */
  broadcastChannelName?: string;
  /** localStorage key for cross-origin fallback (default: "mcp_oauth_result") */
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
): { promise: Promise<OAuthPopupResult>; cleanup: Cleanup } {
  const {
    stateToken,
    useCrossOriginListeners = false,
    broadcastChannelName = "mcp_oauth",
    localStorageKey = "mcp_oauth_result",
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

  if (popup && !popup.closed) {
    popup.location.href = loginUrl;
  } else {
    // Popup was blocked — open in new tab as fallback
    window.open(loginUrl, "_blank");
  }

  // Close popup on abort
  controller.signal.addEventListener("abort", () => {
    if (popup && !popup.closed) popup.close();
  });

  // Clear any stale localStorage entry
  if (useCrossOriginListeners) {
    try {
      localStorage.removeItem(localStorageKey);
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
          const stored = localStorage.getItem(localStorageKey);
          if (stored) {
            const data = JSON.parse(stored);
            localStorage.removeItem(localStorageKey);
            handleResult(data);
          }
        } catch {}
      }, 500);
      controller.signal.addEventListener("abort", () =>
        clearInterval(pollInterval),
      );
    }

    // Timeout
    const timeoutId = setTimeout(() => {
      if (!handled) {
        handled = true;
        reject(new Error("OAuth flow timed out"));
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
  };
}
