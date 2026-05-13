import { OAuthPopupResultMessage } from "./types";
import { NextResponse } from "next/server";

/**
 * Safely encode a value as JSON for embedding in a script tag.
 * Escapes characters that could break out of the script context to prevent XSS.
 */
function safeJsonStringify(value: unknown): string {
  return JSON.stringify(value)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/&/g, "\\u0026");
}

// This route is intended to be used as the callback for integration OAuth flows,
// controlled by the CredentialsInput component. The CredentialsInput opens the login
// page in a pop-up window, which then redirects to this route to close the loop.
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const code = searchParams.get("code");
  const state = searchParams.get("state");

  console.debug("OAuth callback received:", { code, state });

  const message: OAuthPopupResultMessage =
    code && state
      ? { message_type: "oauth_popup_result", success: true, code, state }
      : {
          message_type: "oauth_popup_result",
          success: false,
          message: `Incomplete query: ${searchParams.toString()}`,
        };

  console.debug("Sending message to opener:", message);

  // Emit via three channels so the result reaches the originating page
  // regardless of how the OAuth flow was opened:
  //   1. BroadcastChannel — works across tabs/popups even when COOP headers
  //      have severed window.opener, and is the only channel that works for
  //      the popup-blocked → new-tab fallback path.
  //   2. window.opener.postMessage — fast path for same-origin popups.
  //   3. localStorage — most reliable cross-tab fallback when both above fail.
  //      Key is scoped by state token so concurrent flows don't clobber each
  //      other's slots. BroadcastChannel/postMessage are pub/sub-style so the
  //      listener's state-token filter already prevents cross-talk for them.
  return new NextResponse(
    `<!DOCTYPE html>
<html>
  <body>
    <script>
      (function() {
        var msg = ${safeJsonStringify(message)};
        var state = ${safeJsonStringify(state)};
        try {
          var bc = new BroadcastChannel("oauth_popup");
          bc.postMessage(msg);
          bc.close();
        } catch(e) {}
        try {
          if (window.opener && !window.opener.closed) {
            window.opener.postMessage(msg, window.location.origin);
          }
        } catch(e) {}
        try {
          if (state) {
            localStorage.setItem("oauth_popup_result_" + state, JSON.stringify(msg));
          }
        } catch(e) {}
        window.close();
      })();
    </script>
  </body>
</html>`,
    {
      headers: { "Content-Type": "text/html" },
    },
  );
}
