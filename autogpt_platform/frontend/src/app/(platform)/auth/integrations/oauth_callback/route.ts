import { OAuthPopupResultMessage } from "./types";
import { NextResponse } from "next/server";
import { microsoftCallbackMessage } from "@/lib/microsoft-setup";

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
const HTML_HEADERS = {
  "Content-Type": "text/html",
  "Cache-Control": "no-store",
  "Referrer-Policy": "no-referrer",
};

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Microsoft's organization-wide approval returns here with no code to
// exchange: it opened in its own tab, so all that's left is to say how it
// went. A refusal is marked admin_consent=True too, so the error decides.
function organizationApprovalPage(error: string | null, description: string) {
  const text = error
    ? `${microsoftCallbackMessage(error, description)} You can close this tab.`
    : 'Approved for your organization. You can close this tab and choose "Check again" in AutoGPT.';
  // Only our own messages and an AADSTS number get here, but the number is
  // read out of the URL.
  return `<!DOCTYPE html>
<html>
  <body style="font-family: system-ui, sans-serif; padding: 2rem;">
    <p>${escapeHtml(text)}</p>
  </body>
</html>`;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const providerError = searchParams.get("error");
  const providerDescription = searchParams.get("error_description") ?? "";
  if (searchParams.get("admin_consent") === "True") {
    return new NextResponse(
      organizationApprovalPage(providerError, providerDescription),
      { headers: HTML_HEADERS },
    );
  }
  const code = searchParams.get("code");
  const state = searchParams.get("state");

  const message: OAuthPopupResultMessage =
    code && state
      ? { message_type: "oauth_popup_result", success: true, code, state }
      : {
          message_type: "oauth_popup_result",
          success: false,
          // Echo back `state` when present so the originating page's
          // state-keyed listeners route the failure to the right flow
          // instead of timing out silently.
          ...(state ? { state } : {}),
          message:
            providerError && /AADSTS\d+/.test(providerDescription)
              ? microsoftCallbackMessage(providerError, providerDescription)
              : providerError
                ? "Sign-in was not completed. Return to the connection page to retry or open setup help."
                : "OAuth callback is missing its code or state",
        };

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
    { headers: HTML_HEADERS },
  );
}
