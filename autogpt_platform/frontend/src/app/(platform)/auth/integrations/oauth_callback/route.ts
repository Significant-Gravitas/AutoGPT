import { OAuthPopupResultMessage } from "@/components/renderers/input-renderer/fields/CredentialField/models/OAuthCredentialModal/useOAuthCredentialModal";
import { NextResponse } from "next/server";

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

  // Escape JSON to prevent XSS attacks via </script> injection
  const safeJson = JSON.stringify(message)
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e");

  // Return a response with the message as JSON and a script to close the window
  return new NextResponse(
    `<!DOCTYPE html>
<html>
  <body>
    <script>
      window.opener.postMessage(${safeJson}, '*');
      window.close();
    </script>
  </body>
</html>`,
    {
      headers: {
        "Content-Type": "text/html",
        "Content-Security-Policy":
          "default-src 'none'; script-src 'unsafe-inline'",
      },
    },
  );
}
