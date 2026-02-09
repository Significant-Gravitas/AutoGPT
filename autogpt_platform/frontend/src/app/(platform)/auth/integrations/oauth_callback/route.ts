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

  // Return a response with the message as JSON and a script to close the window
  // Use safeJsonStringify to prevent XSS by escaping <, >, and & characters
  return new NextResponse(
    `
    <html>
      <body>
        <script>
          window.opener.postMessage(${safeJsonStringify(message)});
          window.close();
        </script>
      </body>
    </html>
    `,
    {
      headers: { "Content-Type": "text/html" },
    },
  );
}
