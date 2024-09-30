import { OAuthPopupResultMessage } from "@/components/integrations/credentials-input";
import { NextResponse } from "next/server";

// This route is intended to be used as the callback for integration OAuth flows,
// controlled by the CredentialsInput component. The CredentialsInput opens the login
// page in a pop-up window, which then redirects to this route to close the loop.
export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get("code");
  const state = searchParams.get("state");

  console.log("OAuth callback received:", { code, state });

  const message: OAuthPopupResultMessage =
    code && state
      ? { message_type: "oauth_popup_result", success: true, code, state }
      : {
          message_type: "oauth_popup_result",
          success: false,
          message: `Incomplete query: ${searchParams.toString()}`,
        };

  console.log("Sending message to opener:", message);

  // Return a response with the message as JSON and a script to close the window
  return new NextResponse(
    `
    <html>
      <body>
        <script>
          console.log("Callback page loaded, attempting to send message and close window");
          window.opener.postMessage(${JSON.stringify(message)}, "*");
          console.log("Message sent to opener");
          window.close();
          console.log("Window close attempted");
        </script>
      </body>
    </html>
    `,
    {
      headers: { "Content-Type": "text/html" },
    },
  );
}
