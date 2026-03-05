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

// MCP-specific OAuth callback route.
//
// Unlike the generic oauth_callback which relies on window.opener.postMessage,
// this route uses BroadcastChannel as the PRIMARY communication method.
// This is critical because cross-origin OAuth flows (e.g. Sentry → localhost)
// often lose window.opener due to COOP (Cross-Origin-Opener-Policy) headers.
//
// BroadcastChannel works across all same-origin tabs/popups regardless of opener.
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const code = searchParams.get("code");
  const state = searchParams.get("state");

  const success = Boolean(code && state);
  const message = success
    ? { success: true, code, state }
    : {
        success: false,
        message: `Missing parameters: ${searchParams.toString()}`,
      };

  return new NextResponse(
    `<!DOCTYPE html>
<html>
  <head><title>MCP Sign-in</title></head>
  <body style="font-family: system-ui, -apple-system, sans-serif; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; background: #f9fafb;">
    <div style="text-align: center; max-width: 400px; padding: 2rem;">
      <div id="spinner" style="margin: 0 auto 1rem; width: 32px; height: 32px; border: 3px solid #e5e7eb; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.8s linear infinite;"></div>
      <p id="status" style="color: #374151; font-size: 16px;">Completing sign-in...</p>
    </div>
    <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
    <script>
      (function() {
        var msg = ${safeJsonStringify(message)};
        var sent = false;

        // Method 1: BroadcastChannel (reliable across tabs/popups, no opener needed)
        try {
          var bc = new BroadcastChannel("mcp_oauth");
          bc.postMessage({ type: "mcp_oauth_result", success: msg.success, code: msg.code, state: msg.state, message: msg.message });
          bc.close();
          sent = true;
        } catch(e) { /* BroadcastChannel not supported */ }

        // Method 2: window.opener.postMessage (fallback for same-origin popups)
        try {
          if (window.opener && !window.opener.closed) {
            window.opener.postMessage(
              { message_type: "mcp_oauth_result", success: msg.success, code: msg.code, state: msg.state, message: msg.message },
              window.location.origin
            );
            sent = true;
          }
        } catch(e) { /* opener not available (COOP) */ }

        // Method 3: localStorage (most reliable cross-tab fallback)
        try {
          localStorage.setItem("mcp_oauth_result", JSON.stringify(msg));
          sent = true;
        } catch(e) { /* localStorage not available */ }

        var statusEl = document.getElementById("status");
        var spinnerEl = document.getElementById("spinner");
        spinnerEl.style.display = "none";

        if (msg.success && sent) {
          statusEl.textContent = "Sign-in complete! This window will close.";
          statusEl.style.color = "#059669";
          setTimeout(function() { window.close(); }, 1500);
        } else if (msg.success) {
          statusEl.textContent = "Sign-in successful! You can close this tab and return to the builder.";
          statusEl.style.color = "#059669";
        } else {
          statusEl.textContent = "Sign-in failed: " + (msg.message || "Unknown error");
          statusEl.style.color = "#dc2626";
        }
      })();
    </script>
  </body>
</html>`,
    { headers: { "Content-Type": "text/html" } },
  );
}
