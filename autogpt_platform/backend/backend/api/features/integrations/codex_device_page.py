import html


def render_device_login_page(nonce: str | None = None) -> str:
    page = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="referrer" content="no-referrer">
    <title>Connect ChatGPT for Codex</title>
    <style{{NONCE}}>
      :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui; }
      body { margin: 0; min-height: 100vh; display: grid; place-items: center;
        background: #f7f7f8; color: #202123; }
      main { width: min(420px, calc(100vw - 48px)); padding: 32px;
        border: 1px solid #dedee3; border-radius: 16px; background: white;
        box-shadow: 0 16px 48px rgb(0 0 0 / 8%); }
      h1 { margin: 0 0 10px; font-size: 24px; }
      p { margin: 0 0 20px; color: #5f6068; line-height: 1.5; }
      code { display: block; margin: 16px 0; padding: 16px; border-radius: 10px;
        background: #f0f0f2; font-size: 24px; font-weight: 650;
        letter-spacing: .12em; text-align: center; color: #202123; }
      .actions { display: grid; gap: 10px; }
      a, button { min-height: 44px; border-radius: 10px; border: 1px solid #202123;
        padding: 0 16px; display: inline-flex; align-items: center;
        justify-content: center; font: inherit; font-weight: 600; cursor: pointer; }
      a { background: #202123; color: white; text-decoration: none; }
      button { background: white; color: #202123; }
      #status { margin-top: 18px; font-size: 14px; color: #5f6068; }
      #error { color: #b42318; }
    </style>
  </head>
  <body>
    <main>
      <h1>Connect ChatGPT for Codex</h1>
      <p>Open the ChatGPT verification page, enter this one-time code, then return here.</p>
      <code id="user-code">Loading…</code>
      <div class="actions">
        <a id="verification-link" target="_blank" rel="noopener noreferrer">Open ChatGPT sign-in</a>
        <button id="copy-code" type="button">Copy code</button>
      </div>
      <div id="status" role="status">Waiting for sign-in…</div>
      <div id="error" role="alert"></div>
    </main>
    <script{{NONCE}}>
      (function () {
        let finished = false;
        const fragment = new URLSearchParams(window.location.hash.slice(1));
        const state = fragment.get("state");
        const userCode = fragment.get("user_code");
        const rawVerificationUrl = fragment.get("verification_url");
        const loginID = decodeURIComponent(window.location.pathname.split("/").pop());
        window.history.replaceState(null, "", window.location.pathname);

        const codeElement = document.getElementById("user-code");
        const linkElement = document.getElementById("verification-link");
        const statusElement = document.getElementById("status");
        const errorElement = document.getElementById("error");
        const copyButton = document.getElementById("copy-code");

        function fail(message) {
          statusElement.textContent = "";
          errorElement.textContent = message;
        }

        let verificationUrl;
        try {
          verificationUrl = new URL(rawVerificationUrl);
          if (verificationUrl.protocol !== "https:") throw new Error();
        } catch (_) {
          fail("The ChatGPT verification link is invalid. Close this window and try again.");
          return;
        }
        if (!state || !userCode || !loginID) {
          fail("This sign-in attempt is incomplete. Close this window and try again.");
          return;
        }

        codeElement.textContent = userCode;
        linkElement.href = verificationUrl.toString();
        copyButton.addEventListener("click", async function () {
          try {
            await navigator.clipboard.writeText(userCode);
            copyButton.textContent = "Copied";
          } catch (_) {
            fail("Could not copy automatically. Select the code above instead.");
          }
        });

        async function poll() {
          try {
            const response = await fetch(window.location.pathname + "/status", {
              cache: "no-store",
              credentials: "same-origin",
              headers: { Accept: "application/json" },
            });
            if (!response.ok) throw new Error();
            const result = await response.json();
            if (result.status === "completed") {
              finished = true;
              statusElement.textContent = "Connected. Returning to AutoGPT…";
              const query = new URLSearchParams({ code: loginID, state: state });
              window.location.replace("/auth/integrations/oauth_callback?" + query);
              return;
            }
            if (result.status === "failed" || result.status === "canceled") {
              finished = true;
              fail(result.error || "ChatGPT sign-in failed. Close this window and try again.");
              return;
            }
          } catch (_) {
            statusElement.textContent = "Reconnecting to AutoGPT…";
          }
          window.setTimeout(poll, 1000);
        }

        window.addEventListener("pagehide", function () {
          if (!finished) {
            navigator.sendBeacon(window.location.pathname + "/cancel");
          }
        });
        poll();
      })();
    </script>
  </body>
</html>"""
    nonce_attribute = f' nonce="{html.escape(nonce, quote=True)}"' if nonce else ""
    return page.replace("{{NONCE}}", nonce_attribute)
