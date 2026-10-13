export function mobileConsentPage(
  email: string,
  request: { challenge: string; state: string },
) {
  return new Response(
    `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in to AutoGPT</title></head>
<body>
<main><h1>Continue to AutoGPT</h1>
<p>Sign in to the AutoGPT app as <strong>${escapeHTML(email)}</strong>?</p>
<p>Only continue if you started this request in the AutoGPT app on this device.</p>
<form method="post" action="/api/auth/mobile/approve">
<input type="hidden" name="challenge" value="${request.challenge}">
<input type="hidden" name="state" value="${request.state}">
<button type="submit">Continue to AutoGPT</button>
</form></main></body></html>`,
    {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy":
          "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
        "X-Frame-Options": "DENY",
      },
    },
  );
}

function escapeHTML(value: string) {
  return value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character]!,
  );
}
