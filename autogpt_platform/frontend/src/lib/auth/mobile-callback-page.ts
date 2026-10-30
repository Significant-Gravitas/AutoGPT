import { createHash } from "node:crypto";

const STYLE = `:root{color-scheme:light dark}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#f5f3fc;color:#171925;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:28rem;margin:1.5rem;padding:2rem;background:#fff;border-radius:1rem;box-shadow:0 1px 3px rgba(23,25,37,.08)}
h1{margin:0 0 .75rem;font-size:1.375rem;line-height:1.3}
p{margin:0 0 1.5rem;color:#575072}
a{color:#4d36ad;font-weight:600}
@media (prefers-color-scheme:dark){body{background:#0f0e17;color:#f4f3fa}main{background:#1b1a26}p{color:#b7b3cc}a{color:#b3a4ff}}`;
const STYLE_HASH = createHash("sha256").update(STYLE).digest("base64");

// Shown only when no verified app took the sign-in callback. It never reads
// the request, so the one-time code in the URL cannot end up on the page.
export function mobileCallbackPage() {
  return new Response(
    `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'sha256-${STYLE_HASH}'; base-uri 'none'; form-action 'none'">
<meta name="referrer" content="no-referrer">
<meta name="robots" content="noindex">
<title>Continue in the AutoGPT app</title>
<style>${STYLE}</style></head>
<body><main>
<h1>Continue in the AutoGPT app</h1>
<p>This page finishes signing in to the AutoGPT app on your phone. If the app didn't open, go back to it and tap Sign in again.</p>
<a href="/">Use AutoGPT in your browser instead</a>
</main></body></html>`,
    {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Robots-Tag": "noindex",
      },
    },
  );
}
