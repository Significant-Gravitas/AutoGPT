/**
 * Artifact iframe preview utilities.
 *
 * ===== WHY THERE IS NO CSP =====
 *
 * We intentionally do NOT inject a Content-Security-Policy meta tag into
 * artifact preview iframes. CSP was added and removed multiple times during
 * review — here's why it stays out:
 *
 * 1. `connect-src 'none'` breaks any AI-generated HTML that uses fetch(),
 *    XMLHttpRequest, or WebSocket — dashboards, API-driven charts, data
 *    loaders, etc. all silently fail.
 *
 * 2. The iframe sandbox (`sandbox="allow-scripts"` without `allow-same-origin`)
 *    already provides strong isolation: the iframe gets a unique opaque origin,
 *    so it cannot access the parent page's cookies, localStorage, DOM, or
 *    make same-origin requests to our backend.
 *
 * 3. The only data a script inside the iframe can exfiltrate is the artifact
 *    content itself — which the user already sees in the chat. There is no
 *    secret data available inside the sandbox.
 *
 * 4. Meta-CSP is unreliable in practice: if AI-generated HTML includes its
 *    own <meta http-equiv="Content-Security-Policy"> before ours, the browser
 *    honors the first one and ignores ours.
 *
 * DO NOT re-add CSP without addressing all four points above.
 * ================================================================
 */

// Pinned to a specific version to reduce exposure to unannounced upstream
// changes (SRI is not possible because the JIT runtime is generated on demand).
export const TAILWIND_CDN_URL = "https://cdn.tailwindcss.com/3.4.16";

/**
 * Inject content into the <head> of an HTML document string.
 * If the content has no <head> tag, wraps it in a full document skeleton.
 */
const HEAD_OPEN_RE = /<head(\s[^>]*)?>/i;
export function wrapWithHeadInjection(
  content: string,
  headInjection: string,
): string {
  if (HEAD_OPEN_RE.test(content)) {
    return content.replace(HEAD_OPEN_RE, (match) => `${match}${headInjection}`);
  }
  return `<!doctype html><html><head>${headInjection}</head><body>${content}</body></html>`;
}
