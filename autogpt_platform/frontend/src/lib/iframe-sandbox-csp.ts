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

// Sandboxed srcdoc iframes without `allow-same-origin` resolve `href="#id"` links
// against the parent's URL as base. The default click then either navigates the
// iframe to `<parent-url>#id` (reloading our app inside the iframe) or updates
// the parent window's hash — both of which break the artifact preview.
//
// This script stays inside the iframe document and handles in-page anchor
// navigation locally by scrolling to the element with the matching id.
export const FRAGMENT_LINK_INTERCEPTOR_SCRIPT = `<script>
(function() {
  if (document.__fragmentLinkInterceptor) return;
  function handler(e) {
    var t = e.target;
    if (!t || typeof t.closest !== 'function') return;
    var a = t.closest('a[href^="#"]');
    if (!a) return;
    var href = a.getAttribute('href');
    if (!href || href === '#') return;
    var id;
    try { id = decodeURIComponent(href.slice(1)); } catch (_) { id = href.slice(1); }
    if (!id) return;
    var target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    if (typeof target.scrollIntoView === 'function') {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
  document.__fragmentLinkInterceptor = handler;
  document.addEventListener('click', handler);
})();
</script>`;

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
