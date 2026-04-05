// CSP meta tag injected into every artifact/HTML preview iframe. Blocks
// outbound fetch/XHR (connect-src 'none'), form submissions, and base-URI
// attacks. Keeps the Tailwind CDN whitelisted for script+style and allows
// HTTPS iframe embeds (legitimate chart/video embeds in dashboards).
//
// Used by:
//   - src/app/(platform)/copilot/components/ArtifactPanel/... (artifact preview panel)
//   - src/components/contextual/OutputRenderers/renderers/HTMLRenderer.tsx
export const ARTIFACT_IFRAME_CSP =
  "default-src 'self' data: blob: 'unsafe-inline' 'unsafe-eval';" +
  " script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://unpkg.com;" +
  " style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com;" +
  " img-src 'self' data: blob: https:;" +
  " font-src 'self' data: https://fonts.gstatic.com;" +
  " connect-src 'none';" +
  " form-action 'none';" +
  " frame-src https:;" +
  " object-src 'none';" +
  " base-uri 'none'";

// Pinned to a specific version to reduce exposure to unannounced upstream
// changes (SRI is not possible because the JIT runtime is generated on demand).
export const TAILWIND_CDN_URL = "https://cdn.tailwindcss.com/3.4.16";

// Builds a `<meta>` CSP tag ready to drop into an iframe srcdoc's <head>.
export function cspMetaTag(): string {
  return `<meta http-equiv="Content-Security-Policy" content="${ARTIFACT_IFRAME_CSP}">`;
}

// Wraps HTML content in a full document skeleton if it lacks a <head> tag,
// so the injected CSP meta lives inside <head> where browsers honor it
// (meta-CSP outside <head> is ignored per the HTML spec).
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
