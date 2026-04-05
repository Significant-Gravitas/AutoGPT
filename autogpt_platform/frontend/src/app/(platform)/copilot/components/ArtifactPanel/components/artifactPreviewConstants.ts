// CDN used by AI-generated HTML/React artifacts for Tailwind classes.
// JIT runtime is generated server-side so SRI isn't available, but the
// version is pinned to reduce exposure to unannounced upstream changes.
export const TAILWIND_CDN_URL = "https://cdn.tailwindcss.com/3.4.16";

// CSP meta tag injected into every artifact iframe. Blocks outbound
// fetch/XHR (connect-src 'none'), form submissions, and base-URI attacks.
// Keeps the Tailwind CDN whitelisted for script+style and allows HTTPS
// iframe embeds (legitimate chart/video embeds in dashboards).
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
